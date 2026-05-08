"""
Debug script: isolate why rotation + MXFP4 accuracy degrades in auto-round.

This script runs several diagnostic checks step-by-step:
  1. Baseline FP16 — sanity check
  2. Rotation FP16 — verify rotation alone is lossless
  3. MXFP4 only (model loaded from string by AutoRound)
  4. MXFP4 only (model pre-loaded as object, same as rotation flow)
  5. Rotation + MXFP4 (the failing scenario)
  6. Rotation + MXFP4 with model moved to CPU before quantizing

By comparing #3 vs #4, we isolate whether passing a model object (vs string)
itself causes issues. By comparing #5 vs #6, we test whether the model's
initial device placement matters.

Usage:
    python debug_rotation_mxfp4.py --device cuda:0 --limit 200
"""

import argparse
import gc
import logging
import sys
import time

sys.path.insert(0, "/data/lkk/quarot/auto-round")

logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig, SpinQuantPreprocessor


def evaluate_model(model, tokenizer, tasks, batch_size=8, limit=None, device="cuda:0"):
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size, limit=limit, device=device)
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc_norm,none") or task_results.get("acc,none")
        if acc is not None:
            metrics[task_name] = acc
    return metrics


def apply_rotation(model, rotation_size=None):
    """Apply R1+R2 rotation (QuaRot mode)."""
    config = SpinQuantConfig(
        r1=True, r2=True, r3=False, r4=False,
        rotation_size=rotation_size,
        trainable_rotation=False, trainable_smooth=False,
        fuse_rmsnorm=True, untie_embeddings=True,
    )
    preprocessor = SpinQuantPreprocessor(model, config)
    preprocessor.preprocess(dataloader=None)
    return model


def check_model_devices(model, label=""):
    """Print device distribution of model parameters."""
    device_counts = {}
    for name, param in model.named_parameters():
        d = str(param.device)
        device_counts[d] = device_counts.get(d, 0) + 1
    device_counts_buf = {}
    for name, buf in model.named_buffers():
        d = str(buf.device)
        device_counts_buf[d] = device_counts_buf.get(d, 0) + 1
    print(f"  [{label}] Param devices: {device_counts}, Buffer devices: {device_counts_buf}")
    return device_counts


def check_norm_weights(model, label=""):
    """Check RMSNorm weight values (should be all 1.0 after fusion)."""
    for name, module in model.named_modules():
        if "layernorm" in name.lower() or "norm" in name.lower():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                is_ones = torch.allclose(w, torch.ones_like(w), atol=1e-4)
                if not is_ones:
                    print(f"  [{label}] {name}: weight NOT all-ones (mean={w.mean():.4f}, std={w.std():.4f}, device={w.device})")
                    return
    print(f"  [{label}] All norm weights are ones (fused)")


def quantize_mxfp4_from_model(model, tokenizer, device, scheme="MXFP4"):
    """Quantize a pre-loaded model object."""
    ar = AutoRound(model, tokenizer=tokenizer, scheme=scheme, iters=0,
                   nsamples=128, seqlen=512, device_map=device)
    ar.quantize()
    qmodel = ar.model
    qmodel.eval()
    return qmodel, ar.tokenizer


def quantize_mxfp4_from_string(model_name, device, scheme="MXFP4"):
    """Quantize by letting AutoRound load the model from string."""
    ar = AutoRound(model_name, scheme=scheme, iters=0,
                   nsamples=128, seqlen=512, device_map=device)
    ar.quantize()
    qmodel = ar.model
    qmodel.eval()
    return qmodel, ar.tokenizer


def main():
    parser = argparse.ArgumentParser(description="Debug rotation + MXFP4 accuracy")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tasks", type=str, default="piqa,hellaswag")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rotation-size", type=int, default=128)
    parser.add_argument("--scheme", type=str, default="MXFP4")
    parser.add_argument(
        "--steps", type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated step numbers to run (1-6)"
    )
    args = parser.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(","))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    all_results = {}

    print("=" * 80)
    print("DEBUG: Rotation + MXFP4 Accuracy Investigation")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Limit: {args.limit}")
    print(f"  Device: {args.device}")
    print(f"  Scheme: {args.scheme}")
    print(f"  Rotation size: {args.rotation_size}")
    print(f"  Steps: {sorted(steps)}")
    print("=" * 80)

    # ---- Step 1: Baseline FP16 ----
    if 1 in steps:
        print(f"\n{'='*60}")
        print("  STEP 1: Baseline FP16")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()
        check_model_devices(model, "step1")
        metrics = evaluate_model(model, tokenizer, args.tasks, args.batch_size, args.limit, args.device)
        all_results["1_baseline_fp16"] = metrics
        print(f"  Results: {metrics}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Step 2: Rotation FP16 (verify lossless) ----
    if 2 in steps:
        print(f"\n{'='*60}")
        print("  STEP 2: Rotation FP16 (R1+R2, should match baseline)")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()
        apply_rotation(model, rotation_size=args.rotation_size)
        check_model_devices(model, "step2-after-rotation")
        check_norm_weights(model, "step2")
        metrics = evaluate_model(model, tokenizer, args.tasks, args.batch_size, args.limit, args.device)
        all_results["2_rotation_fp16"] = metrics
        print(f"  Results: {metrics}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Step 3: MXFP4 only (AutoRound loads from string) ----
    if 3 in steps:
        print(f"\n{'='*60}")
        print("  STEP 3: MXFP4 only (model loaded from STRING by AutoRound)")
        print(f"{'='*60}")
        model, tok = quantize_mxfp4_from_string(args.model, args.device, scheme=args.scheme)
        check_model_devices(model, "step3-after-quant")
        model.to(args.device)
        check_model_devices(model, "step3-after-to-device")
        metrics = evaluate_model(model, tok, args.tasks, args.batch_size, args.limit, args.device)
        all_results["3_mxfp4_from_string"] = metrics
        print(f"  Results: {metrics}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Step 4: MXFP4 only (pre-loaded model OBJECT, fp16, on GPU) ----
    if 4 in steps:
        print(f"\n{'='*60}")
        print("  STEP 4: MXFP4 only (pre-loaded model OBJECT, fp16, on GPU)")
        print(f"  This tests whether passing a model object causes issues")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()
        check_model_devices(model, "step4-before-quant")
        model, _ = quantize_mxfp4_from_model(model, tokenizer, args.device, scheme=args.scheme)
        check_model_devices(model, "step4-after-quant")
        model.to(args.device)
        check_model_devices(model, "step4-after-to-device")
        metrics = evaluate_model(model, tokenizer, args.tasks, args.batch_size, args.limit, args.device)
        all_results["4_mxfp4_from_object"] = metrics
        print(f"  Results: {metrics}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Step 5: Rotation + MXFP4 (current broken flow) ----
    if 5 in steps:
        print(f"\n{'='*60}")
        print("  STEP 5: Rotation + MXFP4 (current flow, model on GPU)")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()
        apply_rotation(model, rotation_size=args.rotation_size)
        check_model_devices(model, "step5-after-rotation")
        check_norm_weights(model, "step5-after-rotation")
        model, _ = quantize_mxfp4_from_model(model, tokenizer, args.device, scheme=args.scheme)
        check_model_devices(model, "step5-after-quant")
        model.to(args.device)
        check_model_devices(model, "step5-after-to-device")
        metrics = evaluate_model(model, tokenizer, args.tasks, args.batch_size, args.limit, args.device)
        all_results["5_rotation_mxfp4_gpu"] = metrics
        print(f"  Results: {metrics}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Step 6: Rotation + MXFP4 (move to CPU before quantizing) ----
    if 6 in steps:
        print(f"\n{'='*60}")
        print("  STEP 6: Rotation + MXFP4 (move model to CPU before quantizing)")
        print(f"  This matches AutoRound's normal flow where model starts on CPU")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(args.device).eval()
        apply_rotation(model, rotation_size=args.rotation_size)
        check_model_devices(model, "step6-after-rotation")
        check_norm_weights(model, "step6-after-rotation")
        # KEY DIFFERENCE: move to CPU before quantizing
        model.to("cpu")
        check_model_devices(model, "step6-after-to-cpu")
        model, _ = quantize_mxfp4_from_model(model, tokenizer, args.device, scheme=args.scheme)
        check_model_devices(model, "step6-after-quant")
        model.to(args.device)
        check_model_devices(model, "step6-after-to-device")
        metrics = evaluate_model(model, tokenizer, args.tasks, args.batch_size, args.limit, args.device)
        all_results["6_rotation_mxfp4_cpu"] = metrics
        print(f"  Results: {metrics}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    tasks = sorted(set(t for m in all_results.values() for t in m.keys()))
    configs = list(all_results.keys())

    header = f"{'Step':<30}"
    for task in tasks:
        header += f" | {task:<12}"
    print(header)
    print("-" * len(header))

    for config in configs:
        row = f"{config:<30}"
        for task in tasks:
            val = all_results.get(config, {}).get(task)
            if val is not None:
                row += f" | {val:<12.4f}"
            else:
                row += f" | {'N/A':<12}"
        print(row)

    print("=" * 80)
    print("\nKey comparisons:")
    if "3_mxfp4_from_string" in all_results and "4_mxfp4_from_object" in all_results:
        for task in tasks:
            a = all_results["3_mxfp4_from_string"].get(task, 0)
            b = all_results["4_mxfp4_from_object"].get(task, 0)
            print(f"  String vs Object (no rotation): {task} Δ = {b-a:+.4f}")
    if "5_rotation_mxfp4_gpu" in all_results and "6_rotation_mxfp4_cpu" in all_results:
        for task in tasks:
            a = all_results["5_rotation_mxfp4_gpu"].get(task, 0)
            b = all_results["6_rotation_mxfp4_cpu"].get(task, 0)
            print(f"  GPU vs CPU before quant (with rotation): {task} Δ = {b-a:+.4f}")
    if "4_mxfp4_from_object" in all_results and "5_rotation_mxfp4_gpu" in all_results:
        for task in tasks:
            a = all_results["4_mxfp4_from_object"].get(task, 0)
            b = all_results["5_rotation_mxfp4_gpu"].get(task, 0)
            print(f"  Object-no-rotation vs Object-with-rotation: {task} Δ = {b-a:+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
