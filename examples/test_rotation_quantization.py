"""
Validate Rotation + Quantization on Qwen3-0.6B using lm_eval.

Compares:
  1. Baseline FP16 (no quant, no rotation)
  2. RTN W4A16 only (quantization without rotation)
  3. Rotation(R1+R2+R3+R4) + RTN W4A16 (rotation then quantization)

The goal is to demonstrate that applying rotation BEFORE quantization
reduces quantization-induced accuracy loss by redistributing outliers.

Usage:
    python test_rotation_quantization.py [--tasks piqa,hellaswag] [--limit 200]
    python test_rotation_quantization.py --levels baseline_fp16,rtn_only,rotation_rtn --limit 500
"""

import argparse
import gc
import logging
import sys
import time

sys.path.insert(0, "/data/lkk/quarot/auto-round")

# Enable SpinQuant logging so the transformation summary table is visible
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("auto_round.spinquant").setLevel(logging.INFO)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
)


def evaluate_model(model, tokenizer, tasks, batch_size=8, limit=None, device="cuda:0"):
    """Evaluate model using lm_eval framework."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)

    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = tasks

    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        device=device,
    )
    return results


def extract_metrics(results):
    """Extract key metrics from lm_eval results."""
    metrics = {}
    if "results" not in results:
        return metrics
    for task_name, task_results in results["results"].items():
        acc_norm = task_results.get("acc_norm,none") or task_results.get("acc,none")
        if acc_norm is not None:
            metrics[task_name] = acc_norm
    return metrics


def apply_rotation(model, r1=True, r2=True, r3=True, r4=True):
    """Apply SpinQuant/QuaRot rotation to a model (in-place)."""
    config = SpinQuantConfig(
        r1=r1, r2=r2, r3=r3, r4=r4,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=True,
        untie_embeddings=True,
    )
    preprocessor = SpinQuantPreprocessor(model, config)
    preprocessor.preprocess(dataloader=None)
    return model


def quantize_rtn_w4a16(model_name, device, nsamples=128, seqlen=512):
    """Quantize model using auto-round RTN (iters=0) with W4A16 scheme.

    Returns the quantized model ready for evaluation.
    """
    ar = AutoRound(
        model_name,
        scheme="W4A16",
        iters=0,  # RTN mode — no optimization iterations
        nsamples=nsamples,
        seqlen=seqlen,
        device_map=device,
    )
    ar.quantize()
    # Get the quantized model
    model = ar.model
    model.eval()
    return model, ar.tokenizer


def quantize_rotated_model_rtn_w4a16(model, tokenizer, device, nsamples=128, seqlen=512):
    """Quantize a pre-rotated model using auto-round RTN (iters=0) with W4A16.

    Takes an already-rotated model and applies W4A16 quantization.
    """
    ar = AutoRound(
        model,
        tokenizer=tokenizer,
        scheme="W4A16",
        iters=0,  # RTN mode
        nsamples=nsamples,
        seqlen=seqlen,
        device_map=device,
    )
    ar.quantize()
    model = ar.model
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Rotation + Quantization (RTN W4A16) on Qwen3-0.6B"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--tasks", type=str, default="piqa,hellaswag", help="Comma-separated lm_eval tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (None=full)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for evaluation")
    parser.add_argument(
        "--levels", type=str, default="baseline_fp16,rtn_only,rotation_rtn",
        help="Comma-separated levels: baseline_fp16, rtn_only, rotation_rtn, rotation_fp16"
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Calibration samples for quantization")
    parser.add_argument("--seqlen", type=int, default=512, help="Calibration sequence length")
    parser.add_argument("--r1", action="store_true", default=True, help="Enable R1 rotation (default: True)")
    parser.add_argument("--no-r1", action="store_false", dest="r1", help="Disable R1 rotation")
    parser.add_argument("--r2", action="store_true", default=True, help="Enable R2 rotation (default: True)")
    parser.add_argument("--no-r2", action="store_false", dest="r2", help="Disable R2 rotation")
    parser.add_argument("--r3", action="store_true", default=True, help="Enable R3 rotation (default: True)")
    parser.add_argument("--no-r3", action="store_false", dest="r3", help="Disable R3 rotation")
    parser.add_argument("--r4", action="store_true", default=True, help="Enable R4 rotation (default: True)")
    parser.add_argument("--no-r4", action="store_false", dest="r4", help="Disable R4 rotation")
    args = parser.parse_args()

    levels_to_test = [l.strip() for l in args.levels.split(",")]

    print("=" * 70)
    print("Rotation + Quantization Evaluation")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Tasks:      {args.tasks}")
    print(f"  Limit:      {args.limit or 'full'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device}")
    print(f"  Levels:     {levels_to_test}")
    rotation_desc = "+".join(
        [f"R{i}" for i, enabled in enumerate([args.r1, args.r2, args.r3, args.r4], 1) if enabled]
    ) or "None"
    print(f"  Quant:      RTN W4A16 (iters=0)")
    print(f"  Rotation:   {rotation_desc} (QuaRot mode, deterministic Hadamard)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    all_results = {}

    for level in levels_to_test:
        print(f"\n{'='*70}")
        print(f"  Evaluating: {level}")
        print(f"{'='*70}")

        t0 = time.time()

        if level == "baseline_fp16":
            # Pure FP16 baseline — no quantization, no rotation
            print("  Loading FP16 baseline model...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            model.eval()

        elif level == "rotation_fp16":
            # Rotation only (no quantization) — should match baseline
            print("  Loading model and applying rotation...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            model.eval()
            apply_rotation(model, r1=args.r1, r2=args.r2, r3=args.r3, r4=args.r4)
            print(f"  Rotation applied ({rotation_desc})")

        elif level == "rtn_only":
            # RTN W4A16 quantization without rotation
            print("  Quantizing with RTN W4A16 (no rotation)...")
            model, _ = quantize_rtn_w4a16(
                args.model, args.device,
                nsamples=args.nsamples, seqlen=args.seqlen
            )

        elif level == "rotation_rtn":
            # Rotation THEN RTN W4A16 quantization
            print("  Step 1: Loading model and applying rotation...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            model.eval()
            apply_rotation(model, r1=args.r1, r2=args.r2, r3=args.r3, r4=args.r4)
            print(f"  Rotation applied ({rotation_desc})")
            print("  Step 2: Quantizing rotated model with RTN W4A16...")
            model = quantize_rotated_model_rtn_w4a16(
                model, tokenizer, args.device,
                nsamples=args.nsamples, seqlen=args.seqlen
            )

        else:
            print(f"  [WARN] Unknown level '{level}', skipping")
            continue

        setup_time = time.time() - t0
        print(f"  Setup completed in {setup_time:.1f}s")

        # Evaluate
        t0 = time.time()
        print(f"  Running lm_eval...")
        results = evaluate_model(
            model, tokenizer,
            tasks=args.tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            device=args.device,
        )
        eval_time = time.time() - t0

        metrics = extract_metrics(results)
        all_results[level] = metrics

        for task, acc in metrics.items():
            print(f"    {task}: acc_norm = {acc:.4f}")
        print(f"  Evaluation done in {eval_time:.1f}s")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE — Rotation + Quantization (RTN W4A16)")
    print("=" * 90)

    tasks = sorted(set(t for m in all_results.values() for t in m.keys()))
    configs = list(all_results.keys())

    # Header
    header = f"{'Task':<15}"
    for config in configs:
        header += f" | {config:<15}"
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{task:<15}"
        baseline_val = None
        for config in configs:
            val = all_results.get(config, {}).get(task)
            if val is not None:
                if baseline_val is None:
                    baseline_val = val
                row += f" | {val:<15.4f}"
            else:
                row += f" | {'N/A':<15}"
        print(row)

    # Delta row
    if "baseline_fp16" in all_results and len(configs) > 1:
        print()
        print("Δ vs baseline_fp16:")
        for task in tasks:
            base = all_results.get("baseline_fp16", {}).get(task)
            if base is None:
                continue
            row = f"  {task:<13}"
            for config in configs:
                val = all_results.get(config, {}).get(task)
                if val is not None and config != "baseline_fp16":
                    delta = val - base
                    row += f" | {delta:+.4f}{'':10}"
                elif config == "baseline_fp16":
                    row += f" | {'(ref)':15}"
                else:
                    row += f" | {'N/A':<15}"
            print(row)

    # Summary: did rotation help?
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    if "rtn_only" in all_results and "rotation_rtn" in all_results:
        print("  Does rotation help quantization accuracy?")
        for task in tasks:
            rtn_acc = all_results["rtn_only"].get(task)
            rot_rtn_acc = all_results["rotation_rtn"].get(task)
            if rtn_acc is not None and rot_rtn_acc is not None:
                improvement = rot_rtn_acc - rtn_acc
                status = "✓ YES" if improvement > 0.005 else ("≈ SAME" if abs(improvement) <= 0.005 else "✗ NO")
                print(f"    {task}: RTN={rtn_acc:.4f} → Rotation+RTN={rot_rtn_acc:.4f} "
                      f"(Δ={improvement:+.4f}) {status}")
    print("=" * 90)


if __name__ == "__main__":
    main()
