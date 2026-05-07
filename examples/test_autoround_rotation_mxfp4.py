"""
Auto-Round: Rotation + MXFP4 Quantization evaluation on Qwen3-0.6B.

This script tests auto-round's rotation + MXFP4 quantization pipeline:
  1. Baseline FP16 (no quant, no rotation)
  2. MXFP4 only (quantization without rotation)
  3. Rotation + MXFP4 (rotation then quantization)

This is the auto-round counterpart of test_quark_rotation_mxfp4.py for
apples-to-apples comparison with Quark.

Usage:
    python test_autoround_rotation_mxfp4.py --device cuda:0 --tasks piqa,hellaswag
    python test_autoround_rotation_mxfp4.py --device cuda:7 --tasks piqa,hellaswag --limit 200
"""

import argparse
import gc
import logging
import sys
import time

sys.path.insert(0, "/data/lkk/quarot/auto-round")

# Enable SpinQuant logging for transformation summary table
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


def quantize_mxfp4(model_or_name, tokenizer=None, device="cuda:0", nsamples=128, seqlen=512):
    """Quantize model using auto-round RTN (iters=0) with MXFP4 scheme.

    Args:
        model_or_name: Either a model name string or pre-loaded model object.
        tokenizer: Tokenizer (required if model_or_name is a model object).
        device: Device for quantization.
        nsamples: Number of calibration samples.
        seqlen: Calibration sequence length.

    Returns:
        Quantized model and tokenizer.
    """
    kwargs = dict(
        scheme="MXFP4",
        iters=0,  # RTN mode — no optimization iterations
        nsamples=nsamples,
        seqlen=seqlen,
        device_map=device,
    )

    if isinstance(model_or_name, str):
        ar = AutoRound(model_or_name, **kwargs)
    else:
        ar = AutoRound(model_or_name, tokenizer=tokenizer, **kwargs)

    ar.quantize()
    model = ar.model
    model.eval()
    return model, ar.tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Round: Evaluate Rotation + MXFP4 Quantization on Qwen3-0.6B"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--tasks", type=str, default="piqa,hellaswag", help="Comma-separated lm_eval tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (None=full)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for evaluation")
    parser.add_argument("--nsamples", type=int, default=128, help="Calibration samples for quantization")
    parser.add_argument("--seqlen", type=int, default=512, help="Calibration sequence length")
    parser.add_argument(
        "--levels", type=str, default="baseline_fp16,mxfp4_only,rotation_mxfp4",
        help="Comma-separated levels: baseline_fp16, mxfp4_only, rotation_mxfp4, rotation_fp16"
    )
    parser.add_argument("--r1", action="store_true", default=True)
    parser.add_argument("--no-r1", action="store_false", dest="r1")
    parser.add_argument("--r2", action="store_true", default=True)
    parser.add_argument("--no-r2", action="store_false", dest="r2")
    parser.add_argument("--r3", action="store_false", default=False)
    parser.add_argument("--enable-r3", action="store_true", dest="r3")
    parser.add_argument("--r4", action="store_false", default=False)
    parser.add_argument("--enable-r4", action="store_true", dest="r4")
    args = parser.parse_args()

    levels_to_test = [l.strip() for l in args.levels.split(",")]

    rotation_desc = "+".join(
        [f"R{i}" for i, enabled in enumerate([args.r1, args.r2, args.r3, args.r4], 1) if enabled]
    ) or "None"

    print("=" * 70)
    print("Auto-Round: Rotation + MXFP4 Quantization Evaluation")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Tasks:      {args.tasks}")
    print(f"  Limit:      {args.limit or 'full'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device}")
    print(f"  Levels:     {levels_to_test}")
    print(f"  Quant:      MXFP4 (RTN, iters=0)")
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

        elif level == "mxfp4_only":
            # MXFP4 quantization without rotation
            print("  Quantizing with MXFP4 (no rotation)...")
            model, _ = quantize_mxfp4(
                args.model, device=args.device,
                nsamples=args.nsamples, seqlen=args.seqlen,
            )

        elif level == "rotation_mxfp4":
            # Rotation THEN MXFP4 quantization
            print(f"  Step 1: Loading model and applying rotation ({rotation_desc})...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            model.eval()
            apply_rotation(model, r1=args.r1, r2=args.r2, r3=args.r3, r4=args.r4)
            print("  Step 2: Quantizing rotated model with MXFP4...")
            model, _ = quantize_mxfp4(
                model, tokenizer=tokenizer, device=args.device,
                nsamples=args.nsamples, seqlen=args.seqlen,
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
    print("COMPARISON TABLE — Auto-Round Rotation + MXFP4")
    print("=" * 90)

    tasks = sorted(set(t for m in all_results.values() for t in m.keys()))
    configs = list(all_results.keys())

    header = f"{'Task':<15}"
    for config in configs:
        header += f" | {config:<15}"
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{task:<15}"
        for config in configs:
            val = all_results.get(config, {}).get(task)
            if val is not None:
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
            deltas = []
            for config in configs:
                if config == "baseline_fp16":
                    continue
                val = all_results.get(config, {}).get(task)
                if val is not None:
                    delta = val - base
                    deltas.append(f"  {config}: {delta:+.4f}")
            print(f"  {task}: " + ", ".join(deltas))

    print("=" * 90)


if __name__ == "__main__":
    main()
