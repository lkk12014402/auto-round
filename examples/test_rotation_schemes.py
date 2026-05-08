"""
Auto-Round: Test Rotation + Multiple Quantization Schemes.

Tests rotation (R1, R1+R2, R1+R2+R3, R1+R2+R3+R4) with different quantization
schemes: W4A16 (INT4), NVFP4, and optionally MXFP4. Uses RTN mode (iters=0).

Model: Qwen/Qwen3-0.6B

Usage:
    # Quick test (limit=100):
    python test_rotation_schemes.py --device cuda:7 --limit 100

    # Full eval with specific schemes:
    python test_rotation_schemes.py --device cuda:7 --schemes W4A16,NVFP4

    # Test specific rotation levels only:
    python test_rotation_schemes.py --device cuda:7 --rotations R1,R1+R2+R3+R4

    # Include baseline + scheme-only comparisons:
    python test_rotation_schemes.py --device cuda:7 --include-baselines
"""

import argparse
import gc
import logging
import sys
import time
from collections import OrderedDict

sys.path.insert(0, "/data/lkk/quarot/auto-round")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig, SpinQuantPreprocessor


# ── Rotation level definitions ──────────────────────────────────────────────
ROTATION_LEVELS = OrderedDict([
    ("R1",          dict(r1=True, r2=False, r3=False, r4=False)),
    ("R1+R2",       dict(r1=True, r2=True,  r3=False, r4=False)),
    ("R1+R2+R3",    dict(r1=True, r2=True,  r3=True,  r4=False)),
    ("R1+R2+R3+R4", dict(r1=True, r2=True,  r3=True,  r4=True)),
])

# ── Scheme definitions ──────────────────────────────────────────────────────
SCHEME_CONFIGS = {
    "W4A16": dict(scheme="W4A16", desc="INT4 weight-only, group_size=128"),
    "NVFP4": dict(scheme="NVFP4", desc="NV FP4, W4A4, group_size=16"),
    "MXFP4": dict(scheme="MXFP4_RCEIL", desc="MX FP4, W4A4, block_size=32"),
}


def load_model(model_name, device, dtype=torch.float16):
    """Load a fresh model instance."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model


def apply_rotation(model, rotation_flags, rotation_size=128):
    """Apply SpinQuant rotation to model (in-place)."""
    config = SpinQuantConfig(
        **rotation_flags,
        rotation_size=rotation_size,
        online_r1_rotation=True,
        trainable_rotation=False,
        trainable_smooth=False,
    )
    preprocessor = SpinQuantPreprocessor(model, config)
    return preprocessor.preprocess()


def quantize_model(model, tokenizer, scheme, device, nsamples=128, seqlen=512):
    """Quantize a (possibly rotated) model with given scheme using RTN."""
    ar = AutoRound(
        model,
        tokenizer=tokenizer,
        scheme=scheme,
        iters=0,  # RTN mode
        nsamples=nsamples,
        seqlen=seqlen,
        device_map=device,
    )
    ar.quantize()
    model = ar.model
    model.eval()
    model.to(device)
    return model


def evaluate_model(model, tokenizer, tasks, batch_size=8, limit=None, device="cuda:0"):
    """Evaluate model using lm_eval."""
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


def run_single_test(model_name, tokenizer, scheme_name, scheme_str, rotation_name,
                    rotation_flags, args):
    """Run a single rotation+scheme combination and return metrics."""
    label = f"{rotation_name}+{scheme_name}" if rotation_name else f"{scheme_name}_only"
    logger.info(f"{'='*60}")
    logger.info(f"  Running: {label}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    try:
        model = load_model(model_name, args.device)

        if rotation_flags:
            logger.info(f"  Applying rotation: {rotation_name} (size={args.rotation_size})")
            model = apply_rotation(model, rotation_flags, rotation_size=args.rotation_size)

        logger.info(f"  Quantizing with {scheme_name} ({scheme_str}) RTN...")
        model = quantize_model(model, tokenizer, scheme_str, args.device,
                               nsamples=args.nsamples, seqlen=args.seqlen)

        setup_time = time.time() - t0
        logger.info(f"  Setup done in {setup_time:.1f}s. Evaluating...")

        metrics = evaluate_model(model, tokenizer, args.tasks,
                                 batch_size=args.batch_size, limit=args.limit, device=args.device)

        for task, acc in sorted(metrics.items()):
            logger.info(f"    {task}: {acc:.4f}")

        return label, metrics

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return label, {"error": str(e)}

    finally:
        if 'model' in dir():
            del model
        gc.collect()
        torch.cuda.empty_cache()


def print_results_table(all_results, tasks):
    """Print a formatted comparison table."""
    task_list = sorted(tasks)

    # Group by scheme
    schemes_seen = []
    for label in all_results:
        if "+" in label:
            scheme = label.split("+", 1)[1]
        else:
            scheme = label.replace("_only", "")
        if scheme not in schemes_seen:
            schemes_seen.append(scheme)

    print("\n" + "=" * 100)
    print("RESULTS: Rotation × Quantization Scheme (Auto-Round RTN)")
    print("=" * 100)

    for scheme in schemes_seen:
        print(f"\n── {scheme} {'─'*(80-len(scheme))}")
        relevant = [(k, v) for k, v in all_results.items()
                     if scheme in k and "error" not in v]

        if not relevant:
            print("  (no results)")
            continue

        header = f"  {'Config':<25}"
        for task in task_list:
            header += f" | {task:<12}"
        print(header)
        print("  " + "-" * (25 + len(task_list) * 15))

        for label, metrics in relevant:
            row = f"  {label:<25}"
            for task in task_list:
                val = metrics.get(task)
                if val is not None:
                    row += f" | {val:<12.4f}"
                else:
                    row += f" | {'N/A':<12}"
            print(row)

    # Errors
    errors = [(k, v) for k, v in all_results.items() if "error" in v]
    if errors:
        print(f"\n── ERRORS {'─'*70}")
        for label, v in errors:
            print(f"  {label}: {v['error']}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Test Rotation + Multiple Quantization Schemes")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tasks", default="piqa,hellaswag")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--rotation-size", type=int, default=128)
    parser.add_argument(
        "--schemes", default="W4A16,NVFP4",
        help="Comma-separated schemes to test (W4A16, NVFP4, MXFP4)"
    )
    parser.add_argument(
        "--rotations", default="R1,R1+R2,R1+R2+R3,R1+R2+R3+R4",
        help="Comma-separated rotation levels to test"
    )
    parser.add_argument("--include-baselines", action="store_true",
                        help="Include FP16 baseline and scheme-only (no rotation) results")
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",")]
    rotations = [r.strip() for r in args.rotations.split(",")]

    # Validate
    for s in schemes:
        if s not in SCHEME_CONFIGS:
            print(f"Unknown scheme '{s}'. Available: {list(SCHEME_CONFIGS.keys())}")
            sys.exit(1)
    for r in rotations:
        if r not in ROTATION_LEVELS:
            print(f"Unknown rotation '{r}'. Available: {list(ROTATION_LEVELS.keys())}")
            sys.exit(1)

    logger.info(f"Model:     {args.model}")
    logger.info(f"Schemes:   {schemes}")
    logger.info(f"Rotations: {rotations}")
    logger.info(f"Tasks:     {args.tasks}")
    logger.info(f"Limit:     {args.limit or 'full'}")
    logger.info(f"Device:    {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    all_results = OrderedDict()
    all_tasks = set()

    # Optional: FP16 baseline
    if args.include_baselines:
        logger.info("Running FP16 baseline...")
        model = load_model(args.model, args.device)
        metrics = evaluate_model(model, tokenizer, args.tasks,
                                 batch_size=args.batch_size, limit=args.limit, device=args.device)
        all_results["FP16_baseline"] = metrics
        all_tasks.update(metrics.keys())
        del model; gc.collect(); torch.cuda.empty_cache()

    for scheme_name in schemes:
        scheme_cfg = SCHEME_CONFIGS[scheme_name]
        scheme_str = scheme_cfg["scheme"]

        # Optional: scheme-only (no rotation)
        if args.include_baselines:
            label, metrics = run_single_test(
                args.model, tokenizer, scheme_name, scheme_str,
                None, None, args
            )
            all_results[label] = metrics
            if "error" not in metrics:
                all_tasks.update(metrics.keys())

        # Rotation + scheme combinations
        for rot_name in rotations:
            rot_flags = ROTATION_LEVELS[rot_name]
            label, metrics = run_single_test(
                args.model, tokenizer, scheme_name, scheme_str,
                rot_name, rot_flags, args
            )
            all_results[label] = metrics
            if "error" not in metrics:
                all_tasks.update(metrics.keys())

    print_results_table(all_results, all_tasks)

    # Also save results to a simple text file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"logs_comparison/{timestamp}_rotation_schemes.txt"
    try:
        import os
        os.makedirs("logs_comparison", exist_ok=True)
        with open(result_file, "w") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Schemes: {schemes}\n")
            f.write(f"Rotations: {rotations}\n")
            f.write(f"Limit: {args.limit or 'full'}\n\n")
            for label, metrics in all_results.items():
                f.write(f"{label}: {metrics}\n")
        logger.info(f"Results saved to {result_file}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
