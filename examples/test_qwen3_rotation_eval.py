"""
Validate QuaRot rotation levels on Qwen3-0.6B using lm_eval.

Tests R1, R1+R2, R1+R2+R3, R1+R2+R3+R4 with real accuracy evaluation
to verify the rotation implementation preserves model quality.

Since rotations are mathematically equivalent transforms, the model accuracy
should remain essentially unchanged (within statistical noise) after rotation.

Usage:
    python test_qwen3_rotation_eval.py [--tasks piqa,hellaswag] [--limit 100] [--device cuda:0]
"""

import argparse
import copy
import logging
import sys
import time

sys.path.insert(0, "/data/lkk/quarot/auto-round")

# Enable SpinQuant logging so the transformation summary table is visible
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("auto_round.spinquant").setLevel(logging.INFO)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
)


def apply_rotation(model, tokenizer, r1=True, r2=True, r3=True, r4=True, device="cuda:0"):
    """Apply SpinQuant/QuaRot rotation to a model (in-place)."""
    config = SpinQuantConfig(
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        trainable_rotation=False,  # Fixed Hadamard (QuaRot mode)
        trainable_smooth=False,
        fuse_rmsnorm=True,
        untie_embeddings=True,
    )
    preprocessor = SpinQuantPreprocessor(model, config)
    preprocessor.preprocess(dataloader=None)
    return model


def evaluate_model(model, tokenizer, tasks, batch_size=8, limit=None, device="cuda:0"):
    """Evaluate model using lm_eval framework.

    Args:
        model: Pre-loaded model (already on device)
        tokenizer: Tokenizer
        tasks: Comma-separated task names or list
        batch_size: Evaluation batch size
        limit: Max samples per task (None = full eval)
        device: Device string

    Returns:
        dict with results per task
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    # Wrap model for lm_eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    # Parse tasks
    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = tasks

    # Run evaluation
    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        device=device,
    )

    return results


def extract_metrics(results, tasks):
    """Extract key accuracy metrics from lm_eval results."""
    metrics = {}
    if "results" not in results:
        return metrics

    for task_name, task_results in results["results"].items():
        # Common metric names in lm_eval
        acc = task_results.get("acc,none") or task_results.get("acc")
        acc_norm = task_results.get("acc_norm,none") or task_results.get("acc_norm")
        ppl = task_results.get("word_perplexity,none") or task_results.get("word_perplexity")
        byte_ppl = task_results.get("byte_perplexity,none")

        entry = {}
        if acc is not None:
            entry["acc"] = acc
        if acc_norm is not None:
            entry["acc_norm"] = acc_norm
        if ppl is not None:
            entry["ppl"] = ppl
        if byte_ppl is not None:
            entry["byte_ppl"] = byte_ppl
        # Fallback: grab any metric that looks like accuracy
        if not entry:
            for k, v in task_results.items():
                if isinstance(v, (int, float)) and ("acc" in k or "ppl" in k or "score" in k):
                    entry[k] = v

        metrics[task_name] = entry

    return metrics


def print_results_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Configuration':<25} | {'Task':<20} | {'Metric':<15} | {'Value':<10}")
    print("-" * 90)

    for config_name, metrics in all_results.items():
        for task_name, task_metrics in metrics.items():
            for metric_name, value in task_metrics.items():
                if isinstance(value, float):
                    print(f"{config_name:<25} | {task_name:<20} | {metric_name:<15} | {value:.4f}")
                else:
                    print(f"{config_name:<25} | {task_name:<20} | {metric_name:<15} | {value}")

    print("=" * 90)


def print_comparison_table(all_results):
    """Print a side-by-side comparison table of all rotation levels."""
    # Collect all tasks and metrics
    all_tasks = set()
    all_metric_keys = {}
    for config_name, metrics in all_results.items():
        for task_name, task_metrics in metrics.items():
            all_tasks.add(task_name)
            if task_name not in all_metric_keys:
                all_metric_keys[task_name] = set()
            all_metric_keys[task_name].update(task_metrics.keys())

    configs = list(all_results.keys())

    print("\n" + "=" * 100)
    print("ROTATION LEVEL COMPARISON - Qwen3-0.6B")
    print("=" * 100)

    for task in sorted(all_tasks):
        metrics_for_task = all_metric_keys.get(task, set())
        for metric in sorted(metrics_for_task):
            # Header
            header = f"  {task}/{metric}"
            print(f"\n{header}:")
            print(f"    {'Config':<25} {'Value':<12} {'Δ vs Baseline':<15}")
            print(f"    {'-'*55}")

            baseline_val = None
            for config_name in configs:
                val = all_results.get(config_name, {}).get(task, {}).get(metric)
                if val is not None and isinstance(val, (int, float)):
                    if baseline_val is None:
                        baseline_val = val
                        delta_str = "(baseline)"
                    else:
                        delta = val - baseline_val
                        delta_str = f"{delta:+.4f}"
                    print(f"    {config_name:<25} {val:<12.4f} {delta_str}")
                elif val is not None:
                    print(f"    {config_name:<25} {val}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Evaluate QuaRot rotations on Qwen3-0.6B")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="Model name or path"
    )
    parser.add_argument(
        "--tasks", type=str, default="piqa,hellaswag,arc_easy,winogrande,lambada_standard",
        help="Comma-separated lm_eval tasks"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit samples per task (None=full eval, use small value for quick test)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for evaluation"
    )
    parser.add_argument(
        "--levels", type=str, default="baseline,r1,r1r2,r1r2r3,r1r2r3r4",
        help="Comma-separated rotation levels to test"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    levels_to_test = [l.strip() for l in args.levels.split(",")]

    # Rotation level configurations
    ROTATION_CONFIGS = {
        "baseline": {"r1": False, "r2": False, "r3": False, "r4": False},
        "r1": {"r1": True, "r2": False, "r3": False, "r4": False},
        "r1r2": {"r1": True, "r2": True, "r3": False, "r4": False},
        "r1r2r3": {"r1": True, "r2": True, "r3": True, "r4": False},
        "r1r2r3r4": {"r1": True, "r2": True, "r3": True, "r4": True},
    }

    print("=" * 70)
    print("QuaRot Rotation Evaluation on Qwen3-0.6B")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Tasks:      {args.tasks}")
    print(f"  Limit:      {args.limit or 'full'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Levels:     {levels_to_test}")
    print("=" * 70)

    # Load tokenizer (shared across all evaluations)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_results = {}

    for level_name in levels_to_test:
        if level_name not in ROTATION_CONFIGS:
            print(f"  [WARN] Unknown level '{level_name}', skipping")
            continue

        rot_cfg = ROTATION_CONFIGS[level_name]
        print(f"\n{'='*70}")
        print(f"  Evaluating: {level_name.upper()}")
        print(f"    R1={rot_cfg['r1']}, R2={rot_cfg['r2']}, R3={rot_cfg['r3']}, R4={rot_cfg['r4']}")
        print(f"{'='*70}")

        # Load fresh model for each level
        t0 = time.time()
        print(f"  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=model_dtype,
            trust_remote_code=True,
        )
        model.to(args.device)
        model.eval()
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")

        # Apply rotation (skip for baseline)
        if any(rot_cfg.values()):
            t0 = time.time()
            print(f"  Applying rotation...")
            apply_rotation(
                model, tokenizer,
                r1=rot_cfg["r1"], r2=rot_cfg["r2"],
                r3=rot_cfg["r3"], r4=rot_cfg["r4"],
                device=args.device,
            )
            rot_time = time.time() - t0
            print(f"  Rotation applied in {rot_time:.1f}s")

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
        print(f"  Evaluation done in {eval_time:.1f}s")

        # Extract and store metrics
        metrics = extract_metrics(results, args.tasks)
        all_results[level_name] = metrics

        # Print per-level results
        for task_name, task_metrics in metrics.items():
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in task_metrics.items() if isinstance(v, float))
            print(f"    {task_name}: {metric_str}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print final comparison table
    print_comparison_table(all_results)

    # Summary: check if rotation preserves accuracy
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if "baseline" in all_results:
        baseline_metrics = all_results["baseline"]
        all_pass = True

        for level_name, level_metrics in all_results.items():
            if level_name == "baseline":
                continue

            max_degradation = 0.0
            for task, task_metrics in level_metrics.items():
                for metric, value in task_metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    baseline_val = baseline_metrics.get(task, {}).get(metric)
                    if baseline_val is not None and isinstance(baseline_val, (int, float)):
                        # For perplexity, higher is worse; for accuracy, lower is worse
                        if "ppl" in metric:
                            degradation = value - baseline_val  # positive = worse
                        else:
                            degradation = baseline_val - value  # positive = worse
                        max_degradation = max(max_degradation, degradation)

            # Allow up to 2% degradation (statistical noise + float16 precision)
            status = "✅ PASS" if max_degradation < 0.02 else "❌ FAIL"
            if max_degradation >= 0.02:
                all_pass = False
            print(f"  {level_name:<20} max_degradation={max_degradation:+.4f}  {status}")

        print()
        if all_pass:
            print("  ✅ All rotation levels preserve model accuracy!")
        else:
            print("  ❌ Some rotation levels show degradation > 2%")
    else:
        print("  (No baseline evaluation - cannot validate)")

    print("=" * 70)


if __name__ == "__main__":
    main()
