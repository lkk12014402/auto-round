#!/usr/bin/env python3
"""
Evaluate a saved quantized model from disk using lm_eval (HF backend).

Includes softmax_dtype="float32" fix to match vLLM backend precision.

Usage:
    # Quick test (limit=100):
    python eval_from_disk.py --model_path /path/to/saved/model --limit 100

    # Full evaluation:
    python eval_from_disk.py --model_path /path/to/saved/model

    # Custom tasks:
    python eval_from_disk.py --model_path /path/to/saved/model --tasks hellaswag,piqa,winogrande,arc_easy

    # Compare two models side by side:
    python eval_from_disk.py --model_path /path/to/model_A --model_path2 /path/to/model_B --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_model_from_path(
    model_path: str,
    tasks: str,
    batch_size: int = 8,
    limit: int | None = None,
    device: str = "cuda:0",
    softmax_dtype: str | None = "float32",
) -> dict[str, float]:
    """Load a saved model from disk and evaluate using lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model_path,
        batch_size=batch_size,
        device=device,
        softmax_dtype=softmax_dtype,
        add_bos_token=True,
    )
    task_list = [t.strip() for t in tasks.split(",")]
    results = simple_evaluate(
        model=lm, tasks=task_list, batch_size=batch_size, limit=limit,
    )
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


def print_model_info(model_path: str) -> None:
    """Print key model info from config.json and safetensors."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config", {})
        sq = qcfg.get("spinquant_config", {})
        logger.info(f"  Model type: {cfg.get('model_type', '?')}")
        logger.info(f"  Quant: W{qcfg.get('bits', '?')}A16, group_size={qcfg.get('group_size', '?')}")
        logger.info(f"  iters: {qcfg.get('iters', '?')}")
        if sq:
            logger.info(f"  SpinQuant: R1={sq.get('r1')}, R2={sq.get('r2')}, "
                        f"R3={sq.get('r3')}, R4={sq.get('r4')}")
            logger.info(f"  online_r1: {sq.get('online_r1_rotation')}, "
                        f"rotation_size: {sq.get('rotation_size')}")
        else:
            logger.info("  SpinQuant: not configured")

    # Check safetensors for spinquant keys
    for fn in os.listdir(model_path):
        if fn.endswith(".safetensors"):
            try:
                from safetensors import safe_open
                with safe_open(os.path.join(model_path, fn), framework="pt") as sf:
                    sq_keys = [k for k in sf.keys() if "spinquant" in k]
                    if sq_keys:
                        for k in sq_keys:
                            t = sf.get_tensor(k)
                            logger.info(f"  {k}: shape={t.shape}, dtype={t.dtype}")
            except Exception:
                pass
            break


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved quantized model from disk")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--model_path2", default=None, help="(Optional) Second model for comparison")
    parser.add_argument("--tasks", default="hellaswag,piqa,winogrande",
                        help="Comma-separated lm_eval tasks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per task (for quick tests)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_softmax_fix", action="store_true",
                        help="Disable float32 softmax fix (default: enabled)")
    args = parser.parse_args()

    softmax_dtype = None if args.no_softmax_fix else "float32"

    models = [("Model A", args.model_path)]
    if args.model_path2:
        models.append(("Model B", args.model_path2))

    all_results = {}
    for label, path in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"{label}: {path}")
        logger.info(f"{'='*60}")
        print_model_info(path)

        logger.info(f"Evaluating tasks={args.tasks}, limit={args.limit}, "
                     f"softmax_dtype={softmax_dtype}...")
        t0 = time.time()
        metrics = evaluate_model_from_path(
            path, args.tasks,
            batch_size=args.batch_size, limit=args.limit,
            device=args.device, softmax_dtype=softmax_dtype,
        )
        elapsed = time.time() - t0
        logger.info(f"Evaluation done in {elapsed:.1f}s")
        for task, acc in sorted(metrics.items()):
            logger.info(f"  {task}: {acc:.4f}")
        all_results[label] = metrics

    # Print comparison table
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("COMPARISON")
        logger.info(f"{'='*60}")
        all_tasks = sorted(set().union(*[m.keys() for m in all_results.values()]))
        header = f"{'Task':<20}" + "".join(f"{label:<15}" for label in all_results) + "Delta"
        logger.info(header)
        logger.info("-" * len(header))
        for task in all_tasks:
            vals = [all_results[label].get(task) for label in all_results]
            line = f"{task:<20}"
            for v in vals:
                line += f"{v:<15.4f}" if v is not None else f"{'N/A':<15}"
            if all(v is not None for v in vals):
                delta = vals[1] - vals[0]
                line += f"{delta:+.4f}"
            logger.info(line)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
