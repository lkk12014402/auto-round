#!/usr/bin/env python3
"""
Compare lm_eval results from HF and vLLM backends across rotation configs.

Reads results from eval_hf.sh and eval_vllm.sh output directories,
and displays a unified comparison table.

Usage:
    python compare_results.py
    python compare_results.py --hf-dir ./lm_eval_results_hf --vllm-dir ./lm_eval_results_vllm
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_results(result_dir: str) -> dict:
    """Load lm_eval results from a directory (handles nested structure)."""
    result_dir = Path(result_dir)
    
    # lm_eval saves results in: <output_path>/<model_name>/<results_file>.json
    # or directly as results.json
    candidates = [
        result_dir / "results.json",
    ]
    
    # Also check subdirectories (lm_eval creates model-name subdirs)
    if result_dir.exists():
        for sub in result_dir.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.name.startswith("results") and f.suffix == ".json":
                        candidates.append(f)
            elif sub.suffix == ".json" and sub.name.startswith("results"):
                candidates.append(sub)
    
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if "results" in data:
                    return data["results"]
            except (json.JSONDecodeError, KeyError):
                continue
    
    return {}


def extract_accuracy(task_results: dict) -> float | None:
    """Extract the primary accuracy metric from task results."""
    # Try in order of preference
    for key in ["acc_norm,none", "acc,none", "exact_match,strict-match"]:
        if key in task_results:
            val = task_results[key]
            if isinstance(val, (int, float)):
                return round(val, 4)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", default="./lm_eval_results_hf")
    parser.add_argument("--vllm-dir", default="./lm_eval_results_vllm")
    parser.add_argument("--model-base", default="./rotated_models_Qwen3-0.6B")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    vllm_dir = Path(args.vllm_dir)

    # Collect all config names
    all_configs = set()
    if hf_dir.exists():
        all_configs.update(d.name for d in hf_dir.iterdir() if d.is_dir())
    if vllm_dir.exists():
        all_configs.update(d.name for d in vllm_dir.iterdir() if d.is_dir())
    
    all_configs = sorted(all_configs)
    
    if not all_configs:
        print("No results found. Run eval_hf.sh and/or eval_vllm.sh first.")
        sys.exit(1)

    # Load all results
    hf_results = {}
    vllm_results = {}
    all_tasks = set()

    for config in all_configs:
        hf_path = hf_dir / config
        vllm_path = vllm_dir / config
        
        if hf_path.exists():
            r = load_results(str(hf_path))
            hf_results[config] = r
            all_tasks.update(r.keys())
        
        if vllm_path.exists():
            r = load_results(str(vllm_path))
            vllm_results[config] = r
            all_tasks.update(r.keys())

    tasks = sorted(all_tasks)

    # Print header
    print(f"\n{'═'*100}")
    print(f"  Rotation Config Evaluation Comparison")
    print(f"  HF results: {hf_dir}")
    print(f"  vLLM results: {vllm_dir}")
    print(f"{'═'*100}")

    # Per-task comparison table
    for task in tasks:
        print(f"\n  Task: {task}")
        print(f"  {'Config':<20} {'HF':>10} {'vLLM':>10} {'Δ(vLLM-HF)':>12}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*12}")
        
        for config in all_configs:
            hf_acc = None
            vllm_acc = None
            
            if config in hf_results and task in hf_results[config]:
                hf_acc = extract_accuracy(hf_results[config][task])
            if config in vllm_results and task in vllm_results[config]:
                vllm_acc = extract_accuracy(vllm_results[config][task])
            
            hf_str = f"{hf_acc:.4f}" if hf_acc is not None else "—"
            vllm_str = f"{vllm_acc:.4f}" if vllm_acc is not None else "—"
            
            if hf_acc is not None and vllm_acc is not None:
                delta = vllm_acc - hf_acc
                delta_str = f"{delta:+.4f}"
                if abs(delta) > 0.01:
                    delta_str += " ⚠️"
            else:
                delta_str = "—"
            
            print(f"  {config:<20} {hf_str:>10} {vllm_str:>10} {delta_str:>12}")

    # Average summary
    print(f"\n{'═'*100}")
    print(f"  Average Accuracy (across all tasks)")
    print(f"{'═'*100}")
    print(f"  {'Config':<20} {'HF avg':>10} {'vLLM avg':>10} {'Δ':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")

    for config in all_configs:
        hf_accs = []
        vllm_accs = []
        
        for task in tasks:
            if config in hf_results and task in hf_results[config]:
                acc = extract_accuracy(hf_results[config][task])
                if acc is not None:
                    hf_accs.append(acc)
            if config in vllm_results and task in vllm_results[config]:
                acc = extract_accuracy(vllm_results[config][task])
                if acc is not None:
                    vllm_accs.append(acc)
        
        hf_avg = sum(hf_accs) / len(hf_accs) if hf_accs else None
        vllm_avg = sum(vllm_accs) / len(vllm_accs) if vllm_accs else None
        
        hf_str = f"{hf_avg:.4f}" if hf_avg else "—"
        vllm_str = f"{vllm_avg:.4f}" if vllm_avg else "—"
        
        if hf_avg and vllm_avg:
            delta_str = f"{vllm_avg - hf_avg:+.4f}"
        else:
            delta_str = "—"
        
        print(f"  {config:<20} {hf_str:>10} {vllm_str:>10} {delta_str:>10}")

    print(f"\n{'═'*100}")
    print(f"  Notes:")
    print(f"  - Δ(vLLM-HF) should be ~0.0 if plugin correctly reproduces rotation")
    print(f"  - Large Δ indicates a rotation or quantization mismatch")
    print(f"  - R3 models may show 'N/A' for vLLM (R3 plugin not yet implemented)")
    print(f"{'═'*100}\n")


if __name__ == "__main__":
    main()
