#!/usr/bin/env python3
"""
Rotation Equivalence Test — Verify rotation is mathematically lossless.

Loads Qwen3-0.6B in BF16, applies rotation (R1, R1+R2, R1+R2+R3+R4) via
auto-round's apply_rotation(), and compares rotated model's logits with
the BF16 baseline. Since rotation is an orthogonal transform (Q @ Q^T = I),
the output should be nearly identical — any difference is purely from
floating-point accumulation error.

Expected results:
  - cosine similarity ≈ 1.0 (> 0.999)
  - max absolute diff: small (< 1.0 for BF16)
  - top-5 token overlap: high (usually 4/5 or 5/5)

Usage:
    python test_rotation_equivalence.py
    python test_rotation_equivalence.py --device cuda:0
    python test_rotation_equivalence.py --model meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig


ROTATION_LEVELS = {
    "R1": SpinQuantConfig(
        r1=True, r2=False, r3=False, r4=False,
        online_r1_rotation=True,
        trainable_rotation=False, trainable_smooth=False,
    ),
    "R1+R2": SpinQuantConfig(
        r1=True, r2=True, r3=False, r4=False,
        online_r1_rotation=True,
        trainable_rotation=False, trainable_smooth=False,
    ),
    "R1+R2+R3": SpinQuantConfig(
        r1=True, r2=True, r3=True, r4=False,
        online_r1_rotation=True,
        trainable_rotation=False, trainable_smooth=False,
    ),
    "R1+R2+R3+R4": SpinQuantConfig(
        r1=True, r2=True, r3=True, r4=True,
        online_r1_rotation=True,
        trainable_rotation=False, trainable_smooth=False,
    ),
}

TEST_PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
]


def get_logits(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).logits


def compare_logits(baseline, rotated, label, tokenizer, prompt):
    """Compare two logit tensors and print detailed metrics."""
    # Flatten last-token logits for comparison
    bl = baseline[0, -1, :].float()
    rt = rotated[0, -1, :].float()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(bl.unsqueeze(0), rt.unsqueeze(0)).item()

    # Absolute diff
    abs_diff = (bl - rt).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative diff (avoid div by zero)
    rel_diff = abs_diff / (bl.abs().clamp(min=1e-8))
    mean_rel = rel_diff.mean().item()

    # Top-k token overlap
    top5_bl = bl.topk(5).indices.tolist()
    top5_rt = rt.topk(5).indices.tolist()
    overlap = len(set(top5_bl) & set(top5_rt))

    # Top-1 match
    top1_match = top5_bl[0] == top5_rt[0]

    # Decode top tokens for display
    bl_tokens = [tokenizer.decode([t]) for t in top5_bl]
    rt_tokens = [tokenizer.decode([t]) for t in top5_rt]

    return {
        "label": label,
        "cos_sim": cos_sim,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "mean_rel_diff": mean_rel,
        "top5_overlap": overlap,
        "top1_match": top1_match,
        "baseline_top5": bl_tokens,
        "rotated_top5": rt_tokens,
    }


def print_result(r, prompt_short):
    status = "✓" if r["cos_sim"] > 0.999 and r["top1_match"] else "✗"
    print(f"  {status} {r['label']:<16} | "
          f"cos={r['cos_sim']:.6f} | "
          f"max_diff={r['max_abs_diff']:.4f} | "
          f"mean_diff={r['mean_abs_diff']:.4f} | "
          f"top5={r['top5_overlap']}/5 | "
          f"top1={'✓' if r['top1_match'] else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="Rotation equivalence test")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--levels", default="R1,R1+R2,R1+R2+R3,R1+R2+R3+R4",
                        help="Comma-separated rotation levels to test")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    levels = [l.strip() for l in args.levels.split(",")]

    print(f"{'═' * 80}")
    print(f"  Rotation Equivalence Test")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Dtype:  {args.dtype}")
    print(f"  Levels: {levels}")
    print(f"{'═' * 80}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Baseline: original model
    print("Loading baseline model...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device).eval()

    # Compute baseline logits for all prompts
    baseline_logits = {}
    for prompt in TEST_PROMPTS:
        baseline_logits[prompt] = get_logits(baseline_model, tokenizer, prompt, args.device)

    # Store state_dict for reloading
    baseline_sd = copy.deepcopy(baseline_model.state_dict())
    del baseline_model
    torch.cuda.empty_cache()

    # Test each rotation level
    all_results = []
    for level in levels:
        if level not in ROTATION_LEVELS:
            print(f"  ✗ Unknown level: {level}")
            continue

        print(f"\n{'─' * 80}")
        print(f"  Testing: {level}")
        print(f"{'─' * 80}")

        # Reload fresh model
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True
        ).to(args.device).eval()

        # Apply rotation
        cfg = ROTATION_LEVELS[level]
        model = apply_rotation(model, cfg)

        for prompt in TEST_PROMPTS:
            rotated_logits = get_logits(model, tokenizer, prompt, args.device)
            r = compare_logits(
                baseline_logits[prompt], rotated_logits,
                level, tokenizer, prompt,
            )
            r["prompt"] = prompt[:40]
            all_results.append(r)
            print_result(r, prompt[:40])

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'═' * 80}")
    print(f"  SUMMARY")
    print(f"{'═' * 80}")
    print(f"  {'Level':<16} | {'Avg Cosine':>12} | {'Avg MaxDiff':>12} | {'Top1 Match':>10} | {'Top5 Avg':>8}")
    print(f"  {'─' * 16}-+-{'─' * 12}-+-{'─' * 12}-+-{'─' * 10}-+-{'─' * 8}")

    for level in levels:
        level_results = [r for r in all_results if r["label"] == level]
        if not level_results:
            continue
        avg_cos = sum(r["cos_sim"] for r in level_results) / len(level_results)
        avg_max = sum(r["max_abs_diff"] for r in level_results) / len(level_results)
        top1_rate = sum(1 for r in level_results if r["top1_match"]) / len(level_results)
        avg_top5 = sum(r["top5_overlap"] for r in level_results) / len(level_results)

        status = "✓" if avg_cos > 0.999 else "✗"
        print(f"  {status} {level:<14} | {avg_cos:>12.6f} | {avg_max:>12.4f} | "
              f"{top1_rate:>9.0%} | {avg_top5:>7.1f}/5")

    # Overall pass/fail
    all_cos = [r["cos_sim"] for r in all_results]
    if all_cos:
        min_cos = min(all_cos)
        all_top1 = all(r["top1_match"] for r in all_results)
        passed = min_cos > 0.99 and all_top1

        print(f"\n  Min cosine similarity: {min_cos:.6f}")
        print(f"  All top-1 tokens match: {'Yes' if all_top1 else 'No'}")
        print(f"  Overall: {'✓ PASS — rotation is equivalent to baseline' if passed else '✗ FAIL — rotation changed model behavior'}")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
