#!/usr/bin/env python3
"""
Test selective Hadamard rotation — compare "all" vs "structural" vs "auto" modes.

This extends the test_rotation_module_compare.py framework to include selective
rotation variants, demonstrating that skipping harmful layers (down_proj, o_proj)
improves quantization accuracy.

Examples
--------
  # Quick test on Qwen3-0.6B
  python test_selective_rotation.py --device cuda:4

  # Compare with more tasks
  python test_selective_rotation.py --device cuda:4 --tasks piqa,hellaswag,arc_easy

  # Only selective variants
  python test_selective_rotation.py --device cuda:4 \
      --variants none,had_all,had_structural,had_auto

  # SpinQuant R1 selective (custom include/exclude)
  python test_selective_rotation.py --device cuda:4 \
      --variants none,spinquant_r1,spinquant_r1_no_down
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Variant registry
# ═══════════════════════════════════════════════════════════════════════════════

VARIANTS = {
    # ---- baseline -----------------------------------------------------------
    "none": dict(system="-", label="no rotation (baseline)", spec=None),

    # ---- rotation/ transform: ALL layers (current default) ------------------
    "had_all": dict(
        system="rotation", label="Hadamard ALL layers",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "all",
        }},
    ),

    # ---- rotation/ transform: STRUCTURAL selective --------------------------
    "had_structural": dict(
        system="rotation", label="Hadamard STRUCTURAL selective",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "structural",
        }},
    ),

    # ---- rotation/ transform: AUTO selective (stats + structural) -----------
    "had_auto": dict(
        system="rotation", label="Hadamard AUTO selective",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "auto",
        }},
    ),

    # ---- rotation/ transform: custom exclude only down_proj -----------------
    "had_no_down": dict(
        system="rotation", label="Hadamard skip down_proj only",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "all",
            "exclude_layers": ["*down_proj*"],
        }},
    ),

    # ---- rotation/ transform: custom exclude down+o_proj --------------------
    "had_no_down_oproj": dict(
        system="rotation", label="Hadamard skip down+o_proj",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "all",
            "exclude_layers": ["*down_proj*", "*o_proj*"],
        }},
    ),

    # ---- rotation/ inplace: ALL (QuaRot-style) ------------------------------
    "inplace_all": dict(
        system="rotation", label="Inplace (QuaRot) ALL",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "inplace",
            "fuse_online_to_weight": True,
        }},
    ),

    # ---- spinquant R1 (online, all target modules) --------------------------
    "spinquant_r1": dict(
        system="spinquant", label="SpinQuant R1 (all targets)",
        spec={"kind": "spinquant", "flags": dict(r1=True, r2=False, r3=False, r4=False)},
    ),

    # ---- spinquant R1+R4 (for comparison) -----------------------------------
    "spinquant_r1r4": dict(
        system="spinquant", label="SpinQuant R1+R4",
        spec={"kind": "spinquant", "flags": dict(r1=True, r2=False, r3=False, r4=True)},
    ),

    # ---- MoE-specific: structural handles MoE router/gate automatically -----
    "had_structural_moe": dict(
        system="rotation", label="Hadamard STRUCTURAL (MoE-aware)",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "structural",
            # structural mode already skips: mlp.gate, shared_expert_gate, embed_tokens
        }},
    ),

    # ---- MoE: match ignore_layers from quantization script ------------------
    "had_moe_match_quant": dict(
        system="rotation", label="Hadamard match quant ignore_layers",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "structural",
            # Additional excludes to match quantization's fp_layers exactly
            "exclude_layers": ["*self_attn*", "*linear_attn*"],
        }},
    ),
}

# Default variants for dense models
DEFAULT_VARIANTS = "none,had_all,had_structural,had_auto,had_no_down,had_no_down_oproj"

# MoE-focused variant set
MOE_VARIANTS = "none,had_all,had_structural,had_structural_moe,had_moe_match_quant"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, tokenizer, tasks, batch_size=16, limit=None, device="cuda:0"):
    """Evaluation via lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
              device=device, add_bos_token=True, softmax_dtype="float32")
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size, limit=limit)
    metrics = {}
    print(results.get("results", {}))
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


def build_rotation_config(spec, rotation_size):
    """Turn a variant spec into a `rotation_config` value for AutoRound."""
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "string":
        return spec["value"]
    if kind == "dict":
        return spec["value"]
    if kind == "spinquant":
        from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
        flags = spec["flags"]
        rnd = spec.get("random", False)
        return SpinQuantConfig(
            **flags,
            rotation_size=rotation_size,
            online_r1_rotation=True,
            trainable_rotation=False,
            trainable_smooth=False,
            random_r1=rnd and flags.get("r1", False),
            random_r2=rnd and flags.get("r2", False),
            random_r3=rnd and flags.get("r3", False),
            random_r4=rnd and flags.get("r4", False),
        )
    raise ValueError(f"Unknown spec kind: {kind!r}")


def run_variant(model_name, tokenizer, variant_key, args):
    """Quantize model with the variant's rotation_config + scheme, eval it."""
    from auto_round import AutoRound

    spec = VARIANTS[variant_key]["spec"]
    rotation_config = build_rotation_config(spec, args.rotation_size)
    scheme = args.scheme

    t0 = time.time()
    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    # For large models or device_map=auto, don't set single device
    if args.device == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Build AutoRound kwargs
    ar_kwargs = dict(
        tokenizer=tokenizer,
        rotation_config=rotation_config,
        scheme=scheme,
        iters=args.quant_iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        device_map=args.device,
    )
    # Pass ignore_layers if specified (for MoE models)
    if args.ignore_layers:
        ar_kwargs["ignore_layers"] = args.ignore_layers
    # Low GPU memory mode for large models
    if getattr(args, "low_gpu_mem", False):
        ar_kwargs["low_gpu_mem_usage"] = True

    ar = AutoRound(model, **ar_kwargs)
    ar.quantize()
    model = ar.model.eval()
    if args.device != "auto":
        model = model.to(args.device)
    setup_s = time.time() - t0

    # Print selective decisions if available.
    if hasattr(model, "rotation_decisions"):
        decisions = model.rotation_decisions
        enabled = sum(1 for d in decisions.values() if d.enabled)
        total = len(decisions)
        print(f"    [selective] {enabled}/{total} layers rotated")

    metrics = evaluate_model(
        model, tokenizer, args.tasks,
        batch_size=args.batch_size, limit=args.limit, device=args.device,
    )

    del model, ar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, round(setup_s, 1)


def print_table(results, tasks, model_name, scheme):
    task_list = [t.strip() for t in tasks.split(",")]
    print(f"\n{'═' * 100}")
    print(f"  Selective Rotation Comparison — {model_name} | scheme={scheme}")
    print(f"{'═' * 100}")
    header = f"  {'Variant':<40} {'System':<10}"
    for t in task_list:
        header += f" {t[:12]:>12}"
    header += f" {'time(s)':>9}"
    print(header)
    print(f"  {'-' * 96}")

    baseline_acc = {}
    for key, (metrics, setup_s) in results.items():
        if key == "none":
            baseline_acc = metrics
            break

    for key, (metrics, setup_s) in results.items():
        label = VARIANTS[key]["label"]
        system = VARIANTS[key]["system"]
        row = f"  {label:<40} {system:<10}"
        for t in task_list:
            v = metrics.get(t)
            if v is not None:
                # Show delta from baseline.
                base = baseline_acc.get(t)
                delta_str = ""
                if base is not None and key != "none":
                    delta = v - base
                    delta_str = f" ({'+' if delta >= 0 else ''}{delta:.4f})"
                row += f" {v:.4f}{delta_str:>12}"[:13].rjust(12)
            else:
                row += f" {'—':>12}"
        row += f" {setup_s:>9}"
        print(row)
    print(f"{'═' * 100}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare selective vs all-layer Hadamard rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MoE model example:
  python test_selective_rotation.py \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --moe \\
      --device auto \\
      --variants none,had_all,had_structural,had_moe_match_quant \\
      --tasks piqa --limit 200
        """,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scheme", default="MXFP4")
    parser.add_argument("--variants", default=None,
                        help=f"Comma-separated. Available: {','.join(VARIANTS)}")
    parser.add_argument("--tasks", default="piqa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rotation-size", type=int, default=128)
    parser.add_argument("--quant-iters", type=int, default=0, help="0 = RTN")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ignore-layers", type=str, default="",
                        help="Comma-separated layer patterns to skip quantization (passed to AutoRound)")
    parser.add_argument("--moe", action="store_true",
                        help="MoE model preset: sets ignore_layers and default variants for MoE")
    parser.add_argument("--low-gpu-mem", action="store_true",
                        help="Enable low_gpu_mem_usage for large models")
    args = parser.parse_args()

    # MoE preset: auto-configure ignore_layers and default variants
    if args.moe:
        if not args.ignore_layers:
            args.ignore_layers = "lm_head,mlp.gate,shared_expert_gate,embed_tokens"
        if args.variants is None:
            args.variants = MOE_VARIANTS

    # Fallback default variants
    if args.variants is None:
        args.variants = DEFAULT_VARIANTS

    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variant_keys if v not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}\nAvailable: {list(VARIANTS)}")
        sys.exit(1)

    print(f"{'═' * 100}")
    print(f"  Selective Hadamard Rotation — Comparison Test")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {args.device}")
    print(f"  Scheme:   {args.scheme}  (iters={args.quant_iters}, nsamples={args.nsamples})")
    print(f"  Variants: {variant_keys}")
    print(f"  Tasks:    {args.tasks}  (limit={args.limit})")
    print(f"{'═' * 100}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = {}
    for key in variant_keys:
        print(f"\n>>> [{key}] {VARIANTS[key]['label']} ({VARIANTS[key]['system']})")
        try:
            metrics, setup_s = run_variant(args.model, tokenizer, key, args)
            results[key] = (metrics, setup_s)
            print(f"    metrics={metrics}  setup={setup_s}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = ({"ERROR": str(e)[:60]}, 0.0)

    print_table(results, args.tasks, args.model, args.scheme)


if __name__ == "__main__":
    main()
