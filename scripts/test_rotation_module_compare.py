#!/usr/bin/env python3
"""
Compare auto-round's TWO rotation implementations side-by-side via lm_eval.

auto-round ships two rotation systems (both reached through the same
``AutoRound(rotation_config=...)`` argument):

  1. spinquant/  — SpinQuant/QuaRot R1–R4 residual-stream rotation.
                   Driver: SpinQuantConfig(r1/r2/r3/r4=...).
  2. rotation/   — Hadamard "transform" / "inplace" backends.
                   Driver: string "default"/"hadamard"/"random_hadamard", or
                   a {"algorithm":"hadamard", ...} dict (inplace backend).

This script runs the SAME model + SAME quant scheme (MXFP4 by default) through a
set of rotation variants drawn from BOTH systems, evaluates each on real
datasets with lm_eval, and prints one comparison table. RTN only (iters=0), no
rotation training — so it isolates the effect of each rotation implementation.

See rotation_module_hadamard.md (same dir) for the design write-up.

Examples
--------
  # default: Qwen3-0.6B, MXFP4, piqa, all variants
  python test_rotation_module_compare.py --device cuda:4

  # pick variants + tasks + a small limit for a quick smoke test
  python test_rotation_module_compare.py --device cuda:4 \
      --variants none,spinquant_r1,had_default,had_random \
      --tasks piqa,hellaswag --limit 200

  # NVFP4 instead of MXFP4
  python test_rotation_module_compare.py --device cuda:4 --scheme NVFP4
"""

from __future__ import annotations

import argparse
import gc
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Variant registry
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each variant maps to a `rotation_config` value accepted by AutoRound, plus a
# human label and which subsystem ("spinquant" | "rotation") it exercises.
#
# rotation_config is resolved lazily (some need SpinQuantConfig built at runtime),
# so we store a small spec dict and build the config in `build_rotation_config`.

VARIANTS = {
    # ---- baseline -----------------------------------------------------------
    "none": dict(system="-", label="no rotation", spec=None),

    # ---- spinquant/ (R1–R4) -------------------------------------------------
    "spinquant_r1": dict(
        system="spinquant", label="SpinQuant R1 (online)",
        spec={"kind": "spinquant", "flags": dict(r1=True, r2=False, r3=False, r4=False)},
    ),
    "spinquant_r1234": dict(
        system="spinquant", label="SpinQuant R1+R2+R3+R4",
        spec={"kind": "spinquant", "flags": dict(r1=True, r2=True, r3=True, r4=True)},
    ),
    "spinquant_r1_rand": dict(
        system="spinquant", label="SpinQuant R1 (random Hadamard)",
        spec={"kind": "spinquant", "flags": dict(r1=True, r2=False, r3=False, r4=False),
              "random": True},
    ),

    # ---- rotation/ transform backend (per-Linear block Hadamard) ------------
    "had_default": dict(
        system="rotation", label="Hadamard transform (default, det)",
        spec={"kind": "string", "value": "default"},
    ),
    "had_random": dict(
        system="rotation", label="Hadamard transform (random)",
        spec={"kind": "string", "value": "random_hadamard"},
    ),

    # ---- rotation/ transform: SELECTIVE modes --------------------------------
    "had_structural": dict(
        system="rotation", label="Hadamard STRUCTURAL selective",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "structural",
        }},
    ),
    "had_auto": dict(
        system="rotation", label="Hadamard AUTO selective",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "auto",
        }},
    ),
    "had_no_down": dict(
        system="rotation", label="Hadamard skip down_proj",
        spec={"kind": "dict", "value": {
            "hadamard_type": "hadamard",
            "backend": "transform",
            "layer_selection": "all",
            "exclude_layers": ["*down_proj*"],
        }},
    ),

    # ---- rotation/ inplace backend (QuaRot residual-stream) -----------------
    "had_inplace": dict(
        system="rotation", label="Hadamard inplace (QuaRot, fused)",
        spec={"kind": "dict", "value": {
            "algorithm": "hadamard", "hadamard_type": "hadamard",
            "backend": "inplace", "fuse_online_to_weight": True,
        }},
    ),
}

DEFAULT_VARIANTS = "none,spinquant_r1,spinquant_r1234,had_default,had_structural,had_auto,had_inplace"

# scheme name -> auto-round scheme string
SCHEME_MAP = {
    "MXFP4": "MXFP4",
    "NVFP4": "NVFP4",
    "W4A16": "W4A16",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, tokenizer, tasks, batch_size=16, limit=None, device="cuda:0"):
    """Common evaluation via lm_eval (matches the other scripts in this dir)."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
              device=device, add_bos_token=True, softmax_dtype="float32")
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks
    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        gen_kwargs="max_gen_toks=2048",
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42)
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
    """Quantize `model_name` with the variant's rotation_config + scheme, eval it."""
    from auto_round import AutoRound

    spec = VARIANTS[variant_key]["spec"]
    rotation_config = build_rotation_config(spec, args.rotation_size)
    scheme = SCHEME_MAP.get(args.scheme, args.scheme)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    ar = AutoRound(
        model,
        tokenizer=tokenizer,
        rotation_config=rotation_config,
        scheme=scheme,
        iters=args.quant_iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        device_map=args.device,
    )
    ar.quantize()
    model = ar.model.eval().to(args.device)
    setup_s = time.time() - t0

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
    print(f"\n{'═' * 96}")
    print(f"  Rotation-module comparison — {model_name} | scheme={scheme}")
    print(f"{'═' * 96}")
    header = f"  {'Variant':<34} {'System':<10}"
    for t in task_list:
        header += f" {t[:12]:>12}"
    header += f" {'time(s)':>9}"
    print(header)
    print(f"  {'-' * 92}")
    for key, (metrics, setup_s) in results.items():
        label = VARIANTS[key]["label"]
        system = VARIANTS[key]["system"]
        row = f"  {label:<34} {system:<10}"
        for t in task_list:
            v = metrics.get(t)
            row += f" {('%.4f' % v) if v is not None else '—':>12}"
        row += f" {setup_s:>9}"
        print(row)
    print(f"{'═' * 96}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare auto-round spinquant vs rotation(Hadamard) implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scheme", default="MXFP4",
                        help="MXFP4 | NVFP4 | W4A16 (or a raw auto-round scheme string)")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS,
                        help=f"Comma-separated. Available: {','.join(VARIANTS)}")
    parser.add_argument("--tasks", default="piqa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rotation-size", type=int, default=128,
                        help="Block rotation size for the spinquant variants")
    parser.add_argument("--quant-iters", type=int, default=0, help="0 = RTN")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variant_keys if v not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}\nAvailable: {list(VARIANTS)}")
        sys.exit(1)

    print(f"{'═' * 96}")
    print(f"  auto-round rotation-module comparison")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {args.device}")
    print(f"  Scheme:   {args.scheme}  (iters={args.quant_iters}, nsamples={args.nsamples})")
    print(f"  Variants: {variant_keys}")
    print(f"  Tasks:    {args.tasks}  (limit={args.limit})")
    print(f"{'═' * 96}")

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
