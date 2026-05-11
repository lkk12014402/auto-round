#!/usr/bin/env python3
"""
Auto-Round: Batch Rotation × Quantization Scheme Accuracy Matrix (v2)

v2 changes from v1 (test_rotation_scheme_matrix.py):
  - Uses the AutoRound pipeline integration: rotation_config parameter
    (rotation is automatically applied at Phase 4.5, before quantization)
  - Supports non-power-of-2 dimensions natively (via known Hadamard matrices)
  - rotation_size is no longer required for models like Qwen3-0.6B
  - Adds R1+R2+R3 as a rotation level (v1 skipped from R1+R2 to R1+R2+R3+R4)
  - Supports string config shorthand: "quarot", or dict-based config
  - mmlu added to default evaluation tasks

Pipeline integration:
  Instead of manually calling apply_rotation() then AutoRound(), this script
  passes rotation_config directly to AutoRound(). The BaseCompressor pipeline
  applies rotation automatically at Phase 4.5:

    AutoRound(model, tokenizer, rotation_config=SpinQuantConfig(...), scheme=..., iters=...)

  Supported rotation_config values:
    - "quarot"           — QuaRot defaults (R1+R2+R3+R4, fixed Hadamard)
    - "spinquant"        — SpinQuant defaults (with learnable rotations)
    - SpinQuantConfig()  — Full control over all parameters
    - dict               — {"algorithm": "spinquant", "r1": True, ...}

Modes (via run_rotation_scheme_matrix_v2.sh):
  quick       — limit=100, 3 common schemes (W4A16, MXFP4, NVFP4), 2 tasks.
                Good for ~15min sanity check.
  full        — No limit, 3 common schemes, 5 tasks.
                Full accuracy numbers, ~2 hours.
  full-matrix — No limit, ALL rotations × ALL schemes (11 schemes).
                Exhaustive test, ~8+ hours.
  weight-only — W2/W3/W4/W8 A16 (rotation impact on weight-only quant)
  weight-act  — MXFP4/NVFP4/INT8/MXFP8 (R3/R4 matter most here)
  random      — Same as full but random Hadamard (H×D)
  tuning      — W4A16 with iters=200 (rotation + auto-round tuning)

Rotation levels:
  none        — No rotation (quantization-only baseline)
  R1          — Residual stream rotation (hidden_size)
  R1+R2       — + per-head V/O rotation (head_dim)
  R1+R2+R3    — + Q/K rotation after RoPE (online)
  R1+R2+R3+R4 — + MLP activation rotation (online)

⚠️  This script tests QuaRot (fixed Hadamard rotation) only.
    SpinQuant (trainable rotation) training loop is experimental and not included.

Usage:
    # Quick test:
    python test_rotation_scheme_matrix_v2.py --device cuda:7 --limit 100 \\
        --rotations "R1,R1+R2" --schemes "W4A16"

    # Full matrix:
    python test_rotation_scheme_matrix_v2.py --device cuda:7 --full-matrix

    # All 5 rotation levels with common schemes:
    python test_rotation_scheme_matrix_v2.py --device cuda:7 \\
        --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \\
        --schemes "W4A16,MXFP4,NVFP4"

    # Auto-round tuning (iters=200, better accuracy):
    python test_rotation_scheme_matrix_v2.py --device cuda:7 --quant-iters 200

    # Sweep rotation_sizes (block rotation dimension):
    python test_rotation_scheme_matrix_v2.py --device cuda:7 \\
        --rotation-sizes "16,32,64,128,auto" --rotations "R1,R1+R2+R3+R4"

    # Custom model:
    python test_rotation_scheme_matrix_v2.py --model meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

# ═══════════════════════════════════════════════════════════════════════════════
# Rotation and scheme definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Each rotation level maps to SpinQuantConfig flags.
# Passed to AutoRound(rotation_config=SpinQuantConfig(...))
ROTATION_LEVELS = OrderedDict([
    ("none",        dict(r1=False, r2=False, r3=False, r4=False)),
    ("R1",          dict(r1=True,  r2=False, r3=False, r4=False)),
    ("R1+R2",       dict(r1=True,  r2=True,  r3=False, r4=False)),
    ("R1+R2+R3",    dict(r1=True,  r2=True,  r3=True,  r4=False)),
    ("R1+R2+R3+R4", dict(r1=True,  r2=True,  r3=True,  r4=True)),
])

# Quantization scheme definitions
# (autoround_scheme_name, description, category)
SCHEME_DEFS = OrderedDict([
    # Weight-only integer
    ("W4A16",       ("W4A16",       "INT4 weight-only, gs=128",     "weight_only")),
    ("W3A16",       ("W3A16",       "INT3 weight-only, gs=128",     "weight_only")),
    ("W2A16",       ("W2A16",       "INT2 weight-only, gs=128",     "weight_only")),
    ("W8A16",       ("W8A16",       "INT8 weight-only, gs=128",     "weight_only")),
    # Weight+Activation block float
    ("MXFP4",       ("MXFP4_RCEIL", "MXFP4 W4A4, block=32",        "weight_act")),
    ("NVFP4",       ("NVFP4",       "NVFP4 W4A4, gs=16",           "weight_act")),
    ("MXFP8",       ("MXFP8",       "MXFP8 W8A8, block=32",        "weight_act")),
    # Weight+Activation integer
    ("INT8",        ("INT8",        "INT8 W8A8, per-tensor",        "weight_act")),
    ("INT4",        ("INT4",        "INT4 W4A4, per-tensor",        "weight_act")),
    # FP8
    ("FP8_STATIC",  ("FP8_STATIC",  "FP8 static, per-tensor",      "fp8")),
    ("FP8_BLOCK",   ("FP8_BLOCK",   "FP8 block 128×128",            "fp8")),
])

COMMON_SCHEMES = ["W4A16", "MXFP4", "NVFP4"]
WEIGHT_ONLY_SCHEMES = ["W2A16", "W3A16", "W4A16", "W8A16"]
WEIGHT_ACT_SCHEMES = ["MXFP4", "NVFP4", "INT8", "MXFP8"]

DEFAULT_TASKS = "hellaswag,piqa,winogrande,lambada_openai,mmlu"


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: str, dtype=torch.float16):
    """Load a fresh model instance."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model


def build_rotation_config(
    rotation_flags: dict,
    rotation_size: int | None = None,
    online_r1: bool = True,
    random_r1: bool = False,
    random_r2: bool = False,
) -> SpinQuantConfig | None:
    """Build a SpinQuantConfig for the given rotation flags.

    Returns None if all rotation flags are False (i.e., "none" rotation level).
    The returned config is passed to ``AutoRound(rotation_config=...)`` so that
    rotation is applied automatically at Phase 4.5 in the pipeline.

    Args:
        rotation_flags: dict with r1, r2, r3, r4 booleans.
        rotation_size: Optional custom rotation dimension. With v2, this is
            rarely needed — known Hadamard matrices handle most non-pow2 sizes.
        online_r1: Use online R1 (recommended).
        random_r1: Use random Hadamard for R1.
        random_r2: Use random Hadamard for R2.

    Returns:
        SpinQuantConfig or None.
    """
    if not any(rotation_flags.values()):
        return None
    return SpinQuantConfig(
        **rotation_flags,
        rotation_size=rotation_size,
        online_r1_rotation=online_r1,
        trainable_rotation=False,   # QuaRot: no training
        trainable_smooth=False,
        random_r1=random_r1,
        random_r2=random_r2,
    )


def evaluate_model(
    model,
    tokenizer,
    tasks: str | list[str],
    batch_size: int = 8,
    limit: int | None = None,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Evaluate model accuracy using lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks

    results = simple_evaluate(
        model=lm, tasks=task_list, batch_size=batch_size, limit=limit, device=device
    )
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc_norm,none") or task_results.get("acc,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


def run_single_combination(
    model_name: str,
    tokenizer,
    rotation_name: str,
    rotation_flags: dict,
    scheme_name: str,
    scheme_str: str,
    args,
    rotation_size: int | None = None,
) -> dict[str, Any]:
    """Run one rotation × scheme combination via AutoRound pipeline.

    Uses ``AutoRound(rotation_config=...)`` so that rotation is applied
    automatically at Phase 4.5, before layer-wise quantization.
    """
    label = f"{rotation_name} + {scheme_name}" if rotation_name != "none" else f"{scheme_name} only"
    logger.info(f"\n{'='*70}")
    logger.info(f"  {label}")
    logger.info(f"  Rotation: {rotation_name} | Scheme: {scheme_name} ({scheme_str})")
    logger.info(f"{'='*70}")

    result = {
        "rotation": rotation_name,
        "scheme": scheme_name,
        "scheme_str": scheme_str,
        "label": label,
        "random_hadamard": args.random_hadamard,
        "quant_iters": args.quant_iters,
        "rotation_size": rotation_size,
        "api": "pipeline_v2",
    }

    # Build rotation config (None for "none" rotation)
    rotation_config = build_rotation_config(
        rotation_flags, rotation_size,
        online_r1=args.online_r1,
        random_r1=args.random_hadamard,
        random_r2=args.random_hadamard,
    )

    t0 = time.time()
    model = None
    try:
        model = load_model(model_name, args.device)

        # ── Pipeline: rotation_config + quantization in one call ──
        if rotation_config is not None:
            logger.info(f"  rotation_config = SpinQuantConfig("
                        f"r1={rotation_config.r1}, r2={rotation_config.r2}, "
                        f"r3={rotation_config.r3}, r4={rotation_config.r4}, "
                        f"online_r1={rotation_config.online_r1_rotation}, "
                        f"random_r1={rotation_config.random_r1})")
        else:
            logger.info(f"  rotation_config = None (no rotation)")

        logger.info(f"  AutoRound(rotation_config=..., scheme={scheme_str}, "
                    f"iters={args.quant_iters})")

        ar = AutoRound(
            model,
            tokenizer=tokenizer,
            rotation_config=rotation_config,
            scheme=scheme_str,
            iters=args.quant_iters,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            device_map=args.device,
        )
        ar.quantize()
        model = ar.model
        model.eval()
        model.to(args.device)

        pipeline_time = time.time() - t0
        result["setup_time_s"] = round(pipeline_time, 1)
        logger.info(f"  Pipeline (rotation+quantization) done in {pipeline_time:.1f}s")

        # Evaluate
        logger.info(f"  Evaluating on tasks: {args.tasks}")
        metrics = evaluate_model(
            model, tokenizer, args.tasks,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )

        result["metrics"] = metrics
        result["status"] = "success"
        result["total_time_s"] = round(time.time() - t0, 1)

        for task, acc in sorted(metrics.items()):
            logger.info(f"    {task}: {acc:.4f}")

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "error"
        result["error"] = str(e)
        result["total_time_s"] = round(time.time() - t0, 1)

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FP16 baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_fp16_baseline(model_name: str, tokenizer, args) -> dict[str, Any]:
    """Run FP16 baseline evaluation (no rotation, no quantization)."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  FP16 BASELINE (no rotation, no quantization)")
    logger.info(f"{'='*70}")

    result = {
        "rotation": "none", "scheme": "FP16", "label": "FP16 baseline",
        "random_hadamard": False, "quant_iters": 0, "rotation_size": None,
        "api": "pipeline_v2",
    }

    t0 = time.time()
    model = None
    try:
        model = load_model(model_name, args.device)
        metrics = evaluate_model(
            model, tokenizer, args.tasks,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )
        result["metrics"] = metrics
        result["status"] = "success"
        result["total_time_s"] = round(time.time() - t0, 1)

        for task, acc in sorted(metrics.items()):
            logger.info(f"    {task}: {acc:.4f}")

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        result["status"] = "error"
        result["error"] = str(e)
        result["total_time_s"] = round(time.time() - t0, 1)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Results formatting
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_matrix(all_results: list[dict], tasks: list[str]):
    """Print results as a rotation × scheme matrix table."""
    task_list = sorted(tasks)

    rotations_seen = []
    schemes_seen = []
    for r in all_results:
        rot = r["rotation"]
        sch = r["scheme"]
        if rot not in rotations_seen:
            rotations_seen.append(rot)
        if sch not in schemes_seen:
            schemes_seen.append(sch)

    lookup = {}
    for r in all_results:
        key = (r["rotation"], r["scheme"])
        lookup[key] = r

    # Per-task matrix
    for task in task_list:
        print(f"\n{'═'*90}")
        print(f"  Task: {task}")
        print(f"{'═'*90}")

        header = f"  {'Rotation':<20}"
        for sch in schemes_seen:
            header += f" | {sch:>12}"
        print(header)
        print(f"  {'─'*20}" + "─┼─".join(["─" * 12] * len(schemes_seen)))

        for rot in rotations_seen:
            row = f"  {rot:<20}"
            for sch in schemes_seen:
                r = lookup.get((rot, sch))
                if r and r.get("status") == "success":
                    acc = r["metrics"].get(task, None)
                    if acc is not None:
                        row += f" | {acc:>12.4f}"
                    else:
                        row += f" | {'N/A':>12}"
                elif r and r.get("status") == "error":
                    row += f" | {'ERROR':>12}"
                else:
                    row += f" | {'—':>12}"
            print(row)

    # Average accuracy matrix
    print(f"\n{'═'*90}")
    print(f"  Average Accuracy (across {len(task_list)} tasks)")
    print(f"{'═'*90}")
    header = f"  {'Rotation':<20}"
    for sch in schemes_seen:
        header += f" | {sch:>12}"
    print(header)
    print(f"  {'─'*20}" + "─┼─".join(["─" * 12] * len(schemes_seen)))

    for rot in rotations_seen:
        row = f"  {rot:<20}"
        for sch in schemes_seen:
            r = lookup.get((rot, sch))
            if r and r.get("status") == "success" and r.get("metrics"):
                vals = [v for v in r["metrics"].values() if isinstance(v, (int, float))]
                if vals:
                    avg = sum(vals) / len(vals)
                    row += f" | {avg:>12.4f}"
                else:
                    row += f" | {'N/A':>12}"
            elif r and r.get("status") == "error":
                row += f" | {'ERROR':>12}"
            else:
                row += f" | {'—':>12}"
        print(row)

    # Timing
    print(f"\n{'═'*90}")
    print(f"  Total Time (seconds)")
    print(f"{'═'*90}")
    header = f"  {'Rotation':<20}"
    for sch in schemes_seen:
        header += f" | {sch:>12}"
    print(header)
    print(f"  {'─'*20}" + "─┼─".join(["─" * 12] * len(schemes_seen)))

    for rot in rotations_seen:
        row = f"  {rot:<20}"
        for sch in schemes_seen:
            r = lookup.get((rot, sch))
            if r and "total_time_s" in r:
                row += f" | {r['total_time_s']:>10.1f}s"
            else:
                row += f" | {'—':>12}"
        print(row)


def save_results_json(all_results: list[dict], output_path: str):
    """Save full results to JSON."""
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


def save_results_csv(all_results: list[dict], tasks: list[str], output_path: str):
    """Save results as CSV for spreadsheet import."""
    task_list = sorted(tasks)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rotation", "scheme", "random_hadamard", "quant_iters",
            "rotation_size", "api", "status", "total_time_s"
        ] + task_list + ["avg_accuracy"])

        for r in all_results:
            metrics = r.get("metrics", {})
            accs = [metrics.get(t, "") for t in task_list]
            vals = [v for v in accs if isinstance(v, (int, float))]
            avg = round(sum(vals) / len(vals), 4) if vals else ""
            writer.writerow([
                r.get("rotation", ""),
                r.get("scheme", ""),
                r.get("random_hadamard", ""),
                r.get("quant_iters", ""),
                r.get("rotation_size", ""),
                r.get("api", "unified_v2"),
                r.get("status", ""),
                r.get("total_time_s", ""),
            ] + accs + [avg])

    logger.info(f"CSV saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Rotation × Quantization Scheme Accuracy Matrix (v2 — pipeline API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (via run_rotation_scheme_matrix_v2.sh):
  quick       — limit=100, W4A16/MXFP4/NVFP4, hellaswag+piqa (~15min)
  full        — no limit, W4A16/MXFP4/NVFP4, 5 tasks (~2h)
  full-matrix — no limit, ALL 11 schemes × ALL 5 rotation levels (~8h+)
  weight-only — W2/W3/W4/W8 A16 (rotation impact on weight-only quant)
  weight-act  — MXFP4/NVFP4/INT8/MXFP8 (R3/R4 matter most here)
  random      — same as full but random Hadamard (H×D)
  size-sweep  — sweep rotation_sizes (16,32,64,128,auto) × rotations × schemes
  tuning      — W4A16 with iters=200 (rotation + auto-round tuning)

Pipeline integration:
  Rotation is passed via AutoRound(rotation_config=...) instead of calling
  apply_rotation() manually. The pipeline applies rotation at Phase 4.5
  (after scheme resolution, before layer-wise quantization).

  Supported rotation_config values:
    "quarot"           — QuaRot defaults (R1+R2+R3+R4, fixed Hadamard)
    "spinquant"        — SpinQuant defaults (with learnable rotations)
    SpinQuantConfig()  — Full control over all parameters
    dict               — {"algorithm": "spinquant", "r1": True, ...}

Rotation levels:
  none        — No rotation (quantization-only baseline)
  R1          — Residual stream rotation (hidden_size)
  R1+R2       — + per-head V/O rotation (head_dim)
  R1+R2+R3    — + Q/K rotation after RoPE (online)
  R1+R2+R3+R4 — + MLP activation rotation (online)

Examples:
  # Quick test:
  python test_rotation_scheme_matrix_v2.py --device cuda:7 --limit 100 \\
      --rotations "R1,R1+R2" --schemes "W4A16"

  # Full matrix with all rotation levels:
  python test_rotation_scheme_matrix_v2.py --device cuda:7 --full-matrix

  # Auto-round tuning (better accuracy):
  python test_rotation_scheme_matrix_v2.py --device cuda:7 --quant-iters 200

  # Common schemes with all 5 rotation levels:
  python test_rotation_scheme_matrix_v2.py --device cuda:7 \\
      --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" --schemes "W4A16,MXFP4,NVFP4"
        """,
    )

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name or path (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--device", type=str, default="cuda:7",
                        help="Device (default: cuda:7)")

    # Rotation config
    parser.add_argument("--rotations", type=str,
                        default="none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4",
                        help="Comma-separated rotation levels (default includes all 5 levels)")
    parser.add_argument("--rotation-size", type=int, default=None,
                        help="Custom rotation dimension. Rarely needed with v2 — known "
                             "Hadamard matrices handle most non-pow2 sizes (768, 1536, 3072, "
                             "5120, 7168, 11008, 14336). Only set if the preprocessor warns "
                             "about unsupported dimensions.")
    parser.add_argument("--rotation-sizes", type=str, default=None,
                        help="Comma-separated rotation sizes to sweep, e.g. '16,32,64,128,auto'. "
                             "'auto'/'none' means automatic detection (known Hadamard). "
                             "Overrides --rotation-size when set.")
    parser.add_argument("--online-r1", action="store_true", default=True,
                        help="Use online R1 rotation (default: True, recommended)")
    parser.add_argument("--offline-r1", dest="online_r1", action="store_false",
                        help="Use offline R1 rotation (⚠️ may degrade accuracy with quantization)")
    parser.add_argument("--random-hadamard", action="store_true", default=False,
                        help="Use random Hadamard (H×D) for R1/R2. "
                             "R3/R4 always use deterministic regardless.")

    # Quantization config
    parser.add_argument("--schemes", type=str, default=",".join(COMMON_SCHEMES),
                        help=f"Comma-separated schemes (default: {','.join(COMMON_SCHEMES)}). "
                             f"Available: {','.join(SCHEME_DEFS.keys())}")
    parser.add_argument("--quant-iters", type=int, default=0,
                        help="AutoRound optimization iterations. "
                             "0=RTN (fast, no calibration), 200=auto-round tuning (slower, better)")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Calibration samples for quantization (default: 128)")
    parser.add_argument("--seqlen", type=int, default=512,
                        help="Sequence length for calibration (default: 512)")

    # Matrix presets
    parser.add_argument("--full-matrix", action="store_true",
                        help="Run all 5 rotation levels × all 11 schemes (exhaustive)")
    parser.add_argument("--weight-only", action="store_true",
                        help="Only weight-only schemes: W2A16, W3A16, W4A16, W8A16")
    parser.add_argument("--weight-act", action="store_true",
                        help="Only weight+activation schemes: MXFP4, NVFP4, INT8, MXFP8")

    # Evaluation config
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help=f"Comma-separated lm_eval tasks (default: {DEFAULT_TASKS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per task (default: None=full eval, 100=quick test)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="lm_eval batch size (default: 8)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: auto-generated with timestamp)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip FP16 baseline evaluation")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to files")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve presets
    if args.full_matrix:
        rotation_names = list(ROTATION_LEVELS.keys())
        scheme_names = list(SCHEME_DEFS.keys())
    elif args.weight_only:
        rotation_names = [r.strip() for r in args.rotations.split(",")]
        scheme_names = WEIGHT_ONLY_SCHEMES
    elif args.weight_act:
        rotation_names = [r.strip() for r in args.rotations.split(",")]
        scheme_names = WEIGHT_ACT_SCHEMES
    else:
        rotation_names = [r.strip() for r in args.rotations.split(",")]
        scheme_names = [s.strip() for s in args.schemes.split(",")]

    # Validate rotation names
    for rn in rotation_names:
        if rn not in ROTATION_LEVELS:
            logger.error(f"Unknown rotation level: '{rn}'. "
                         f"Available: {list(ROTATION_LEVELS.keys())}")
            sys.exit(1)

    # Validate scheme names
    for sn in scheme_names:
        if sn not in SCHEME_DEFS:
            logger.error(f"Unknown scheme: '{sn}'. Available: {list(SCHEME_DEFS.keys())}")
            sys.exit(1)

    # Resolve rotation_sizes to sweep
    if args.rotation_sizes:
        rotation_sizes = []
        for s in args.rotation_sizes.split(","):
            s = s.strip().lower()
            if s in ("auto", "none", "null", ""):
                rotation_sizes.append(None)
            else:
                rotation_sizes.append(int(s))
    else:
        rotation_sizes = [args.rotation_size]
    multi_size = len(rotation_sizes) > 1

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        iters_tag = f"_iters{args.quant_iters}" if args.quant_iters > 0 else "_rtn"
        size_tag = f"_rs{'_'.join(str(s) if s else 'auto' for s in rotation_sizes)}" if multi_size else ""
        args.output_dir = f"results_v2_{model_short}{iters_tag}{size_tag}_{timestamp}"

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Print test plan
    # Calculate total combinations (deduplicate "none" across rotation_sizes)
    n_rot_with_size = len([r for r in rotation_names if r != "none"])
    n_none = 1 if "none" in rotation_names else 0
    n_combos = (n_rot_with_size * len(rotation_sizes) + n_none) * len(scheme_names)
    total_combos = n_combos + (0 if args.no_baseline else 1)
    task_list = [t.strip() for t in args.tasks.split(",")]

    logger.info(f"\n{'═'*70}")
    logger.info(f"  ROTATION × QUANTIZATION ACCURACY MATRIX (v2)")
    logger.info(f"{'═'*70}")
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Device:         {args.device}")
    logger.info(f"  Rotations:      {rotation_names}")
    logger.info(f"  Schemes:        {scheme_names}")
    logger.info(f"  Total combos:   {total_combos}")
    logger.info(f"  Tasks:          {task_list}")
    logger.info(f"  Eval limit:     {args.limit or 'full'}")
    logger.info(f"  Quant iters:    {args.quant_iters} "
                f"({'RTN' if args.quant_iters == 0 else 'auto-round tuning'})")
    if multi_size:
        rs_display = [str(s) if s is not None else "auto" for s in rotation_sizes]
        logger.info(f"  rotation_sizes: {rs_display}")
    else:
        logger.info(f"  rotation_size:  {args.rotation_size or 'auto (known Hadamard)'}")
    logger.info(f"  Hadamard type:  {'Random' if args.random_hadamard else 'Deterministic'}")
    logger.info(f"  Online R1:      {args.online_r1}")
    logger.info(f"  API:            AutoRound(rotation_config=...) pipeline (v2)")
    logger.info(f"  Output dir:     {args.output_dir if not args.no_save else 'disabled'}")
    logger.info(f"{'═'*70}\n")

    # Load tokenizer (shared across all runs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_results = []
    t_start = time.time()

    # FP16 baseline
    if not args.no_baseline:
        baseline = run_fp16_baseline(args.model, tokenizer, args)
        all_results.append(baseline)

    # Run all combinations
    combo_idx = 0
    none_done = False
    for rs in rotation_sizes:
        rs_label = str(rs) if rs is not None else "auto"
        if multi_size:
            logger.info(f"\n{'─'*70}")
            logger.info(f"  ▶ rotation_size = {rs_label}")
            logger.info(f"{'─'*70}")

        for rot_name in rotation_names:
            # Skip duplicate "none" runs — rotation_size is irrelevant for no-rotation
            if rot_name == "none" and none_done:
                continue
            if rot_name == "none":
                none_done = True

            rot_flags = ROTATION_LEVELS[rot_name]

            # Display name includes rotation_size when sweeping multiple sizes
            rot_display = rot_name
            if multi_size and rot_name != "none":
                rot_display = f"{rot_name} (rs={rs_label})"

            for sch_name in scheme_names:
                combo_idx += 1
                sch_str, sch_desc, sch_cat = SCHEME_DEFS[sch_name]

                logger.info(f"\n[{combo_idx}/{n_combos}] "
                            f"{rot_display} × {sch_name} ({sch_desc})")

                result = run_single_combination(
                    args.model, tokenizer,
                    rot_display, rot_flags,
                    sch_name, sch_str,
                    args,
                    rotation_size=rs,
                )
                all_results.append(result)

                # Save intermediate results
                if not args.no_save:
                    save_results_json(all_results, os.path.join(args.output_dir, "results.json"))

    total_time = time.time() - t_start

    # Print summary
    print_results_matrix(all_results, task_list)

    print(f"\n{'═'*70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Combinations tested: {len(all_results)}")
    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_error = sum(1 for r in all_results if r.get("status") == "error")
    print(f"  Success: {n_success}, Errors: {n_error}")
    print(f"  API: AutoRound(rotation_config=...) pipeline (v2)")
    print(f"{'═'*70}")

    # Save final results
    if not args.no_save:
        save_results_json(all_results, os.path.join(args.output_dir, "results.json"))
        save_results_csv(all_results, task_list, os.path.join(args.output_dir, "results.csv"))

        config_data = {
            "model": args.model, "device": args.device,
            "rotations": rotation_names, "schemes": scheme_names,
            "tasks": task_list, "limit": args.limit,
            "quant_iters": args.quant_iters,
            "rotation_sizes": [str(s) if s is not None else "auto" for s in rotation_sizes],
            "random_hadamard": args.random_hadamard, "online_r1": args.online_r1,
            "api": "pipeline_v2",
            "total_time_s": round(total_time, 1),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"\n  Results saved to: {args.output_dir}/")
        print(f"    - results.json  (full results)")
        print(f"    - results.csv   (for spreadsheet)")
        print(f"    - config.json   (run configuration)")


if __name__ == "__main__":
    main()
