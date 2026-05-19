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
  layerwise   — Block-wise rotation (layerwise_rotation=True), RTN
  layerwise-tuning — Block-wise rotation + iters=200 auto-round tuning

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

    # Multi-GPU for large models (32B/70B/122B):
    python test_rotation_scheme_matrix_v2.py --device "0,1,2,3" \\
        --model meta-llama/Llama-3.1-70B --rotations "R1,R1+R2+R3+R4"

    # Multi-GPU with auto device mapping:
    python test_rotation_scheme_matrix_v2.py --device auto \\
        --model Qwen/Qwen2.5-32B --layerwise

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
import random
import shutil
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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def is_multi_gpu(device: str) -> bool:
    """Check if device spec indicates multi-GPU usage.

    Multi-GPU formats:
      - "auto"       — accelerate auto device_map
      - "0,1,2,3"   — explicit GPU list
      - "0,1"       — two GPUs
    Single-GPU formats:
      - "cuda:0"    — single GPU
      - "cuda"      — default GPU
      - "cpu"       — CPU only
    """
    if device is None:
        return False
    device = str(device).strip()
    if device == "auto":
        return True
    if "," in device:
        return True
    return False


def get_primary_device(device: str) -> str:
    """Get the primary device for single-tensor operations.

    For multi-GPU: returns "cuda:0" (first GPU).
    For single-GPU: returns the device as-is.
    """
    if not is_multi_gpu(device):
        return device
    device = str(device).strip()
    if device == "auto":
        return "cuda:0"
    # "0,1,2,3" → "cuda:0"
    first_gpu = device.split(",")[0].strip()
    return f"cuda:{first_gpu}"


def load_model(model_name: str, device: str, dtype=torch.bfloat16,
               for_eval_only: bool = False):
    """Load a fresh model instance.

    Args:
        dtype: Model dtype. Default is bfloat16 which is safer for large models
            (float16 can overflow in attention scores for 30B+ models).
        for_eval_only: If True, load for direct evaluation (not for AutoRound).
            Multi-GPU models are loaded with device_map="auto".
            If False (default), multi-GPU models stay on CPU for AutoRound to handle.

    For multi-GPU (device="auto" or "0,1,2,3"):
      - for_eval_only=True:  device_map="auto" (model distributed across GPUs)
      - for_eval_only=False: stays on CPU (AutoRound dispatches internally)
    For single-GPU (device="cuda:X"):
      Loads to the specified device directly.
    """
    if is_multi_gpu(device):
        if for_eval_only:
            logger.info(f"  Loading model with device_map='auto' for evaluation...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=True,
                device_map="auto",
            )
        else:
            logger.info(f"  Loading model to CPU (AutoRound will dispatch to multi-GPU)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, trust_remote_code=True,
            )
    else:
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
    random_r3: bool = False,
    random_r4: bool = False,
) -> SpinQuantConfig | None:
    """Build a SpinQuantConfig for the given rotation flags.

    Returns None if all rotation flags are False (i.e., "none" rotation level).
    The returned config is passed to ``AutoRound(rotation_config=...)`` so that
    rotation is applied automatically at Phase 4.5 in the pipeline.

    Args:
        rotation_flags: dict with r1, r2, r3, r4 booleans.
        rotation_size: Optional custom rotation dimension.
        online_r1: Use online R1 (recommended).
        random_r1: Use random Hadamard for R1.
        random_r2: Use random Hadamard for R2.
        random_r3: Use random Hadamard for R3.
        random_r4: Use random Hadamard for R4.

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
        random_r3=random_r3,
        random_r4=random_r4,
    )


def evaluate_model(
    model,
    tokenizer,
    tasks: str | list[str],
    batch_size: int = 8,
    limit: int | None = None,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Evaluate model accuracy using lm_eval.

    For multi-GPU: model is already distributed, HFLM uses it as-is.
    For single-GPU: passes device to HFLM normally.

    Uses add_bos_token=True and softmax_dtype="float32" to match vLLM backend
    precision and ensure correct logprob computation for models that require BOS.
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    multi_gpu = is_multi_gpu(device)
    primary_dev = get_primary_device(device)

    common_kwargs = dict(
        pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
        add_bos_token=True, softmax_dtype="float32",
    )
    if multi_gpu:
        lm = HFLM(**common_kwargs)
    else:
        lm = HFLM(**common_kwargs, device=primary_dev)
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks

    results = simple_evaluate(
        model=lm, tasks=task_list, batch_size=batch_size, limit=limit,
        gen_kwargs="max_gen_toks=2048",
    )
    print("==results: ", results.get("results", {}))
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


def evaluate_model_from_path(
    model_path: str,
    tasks: str | list[str],
    batch_size: int = 8,
    limit: int | None = None,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Load a saved model from disk and evaluate using lm_eval.

    For multi-GPU: uses parallelize=True so lm_eval distributes the model.
    For single-GPU: loads to the specified device.

    Uses dtype=bfloat16 (safer for large models), add_bos_token=True, and
    softmax_dtype="float32" to match vLLM backend precision.
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    multi_gpu = is_multi_gpu(device)
    primary_dev = get_primary_device(device)

    common_kwargs = dict(
        pretrained=model_path, batch_size=batch_size,
        dtype="bfloat16", trust_remote_code=True,
        add_bos_token=True, softmax_dtype="float32",
    )
    if multi_gpu:
        lm = HFLM(**common_kwargs, parallelize=True)
    else:
        lm = HFLM(**common_kwargs, device=primary_dev)
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks

    results = simple_evaluate(
        model=lm, tasks=task_list, batch_size=batch_size, limit=limit,
        gen_kwargs="max_gen_toks=2048",
    )
    print("==results: ", results.get("results", {}))
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
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
    random_override: tuple[bool, bool, bool, bool] | None = None,
    matrix_mode: str = "det",
) -> dict[str, Any]:
    """Run one rotation × scheme combination via AutoRound pipeline.

    Uses ``AutoRound(rotation_config=...)`` so that rotation is applied
    automatically at Phase 4.5, before layer-wise quantization.

    When ``args.save_load`` is True, also saves the model to disk, loads it
    back, and evaluates from disk for roundtrip verification.

    Args:
        random_override: If provided, overrides args.random_r1/r2/r3/r4.
            Used by --compare-random to run det and random variants.
        matrix_mode: "det" or "random" — recorded in results for grouping.
    """
    if random_override is not None:
        rr1, rr2, rr3, rr4 = random_override
    else:
        rr1 = args.random_r1
        rr2 = args.random_r2
        rr3 = args.random_r3
        rr4 = args.random_r4

    is_random = rr1 or rr2 or rr3 or rr4
    rot_mode = "random" if is_random else "deterministic"
    label = (f"{rotation_name} + {scheme_name} ({rot_mode})"
             if rotation_name != "none" else f"{scheme_name} only")
    logger.info(f"\n{'='*70}")
    logger.info(f"  {label}")
    logger.info(f"  Rotation: {rotation_name} | Scheme: {scheme_name} "
                f"({scheme_str}) | Mode: {rot_mode}")
    logger.info(f"{'='*70}")

    result = {
        "rotation": rotation_name,
        "scheme": scheme_name,
        "scheme_str": scheme_str,
        "label": label,
        "rotation_mode": rot_mode,
        "matrix_mode": matrix_mode,
        "random_r1": rr1,
        "random_r2": rr2,
        "random_r3": rr3,
        "random_r4": rr4,
        "quant_iters": args.quant_iters,
        "rotation_size": rotation_size,
        "layerwise": args.layerwise,
        "api": "pipeline_v2",
    }

    # Build rotation config (None for "none" rotation)
    rotation_config = build_rotation_config(
        rotation_flags, rotation_size,
        online_r1=args.online_r1,
        random_r1=rr1,
        random_r2=rr2,
        random_r3=rr3,
        random_r4=rr4,
    )

    t0 = time.time()
    model = None
    save_dir = None
    try:
        model = load_model(model_name, args.device)

        # ── Pipeline: rotation_config + quantization in one call ──
        if rotation_config is not None:
            logger.info(
                f"  rotation_config = SpinQuantConfig("
                f"r1={rotation_config.r1}, r2={rotation_config.r2}, "
                f"r3={rotation_config.r3}, r4={rotation_config.r4}, "
                f"online_r1={rotation_config.online_r1_rotation}, "
                f"random_r1={rotation_config.random_r1}, "
                f"random_r2={rotation_config.random_r2}, "
                f"random_r3={rotation_config.random_r3}, "
                f"random_r4={rotation_config.random_r4})"
            )
            if args.layerwise:
                logger.info("  layerwise_rotation = True (block-wise rotation)")
        else:
            logger.info("  rotation_config = None (no rotation)")

        logger.info(f"  AutoRound(rotation_config=..., scheme={scheme_str}, "
                    f"iters={args.quant_iters}, layerwise={args.layerwise})")

        ar = AutoRound(
            model,
            tokenizer=tokenizer,
            rotation_config=rotation_config,
            scheme=scheme_str,
            iters=args.quant_iters,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            device_map=args.device,
            layerwise_rotation=args.layerwise,
        )
        ar.quantize()
        model = ar.model

        pipeline_time = time.time() - t0
        result["setup_time_s"] = round(pipeline_time, 1)
        logger.info(f"  Pipeline (rotation+quantization) done in "
                    f"{pipeline_time:.1f}s")

        # ── In-memory evaluation (optional) ──
        if not args.no_in_memory_eval:
            model.eval()
            if is_multi_gpu(args.device):
                from accelerate import dispatch_model, infer_auto_device_map
                from accelerate.utils import get_max_memory
                no_split = list(getattr(model, "_no_split_modules", None) or [])
                max_memory = get_max_memory()
                # Reserve 20% headroom for activations / KV cache
                max_memory = {k: int(v * 0.85) for k, v in max_memory.items()}
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory,
                    no_split_module_classes=no_split,
                )
                dispatch_model(model, device_map=device_map)
                logger.info(f"  Model re-dispatched to multi-GPU for eval")
            else:
                model.to(args.device)

            logger.info(f"  Evaluating in-memory on tasks: {args.tasks}")
            metrics = evaluate_model(
                model, tokenizer, args.tasks,
                batch_size=args.batch_size, limit=args.limit, device=args.device,
            )
            result["metrics"] = metrics
            logger.info("  In-memory results:")
            for task, acc in sorted(metrics.items()):
                logger.info(f"    {task}: {acc:.4f}")
        else:
            logger.info("  Skipping in-memory evaluation (--no-in-memory-eval)")
            metrics = {}

        # ── Save/Load roundtrip ──
        if args.save_load:
            # Save
            safe_rot_name = rotation_name.replace("+", "_").replace(
                " ", "_").replace("(", "").replace(")", "").replace("=", "")
            save_dir = os.path.join(
                args.output_dir or ".",
                f"_tmp_{safe_rot_name}_{scheme_name}_{rot_mode}"
            )
            logger.info(f"  Saving model to {save_dir} ...")
            ar.save_quantized(save_dir)

            # Free in-memory model
            del model, ar
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load from disk and evaluate
            logger.info(f"  Loading from disk and evaluating ...")
            metrics_disk = evaluate_model_from_path(
                save_dir, args.tasks,
                batch_size=args.batch_size, limit=args.limit,
                device=args.device,
            )
            result["metrics_disk"] = metrics_disk

            # When in-memory eval was skipped, use disk metrics as primary
            if args.no_in_memory_eval:
                result["metrics"] = metrics_disk
                logger.info("  From-disk results (primary):")
                for task, acc in sorted(metrics_disk.items()):
                    logger.info(f"    {task}: {acc:.4f}")
            else:
                # Compare in-memory vs from-disk
                match = True
                max_diff = 0.0
                avg_diff = 0.0
                n_compared = 0
                logger.info("  From-disk results:")
                for task in sorted(set(list(metrics.keys())
                                       + list(metrics_disk.keys()))):
                    mem_val = metrics.get(task)
                    disk_val = metrics_disk.get(task)
                    if mem_val is not None and disk_val is not None:
                        diff = abs(mem_val - disk_val)
                        status = "✓" if diff < 5e-3 else "✗"
                        logger.info(f"    {task}: mem={mem_val:.4f} "
                                    f"disk={disk_val:.4f} diff={diff:.6f} "
                                    f"{status}")
                        if diff >= 5e-3:
                            match = False
                        max_diff = max(max_diff, diff)
                        avg_diff += diff
                        n_compared += 1
                    else:
                        logger.info(f"    {task}: mem={mem_val} disk={disk_val}")
                result["roundtrip_match"] = match
                result["roundtrip_max_diff"] = round(max_diff, 6)
                result["roundtrip_avg_diff"] = (
                    round(avg_diff / n_compared, 6) if n_compared > 0 else 0.0
                )

            # Cleanup saved model
            if args.cleanup and save_dir and os.path.exists(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
                logger.info(f"  Cleaned up {save_dir}")

        result["status"] = "success"
        result["total_time_s"] = round(time.time() - t0, 1)

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
        # Cleanup on error too
        if (args.save_load and args.cleanup and save_dir
                and os.path.exists(save_dir)):
            shutil.rmtree(save_dir, ignore_errors=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FP16 baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_fp16_baseline(model_name: str, tokenizer, args) -> dict[str, Any]:
    """Run bf16 baseline evaluation (no rotation, no quantization)."""
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
        model = load_model(model_name, args.device, for_eval_only=True)
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

def _compute_avg(metrics: dict) -> float | None:
    """Compute average of numeric values in a metrics dict."""
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def print_results_matrix(all_results: list[dict], tasks: list[str],
                         has_disk: bool = False,
                         compare_random: bool = False):
    """Print results as a rotation × scheme matrix table.

    When compare_random=True, shows deterministic and random side by side
    with Δ(bp) columns for each scheme.

    When has_disk=True, shows roundtrip match table.
    """
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

    # Build lookup: (rotation, scheme, matrix_mode) → result
    lookup = {}
    for r in all_results:
        mm = r.get("matrix_mode", "det")
        key = (r["rotation"], r["scheme"], mm)
        lookup[key] = r
        # Also store with plain key for non-compare-random mode
        lookup[(r["rotation"], r["scheme"])] = r

    def _fmt(val):
        if val is None:
            return "—"
        return f"{val:.4f}"

    def _fmt_delta(det_val, rand_val):
        if det_val is None or rand_val is None:
            return "—"
        bp = round((rand_val - det_val) * 10000)
        return f"{bp:+d}"

    if compare_random:
        _print_compare_random_tables(
            all_results, task_list, rotations_seen, schemes_seen,
            lookup, has_disk, _fmt, _fmt_delta)
    else:
        _print_standard_tables(
            all_results, task_list, rotations_seen, schemes_seen,
            lookup, has_disk, _fmt)


def _print_compare_random_tables(
    all_results, task_list, rotations_seen, schemes_seen,
    lookup, has_disk, _fmt, _fmt_delta,
):
    """Print det vs random side-by-side tables."""
    col_w = 8  # width for each value column

    def _header_line(schemes, label_width=26):
        h = f"  {'Rotation':<{label_width}}"
        for sch in schemes:
            h += f" │ {sch:^26}"
        return h

    def _sub_header(schemes, label_width=26):
        h = f"  {'':<{label_width}}"
        for _ in schemes:
            h += f" │ {'det':>{col_w}} {'rand':>{col_w}} {'Δ(bp)':>{col_w}}"
        return h

    def _sep(schemes, label_width=26, char="─"):
        s = f"  {char*label_width}"
        for _ in schemes:
            s += f"─┼─{char*(3*col_w+2)}"
        return s

    def _get_task_acc(r, task):
        if r and r.get("status") == "success":
            return r.get("metrics", {}).get(task)
        return None

    def _get_avg(r):
        if r and r.get("status") == "success" and r.get("metrics"):
            return _compute_avg(r["metrics"])
        return None

    def _is_none_rot(rot):
        """Check if this rotation is a 'none' variant."""
        return rot == "none" or rot.startswith("FP16")

    # ── Per-task tables ──
    for task in task_list:
        print(f"\n{'═'*120}")
        print(f"  Task: {task}  (det vs random)")
        print(f"{'═'*120}")
        print(_header_line(schemes_seen))
        print(_sub_header(schemes_seen))
        print(_sep(schemes_seen))

        for rot in rotations_seen:
            row = f"  {rot:<26}"
            for sch in schemes_seen:
                r_det = lookup.get((rot, sch, "det"))
                r_rand = lookup.get((rot, sch, "random"))
                det_v = _get_task_acc(r_det, task)
                rand_v = _get_task_acc(r_rand, task)
                if _is_none_rot(rot):
                    rand_v = None  # no random for none/FP16
                row += (f" │ {_fmt(det_v):>{col_w}}"
                        f" {_fmt(rand_v):>{col_w}}"
                        f" {_fmt_delta(det_v, rand_v):>{col_w}}")
            print(row)

    # ── Average accuracy table ──
    print(f"\n{'═'*120}")
    print(f"  Average Accuracy  (det vs random, Δ in basis points)")
    print(f"{'═'*120}")
    print(_header_line(schemes_seen))
    print(_sub_header(schemes_seen))
    print(_sep(schemes_seen))

    for rot in rotations_seen:
        row = f"  {rot:<26}"
        for sch in schemes_seen:
            r_det = lookup.get((rot, sch, "det"))
            r_rand = lookup.get((rot, sch, "random"))
            det_v = _get_avg(r_det)
            rand_v = _get_avg(r_rand)
            if _is_none_rot(rot):
                rand_v = None
            row += (f" │ {_fmt(det_v):>{col_w}}"
                    f" {_fmt(rand_v):>{col_w}}"
                    f" {_fmt_delta(det_v, rand_v):>{col_w}}")
        print(row)

    # ── Roundtrip table ──
    if has_disk:
        print(f"\n{'═'*120}")
        print(f"  Roundtrip Save/Load — max|task diff  (det / random, threshold < 0.5%)")
        print(f"{'═'*120}")
        h = f"  {'Rotation':<26}"
        for sch in schemes_seen:
            h += f" │ {'det':>8} {'rand':>8}"
        print(h)
        print(_sep(schemes_seen))

        for rot in rotations_seen:
            row = f"  {rot:<26}"
            for sch in schemes_seen:
                for mm in ("det", "random"):
                    r = lookup.get((rot, sch, mm))
                    if _is_none_rot(rot) and mm == "random":
                        row += f" {'—':>8}"
                    elif r and r.get("status") == "success":
                        md = r.get("roundtrip_max_diff")
                        if md is not None:
                            pct = md * 100
                            sym = "✓" if r.get("roundtrip_match") else "✗"
                            row += f" {sym}{pct:>5.2f}%"
                        else:
                            row += f" {'N/A':>8}"
                    elif r and r.get("status") == "error":
                        row += f" {'ERR':>8}"
                    else:
                        row += f" {'—':>8}"
                row += " │" if sch != schemes_seen[-1] else ""
            print(row)

    # ── Timing table ──
    print(f"\n{'═'*120}")
    print(f"  Timing (seconds)  (det / random)")
    print(f"{'═'*120}")
    h = f"  {'Rotation':<26}"
    for sch in schemes_seen:
        h += f" │ {'det':>8} {'rand':>8}"
    print(h)
    print(_sep(schemes_seen))

    for rot in rotations_seen:
        row = f"  {rot:<26}"
        for sch in schemes_seen:
            for mm in ("det", "random"):
                r = lookup.get((rot, sch, mm))
                if _is_none_rot(rot) and mm == "random":
                    row += f" {'—':>8}"
                elif r and "total_time_s" in r:
                    row += f" {r['total_time_s']:>7.0f}s"
                else:
                    row += f" {'—':>8}"
            row += " │" if sch != schemes_seen[-1] else ""
        print(row)


def _print_standard_tables(
    all_results, task_list, rotations_seen, schemes_seen,
    lookup, has_disk, _fmt,
):
    """Print standard (non-compare-random) tables."""
    def _fmt_acc(val):
        if val is None:
            return f"{'N/A':>10}"
        return f"{val:>10.4f}"

    # Per-task matrix
    for task in task_list:
        print(f"\n{'═'*120}")
        print(f"  Task: {task}")
        print(f"{'═'*120}")

        if has_disk:
            header = f"  {'Rotation':<20}"
            for sch in schemes_seen:
                header += f" | {sch+' mem':>14} {sch+' disk':>14}"
            print(header)
        else:
            header = f"  {'Rotation':<20}"
            for sch in schemes_seen:
                header += f" | {sch:>12}"
            print(header)

        for rot in rotations_seen:
            row = f"  {rot:<20}"
            for sch in schemes_seen:
                r = lookup.get((rot, sch))
                if r and r.get("status") == "success":
                    mem_acc = r["metrics"].get(task)
                    if has_disk:
                        disk_acc = r.get("metrics_disk", {}).get(task)
                        row += (f" | {_fmt_acc(mem_acc)}"
                                f" {_fmt_acc(disk_acc)}")
                    else:
                        row += f" | {_fmt_acc(mem_acc):>12}"
                elif r and r.get("status") == "error":
                    err_w = " | " + f"{'ERROR':>12}"
                    if has_disk:
                        err_w += f" {'':>14}"
                    row += err_w
                else:
                    dash_w = " | " + f"{'—':>12}"
                    if has_disk:
                        dash_w += f" {'':>14}"
                    row += dash_w
            print(row)

    # Average accuracy matrix
    print(f"\n{'═'*120}")
    title = "Average Accuracy"
    if has_disk:
        title += " (mem = in-memory, disk = from saved model)"
    print(f"  {title}")
    print(f"{'═'*120}")

    if has_disk:
        header = f"  {'Rotation':<20}"
        for sch in schemes_seen:
            header += f" | {sch+' mem':>14} {sch+' disk':>14}"
        print(header)
    else:
        header = f"  {'Rotation':<20}"
        for sch in schemes_seen:
            header += f" | {sch:>12}"
        print(header)

    for rot in rotations_seen:
        row = f"  {rot:<20}"
        for sch in schemes_seen:
            r = lookup.get((rot, sch))
            if r and r.get("status") == "success" and r.get("metrics"):
                avg_mem = _compute_avg(r["metrics"])
                if has_disk and r.get("metrics_disk"):
                    avg_disk = _compute_avg(r["metrics_disk"])
                    row += (f" | {_fmt_acc(avg_mem)}"
                            f" {_fmt_acc(avg_disk)}")
                else:
                    row += f" | {_fmt_acc(avg_mem):>12}"
            elif r and r.get("status") == "error":
                err_w = " | " + f"{'ERROR':>12}"
                if has_disk:
                    err_w += f" {'':>14}"
                row += err_w
            else:
                dash_w = " | " + f"{'—':>12}"
                if has_disk:
                    dash_w += f" {'':>14}"
                row += dash_w
        print(row)

    # Roundtrip match table
    if has_disk:
        print(f"\n{'═'*90}")
        print(f"  Roundtrip Save/Load — max|task diff "
              f"(threshold < 0.5%)")
        print(f"{'═'*90}")
        header = f"  {'Rotation':<20}"
        for sch in schemes_seen:
            header += f" | {sch:>12}"
        print(header)

        for rot in rotations_seen:
            row = f"  {rot:<20}"
            for sch in schemes_seen:
                r = lookup.get((rot, sch))
                if r and r.get("status") == "success":
                    md = r.get("roundtrip_max_diff")
                    if md is not None:
                        pct = md * 100
                        sym = "✓" if r.get("roundtrip_match") else "✗"
                        row += f" | {sym} {pct:>6.2f}%  "
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


def save_results_csv(all_results: list[dict], tasks: list[str],
                     output_path: str, has_disk: bool = False):
    """Save results as CSV for spreadsheet import."""
    task_list = sorted(tasks)
    # Build header
    header = [
        "rotation", "scheme", "rotation_mode", "matrix_mode",
        "random_r1", "random_r2", "random_r3", "random_r4",
        "quant_iters", "rotation_size", "layerwise", "status", "total_time_s",
    ]
    for t in task_list:
        header.append(f"{t}_mem")
    header.append("avg_mem")
    if has_disk:
        for t in task_list:
            header.append(f"{t}_disk")
        header.append("avg_disk")
        header.append("roundtrip_match")
        header.append("roundtrip_max_diff")
        header.append("roundtrip_avg_diff")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for r in all_results:
            metrics = r.get("metrics", {})
            accs_mem = [metrics.get(t, "") for t in task_list]
            vals_mem = [v for v in accs_mem if isinstance(v, (int, float))]
            avg_mem = round(sum(vals_mem) / len(vals_mem), 4) if vals_mem else ""

            row = [
                r.get("rotation", ""),
                r.get("scheme", ""),
                r.get("rotation_mode", "deterministic"),
                r.get("matrix_mode", "det"),
                r.get("random_r1", False),
                r.get("random_r2", False),
                r.get("random_r3", False),
                r.get("random_r4", False),
                r.get("quant_iters", ""),
                r.get("rotation_size", ""),
                r.get("layerwise", False),
                r.get("status", ""),
                r.get("total_time_s", ""),
            ] + accs_mem + [avg_mem]

            if has_disk:
                metrics_d = r.get("metrics_disk", {})
                accs_disk = [metrics_d.get(t, "") for t in task_list]
                vals_disk = [v for v in accs_disk
                             if isinstance(v, (int, float))]
                avg_disk = (round(sum(vals_disk) / len(vals_disk), 4)
                            if vals_disk else "")
                row += accs_disk + [avg_disk,
                       r.get("roundtrip_match", ""),
                       r.get("roundtrip_max_diff", ""),
                       r.get("roundtrip_avg_diff", "")]

            writer.writerow(row)

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

  # Block-wise (layer-wise) rotation — same as full-model but rotation applied per-block:
  python test_rotation_scheme_matrix_v2.py --device cuda:7 --layerwise

  # Block-wise + tuning (simulates 70B+ model workflow):
  python test_rotation_scheme_matrix_v2.py --device cuda:7 --layerwise --quant-iters 200

  # Common schemes with all 5 rotation levels:
  python test_rotation_scheme_matrix_v2.py --device cuda:7 \\
      --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" --schemes "W4A16,MXFP4,NVFP4"
        """,
    )

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name or path (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--device", type=str, default="cuda:7",
                        help="Device specification. Single GPU: 'cuda:0'. "
                             "Multi-GPU: '0,1,2,3' or 'auto'. "
                             "Multi-GPU uses accelerate device_map for model parallelism. "
                             "(default: cuda:7)")

    # Rotation config
    parser.add_argument("--rotations", type=str,
                        default="none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4",
                        help="Comma-separated rotation levels (default includes all 5 levels)")
    parser.add_argument("--rotation-size", type=int, default=128,
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
                        help="Use offline R1 rotation")
    parser.add_argument("--random-hadamard", action="store_true", default=False,
                        help="Use random Hadamard for ALL rotations (R1/R2/R3/R4). "
                             "For per-rotation control, use --random-r1/r2/r3/r4.")
    parser.add_argument("--random-r1", action="store_true", default=False,
                        help="Use random Hadamard for R1 only")
    parser.add_argument("--random-r2", action="store_true", default=False,
                        help="Use random Hadamard for R2 only")
    parser.add_argument("--random-r3", action="store_true", default=False,
                        help="Use random Hadamard for R3 only")
    parser.add_argument("--random-r4", action="store_true", default=False,
                        help="Use random Hadamard for R4 only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

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
                        help="Output directory for results (default: auto-generated)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip FP16 baseline evaluation")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to files")
    parser.add_argument("--save-load", action="store_true", default=False,
                        help="Enable save/load roundtrip: save model to disk, "
                             "load back, evaluate from disk, compare with "
                             "in-memory results")
    parser.add_argument("--no-in-memory-eval", action="store_true", default=False,
                        help="Skip in-memory evaluation after quantization. "
                             "Useful for large models where GPU memory is tight. "
                             "When combined with --save-load, only from-disk "
                             "evaluation is performed.")
    parser.add_argument("--compare-random", action="store_true", default=False,
                        help="Run each rotation combo twice (deterministic + "
                             "random Hadamard) for side-by-side comparison. "
                             "Overrides --random-hadamard/--random-r* flags.")
    parser.add_argument("--layerwise", action="store_true", default=False,
                        help="Enable block-wise (layer-wise) rotation. Rotation "
                             "is applied per-block inside the compressor loop "
                             "via _on_block_ready hook, instead of once upfront. "
                             "This enables rotation+quantization for 70B+ models "
                             "with limited GPU memory. Requires online_r1=True.")
    parser.add_argument("--cleanup", action="store_true", default=True,
                        help="Clean up saved model after roundtrip (default: True)")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                        help="Keep saved model after roundtrip")

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    set_seed(args.seed)

    # --compare-random overrides individual random flags
    if args.compare_random:
        # Clear any random flags — we'll set them per-run in the loop
        args.random_r1 = False
        args.random_r2 = False
        args.random_r3 = False
        args.random_r4 = False
        args.random_hadamard = False
    elif args.random_hadamard:
        args.random_r1 = True
        args.random_r2 = True
        args.random_r3 = True
        args.random_r4 = True

    # --layerwise requires online_r1 (offline R1 modifies embed/lm_head,
    # incompatible with pre-cached inputs in block-wise loop)
    if args.layerwise and not args.online_r1:
        logger.error("--layerwise requires --online-r1 (default). "
                     "Offline R1 is incompatible with block-wise rotation.")
        sys.exit(1)

    # --no-in-memory-eval without --save-load means no evaluation at all
    if args.no_in_memory_eval and not args.save_load:
        logger.warning("--no-in-memory-eval without --save-load: "
                       "no evaluation will be performed. Adding --save-load.")
        args.save_load = True

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
            logger.error(f"Unknown scheme: '{sn}'. "
                         f"Available: {list(SCHEME_DEFS.keys())}")
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
        iters_tag = (f"_iters{args.quant_iters}"
                     if args.quant_iters > 0 else "_rtn")
        size_tag = ""
        if multi_size:
            parts = [str(s) if s else "auto" for s in rotation_sizes]
            size_tag = f"_rs{'_'.join(parts)}"
        cmp_tag = "_cmprand" if args.compare_random else ""
        lw_tag = "_layerwise" if args.layerwise else ""
        args.output_dir = (
            f"results_v2_{model_short}{iters_tag}"
            f"{size_tag}{cmp_tag}{lw_tag}_{timestamp}"
        )

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine matrix modes for this run
    # "det" = deterministic, "random" = random Hadamard (H×diag(±1))
    if args.compare_random:
        matrix_modes = ["det", "random"]
    elif (args.random_r1 or args.random_r2
          or args.random_r3 or args.random_r4):
        matrix_modes = ["random"]
    else:
        matrix_modes = ["det"]

    # Calculate total combinations
    n_rot_with_size = len([r for r in rotation_names if r != "none"])
    n_none = 1 if "none" in rotation_names else 0
    n_modes_per_rot = len(matrix_modes)
    # "none" rotation: only 1 mode (det), never random
    n_combos = (
        (n_rot_with_size * len(rotation_sizes) * n_modes_per_rot
         + n_none) * len(scheme_names)
    )
    total_combos = n_combos + (0 if args.no_baseline else 1)
    task_list = [t.strip() for t in args.tasks.split(",")]

    # Rotation mode display
    if args.compare_random:
        rot_mode_str = "compare (det vs random)"
    elif args.random_hadamard:
        rot_mode_str = "random (all)"
    elif args.random_r1 or args.random_r2 or args.random_r3 or args.random_r4:
        parts = []
        for flag, name in [(args.random_r1, "R1"), (args.random_r2, "R2"),
                           (args.random_r3, "R3"), (args.random_r4, "R4")]:
            if flag:
                parts.append(name)
        rot_mode_str = f"random ({'+'.join(parts)})"
    else:
        rot_mode_str = "deterministic"

    logger.info(f"\n{'═'*70}")
    logger.info(f"  ROTATION × QUANTIZATION ACCURACY MATRIX (v2)")
    logger.info(f"{'═'*70}")
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Device:         {args.device}")
    logger.info(f"  Rotations:      {rotation_names}")
    logger.info(f"  Schemes:        {scheme_names}")
    logger.info(f"  Matrix modes:   {matrix_modes}")
    logger.info(f"  Total combos:   {total_combos}")
    logger.info(f"  Tasks:          {task_list}")
    logger.info(f"  Eval limit:     {args.limit or 'full'}")
    logger.info(f"  Quant iters:    {args.quant_iters} "
                f"({'RTN' if args.quant_iters == 0 else 'auto-round tuning'})")
    if multi_size:
        rs_display = [str(s) if s is not None else "auto"
                      for s in rotation_sizes]
        logger.info(f"  rotation_sizes: {rs_display}")
    else:
        logger.info(f"  rotation_size:  "
                    f"{args.rotation_size or 'auto (known Hadamard)'}")
    logger.info(f"  Rotation mode:  {rot_mode_str}")
    logger.info(f"  Online R1:      {args.online_r1}")
    logger.info(f"  Layerwise:      {args.layerwise}")
    logger.info(f"  Save/Load:      {args.save_load}")
    logger.info(f"  In-memory eval: {not args.no_in_memory_eval}")
    logger.info(f"  Compare random: {args.compare_random}")
    logger.info(f"  Seed:           {args.seed}")
    logger.info(f"  Output dir:     "
                f"{args.output_dir if not args.no_save else 'disabled'}")
    logger.info(f"{'═'*70}\n")

    # Load tokenizer (shared across all runs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)

    all_results = []
    t_start = time.time()

    # FP16 baseline (always "det" mode)
    if not args.no_baseline:
        baseline = run_fp16_baseline(args.model, tokenizer, args)
        baseline["matrix_mode"] = "det"
        all_results.append(baseline)

    # Run all combinations: rotation_size × rotation × scheme × matrix_mode
    combo_idx = 0
    none_done = False
    for rs in rotation_sizes:
        rs_label = str(rs) if rs is not None else "auto"
        if multi_size:
            logger.info(f"\n{'─'*70}")
            logger.info(f"  ▶ rotation_size = {rs_label}")
            logger.info(f"{'─'*70}")

        for rot_name in rotation_names:
            # Skip duplicate "none" runs across rotation_sizes
            if rot_name == "none" and none_done:
                continue
            if rot_name == "none":
                none_done = True

            rot_flags = ROTATION_LEVELS[rot_name]

            # Display name includes rotation_size when sweeping
            rot_display = rot_name
            if multi_size and rot_name != "none":
                rot_display = f"{rot_name} (rs={rs_label})"

            # Determine which matrix modes to run for this rotation
            if rot_name == "none":
                modes_for_rot = ["det"]  # no random for "none"
            else:
                modes_for_rot = matrix_modes

            for mm in modes_for_rot:
                # Build random override based on matrix mode
                if mm == "random":
                    random_override = (
                        rot_flags.get("r1", False),
                        rot_flags.get("r2", False),
                        rot_flags.get("r3", False),
                        rot_flags.get("r4", False),
                    )
                else:
                    random_override = (False, False, False, False)

                for sch_name in scheme_names:
                    combo_idx += 1
                    sch_str, sch_desc, sch_cat = SCHEME_DEFS[sch_name]

                    mm_label = f" [{mm}]" if len(matrix_modes) > 1 else ""
                    logger.info(
                        f"\n[{combo_idx}/{n_combos}] "
                        f"{rot_display} × {sch_name} "
                        f"({sch_desc}){mm_label}"
                    )

                    result = run_single_combination(
                        args.model, tokenizer,
                        rot_display, rot_flags,
                        sch_name, sch_str,
                        args,
                        rotation_size=rs,
                        random_override=random_override,
                        matrix_mode=mm,
                    )
                    all_results.append(result)

                    # Save intermediate results
                    if not args.no_save:
                        save_results_json(
                            all_results,
                            os.path.join(args.output_dir, "results.json"),
                        )

    total_time = time.time() - t_start

    # Print summary tables
    print_results_matrix(
        all_results, task_list,
        has_disk=args.save_load,
        compare_random=args.compare_random,
    )

    print(f"\n{'═'*70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Combinations tested: {len(all_results)}")
    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_error = sum(1 for r in all_results if r.get("status") == "error")
    print(f"  Success: {n_success}, Errors: {n_error}")
    if args.save_load:
        n_match = sum(1 for r in all_results
                      if r.get("roundtrip_match") is True)
        n_mismatch = sum(1 for r in all_results
                         if r.get("roundtrip_match") is False)
        diffs = [r["roundtrip_max_diff"] for r in all_results
                 if r.get("roundtrip_max_diff") is not None]
        max_d = max(diffs) if diffs else 0
        avg_d = sum(diffs) / len(diffs) if diffs else 0
        print(f"  Roundtrip (<0.5%): {n_match} match, {n_mismatch} mismatch"
              f"  (max_diff={max_d*100:.2f}%, avg={avg_d*100:.2f}%)")
    if args.compare_random:
        n_det = sum(1 for r in all_results
                    if r.get("matrix_mode") == "det"
                    and r.get("status") == "success")
        n_rand = sum(1 for r in all_results
                     if r.get("matrix_mode") == "random"
                     and r.get("status") == "success")
        print(f"  Det runs: {n_det}, Random runs: {n_rand}")
    print(f"{'═'*70}")

    # Save final results
    if not args.no_save:
        save_results_json(all_results,
                          os.path.join(args.output_dir, "results.json"))
        save_results_csv(all_results, task_list,
                         os.path.join(args.output_dir, "results.csv"),
                         has_disk=args.save_load)

        config_data = {
            "model": args.model, "device": args.device,
            "rotations": rotation_names, "schemes": scheme_names,
            "tasks": task_list, "limit": args.limit,
            "quant_iters": args.quant_iters,
            "rotation_sizes": [str(s) if s is not None else "auto"
                               for s in rotation_sizes],
            "matrix_modes": matrix_modes,
            "compare_random": args.compare_random,
            "layerwise_rotation": args.layerwise,
            "random_r1": args.random_r1, "random_r2": args.random_r2,
            "random_r3": args.random_r3, "random_r4": args.random_r4,
            "online_r1": args.online_r1,
            "save_load": args.save_load,
            "seed": args.seed,
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
