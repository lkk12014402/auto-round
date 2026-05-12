#!/usr/bin/env python3
"""
Auto-Round: Save/Load Roundtrip Verification for Rotation × Quantization

This script verifies that rotated+quantized models can be:
  1. Saved to disk (safetensors + config.json with spinquant_config)
  2. Loaded back from disk (buffers restored, forward patched)
  3. Produce identical lm_eval accuracy as the in-memory quantized model

For each rotation×scheme×iters×rotation_size combination, two evaluations
are performed:
  A) "in-memory"  — quantize, then evaluate the model object directly
                     (same as test_rotation_scheme_matrix_v2.py)
  B) "from-disk"  — quantize_and_save to disk, load back via
                     AutoModelForCausalLM.from_pretrained, then evaluate

If the accuracy matches between A and B, the save/load roundtrip is correct.

Supported dimensions:
  - Rotation levels: none, R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
  - Rotation modes: hadamard (deterministic), random (random Hadamard H×D)
  - Schemes: W4A16, W3A16, W8A16 (INT), MXFP4, NVFP4 (FP4), FP8_STATIC
  - Iters: 0 (RTN) or >0 (auto-round tuning)
  - Rotation sizes: auto (full dim) or custom block sizes (64, 128, etc.)

Usage:
    # Quick test (R1+R2 only, W4A16, limit=100):
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1+R2" --schemes "W4A16"

    # Hadamard vs random rotation (all rotations use random):
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1+R2" --schemes "W4A16" \
        --rotation-modes "hadamard,random"

    # Per-rotation random control (only R2 uses random, R1 stays hadamard):
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1+R2" --schemes "W4A16" \
        --random-rotations "R2"

    # Multi-scheme coverage (INT + FP paths):
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1+R2" --schemes "W4A16,MXFP4,NVFP4,FP8_STATIC"

    # RTN vs tuning:
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1+R2" --schemes "W4A16" --quant-iters-list "0,200"

    # Rotation size sweep:
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1+R2+R3+R4" --schemes "W4A16" \
        --rotation-sizes "64,128,auto"

    # Full coverage matrix (with random):
    python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
        --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
        --schemes "W4A16,MXFP4,NVFP4,FP8_STATIC" \
        --rotation-modes "hadamard,random" \
        --quant-iters-list "0,200"
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

# ═══════════════════════════════════════════════════════════════════════════════
# Rotation and scheme definitions (same as test_rotation_scheme_matrix_v2.py)
# ═══════════════════════════════════════════════════════════════════════════════

ROTATION_LEVELS = OrderedDict([
    ("none",        dict(r1=False, r2=False, r3=False, r4=False)),
    ("R1",          dict(r1=True,  r2=False, r3=False, r4=False)),
    ("R1+R2",       dict(r1=True,  r2=True,  r3=False, r4=False)),
    ("R1+R2+R3",    dict(r1=True,  r2=True,  r3=True,  r4=False)),
    ("R1+R2+R3+R4", dict(r1=True,  r2=True,  r3=True,  r4=True)),
])

SCHEME_DEFS = OrderedDict([
    ("W4A16",      ("W4A16",       "INT4 weight-only, gs=128")),
    ("W3A16",      ("W3A16",       "INT3 weight-only, gs=128")),
    ("W8A16",      ("W8A16",       "INT8 weight-only, gs=128")),
    ("MXFP4",     ("MXFP4_RCEIL", "MXFP4 W4A4, block=32")),
    ("NVFP4",     ("NVFP4",       "NVFP4 W4A4, gs=16")),
    ("FP8_STATIC", ("FP8_STATIC", "FP8 static, per-tensor")),
    ("FP16",      ("FP16",        "No quantization (FP16 baseline)")),
])

DEFAULT_TASKS = "hellaswag,piqa"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

def build_rotation_config(rotation_flags: dict, **kwargs) -> SpinQuantConfig | None:
    if not any(rotation_flags.values()):
        return None
    rotation_mode = kwargs.get("rotation_mode", "hadamard")
    is_random = rotation_mode == "random"
    # Per-rotation random overrides (from --random-rotations)
    random_set = kwargs.get("random_set", None)  # e.g. {"r1", "r2"}
    if random_set is not None:
        random_r1 = "r1" in random_set
        random_r2 = "r2" in random_set
        random_r3 = "r3" in random_set
        random_r4 = "r4" in random_set
    else:
        random_r1 = is_random or kwargs.get("random_r1", False)
        random_r2 = is_random or kwargs.get("random_r2", False)
        random_r3 = is_random or kwargs.get("random_r3", False)
        random_r4 = is_random or kwargs.get("random_r4", False)
    return SpinQuantConfig(
        **rotation_flags,
        rotation_size=kwargs.get("rotation_size", None),
        online_r1_rotation=kwargs.get("online_r1", True),
        trainable_rotation=False,
        trainable_smooth=False,
        random_r1=random_r1,
        random_r2=random_r2,
        random_r3=random_r3,
        random_r4=random_r4,
    )


def evaluate_model_object(
    model, tokenizer, tasks: str, batch_size: int = 8,
    limit: int | None = None, device: str = "cuda:0",
) -> dict[str, float]:
    """Evaluate an in-memory model object using lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    task_list = [t.strip() for t in tasks.split(",")]
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size,
                              limit=limit, device=device)
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        # Accuracy (acc preferred, fallback to acc_norm)
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
        # Perplexity (word_perplexity or byte_perplexity for wikitext)
        ppl = task_results.get("word_perplexity,none") or task_results.get("byte_perplexity,none")
        if ppl is not None:
            metrics[f"{task_name}_ppl"] = round(ppl, 4)
    return metrics


def evaluate_model_from_path(
    model_path: str, tasks: str, batch_size: int = 8,
    limit: int | None = None, device: str = "cuda:0",
) -> dict[str, float]:
    """Load a saved model from disk and evaluate using lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model_path, batch_size=batch_size, device=device)
    task_list = [t.strip() for t in tasks.split(",")]
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size,
                              limit=limit, device=device)
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
        ppl = task_results.get("word_perplexity,none") or task_results.get("byte_perplexity,none")
        if ppl is not None:
            metrics[f"{task_name}_ppl"] = round(ppl, 4)
    return metrics


def check_saved_model(save_dir: str, rotation_name: str) -> dict[str, Any]:
    """Verify saved model has expected spinquant artifacts."""
    info = {"save_dir": save_dir, "checks": {}}

    # Check config.json for spinquant_config
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        qcfg = config.get("quantization_config", {})
        sc = qcfg.get("spinquant_config")
        info["checks"]["config_has_spinquant"] = sc is not None
        if sc:
            info["checks"]["config_r1"] = sc.get("r1", False)
            info["checks"]["config_r2"] = sc.get("r2", False)
            info["checks"]["config_r3"] = sc.get("r3", False)
            info["checks"]["config_r4"] = sc.get("r4", False)
    else:
        info["checks"]["config_has_spinquant"] = False

    # Check safetensors for spinquant keys
    spinquant_keys = []
    for f in os.listdir(save_dir):
        if f.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(os.path.join(save_dir, f), framework="pt") as sf:
                for key in sf.keys():
                    if "spinquant" in key:
                        spinquant_keys.append(key)

    info["checks"]["n_spinquant_keys"] = len(spinquant_keys)
    info["spinquant_keys_sample"] = spinquant_keys[:10]

    # Expected buffer count (rough check)
    if rotation_name != "none":
        # R1 online → buffers on q/k/v/gate/up = 5 per layer × 2 (type+size)
        # R4 → buffers on down = 1 per layer × 2 (type+size)
        info["checks"]["has_expected_buffers"] = len(spinquant_keys) > 0
    else:
        info["checks"]["has_expected_buffers"] = True  # none = no buffers expected

    return info


def find_model_subdir(output_dir: str) -> str:
    """Find the actual model subdirectory inside the output_dir.

    AutoRound.quantize_and_save creates a subdirectory like
    'ModelName-w4g128' inside the given output_dir.
    """
    if os.path.exists(os.path.join(output_dir, "config.json")):
        return output_dir

    for entry in os.listdir(output_dir):
        candidate = os.path.join(output_dir, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate

    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Single combination runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_roundtrip(
    model_name: str,
    tokenizer,
    rotation_name: str,
    rotation_flags: dict,
    scheme_name: str,
    scheme_str: str,
    args,
    rotation_size: int | None = None,
    quant_iters: int | None = None,
    rotation_mode: str = "hadamard",
    random_set: set[str] | None = None,
) -> dict[str, Any]:
    """Run save/load roundtrip test for one rotation × scheme combination.

    Steps:
      1. Load fresh model
      2. quantize_and_save to disk
      3. Evaluate in-memory model (path A)
      4. Load from disk, evaluate (path B)
      5. Compare A vs B
    """
    iters = quant_iters if quant_iters is not None else args.quant_iters
    label = f"{rotation_name} + {scheme_name}" if rotation_name != "none" else f"{scheme_name} only"
    if random_set:
        rand_parts = sorted(r.upper() for r in random_set)
        label += f" [random:{'+'.join(rand_parts)}]"
    elif rotation_mode == "random":
        label += " [random:all]"
    if iters > 0:
        label += f" (iters={iters})"
    if rotation_size is not None:
        label += f" (rs={rotation_size})"
    logger.info(f"\n{'='*70}")
    logger.info(f"  ROUNDTRIP: {label}")
    logger.info(f"{'='*70}")

    result = {
        "rotation": rotation_name,
        "scheme": scheme_name,
        "rotation_mode": rotation_mode,
        "random_set": sorted(random_set) if random_set else None,
        "label": label,
        "quant_iters": iters,
        "rotation_size": rotation_size,
    }

    rotation_config = build_rotation_config(
        rotation_flags,
        rotation_size=rotation_size,
        online_r1=args.online_r1,
        rotation_mode=rotation_mode,
        random_set=random_set,
    )

    # Build save directory
    model_short = model_name.split("/")[-1]
    safe_rot = rotation_name.replace("+", "_")
    if random_set:
        rand_tag = "-rand_" + "_".join(sorted(r.upper() for r in random_set))
    elif rotation_mode == "random":
        rand_tag = "-rand"
    else:
        rand_tag = ""
    mode_tag = rand_tag
    iters_tag = f"-iters{iters}" if iters > 0 else ""
    rs_tag = f"-rs{rotation_size}" if rotation_size else ""
    save_dir = os.path.join(args.output_dir, f"{model_short}-{safe_rot}-{scheme_name}{mode_tag}{iters_tag}{rs_tag}")

    t0 = time.time()
    model = None

    # Set seed for reproducibility
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
        result["seed"] = args.seed

    is_fp16_baseline = (scheme_name == "FP16")

    try:
        # ── Step 1: Load fresh model ──
        logger.info(f"  [1/5] Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        model.eval()

        # ── Step 2: Quantize and save (or FP16 baseline) ──
        if is_fp16_baseline:
            # FP16 baseline: apply rotation only, no quantization
            if rotation_config is not None:
                logger.info(f"  [2/5] FP16 baseline with rotation — applying rotation only (no quant)")
                from auto_round.algorithms.transforms import apply_rotation
                model = apply_rotation(model, rotation_config)
            else:
                logger.info(f"  [2/5] FP16 baseline — no rotation, no quant")
            model.eval()
            model.to(args.device)

            # Evaluate directly (no save/load for FP16)
            logger.info(f"  [3/5] N/A (FP16 baseline — no save)")
            logger.info(f"  [4/5] Evaluating FP16 model directly...")
            t_eval = time.time()
            metrics_disk = evaluate_model_object(
                model, tokenizer, args.tasks,
                batch_size=args.batch_size, limit=args.limit, device=args.device,
            )
            eval_time = time.time() - t_eval
            result["metrics_from_disk"] = metrics_disk
            result["metrics_in_memory"] = metrics_disk  # same for FP16
            result["roundtrip_match"] = True  # trivially true
            logger.info(f"    FP16 eval done in {eval_time:.1f}s")
            for task, acc in sorted(metrics_disk.items()):
                logger.info(f"      {task}: {acc:.4f}")

            result["status"] = "success"
            result["total_time_s"] = round(time.time() - t0, 1)
            result["save_dir"] = None
            result["save_checks"] = {"config_has_spinquant": False, "n_spinquant_keys": 0}

            del model
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result

        if rotation_config is not None:
            logger.info(f"  [2/5] AutoRound(rotation_config=SpinQuantConfig("
                        f"r1={rotation_config.r1}, r2={rotation_config.r2}, "
                        f"r3={rotation_config.r3}, r4={rotation_config.r4}, "
                        f"rotation_size={rotation_size}), "
                        f"scheme={scheme_str}, iters={iters})")
        else:
            logger.info(f"  [2/5] AutoRound(scheme={scheme_str}, iters={iters}) — no rotation")

        ar = AutoRound(
            model,
            tokenizer=tokenizer,
            rotation_config=rotation_config,
            scheme=scheme_str,
            iters=iters,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            device_map=args.device,
        )
        ar.quantize_and_save(output_dir=save_dir, format="auto_round")
        quant_time = time.time() - t0
        logger.info(f"  Quantize+save done in {quant_time:.1f}s → {save_dir}")

        # Find actual model subdir
        actual_save_dir = find_model_subdir(save_dir)
        result["save_dir"] = actual_save_dir

        # ── Step 3: Check saved artifacts ──
        logger.info(f"  [3/5] Checking saved model artifacts...")
        check = check_saved_model(actual_save_dir, rotation_name)
        result["save_checks"] = check["checks"]
        logger.info(f"    config has spinquant_config: {check['checks'].get('config_has_spinquant')}")
        logger.info(f"    spinquant keys in safetensors: {check['checks'].get('n_spinquant_keys')}")
        if check.get("spinquant_keys_sample"):
            for k in check["spinquant_keys_sample"][:5]:
                logger.info(f"      {k}")

        # Clean up in-memory model before evaluation
        del ar
        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 4: Evaluate from disk ──
        logger.info(f"  [4/5] Evaluating model loaded FROM DISK: {actual_save_dir}")
        t_eval_disk = time.time()
        metrics_disk = evaluate_model_from_path(
            actual_save_dir, args.tasks,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )
        eval_disk_time = time.time() - t_eval_disk
        result["metrics_from_disk"] = metrics_disk
        logger.info(f"    From-disk eval done in {eval_disk_time:.1f}s")
        for task, acc in sorted(metrics_disk.items()):
            logger.info(f"      {task}: {acc:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 5: Evaluate in-memory (fresh quantize, no save) ──
        if not args.skip_inmemory:
            logger.info(f"  [5/5] Evaluating IN-MEMORY model (fresh quantize, same config)...")
            t_inmem = time.time()

            model2 = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True
            )
            model2.eval()

            ar2 = AutoRound(
                model2,
                tokenizer=tokenizer,
                rotation_config=rotation_config,
                scheme=scheme_str,
                iters=iters,
                nsamples=args.nsamples,
                seqlen=args.seqlen,
                device_map=args.device,
            )
            ar2.quantize()
            model2 = ar2.model
            model2.eval()
            model2.to(args.device)

            metrics_inmem = evaluate_model_object(
                model2, tokenizer, args.tasks,
                batch_size=args.batch_size, limit=args.limit, device=args.device,
            )
            inmem_time = time.time() - t_inmem
            result["metrics_in_memory"] = metrics_inmem
            logger.info(f"    In-memory eval done in {inmem_time:.1f}s")
            for task, acc in sorted(metrics_inmem.items()):
                logger.info(f"      {task}: {acc:.4f}")

            del ar2, model2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Compare ──
            all_match = True
            for task in metrics_disk:
                disk_acc = metrics_disk.get(task, -1)
                mem_acc = metrics_inmem.get(task, -1)
                diff = abs(disk_acc - mem_acc)
                match = diff < 1e-4
                if not match:
                    all_match = False
                status = "✓ MATCH" if match else f"✗ DIFF={diff:.4f}"
                logger.info(f"    {task}: disk={disk_acc:.4f}  mem={mem_acc:.4f}  {status}")

            result["roundtrip_match"] = all_match
        else:
            logger.info(f"  [5/5] SKIPPED (--skip-inmemory)")
            result["roundtrip_match"] = None

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

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Results formatting
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list[dict], task_list: list[str]):
    """Print roundtrip comparison summary."""
    print(f"\n{'═'*90}")
    print(f"  SAVE/LOAD ROUNDTRIP VERIFICATION SUMMARY")
    print(f"{'═'*90}")

    header = f"  {'Rotation × Scheme':<25} | {'Status':>8}"
    for t in task_list:
        header += f" | {t+' (disk)':>16} | {t+' (mem)':>16}"
    header += f" | {'Match':>7}"
    print(header)
    print(f"  {'─'*25}─┼─{'─'*8}" + "".join(f"─┼─{'─'*16}─┼─{'─'*16}" for _ in task_list) + f"─┼─{'─'*7}")

    for r in all_results:
        label = f"{r['rotation']} + {r['scheme']}"
        status = r.get("status", "?")
        row = f"  {label:<25} | {status:>8}"

        for t in task_list:
            disk = r.get("metrics_from_disk", {}).get(t)
            mem = r.get("metrics_in_memory", {}).get(t)
            disk_str = f"{disk:.4f}" if disk is not None else "N/A"
            mem_str = f"{mem:.4f}" if mem is not None else "N/A"
            row += f" | {disk_str:>16} | {mem_str:>16}"

        match = r.get("roundtrip_match")
        if match is True:
            match_str = "✓ PASS"
        elif match is False:
            match_str = "✗ FAIL"
        else:
            match_str = "SKIP"
        row += f" | {match_str:>7}"
        print(row)

    # Save artifact check summary
    print(f"\n{'═'*90}")
    print(f"  SAVE ARTIFACT CHECKS")
    print(f"{'═'*90}")
    print(f"  {'Rotation × Scheme':<25} | {'config':>7} | {'n_keys':>7} | {'buffers':>8}")
    print(f"  {'─'*25}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}")

    for r in all_results:
        label = f"{r['rotation']} + {r['scheme']}"
        checks = r.get("save_checks", {})
        cfg = "✓" if checks.get("config_has_spinquant") else "✗"
        nkeys = str(checks.get("n_spinquant_keys", "?"))
        bufs = "✓" if checks.get("has_expected_buffers") else "✗"
        print(f"  {label:<25} | {cfg:>7} | {nkeys:>7} | {bufs:>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save/Load Roundtrip Verification for Rotation × Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick: one rotation, one scheme, limit=100
  python test_save_load_roundtrip.py --device cuda:0 --limit 100 \\
      --rotations "R1+R2" --schemes "W4A16"

  # All rotations × W4A16 (RTN):
  python test_save_load_roundtrip.py --device cuda:0 --limit 100 \\
      --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" --schemes "W4A16"

  # Multi-scheme (INT + FP):
  python test_save_load_roundtrip.py --device cuda:0 --limit 100 \\
      --rotations "R1+R2" --schemes "W4A16,MXFP4,NVFP4,FP8_STATIC"

  # RTN vs tuning comparison:
  python test_save_load_roundtrip.py --device cuda:0 --limit 100 \\
      --rotations "R1+R2" --schemes "W4A16" --quant-iters-list "0,200"

  # Rotation size sweep:
  python test_save_load_roundtrip.py --device cuda:0 --limit 100 \\
      --rotations "R1+R2+R3+R4" --schemes "W4A16" \\
      --rotation-sizes "64,128,auto"

  # Full coverage:
  python test_save_load_roundtrip.py --device cuda:0 --limit 100 \\
      --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \\
      --schemes "W4A16,MXFP4,NVFP4,FP8_STATIC" \\
      --quant-iters-list "0,200"
        """,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rotations", type=str,
                        default="R1,R1+R2,R1+R2+R3,R1+R2+R3+R4",
                        help="Comma-separated rotation levels")
    parser.add_argument("--schemes", type=str, default="W4A16",
                        help="Comma-separated quantization schemes "
                             f"(available: {','.join(SCHEME_DEFS.keys())})")
    parser.add_argument("--quant-iters", type=int, default=0,
                        help="AutoRound iters (0=RTN). Overridden by --quant-iters-list")
    parser.add_argument("--quant-iters-list", type=str, default=None,
                        help="Comma-separated list of iters values to test, e.g. '0,200'. "
                             "Each rotation×scheme combo is tested with each iters value.")
    parser.add_argument("--rotation-size", type=int, default=None,
                        help="Custom rotation_size (block dimension)")
    parser.add_argument("--rotation-sizes", type=str, default=None,
                        help="Comma-separated rotation sizes to sweep, e.g. '64,128,auto'. "
                             "'auto'/'none' = default (full hidden/intermediate size).")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--online-r1", action="store_true", default=True)
    parser.add_argument("--offline-r1", dest="online_r1", action="store_false")
    parser.add_argument("--random-hadamard", action="store_true", default=False,
                        help="(Legacy) Enable random Hadamard for all combos. "
                             "Prefer --rotation-modes for sweep.")
    parser.add_argument("--rotation-modes", type=str, default=None,
                        help="Comma-separated rotation modes: 'hadamard,random'. "
                             "Sweeps each combo with each mode. "
                             "If not set, uses 'hadamard' (or 'random' if --random-hadamard).")
    parser.add_argument("--random-rotations", type=str, default=None,
                        help="Per-rotation random control. Comma-separated list of which "
                             "rotations should use random Hadamard, e.g. 'R2', 'R1,R2', "
                             "'R1,R2,R3,R4'. Overrides --rotation-modes for fine-grained "
                             "control. Example: --random-rotations 'R2' means only R2 is "
                             "random while R1/R3/R4 stay deterministic.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-inmemory", action="store_true", default=False,
                        help="Skip in-memory comparison (only test save+load from disk)")
    parser.add_argument("--cleanup", action="store_true", default=False,
                        help="Remove saved model directories after verification")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible random rotations. "
                             "Sets torch/numpy/python random seeds before each combo.")
    return parser.parse_args()


def main():
    args = parse_args()

    rotation_names = [r.strip() for r in args.rotations.split(",")]
    scheme_names = [s.strip() for s in args.schemes.split(",")]
    task_list = [t.strip() for t in args.tasks.split(",")]

    for rn in rotation_names:
        if rn not in ROTATION_LEVELS:
            logger.error(f"Unknown rotation: '{rn}'. Available: {list(ROTATION_LEVELS.keys())}")
            sys.exit(1)
    for sn in scheme_names:
        if sn not in SCHEME_DEFS:
            logger.error(f"Unknown scheme: '{sn}'. Available: {list(SCHEME_DEFS.keys())}")
            sys.exit(1)

    # Resolve iters list
    if args.quant_iters_list:
        iters_list = [int(x.strip()) for x in args.quant_iters_list.split(",")]
    else:
        iters_list = [args.quant_iters]

    # Resolve rotation_sizes
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

    # Resolve rotation_modes
    # --random-rotations takes precedence over --rotation-modes
    random_set = None
    if args.random_rotations:
        random_set = {r.strip().lower() for r in args.random_rotations.split(",")}
        rotation_modes = ["per-rotation"]  # single pass, random_set controls per-rotation
    elif args.rotation_modes:
        rotation_modes = [m.strip().lower() for m in args.rotation_modes.split(",")]
    elif args.random_hadamard:
        rotation_modes = ["random"]
    else:
        rotation_modes = ["hadamard"]

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        args.output_dir = f"roundtrip_{model_short}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Compute total combos (none rotation doesn't vary by rotation_size/mode)
    n_rot_with_size = len([r for r in rotation_names if r != "none"])
    n_none = 1 if "none" in rotation_names else 0
    n_combos = (
        (n_rot_with_size * len(rotation_sizes) * len(rotation_modes) + n_none)
        * len(scheme_names) * len(iters_list)
    )

    logger.info(f"\n{'═'*70}")
    logger.info(f"  SAVE/LOAD ROUNDTRIP VERIFICATION")
    logger.info(f"{'═'*70}")
    logger.info(f"  Model:           {args.model}")
    logger.info(f"  Device:          {args.device}")
    logger.info(f"  Rotations:       {rotation_names}")
    if random_set:
        logger.info(f"  Random rotations:{sorted(r.upper() for r in random_set)}")
    else:
        logger.info(f"  Rotation modes:  {rotation_modes}")
    logger.info(f"  Schemes:         {scheme_names}")
    logger.info(f"  Iters:           {iters_list}")
    rs_display = [str(s) if s is not None else "auto" for s in rotation_sizes]
    logger.info(f"  Rotation sizes:  {rs_display}")
    logger.info(f"  Total combos:    {n_combos}")
    logger.info(f"  Tasks:           {task_list}")
    logger.info(f"  Eval limit:      {args.limit or 'full'}")
    logger.info(f"  In-memory:       {'skip' if args.skip_inmemory else 'enabled'}")
    logger.info(f"  Seed:            {args.seed if args.seed is not None else 'none (non-reproducible)'}")
    logger.info(f"  Output dir:      {args.output_dir}")
    logger.info(f"{'═'*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    all_results = []
    t_start = time.time()

    combo_idx = 0
    none_done_iters = set()  # track which iters have been done for "none"

    for iters in iters_list:
        for rs in rotation_sizes:
            for rot_mode in rotation_modes:
                for rot_name in rotation_names:
                    # "none" doesn't vary by rotation_size/mode — deduplicate
                    if rot_name == "none":
                        if iters in none_done_iters:
                            continue
                        none_done_iters.add(iters)

                    rot_flags = ROTATION_LEVELS[rot_name]

                    for sch_name in scheme_names:
                        combo_idx += 1
                        sch_str, sch_desc = SCHEME_DEFS[sch_name]

                        rs_label = str(rs) if rs is not None else "auto"
                        if random_set:
                            mode_label = f", random={'+'.join(sorted(r.upper() for r in random_set))}"
                        elif rot_mode != "hadamard":
                            mode_label = f", mode={rot_mode}"
                        else:
                            mode_label = ""
                        logger.info(f"\n[{combo_idx}/{n_combos}] "
                                    f"{rot_name} × {sch_name} (iters={iters}, rs={rs_label}{mode_label})")

                        result = run_roundtrip(
                            args.model, tokenizer,
                            rot_name, rot_flags,
                            sch_name, sch_str,
                            args,
                            rotation_size=rs if rot_name != "none" else None,
                            quant_iters=iters,
                            rotation_mode=rot_mode if rot_name != "none" else "hadamard",
                            random_set=random_set if rot_name != "none" else None,
                        )
                        all_results.append(result)

                    # Save intermediate
                    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                        json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - t_start

    # Print summary
    print_summary(all_results, task_list)

    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_error = sum(1 for r in all_results if r.get("status") == "error")
    n_match = sum(1 for r in all_results if r.get("roundtrip_match") is True)
    n_mismatch = sum(1 for r in all_results if r.get("roundtrip_match") is False)

    print(f"\n{'═'*70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Combinations: {len(all_results)}")
    print(f"  Success: {n_success}, Errors: {n_error}")
    if not args.skip_inmemory:
        print(f"  Roundtrip match: {n_match}, Mismatch: {n_mismatch}")
    print(f"  Results saved to: {os.path.join(args.output_dir, 'results.json')}")
    print(f"{'═'*70}")

    # Cleanup if requested
    if args.cleanup:
        for r in all_results:
            sd = r.get("save_dir")
            if sd and os.path.isdir(sd):
                parent = os.path.dirname(sd) if sd != args.output_dir else sd
                if parent != args.output_dir and os.path.isdir(parent):
                    shutil.rmtree(parent, ignore_errors=True)
                elif os.path.isdir(sd):
                    shutil.rmtree(sd, ignore_errors=True)
        logger.info("Cleaned up saved model directories")

    # Exit code: non-zero if any errors or mismatches
    if n_error > 0 or n_mismatch > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
