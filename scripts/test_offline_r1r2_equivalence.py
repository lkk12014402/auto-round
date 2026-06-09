#!/usr/bin/env python3
"""
Auto-Round: R1+R2 Offline Rotation Equivalence & Save/Load Roundtrip Verification

This script verifies two key properties of offline (weight-fused) rotation:

  1. **Rotation equivalence**: R1(offline)+R2 rotation should NOT change model
     accuracy. Since R1 and R2 are both fully fused into weights (offline),
     applying them is a mathematically lossless orthogonal transform. The
     rotated bf16 model should produce the same (or near-identical) lm_eval
     accuracy as the original bf16 model.

  2. **Save/load roundtrip**: After rotation + quantization, saving the model
     to disk and reloading it should produce identical accuracy as the
     in-memory quantized model.

Rotation levels tested:
  - bf16 baseline (no rotation)
  - R1(offline) + R2
  - R1(offline) + R2 + quantization (W4A16)
  - (optional) R1(offline) + R2 + quantization (MXFP4, NVFP4)

What "offline R1" means:
  online_r1_rotation=False → R1 is fused directly into weights:
    embed_tokens, all linear weights, lm_head, RMSNorm gamma
  This is identical to what llm-compressor does. No runtime hooks needed.
  R2 is always offline (fused into v_proj/o_proj weights).

Usage:
    # Quick test (limit=100 samples):
    python test_offline_r1r2_equivalence.py --device cuda:0 --limit 100

    # Full evaluation (no limit):
    python test_offline_r1r2_equivalence.py --device cuda:0

    # Custom model:
    python test_offline_r1r2_equivalence.py --model meta-llama/Llama-3.2-1B --device cuda:0

    # With quantization schemes:
    python test_offline_r1r2_equivalence.py --device cuda:0 --schemes W4A16,MXFP4

    # With auto-round tuning (iters > 0):
    python test_offline_r1r2_equivalence.py --device cuda:0 --quant-iters 200
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
from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TASKS = "hellaswag,piqa,winogrande"
DEFAULT_SCHEMES = ["W4A16"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory(model=None):
    """Release GPU memory aggressively."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model_object(
    model, tokenizer, tasks: str, batch_size: int = 8,
    limit: int | None = None, device: str = "cuda:0",
    softmax_dtype: str | None = "float32",
) -> dict[str, float]:
    """Evaluate an in-memory model object using lm_eval with HF backend.

    Args:
        softmax_dtype: dtype for log_softmax computation. "float32" recommended
            to match vLLM backend precision.
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
        device=device, softmax_dtype=softmax_dtype, add_bos_token=True,
    )
    task_list = [t.strip() for t in tasks.split(",")]
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size,
                              limit=limit)
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


def evaluate_model_from_path(
    model_path: str, tasks: str, batch_size: int = 8,
    limit: int | None = None, device: str = "cuda:0",
    softmax_dtype: str | None = "float32",
) -> dict[str, float]:
    """Load a saved model from disk and evaluate using lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model_path, batch_size=batch_size, device=device,
        softmax_dtype=softmax_dtype, add_bos_token=True,
    )
    task_list = [t.strip() for t in tasks.split(",")]
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size,
                              limit=limit)
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


def find_model_subdir(output_dir: str) -> str:
    """Find the actual model subdirectory (AutoRound creates a subdir)."""
    if os.path.exists(os.path.join(output_dir, "config.json")):
        return output_dir
    for entry in os.listdir(output_dir):
        candidate = os.path.join(output_dir, entry)
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "config.json")
        ):
            return candidate
    return output_dir


def check_saved_model(save_dir: str) -> dict[str, Any]:
    """Verify saved model has expected rotation artifacts."""
    info = {}
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        sc = config.get("quantization_config", {}).get("spinquant_config")
        info["config_has_spinquant"] = sc is not None
        if sc:
            info["r1"] = sc.get("r1", False)
            info["r2"] = sc.get("r2", False)
            info["online_r1"] = sc.get("online_r1_rotation", True)

    spinquant_keys = []
    for f in os.listdir(save_dir):
        if f.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(os.path.join(save_dir, f), framework="pt") as sf:
                for key in sf.keys():
                    if "spinquant" in key:
                        spinquant_keys.append(key)
    info["n_spinquant_keys"] = len(spinquant_keys)
    info["spinquant_keys_sample"] = spinquant_keys[:5]
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Rotation Equivalence (bf16 vs rotated-bf16)
# ═══════════════════════════════════════════════════════════════════════════════

def test_rotation_equivalence(
    model_name: str, tokenizer, args,
) -> dict[str, Any]:
    """Verify that offline R1+R2 rotation does not change bf16 accuracy.

    Steps:
      1. Evaluate original bf16 model → baseline metrics
      2. Apply R1(offline)+R2 rotation to a fresh copy → rotated metrics
      3. Compare: they should match (within floating-point tolerance)
    """
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: Rotation Equivalence (bf16 vs R1_offline+R2 bf16)")
    logger.info("=" * 70)

    result = {"phase": "rotation_equivalence"}

    # ── 1a: BF16 baseline ──
    logger.info("\n  [1/2] Evaluating original bf16 model (no rotation)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)
    model.eval()

    metrics_baseline = evaluate_model_object(
        model, tokenizer, args.tasks,
        batch_size=args.batch_size, limit=args.limit, device=args.device,
    )
    baseline_time = time.time() - t0
    result["metrics_baseline"] = metrics_baseline
    logger.info(f"    Baseline eval done in {baseline_time:.1f}s")
    for task, acc in sorted(metrics_baseline.items()):
        logger.info(f"      {task}: {acc:.4f}")

    free_memory(model)

    # ── 1b: R1(offline) + R2 rotated bf16 ──
    logger.info("\n  [2/2] Evaluating R1(offline)+R2 rotated bf16 model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()

    rotation_config = SpinQuantConfig(
        r1=True, r2=True, r3=False, r4=False,
        online_r1_rotation=False,  # offline: fuse R1 into weights
        trainable_rotation=False,
        trainable_smooth=False,
    )
    model = apply_rotation(model, rotation_config)
    model.to(args.device)
    model.eval()

    metrics_rotated = evaluate_model_object(
        model, tokenizer, args.tasks,
        batch_size=args.batch_size, limit=args.limit, device=args.device,
    )
    rotated_time = time.time() - t0
    result["metrics_rotated"] = metrics_rotated
    logger.info(f"    Rotated eval done in {rotated_time:.1f}s")
    for task, acc in sorted(metrics_rotated.items()):
        logger.info(f"      {task}: {acc:.4f}")

    free_memory(model)

    # ── Compare ──
    logger.info("\n  ── Rotation Equivalence Comparison ──")
    all_equiv = True
    max_diff = 0.0
    for task in sorted(set(metrics_baseline) | set(metrics_rotated)):
        base_acc = metrics_baseline.get(task, -1)
        rot_acc = metrics_rotated.get(task, -1)
        diff = abs(base_acc - rot_acc)
        max_diff = max(max_diff, diff)
        # Allow small floating-point tolerance (< 1%)
        equiv = diff < 0.01
        if not equiv:
            all_equiv = False
        status = "✓ EQUIV" if equiv else f"✗ DIFF={diff:.4f}"
        logger.info(f"    {task}: baseline={base_acc:.4f}  rotated={rot_acc:.4f}  {status}")

    result["equivalent"] = all_equiv
    result["max_diff"] = round(max_diff, 4)
    logger.info(f"\n  Max difference: {max_diff:.4f}")
    logger.info(f"  Rotation equivalence: {'✓ PASS' if all_equiv else '✗ FAIL'}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Rotation + Quantization + Save/Load Roundtrip
# ═══════════════════════════════════════════════════════════════════════════════

def test_quant_roundtrip(
    model_name: str, tokenizer, scheme_name: str, scheme_str: str, args,
) -> dict[str, Any]:
    """Verify save/load roundtrip for R1(offline)+R2 + quantization.

    Steps:
      1. Load model → apply R1(offline)+R2 rotation + quantization → save
      2. Evaluate in-memory quantized model → in-memory metrics
      3. Load from disk → evaluate → from-disk metrics
      4. Compare: in-memory vs from-disk should match exactly
    """
    iters = args.quant_iters
    label = f"R1_offline+R2 + {scheme_name}"
    if iters > 0:
        label += f" (iters={iters})"

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PHASE 2: Roundtrip — {label}")
    logger.info(f"{'=' * 70}")

    result = {"phase": "quant_roundtrip", "scheme": scheme_name, "quant_iters": iters}

    rotation_config = SpinQuantConfig(
        r1=True, r2=True, r3=False, r4=False,
        online_r1_rotation=False,
        trainable_rotation=False,
        trainable_smooth=False,
    )

    model_short = model_name.split("/")[-1]
    iters_tag = f"-iters{iters}" if iters > 0 else ""
    save_dir = os.path.join(
        args.output_dir, f"{model_short}-R1off_R2-{scheme_name}{iters_tag}"
    )

    t0 = time.time()

    try:
        # ── Step 1: Quantize and save ──
        logger.info(f"  [1/4] AutoRound(rotation_config=R1_offline+R2, scheme={scheme_str}, iters={iters})")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.eval()

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
        logger.info(f"    Quantize+save done in {quant_time:.1f}s → {save_dir}")

        actual_save_dir = find_model_subdir(save_dir)
        result["save_dir"] = actual_save_dir

        # ── Step 2: Check saved artifacts ──
        logger.info(f"  [2/4] Checking saved model artifacts...")
        check = check_saved_model(actual_save_dir)
        result["save_checks"] = check
        logger.info(f"    config has spinquant_config: {check.get('config_has_spinquant')}")
        logger.info(f"    online_r1_rotation: {check.get('online_r1')}")
        logger.info(f"    spinquant keys in safetensors: {check.get('n_spinquant_keys')}")
        for k in check.get("spinquant_keys_sample", []):
            logger.info(f"      {k}")

        # Free in-memory model before evaluations
        del ar
        free_memory(model)
        model = None

        # ── Step 3: Evaluate from disk ──
        logger.info(f"  [3/4] Evaluating model FROM DISK: {actual_save_dir}")
        t_eval = time.time()
        metrics_disk = evaluate_model_from_path(
            actual_save_dir, args.tasks,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )
        eval_disk_time = time.time() - t_eval
        result["metrics_from_disk"] = metrics_disk
        logger.info(f"    From-disk eval done in {eval_disk_time:.1f}s")
        for task, acc in sorted(metrics_disk.items()):
            logger.info(f"      {task}: {acc:.4f}")

        free_memory()

        # ── Step 4: Evaluate in-memory (fresh quantize, same config) ──
        logger.info(f"  [4/4] Evaluating IN-MEMORY model (fresh quantize, same config)...")
        t_inmem = time.time()

        # Reset seed for reproducibility
        set_seed(args.seed)

        model2 = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
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

        metrics_inmem = evaluate_model_object(
            model2, tokenizer, args.tasks,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )
        inmem_time = time.time() - t_inmem
        result["metrics_in_memory"] = metrics_inmem
        logger.info(f"    In-memory eval done in {inmem_time:.1f}s")
        for task, acc in sorted(metrics_inmem.items()):
            logger.info(f"      {task}: {acc:.4f}")

        del ar2
        free_memory(model2)

        # ── Compare ──
        logger.info("\n  ── Save/Load Roundtrip Comparison ──")
        all_match = True
        for task in sorted(set(metrics_disk) | set(metrics_inmem)):
            disk_acc = metrics_disk.get(task, -1)
            mem_acc = metrics_inmem.get(task, -1)
            diff = abs(disk_acc - mem_acc)
            match = diff < 1e-4
            if not match:
                all_match = False
            status = "✓ MATCH" if match else f"✗ DIFF={diff:.4f}"
            logger.info(f"    {task}: disk={disk_acc:.4f}  mem={mem_acc:.4f}  {status}")

        result["roundtrip_match"] = all_match
        result["status"] = "success"
        logger.info(f"  Roundtrip: {'✓ PASS' if all_match else '✗ FAIL'}")

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "error"
        result["error"] = str(e)

    finally:
        free_memory()

    result["total_time_s"] = round(time.time() - t0, 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(equiv_result: dict, roundtrip_results: list[dict], task_list: list[str]):
    """Print final summary table."""
    print(f"\n{'═' * 90}")
    print(f"  SUMMARY — Offline R1+R2 Equivalence & Roundtrip Verification")
    print(f"{'═' * 90}")

    # Phase 1: Rotation equivalence
    print(f"\n  Phase 1: Rotation Equivalence (bf16 vs R1_offline+R2 bf16)")
    print(f"  {'─' * 60}")
    header = f"  {'Task':<20}"
    header += f"{'Baseline':>10}{'Rotated':>10}{'Diff':>8}{'Status':>10}"
    print(header)

    metrics_base = equiv_result.get("metrics_baseline", {})
    metrics_rot = equiv_result.get("metrics_rotated", {})
    for task in task_list:
        base = metrics_base.get(task, -1)
        rot = metrics_rot.get(task, -1)
        diff = abs(base - rot)
        status = "✓" if diff < 0.01 else "✗"
        print(f"  {task:<20}{base:>10.4f}{rot:>10.4f}{diff:>8.4f}{status:>10}")

    equiv_pass = equiv_result.get("equivalent", False)
    print(f"\n  Rotation equivalence: {'✓ PASS' if equiv_pass else '✗ FAIL'}"
          f"  (max diff: {equiv_result.get('max_diff', -1):.4f})")

    # Phase 2: Roundtrip
    if roundtrip_results:
        print(f"\n  Phase 2: Save/Load Roundtrip (rotation + quantization)")
        print(f"  {'─' * 60}")
        for rr in roundtrip_results:
            scheme = rr.get("scheme", "?")
            status = rr.get("status", "?")
            match = rr.get("roundtrip_match")
            print(f"\n  Scheme: {scheme}  (iters={rr.get('quant_iters', 0)})")

            if status == "error":
                print(f"    ✗ ERROR: {rr.get('error', 'unknown')}")
                continue

            header = f"    {'Task':<20}{'From-Disk':>10}{'In-Memory':>10}{'Diff':>8}{'Status':>10}"
            print(header)
            m_disk = rr.get("metrics_from_disk", {})
            m_mem = rr.get("metrics_in_memory", {})
            for task in task_list:
                d = m_disk.get(task, -1)
                m = m_mem.get(task, -1)
                diff = abs(d - m)
                s = "✓" if diff < 1e-4 else "✗"
                print(f"    {task:<20}{d:>10.4f}{m:>10.4f}{diff:>8.4f}{s:>10}")
            print(f"    Roundtrip: {'✓ MATCH' if match else '✗ MISMATCH'}")

    # Overall
    print(f"\n{'═' * 90}")
    all_pass = equiv_pass and all(
        r.get("roundtrip_match", False) for r in roundtrip_results
        if r.get("status") == "success"
    )
    print(f"  OVERALL: {'✓ ALL PASSED' if all_pass else '✗ SOME FAILED'}")
    print(f"{'═' * 90}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

SCHEME_MAP = {
    "W4A16":      "W4A16",
    "W3A16":      "W3A16",
    "W8A16":      "W8A16",
    "MXFP4":      "MXFP4_RCEIL",
    "NVFP4":      "NVFP4",
    "FP8_STATIC": "FP8_STATIC",
}


def main():
    parser = argparse.ArgumentParser(
        description="Verify offline R1+R2 rotation equivalence and save/load roundtrip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test:
  python test_offline_r1r2_equivalence.py --device cuda:0 --limit 100

  # Full evaluation with multiple quant schemes:
  python test_offline_r1r2_equivalence.py --device cuda:0 --schemes W4A16,MXFP4

  # With auto-round tuning:
  python test_offline_r1r2_equivalence.py --device cuda:0 --quant-iters 200

  # Skip roundtrip (only test rotation equivalence):
  python test_offline_r1r2_equivalence.py --device cuda:0 --skip-roundtrip
""",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help=f"Comma-separated lm_eval tasks (default: {DEFAULT_TASKS})")
    parser.add_argument("--schemes", type=str, default="W4A16",
                        help="Comma-separated quant schemes for roundtrip test "
                             f"(choices: {','.join(SCHEME_MAP)})")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit lm_eval samples (None=full eval)")
    parser.add_argument("--quant-iters", type=int, default=0,
                        help="AutoRound tuning iterations (0=RTN)")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output dir for saved models")
    parser.add_argument("--skip-roundtrip", action="store_true",
                        help="Skip Phase 2 (roundtrip), only test rotation equivalence")
    parser.add_argument("--skip-equivalence", action="store_true",
                        help="Skip Phase 1 (equivalence), only test roundtrip")
    parser.add_argument("--keep-models", action="store_true",
                        help="Keep saved model directories after test")

    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        args.output_dir = f"offline_r1r2_{model_short}_{ts}"

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger.info(f"Model:      {args.model}")
    logger.info(f"Device:     {args.device}")
    logger.info(f"Tasks:      {args.tasks}")
    logger.info(f"Limit:      {args.limit}")
    logger.info(f"Schemes:    {args.schemes}")
    logger.info(f"Quant iters: {args.quant_iters}")
    logger.info(f"Output dir: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    task_list = [t.strip() for t in args.tasks.split(",")]

    # ── Phase 1: Rotation equivalence ──
    equiv_result = {}
    if not args.skip_equivalence:
        equiv_result = test_rotation_equivalence(args.model, tokenizer, args)
    else:
        logger.info("\n  Phase 1 skipped (--skip-equivalence)")
        equiv_result = {"equivalent": True, "max_diff": 0.0,
                        "metrics_baseline": {}, "metrics_rotated": {}}

    # ── Phase 2: Roundtrip tests ──
    roundtrip_results = []
    if not args.skip_roundtrip:
        schemes = [s.strip() for s in args.schemes.split(",")]
        for scheme_name in schemes:
            if scheme_name not in SCHEME_MAP:
                logger.warning(f"Unknown scheme '{scheme_name}', skipping. "
                               f"Choices: {list(SCHEME_MAP.keys())}")
                continue
            scheme_str = SCHEME_MAP[scheme_name]
            set_seed(args.seed)
            rr = test_quant_roundtrip(
                args.model, tokenizer, scheme_name, scheme_str, args,
            )
            roundtrip_results.append(rr)
    else:
        logger.info("\n  Phase 2 skipped (--skip-roundtrip)")

    # ── Summary ──
    all_pass = print_summary(equiv_result, roundtrip_results, task_list)

    # Save results JSON
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "tasks": args.tasks,
            "limit": args.limit,
            "seed": args.seed,
            "rotation_equivalence": equiv_result,
            "roundtrip_tests": roundtrip_results,
            "all_pass": all_pass,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Cleanup saved models if requested
    if not args.keep_models:
        for rr in roundtrip_results:
            sd = rr.get("save_dir")
            if sd and os.path.isdir(sd):
                parent = os.path.dirname(sd)
                if parent != args.output_dir and os.path.isdir(parent):
                    shutil.rmtree(parent, ignore_errors=True)
                elif os.path.isdir(sd):
                    shutil.rmtree(sd, ignore_errors=True)
        logger.info("Cleaned up saved model directories (use --keep-models to retain)")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
