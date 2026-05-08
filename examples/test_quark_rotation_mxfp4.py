"""
Quark: Rotation + MXFP4 Quantization evaluation on Qwen3-0.6B.

This script tests Quark's rotation + MXFP4 quantization pipeline:
  1. Baseline MXFP4 (no rotation)
  2. Rotation (R1+R2, Hadamard, no training) + MXFP4

Uses Quark's ModelQuantizer + RotationProcessor for an apples-to-apples
comparison with auto-round's equivalent test.

Usage:
    python test_quark_rotation_mxfp4.py --device cuda:0 --tasks piqa,hellaswag
    python test_quark_rotation_mxfp4.py --device cuda:7 --tasks piqa,hellaswag --limit 200
"""

import argparse
import copy
import gc
import sys
import time

sys.path.insert(0, "/data/lkk/quarot/Quark")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.quantization.config.config import RotationConfig, load_quant_algo_config_from_file
from quark.torch.utils.llm import get_calib_dataloader, get_model, get_tokenizer, prepare_for_moe_quant


def evaluate_model(model, tokenizer, tasks, batch_size=8, limit=None, device="cuda:0"):
    """Evaluate model using lm_eval framework."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)

    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = tasks

    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        device=device,
    )
    return results


def extract_metrics(results):
    """Extract key metrics from lm_eval results."""
    metrics = {}
    if "results" not in results:
        return metrics
    for task_name, task_results in results["results"].items():
        acc_norm = task_results.get("acc_norm,none") or task_results.get("acc,none")
        if acc_norm is not None:
            metrics[task_name] = acc_norm
    return metrics


def get_rotation_config(r1=True, r2=True, r3=False, r4=False, online_r1=True, rotation_size=128):
    """Build a Quark RotationConfig object for the rotation algorithm.

    Note: Quark's R3 does NOT support custom rotation_size. If r3=True and
    rotation_size is set, we pass rotation_size=None to Quark (R3 always uses
    head_dim internally).
    """
    # Quark R3 raises NotImplementedError with custom rotation_size
    effective_rotation_size = None if r3 else rotation_size
    scaling_layers = {
        "first_layer": [
            {
                "prev_modules": ["model.embed_tokens"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
            },
        ],
        "middle_layers": [
            {
                "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
            },
        ],
        "last_layer": [
            {
                "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                "norm_module": "model.norm",
                "next_modules": ["lm_head"],
                "target_modules": ["lm_head"],
            }
        ],
    }

    config = RotationConfig(
        backbone="model",
        model_decoder_layers="model.layers",
        v_proj="self_attn.v_proj",
        o_proj="self_attn.o_proj",
        self_attn="self_attn",
        mlp="mlp",
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        trainable=False,
        online_r1_rotation=online_r1,
        rotation_size=effective_rotation_size,
        scaling_layers=scaling_layers,
    )
    return config


def quantize_with_quark_mxfp4(model, tokenizer, device, with_rotation=False,
                               r1=True, r2=True, r3=False, r4=False,
                               rotation_size=128,
                               num_calib_data=128, seqlen=512):
    """Quantize model using Quark's MXFP4 scheme, optionally with rotation.

    Args:
        model: Pre-loaded model.
        tokenizer: Tokenizer.
        device: Device string.
        with_rotation: If True, apply rotation before quantization.
        r1, r2, r3, r4: Which rotations to enable.
        rotation_size: Rotation size for R1 (default: 128).
        num_calib_data: Number of calibration samples.
        seqlen: Calibration sequence length.

    Returns:
        Quantized (and optionally rotated) model.
    """
    model_type = model.config.model_type
    template = LLMTemplate.get(model_type)

    # Build quantization config
    if with_rotation:
        rotation_config = get_rotation_config(r1=r1, r2=r2, r3=r3, r4=r4,
                                              rotation_size=rotation_size)
        algo_configs = {"rotation": rotation_config}
        quant_config = template.get_config(
            scheme="mxfp4",
            algorithm=["rotation"],
            layer_config={},
            algo_configs=algo_configs,
        )
    else:
        quant_config = template.get_config(
            scheme="mxfp4",
            algorithm=[],
            layer_config={},
        )

    # Calibration data
    calib_dataloader = get_calib_dataloader(
        dataset_name="pileval_for_awq_benchmark",
        tokenizer=tokenizer,
        batch_size=1,
        num_calib_data=num_calib_data,
        seqlen=seqlen,
        device=device,
    )

    # Quantize
    quantizer = ModelQuantizer(quant_config)
    with torch.no_grad():
        model = quantizer.quantize_model(model, calib_dataloader)
    model = quantizer.freeze(model)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Quark: Evaluate Rotation + MXFP4 Quantization on Qwen3-0.6B"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--tasks", type=str, default="piqa,hellaswag", help="Comma-separated lm_eval tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (None=full)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for evaluation")
    parser.add_argument("--num_calib_data", type=int, default=128, help="Calibration samples")
    parser.add_argument("--seqlen", type=int, default=512, help="Calibration sequence length")
    parser.add_argument(
        "--levels", type=str, default="baseline_fp16,mxfp4_only,rotation_mxfp4",
        help="Comma-separated levels: baseline_fp16, mxfp4_only, rotation_mxfp4"
    )
    parser.add_argument("--r1", action="store_true", default=True)
    parser.add_argument("--no-r1", action="store_false", dest="r1")
    parser.add_argument("--r2", action="store_true", default=True)
    parser.add_argument("--no-r2", action="store_false", dest="r2")
    parser.add_argument("--r3", action="store_false", default=False)
    parser.add_argument("--enable-r3", action="store_true", dest="r3")
    parser.add_argument("--r4", action="store_false", default=False)
    parser.add_argument("--enable-r4", action="store_true", dest="r4")
    parser.add_argument("--rotation-size", type=int, default=128,
                        help="Rotation size for R1 (default: 128, matching Quark default)")
    args = parser.parse_args()

    levels_to_test = [l.strip() for l in args.levels.split(",")]

    rotation_desc = "+".join(
        [f"R{i}" for i, enabled in enumerate([args.r1, args.r2, args.r3, args.r4], 1) if enabled]
    ) or "None"

    print("=" * 70)
    print("Quark: Rotation + MXFP4 Quantization Evaluation")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Tasks:      {args.tasks}")
    print(f"  Limit:      {args.limit or 'full'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device:     {args.device}")
    print(f"  Levels:     {levels_to_test}")
    print(f"  Quant:      MXFP4 (W4A4, weight static + activation dynamic)")
    print(f"  Rotation:   {rotation_desc} (Hadamard, no training)")
    print(f"  Rot. size:  {args.rotation_size}")
    print("=" * 70)

    tokenizer = get_tokenizer(args.model, model_type="qwen3", trust_remote_code=False)
    all_results = {}

    for level in levels_to_test:
        print(f"\n{'='*70}")
        print(f"  Evaluating: {level}")
        print(f"{'='*70}")

        t0 = time.time()

        if level == "baseline_fp16":
            # Pure FP16 baseline — no quantization, no rotation
            print("  Loading FP16 baseline model...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            model.eval()

        elif level == "mxfp4_only":
            # MXFP4 quantization without rotation
            print("  Loading model and applying MXFP4 quantization (no rotation)...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            prepare_for_moe_quant(model)
            model = quantize_with_quark_mxfp4(
                model, tokenizer, args.device,
                with_rotation=False,
                num_calib_data=args.num_calib_data,
                seqlen=args.seqlen,
            )

        elif level == "rotation_mxfp4":
            # Rotation + MXFP4 quantization
            print(f"  Loading model and applying rotation ({rotation_desc}) + MXFP4...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, trust_remote_code=True
            ).to(args.device)
            prepare_for_moe_quant(model)
            model = quantize_with_quark_mxfp4(
                model, tokenizer, args.device,
                with_rotation=True,
                r1=args.r1, r2=args.r2, r3=args.r3, r4=args.r4,
                rotation_size=args.rotation_size,
                num_calib_data=args.num_calib_data,
                seqlen=args.seqlen,
            )

        else:
            print(f"  [WARN] Unknown level '{level}', skipping")
            continue

        setup_time = time.time() - t0
        print(f"  Setup completed in {setup_time:.1f}s")

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

        metrics = extract_metrics(results)
        all_results[level] = metrics

        for task, acc in metrics.items():
            print(f"    {task}: acc_norm = {acc:.4f}")
        print(f"  Evaluation done in {eval_time:.1f}s")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE — Quark Rotation + MXFP4")
    print("=" * 90)

    tasks = sorted(set(t for m in all_results.values() for t in m.keys()))
    configs = list(all_results.keys())

    header = f"{'Task':<15}"
    for config in configs:
        header += f" | {config:<15}"
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{task:<15}"
        for config in configs:
            val = all_results.get(config, {}).get(task)
            if val is not None:
                row += f" | {val:<15.4f}"
            else:
                row += f" | {'N/A':<15}"
        print(row)

    # Delta row
    if "baseline_fp16" in all_results and len(configs) > 1:
        print()
        print("Δ vs baseline_fp16:")
        for task in tasks:
            base = all_results.get("baseline_fp16", {}).get(task)
            if base is None:
                continue
            deltas = []
            for config in configs:
                if config == "baseline_fp16":
                    continue
                val = all_results.get(config, {}).get(task)
                if val is not None:
                    delta = val - base
                    deltas.append(f"  {config}: {delta:+.4f}")
            print(f"  {task}: " + ", ".join(deltas))

    print("=" * 90)


if __name__ == "__main__":
    main()
