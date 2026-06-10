#!/usr/bin/env python
"""Test: selective per-linear Hadamard rotation → save → vLLM-compatible model.

Usage:
    # Quantize with selective rotation and save
    python scripts/test_selective_rotation_save_vllm.py \
        --model Qwen/Qwen3-0.6B \
        --output /tmp/qwen3-0.6b-mxfp4-selective

    # Then load in vLLM:
    #   pip install -e /data/lkk/quarot/latest/new_commit/auto-round
    #   python -c "from vllm import LLM; llm = LLM('/tmp/qwen3-0.6b-mxfp4-selective')"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/data/lkk/quarot/latest/new_commit/auto-round")

import torch


def main():
    parser = argparse.ArgumentParser(description="Selective rotation + MXFP4 quantization + save")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=str, default="./selective_rotation_mxfp4_model")
    parser.add_argument("--layer-selection", type=str, default="structural",
                        choices=["all", "structural", "auto", "none"])
    parser.add_argument("--hadamard-type", type=str, default="hadamard",
                        choices=["hadamard", "random_hadamard"])
    parser.add_argument("--iters", type=int, default=0, help="AutoRound tuning iters (0=RTN)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from auto_round import AutoRound

    # Configure selective rotation
    rotation_config = {
        "hadamard_type": args.hadamard_type,
        "block_size": 32,
        "layer_selection": args.layer_selection,
        "backend": "transform",
    }

    #rotation_config = "quarot"

    print(f"\nRotation config: {rotation_config}")
    print(f"Quantization: MXFP4, iters={args.iters}")

    ar = AutoRound(
        args.model,
        scheme="MXFP4",
        iters=0,
        device_map="auto",
        low_gpu_mem_usage=True,
        rotation_config=rotation_config,
    )
    ar.quantize_and_save(output_dir=args.output, format="auto_round")


    # Verify saved config
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        qcfg = config.get("quantization_config", {})
        sq_cfg = qcfg.get("spinquant_config", {})
        if sq_cfg:
            print(f"\n✅ spinquant_config saved in config.json:")
            print(f"   algorithm: {sq_cfg.get('algorithm')}")
            print(f"   online_r1_rotation: {sq_cfg.get('online_r1_rotation')}")
            print(f"   selective_rotation: {sq_cfg.get('selective_rotation', False)}")
            if sq_cfg.get("selective_rotation"):
                print(f"   num_rotated_layers: {sq_cfg.get('num_rotated_layers')}")
                rotated = sq_cfg.get("rotated_layers", [])
                print(f"   rotated_layers (first 5): {rotated[:5]}")
        else:
            print("⚠️  No spinquant_config found in config.json")
            print(f"   Keys in quantization_config: {list(qcfg.keys())}")

    # Quick check: verify rotation buffers in safetensors
    import glob
    st_files = glob.glob(os.path.join(args.output, "*.safetensors"))
    if st_files:
        from safetensors import safe_open
        with safe_open(st_files[0], framework="pt") as f:
            keys = f.keys()
            r1_keys = [k for k in keys if "spinquant_r1" in k]
            print(f"\n✅ Rotation buffer keys in safetensors: {len(r1_keys)}")
            if r1_keys:
                print(f"   Examples: {r1_keys[:4]}")
    else:
        print("⚠️  No safetensors files found")

    print(f"\n{'='*60}")
    print(f"Model saved to: {args.output}")
    print(f"To use with vLLM:")
    print(f"  pip install -e /data/lkk/quarot/latest/new_commit/auto-round")
    print(f"  python -c \"from vllm import LLM; llm = LLM('{args.output}')\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
