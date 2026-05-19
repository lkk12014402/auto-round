#!/usr/bin/env python3
"""
Test R4 rotation export and vLLM plugin compatibility.

Steps:
1. Export a model with R1+R4 rotation + MXFP4 quantization using auto-round
2. Verify the checkpoint contains spinquant_r4_type/size buffers on down_proj
3. Verify the vLLM plugin can parse the config correctly
4. (Optional) Load with vLLM plugin and verify rotation matrices are built

Usage:
    python test_r4_export_and_plugin.py --device cuda:7
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-round"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig


def test_r4_export(model_name: str, device: str, output_dir: str):
    """Export model with R1+R4 rotation and MXFP4 quantization."""
    print(f"\n{'='*70}")
    print(f"  Step 1: Export model with R1+R4 + MXFP4")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()

    # R1 + R4 (no R2/R3 for simplicity)
    rotation_config = SpinQuantConfig(
        r1=True, r2=False, r3=False, r4=True,
        online_r1_rotation=True,
        trainable_rotation=False,
        trainable_smooth=False,
    )
    print(f"  rotation_config = {rotation_config}")

    ar = AutoRound(
        model,
        tokenizer=tokenizer,
        rotation_config=rotation_config,
        scheme="MXFP4_RCEIL",
        iters=0,  # No tuning, just rotation + quantization
        nsamples=16,
        seqlen=128,
        device_map=device,
    )
    ar.quantize()

    # Save
    ar.save_quantized(output_dir, format="auto_round")
    print(f"\n  ✅ Model saved to {output_dir}")
    return tokenizer


def test_checkpoint_contents(output_dir: str):
    """Verify the checkpoint has R4 rotation buffers."""
    print(f"\n{'='*70}")
    print(f"  Step 2: Verify checkpoint contents")
    print(f"{'='*70}\n")

    # Check config.json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization_config", {})
    sq_config = quant_config.get("spinquant_config", {})

    print(f"  quantization_config.quant_method: {quant_config.get('quant_method')}")
    print(f"  quantization_config.data_type: {quant_config.get('data_type')}")
    print(f"  spinquant_config: {json.dumps(sq_config, indent=4)}")

    assert sq_config.get("r1") == True, "R1 should be enabled"
    assert sq_config.get("r4") == True, "R4 should be enabled"
    assert sq_config.get("online_r1_rotation") == True, "Online R1 should be True"
    print(f"\n  ✅ Config has R1=True, R4=True, online_r1_rotation=True")

    # Check safetensors for R4 buffers
    import safetensors
    from safetensors import safe_open

    safetensor_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
    print(f"\n  Safetensor files: {safetensor_files}")

    r1_keys = []
    r4_keys = []
    for sf_file in safetensor_files:
        with safe_open(os.path.join(output_dir, sf_file), framework="pt") as f:
            for key in f.keys():
                if "spinquant_r1" in key:
                    r1_keys.append(key)
                if "spinquant_r4" in key:
                    r4_keys.append(key)

    print(f"\n  R1 buffer keys ({len(r1_keys)}):")
    for k in r1_keys[:6]:
        print(f"    {k}")
    if len(r1_keys) > 6:
        print(f"    ... ({len(r1_keys)} total)")

    print(f"\n  R4 buffer keys ({len(r4_keys)}):")
    for k in r4_keys[:6]:
        print(f"    {k}")
    if len(r4_keys) > 6:
        print(f"    ... ({len(r4_keys)} total)")

    assert len(r4_keys) > 0, "No R4 keys found in safetensors!"
    # down_proj should have spinquant_r4_type and spinquant_r4_size
    r4_type_keys = [k for k in r4_keys if "r4_type" in k]
    r4_size_keys = [k for k in r4_keys if "r4_size" in k]
    assert len(r4_type_keys) > 0, "No spinquant_r4_type keys found"
    assert len(r4_size_keys) > 0, "No spinquant_r4_size keys found"
    print(f"\n  ✅ R4 buffers present: {len(r4_type_keys)} type + {len(r4_size_keys)} size keys")

    # Check R4 values
    with safe_open(os.path.join(output_dir, safetensor_files[0]), framework="pt") as f:
        if r4_type_keys:
            r4_type_val = f.get_tensor(r4_type_keys[0])
            r4_size_val = f.get_tensor(r4_size_keys[0])
            print(f"\n  Sample R4 values:")
            print(f"    {r4_type_keys[0]}: {r4_type_val.item()}")
            print(f"    {r4_size_keys[0]}: {r4_size_val.item()}")

    return sq_config


def test_plugin_config_parsing(sq_config: dict):
    """Verify the vLLM plugin correctly parses R4 config."""
    print(f"\n{'='*70}")
    print(f"  Step 3: Verify vLLM plugin config parsing")
    print(f"{'='*70}\n")

    from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config

    # Simulate what vLLM does: pass full quantization_config dict
    full_config = {
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 32,
        "data_type": "mx_fp",
        "spinquant_config": sq_config,
    }

    # Test override_quantization_method detection
    method = SpinQuantMXFP4Config.override_quantization_method(full_config, None)
    print(f"  override_quantization_method(config, None) = {method}")
    assert method == "spinquant_mxfp4", f"Expected 'spinquant_mxfp4', got {method}"

    # Test from_config parsing
    plugin_config = SpinQuantMXFP4Config.from_config(full_config)
    print(f"\n  Parsed plugin config:")
    print(f"    online_r1: {plugin_config.online_r1}")
    print(f"    online_r4: {plugin_config.online_r4}")
    print(f"    online_r3: {plugin_config.online_r3}")
    print(f"    r1_type: {plugin_config.r1_type}")
    print(f"    r4_type: {plugin_config.r4_type}")
    print(f"    rotation_size: {plugin_config.rotation_size}")
    print(f"    hidden_size: {plugin_config.hidden_size}")
    print(f"    intermediate_size: {plugin_config.intermediate_size}")
    print(f"    head_dim: {plugin_config.head_dim}")

    assert plugin_config.online_r1 == True
    assert plugin_config.online_r4 == True
    assert plugin_config.r1_type == "hadamard"
    assert plugin_config.r4_type == "hadamard"
    print(f"\n  ✅ Plugin correctly parses R4 config")
    return plugin_config


def test_plugin_rotation_build(plugin_config, output_dir: str, device: str):
    """Test that the plugin can build rotation matrices from checkpoint data."""
    print(f"\n{'='*70}")
    print(f"  Step 4: Verify plugin rotation matrix building")
    print(f"{'='*70}\n")

    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        ROTATION_TYPE_HADAMARD,
        SpinQuantMXFP4LinearMethod,
        _build_full_hadamard,
    )

    method = SpinQuantMXFP4LinearMethod(plugin_config)

    # Simulate what happens during model loading:
    # Create a mock layer with the buffers that would come from checkpoint
    import safetensors
    from safetensors import safe_open

    safetensor_files = [f for f in os.listdir(output_dir) if f.endswith(".safetensors")]
    with safe_open(os.path.join(output_dir, safetensor_files[0]), framework="pt") as f:
        all_keys = list(f.keys())

    # Find a down_proj layer with R4 buffers
    r4_type_keys = [k for k in all_keys if "spinquant_r4_type" in k]
    if not r4_type_keys:
        print("  ⚠️  No R4 type keys found, skipping rotation build test")
        return

    # Get the R4 rotation size from checkpoint
    with safe_open(os.path.join(output_dir, safetensor_files[0]), framework="pt") as f:
        r4_type = f.get_tensor(r4_type_keys[0])
        r4_size_key = r4_type_keys[0].replace("_type", "_size")
        r4_size = f.get_tensor(r4_size_key)

    r4_type_val = int(r4_type.item())
    r4_size_val = int(r4_size.item())
    print(f"  R4 from checkpoint: type={r4_type_val}, size={r4_size_val}")

    # Similarly find R1
    r1_type_keys = [k for k in all_keys if "spinquant_r1_type" in k]
    if r1_type_keys:
        with safe_open(os.path.join(output_dir, safetensor_files[0]), framework="pt") as f:
            r1_type = f.get_tensor(r1_type_keys[0])
            r1_size_key = r1_type_keys[0].replace("_type", "_size")
            r1_size = f.get_tensor(r1_size_key)
        r1_type_val = int(r1_type.item())
        r1_size_val = int(r1_size.item())
        print(f"  R1 from checkpoint: type={r1_type_val}, size={r1_size_val}")

    # Test Hadamard matrix building
    if r4_size_val > 0 and r4_type_val == ROTATION_TYPE_HADAMARD:
        R4 = _build_full_hadamard(r4_size_val, torch.device(device))
        print(f"\n  Built R4 Hadamard matrix: shape={R4.shape}, dtype={R4.dtype}")
        # Verify orthogonality: R @ R^T ≈ I
        eye_check = (R4 @ R4.T)
        identity = torch.eye(r4_size_val, device=R4.device, dtype=R4.dtype)
        diff = (eye_check - identity).abs().max().item()
        print(f"  R4 orthogonality check: max|R@R^T - I| = {diff:.2e}")
        assert diff < 1e-2, f"R4 not orthogonal! max diff = {diff}"
        print(f"  ✅ R4 rotation matrix is orthogonal (tol=1e-2 for large non-pow2 dim)")

    if r1_size_val > 0 and r1_type_val == ROTATION_TYPE_HADAMARD:
        R1 = _build_full_hadamard(r1_size_val, torch.device(device))
        print(f"\n  Built R1 Hadamard matrix: shape={R1.shape}, dtype={R1.dtype}")
        eye_check = (R1 @ R1.T)
        identity = torch.eye(r1_size_val, device=R1.device, dtype=R1.dtype)
        diff = (eye_check - identity).abs().max().item()
        print(f"  R1 orthogonality check: max|R@R^T - I| = {diff:.2e}")
        assert diff < 1e-3, f"R1 not orthogonal! max diff = {diff}"
        print(f"  ✅ R1 rotation matrix is orthogonal")

    # Test _process_rotation logic with a mock layer
    class MockLayer(torch.nn.Module):
        pass

    mock = MockLayer()
    mock.weight_packed = torch.zeros(1, device=device)  # for .device
    mock.spinquant_r4_type = torch.nn.Parameter(
        torch.tensor(r4_type_val, dtype=torch.int32), requires_grad=False
    )
    mock.spinquant_r4_size = torch.nn.Parameter(
        torch.tensor(r4_size_val, dtype=torch.int32), requires_grad=False
    )
    mock.spinquant_r4_matrix = torch.nn.Parameter(
        torch.zeros(r4_size_val, r4_size_val, dtype=torch.float32), requires_grad=False
    )

    method._process_rotation(mock, prefix="r4")
    print(f"\n  After _process_rotation(prefix='r4'):")
    print(f"    mock._r4_rotation_matrix: {mock._r4_rotation_matrix is not None}")
    if mock._r4_rotation_matrix is not None:
        print(f"    shape: {mock._r4_rotation_matrix.shape}, dtype: {mock._r4_rotation_matrix.dtype}")
        print(f"    _r4_rot_size: {mock._r4_rot_size}")
    assert mock._r4_rotation_matrix is not None, "R4 rotation matrix should be built"
    assert mock._r4_rotation_matrix.shape == (r4_size_val, r4_size_val)
    print(f"  ✅ _process_rotation correctly builds R4 matrix")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model to test")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: auto temp dir)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export step (use existing output-dir)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "/data/lkk/quarot/latest/new_commit",
            "test_r4_export_output"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_export:
        tokenizer = test_r4_export(args.model, args.device, args.output_dir)
    else:
        print(f"  Skipping export, using existing dir: {args.output_dir}")

    sq_config = test_checkpoint_contents(args.output_dir)
    plugin_config = test_plugin_config_parsing(sq_config)
    test_plugin_rotation_build(plugin_config, args.output_dir, args.device)

    print(f"\n{'='*70}")
    print(f"  ALL TESTS PASSED ✅")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
