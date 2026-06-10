#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test script for the vLLM SpinQuant MXFP4 plugin.

Validates:
  1. Plugin registration (import auto_round.vllm_plugin registers the config)
  2. Config parsing from model's quantization_config
  3. Weight creation and shape verification
  4. Forward pass with rotation + dequant + GEMM
"""

import sys
import os
import torch

# auto-round path (vllm from pip)
sys.path.insert(0, "/data/lkk/quarot/latest/new_commit/auto-round")


def test_plugin_registration():
    """Test that importing the plugin registers 'spinquant_mxfp4'."""
    import auto_round.vllm_plugin  # noqa: F401
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    assert "spinquant_mxfp4" in QUANTIZATION_METHODS, (
        f"spinquant_mxfp4 not in QUANTIZATION_METHODS: {QUANTIZATION_METHODS}"
    )

    config_cls = get_quantization_config("spinquant_mxfp4")
    assert config_cls.__name__ == "SpinQuantMXFP4Config"
    print("[PASS] Plugin registration")


def test_config_parsing():
    """Test config parsing from quantization_config dict."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config

    config_dict = {
        "bits": 4,
        "group_size": 32,
        "data_type": "mxfp4",
        "spinquant_config": {
            "r1": True,
            "r2": False,
            "r3": False,
            "r4": False,
            "online_r1_rotation": True,
            "rotation_size": None,
            "random_r1": False,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "head_dim": 128,
        },
    }

    cfg = SpinQuantMXFP4Config.from_config(config_dict)
    assert cfg.bits == 4
    assert cfg.group_size == 32
    assert cfg.online_r1 is True
    assert cfg.r1_type == "hadamard"
    assert cfg.hidden_size == 1024
    print("[PASS] Config parsing")


def test_weight_creation():
    """Test that create_weights produces correct parameter shapes."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        SpinQuantMXFP4Config,
        SpinQuantMXFP4LinearMethod,
    )

    cfg = SpinQuantMXFP4Config(
        bits=4, group_size=32, online_r1=True, r1_type="hadamard"
    )
    method = SpinQuantMXFP4LinearMethod(cfg)

    # Simulate a linear layer
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=1024,
        output_partition_sizes=[1024],
        input_size=1024,
        output_size=1024,
        params_dtype=torch.bfloat16,
    )

    assert hasattr(layer, "weight_packed")
    assert layer.weight_packed.shape == (1024, 512)  # [N, K//2]
    assert layer.weight_packed.dtype == torch.uint8

    assert hasattr(layer, "weight_scale")
    assert layer.weight_scale.shape == (1024, 32)  # [N, K//32]
    assert layer.weight_scale.dtype == torch.uint8

    assert hasattr(layer, "spinquant_r1_type")
    assert hasattr(layer, "spinquant_r1_size")
    print("[PASS] Weight creation")


def test_forward_pass():
    """Test the full forward: rotation + dequant + GEMM."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        SpinQuantMXFP4Config,
        SpinQuantMXFP4LinearMethod,
        ROTATION_TYPE_HADAMARD,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 256, 256
    group_size = 32

    cfg = SpinQuantMXFP4Config(bits=4, group_size=group_size, online_r1=True)
    method = SpinQuantMXFP4LinearMethod(cfg)

    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
    )

    # Fill with dummy data
    layer.weight_packed.data = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    layer.weight_scale.data = torch.randint(120, 135, (N, K // group_size), dtype=torch.uint8)
    layer.spinquant_r1_type.data = torch.tensor(ROTATION_TYPE_HADAMARD, dtype=torch.int32)
    layer.spinquant_r1_size.data = torch.tensor(K, dtype=torch.int32)

    layer = layer.to(device)

    # Process weights (build rotation matrix)
    method.process_weights_after_loading(layer)
    assert hasattr(layer, "_rotation_matrix"), "Rotation matrix not built"
    assert layer._rotation_matrix is not None, "Rotation matrix is None"
    assert layer._rotation_matrix.shape == (K, K), f"Expected ({K},{K}), got {layer._rotation_matrix.shape}"

    # Forward pass
    x = torch.randn(2, K, dtype=torch.bfloat16, device=device)
    output = method.apply(layer, x, bias=None)

    assert output.shape == (2, N), f"Expected (2, {N}), got {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    print(f"[PASS] Forward pass (device={device}, output_range=[{output.min():.3f}, {output.max():.3f}])")


def test_override_detection():
    """Test auto-detection of spinquant_mxfp4 models."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config

    # Should detect
    cfg1 = {
        "quant_method": "auto-round",
        "data_type": "mxfp4",
        "spinquant_config": {"online_r1_rotation": True},
    }
    result = SpinQuantMXFP4Config.override_quantization_method(cfg1, None)
    assert result == "spinquant_mxfp4", f"Expected detection, got {result}"

    # Should NOT detect (no spinquant_config)
    cfg2 = {"quant_method": "auto-round", "data_type": "int"}
    result2 = SpinQuantMXFP4Config.override_quantization_method(cfg2, None)
    assert result2 is None, f"Should not detect, got {result2}"

    print("[PASS] Override detection")


if __name__ == "__main__":
    test_plugin_registration()
    test_config_parsing()
    test_weight_creation()
    test_forward_pass()
    test_override_detection()
    print("\n✅ All vLLM plugin tests passed!")
