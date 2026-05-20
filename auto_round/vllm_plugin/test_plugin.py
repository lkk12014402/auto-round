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
    assert cfg.runtime_backend == "packed_fused"
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
    """Test the full forward: rotation + activation qdq + dequant + GEMM."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        ROTATION_RUNTIME_HADAMARD,
        SpinQuantMXFP4Config,
        SpinQuantMXFP4LinearMethod,
        ROTATION_TYPE_HADAMARD,
        _mxfp4_dequant_linear_fallback,
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
    packed_before = layer.weight_packed.detach().clone()

    layer = layer.to(device)

    # Load-time preprocessing prepares activation rotation state only.
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight_packed.cpu(), packed_before), "Packed weights changed during load-time prep"
    assert layer._r1_rotation_runtime == ROTATION_RUNTIME_HADAMARD
    assert layer._r1_rotation_matrix is None
    assert layer._r1_hadamard_K is not None
    assert layer._r1_rot_size == K

    # Forward pass
    x = torch.randn(2, K, dtype=torch.bfloat16, device=device)
    output = method.apply(layer, x, bias=None)
    rotated = method._apply_rotation(layer, x, prefix="r1")
    rotated_qdq = torch.ops.auto_round.spinquant_mxfp4_act_qdq(rotated, group_size)
    expected = _mxfp4_dequant_linear_fallback(rotated_qdq, layer.weight_packed, layer.weight_scale, group_size)

    assert output.shape == (2, N), f"Expected (2, {N}), got {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
    print(f"[PASS] Forward pass (device={device}, output_range=[{output.min():.3f}, {output.max():.3f}])")


def test_hadamard_runtime_matches_explicit_matrix():
    """Test full-size Hadamard load-time prep against an explicit matrix."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        ROTATION_TYPE_HADAMARD,
        SpinQuantMXFP4Config,
        SpinQuantMXFP4LinearMethod,
        _build_full_hadamard,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = 64

    cfg = SpinQuantMXFP4Config(bits=4, group_size=32, online_r1=True)
    method = SpinQuantMXFP4LinearMethod(cfg)

    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[K],
        input_size=K,
        output_size=K,
        params_dtype=torch.bfloat16,
    )
    layer.spinquant_r1_type.data = torch.tensor(ROTATION_TYPE_HADAMARD, dtype=torch.int32)
    layer.spinquant_r1_size.data = torch.tensor(K, dtype=torch.int32)
    layer = layer.to(device)
    method.process_weights_after_loading(layer)

    x = torch.randn(3, K, dtype=torch.float64, device=device)
    expected = x @ _build_full_hadamard(K, x.device).to(dtype=x.dtype)
    actual = method._apply_rotation(layer, x, prefix="r1")

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    print(f"[PASS] Hadamard runtime path matches explicit matrix (device={device})")


def test_activation_qdq_matches_vllm_ext_reference():
    """Test activation qdq alignment with the vllm_ext reference implementation."""
    from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(5, 64, dtype=torch.float32, device=device)

    actual = torch.ops.auto_round.spinquant_mxfp4_act_qdq(x, 32)
    expected = qdq_mxfp4(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    print(f"[PASS] Activation qdq matches vllm_ext reference (device={device})")


def test_local_activation_qdq_kernel_matches_reference():
    """Test the vendored local CUDA QDQ kernel against the PyTorch reference."""
    from auto_round.vllm_plugin._mxfp4_qdq_ext import qdq_mxfp4 as local_qdq_mxfp4
    from auto_round.vllm_plugin.spinquant_mxfp4 import _mxfp4_act_qdq_fallback

    if not torch.cuda.is_available():
        print("[SKIP] Local activation qdq kernel test requires CUDA")
        return

    x = torch.randn(7, 64, dtype=torch.bfloat16, device="cuda")
    actual = local_qdq_mxfp4(x, 32)
    if actual is None:
        print("[SKIP] Local activation qdq kernel extension unavailable")
        return
    expected = _mxfp4_act_qdq_fallback(x, 32)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print("[PASS] Local activation qdq kernel matches reference")


def test_quark_like_dense_backend():
    """Test load-time dense backend preparation and forward path."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        SpinQuantMXFP4Config,
        SpinQuantMXFP4LinearMethod,
        RUNTIME_BACKEND_QUARK_LIKE_DENSE,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 64, 64
    group_size = 32

    cfg = SpinQuantMXFP4Config(
        bits=4,
        group_size=group_size,
        online_r1=False,
        runtime_backend=RUNTIME_BACKEND_QUARK_LIKE_DENSE,
    )
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
    layer.weight_packed.data = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    layer.weight_scale.data = torch.randint(120, 135, (N, K // group_size), dtype=torch.uint8)
    layer = layer.to(device)

    method.process_weights_after_loading(layer)
    assert layer.weight_packed is None
    assert layer.weight_scale is None
    assert "weight_packed" in layer._parameters and layer._parameters["weight_packed"] is None
    assert "weight_scale" in layer._parameters and layer._parameters["weight_scale"] is None
    assert hasattr(layer, "weight_dense_qdq")
    assert layer.weight_dense_qdq.shape == (N, K)
    assert layer.weight_dense_qdq.dtype == torch.bfloat16

    x = torch.randn(3, K, dtype=torch.bfloat16, device=device)
    x_qdq = torch.ops.auto_round.spinquant_mxfp4_act_qdq(x, group_size)
    expected = torch.nn.functional.linear(x_qdq, layer.weight_dense_qdq, None)
    output = method.apply(layer, x, bias=None)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    print(f"[PASS] quark_like_dense backend (device={device})")


def test_preunpack_fp8_backend():
    """Test load-time FP8 pre-unpack backend and forward path."""
    from auto_round.vllm_plugin.spinquant_mxfp4 import (
        SpinQuantMXFP4Config,
        SpinQuantMXFP4LinearMethod,
        RUNTIME_BACKEND_PREUNPACK_FP8,
        _dequant_preunpacked_mxfp4_weight,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 64, 64
    group_size = 32

    cfg = SpinQuantMXFP4Config(
        bits=4,
        group_size=group_size,
        online_r1=False,
        runtime_backend=RUNTIME_BACKEND_PREUNPACK_FP8,
    )
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
    layer.weight_packed.data = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    layer.weight_scale.data = torch.randint(120, 135, (N, K // group_size), dtype=torch.uint8)
    layer = layer.to(device)

    method.process_weights_after_loading(layer)
    assert layer.weight_packed is None
    assert layer.weight_scale is None
    assert "weight_packed" in layer._parameters and layer._parameters["weight_packed"] is None
    assert "weight_scale" in layer._parameters and layer._parameters["weight_scale"] is None
    assert hasattr(layer, "weight_unpacked_fp8")
    assert hasattr(layer, "weight_scale_bf16")
    assert layer.weight_unpacked_fp8.shape == (N, K)
    assert layer.weight_unpacked_fp8.dtype == torch.float8_e4m3fn

    x = torch.randn(3, K, dtype=torch.bfloat16, device=device)
    x_qdq = torch.ops.auto_round.spinquant_mxfp4_act_qdq(x, group_size)
    dense_weight = _dequant_preunpacked_mxfp4_weight(
        layer.weight_unpacked_fp8,
        layer.weight_scale_bf16,
        target_dtype=torch.bfloat16,
    )
    expected = torch.nn.functional.linear(x_qdq, dense_weight, None)
    output = method.apply(layer, x, bias=None)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    print(f"[PASS] preunpack_fp8 backend (device={device})")


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
    test_hadamard_runtime_matches_explicit_matrix()
    test_activation_qdq_matches_vllm_ext_reference()
    test_quark_like_dense_backend()
    test_preunpack_fp8_backend()
    test_override_detection()
    print("\n✅ All vLLM plugin tests passed!")
