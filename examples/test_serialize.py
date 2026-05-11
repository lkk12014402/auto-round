#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test SpinQuant rotation serialization (save → load → inference).

Verifies that:
1. inject_spinquant_buffers() correctly registers rotation buffers on QuantLinear
2. rebuild_spinquant_online() correctly patches QuantLinear.forward()
3. The rotation is correctly applied during inference after load
4. config.json contains spinquant_config for R3 reconstruction

Usage:
    python test_serialize.py
"""

import math
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auto_round.algorithms.transforms.spinquant.serialize import (
    ROTATION_TYPE_HADAMARD,
    ROTATION_TYPE_RANDOM,
    ROTATION_TYPE_TRAINED,
    _R1_PREFIX,
    _R4_PREFIX,
    _apply_rotation_from_buffer,
    _inject_rotation_buffers,
    _monkey_patch_forward,
    _has_spinquant_buffers,
    inject_spinquant_buffers,
    rebuild_spinquant_online,
    save_spinquant_config,
)
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    get_hadamard_K,
    matmul_hadU,
)


def test_inject_deterministic_hadamard():
    """Test buffer injection for deterministic Hadamard (type=0)."""
    print("=" * 60)
    print("Test 1: Inject deterministic Hadamard buffers")
    print("=" * 60)

    # Create a mock QuantLinear-like module
    class MockQuantLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.randn(out_f, in_f))

        def forward(self, x):
            return x @ self.weight.T

    module = MockQuantLinear(128, 256)

    # Inject R1 deterministic Hadamard buffers
    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=128,
        random=False, is_trained=False, rotation_matrix=None,
    )

    # Verify buffers
    assert hasattr(module, f"{_R1_PREFIX}_type"), "Missing type buffer"
    assert hasattr(module, f"{_R1_PREFIX}_size"), "Missing size buffer"
    assert not hasattr(module, f"{_R1_PREFIX}_matrix"), "Should NOT store matrix for deterministic"

    assert int(module.spinquant_r1_type) == ROTATION_TYPE_HADAMARD
    assert int(module.spinquant_r1_size) == 128

    # Verify it appears in state_dict
    sd = module.state_dict()
    assert "spinquant_r1_type" in sd
    assert "spinquant_r1_size" in sd

    print("  ✓ Deterministic Hadamard: type=0, size=128, no matrix stored")
    print("  ✓ Buffers in state_dict")
    print()


def test_inject_random_hadamard():
    """Test buffer injection for random Hadamard (type=1, int8 matrix)."""
    print("=" * 60)
    print("Test 2: Inject random Hadamard buffers")
    print("=" * 60)

    class MockQuantLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.randn(out_f, in_f))

        def forward(self, x):
            return x @ self.weight.T

    module = MockQuantLinear(64, 128)

    # Create a random Hadamard matrix (±1 values)
    had_K, K = get_hadamard_K(64)
    signs = torch.randint(0, 2, (64,)) * 2 - 1  # ±1
    random_matrix = had_K * signs.unsqueeze(0).float()  # H @ diag(signs)

    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=64,
        random=True, is_trained=False, rotation_matrix=random_matrix,
    )

    assert int(module.spinquant_r1_type) == ROTATION_TYPE_RANDOM
    assert int(module.spinquant_r1_size) == 64
    assert hasattr(module, f"{_R1_PREFIX}_matrix")
    assert module.spinquant_r1_matrix.dtype == torch.int8
    assert module.spinquant_r1_matrix.shape == (64, 64)

    # Verify all values are ±1
    unique_vals = module.spinquant_r1_matrix.unique()
    assert set(unique_vals.tolist()) <= {-1, 1}

    print("  ✓ Random Hadamard: type=1, size=64, int8 matrix stored")
    print(f"  ✓ Matrix shape: {module.spinquant_r1_matrix.shape}")
    print(f"  ✓ Unique values: {unique_vals.tolist()}")
    print()


def test_inject_trained_orthogonal():
    """Test buffer injection for trained orthogonal matrix (type=2, float32)."""
    print("=" * 60)
    print("Test 3: Inject trained orthogonal buffers")
    print("=" * 60)

    class MockQuantLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.randn(out_f, in_f))

        def forward(self, x):
            return x @ self.weight.T

    module = MockQuantLinear(32, 64)

    # Create a random orthogonal matrix (simulating trained)
    Q, _ = torch.linalg.qr(torch.randn(32, 32))

    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=32,
        random=False, is_trained=True, rotation_matrix=Q,
    )

    assert int(module.spinquant_r1_type) == ROTATION_TYPE_TRAINED
    assert int(module.spinquant_r1_size) == 32
    assert hasattr(module, f"{_R1_PREFIX}_matrix")
    assert module.spinquant_r1_matrix.dtype == torch.float32
    assert module.spinquant_r1_matrix.shape == (32, 32)

    # Verify orthogonality preserved
    R = module.spinquant_r1_matrix
    identity = R @ R.T
    assert torch.allclose(identity, torch.eye(32), atol=1e-5), "Orthogonality lost"

    print("  ✓ Trained orthogonal: type=2, size=32, float32 matrix stored")
    print(f"  ✓ Orthogonality preserved: ||RR^T - I|| = {(identity - torch.eye(32)).abs().max():.2e}")
    print()


def test_forward_patch_deterministic():
    """Test that patched forward correctly applies deterministic Hadamard."""
    print("=" * 60)
    print("Test 4: Forward patch with deterministic Hadamard")
    print("=" * 60)

    class MockQuantLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.eye(in_f)[:out_f])  # identity-like

        def forward(self, x):
            return x @ self.weight.T

    module = MockQuantLinear(128, 128)

    # Inject R1 buffer
    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=128,
        random=False, is_trained=False, rotation_matrix=None,
    )

    # Input
    x = torch.randn(2, 128)

    # Expected: matmul_hadU(x) then linear
    had_K, K = get_hadamard_K(128)
    x_rotated_expected = matmul_hadU(x, hadamard_K=had_K, K=K)
    y_expected = x_rotated_expected @ module.weight.T

    # Patch forward
    _monkey_patch_forward(type(module))

    # Actual
    y_actual = module(x)

    diff = (y_actual - y_expected).abs().max().item()
    assert diff < 1e-4, f"Forward mismatch: max diff = {diff}"

    print(f"  ✓ Patched forward matches manual Hadamard rotation")
    print(f"  ✓ Max difference: {diff:.2e}")
    print()


def test_forward_patch_trained():
    """Test that patched forward correctly applies trained orthogonal rotation."""
    print("=" * 60)
    print("Test 5: Forward patch with trained rotation")
    print("=" * 60)

    class MockQuantLinear2(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.eye(in_f)[:out_f])

        def forward(self, x):
            return x @ self.weight.T

    module = MockQuantLinear2(32, 32)

    # Create orthogonal matrix
    Q, _ = torch.linalg.qr(torch.randn(32, 32))

    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=32,
        random=False, is_trained=True, rotation_matrix=Q,
    )

    # Input
    x = torch.randn(2, 32)

    # Expected: x @ Q then linear
    x_rotated_expected = x @ Q.float()
    y_expected = x_rotated_expected @ module.weight.T

    # Patch forward
    _monkey_patch_forward(type(module))

    # Actual
    y_actual = module(x)

    diff = (y_actual - y_expected).abs().max().item()
    assert diff < 1e-4, f"Forward mismatch: max diff = {diff}"

    print(f"  ✓ Patched forward applies trained rotation correctly")
    print(f"  ✓ Max difference: {diff:.2e}")
    print()


def test_save_load_state_dict():
    """Test that spinquant buffers survive save/load via state_dict."""
    print("=" * 60)
    print("Test 6: Save/Load via state_dict")
    print("=" * 60)

    class MockQuantLinear3(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.randn(out_f, in_f))

        def forward(self, x):
            return x @ self.weight.T

    # Create and inject
    module_save = MockQuantLinear3(64, 64)
    _inject_rotation_buffers(
        module_save, _R1_PREFIX, rotation_size=64,
        random=False, is_trained=False, rotation_matrix=None,
    )
    _inject_rotation_buffers(
        module_save, _R4_PREFIX, rotation_size=64,
        random=False, is_trained=False, rotation_matrix=None,
    )

    # Save state_dict
    sd = module_save.state_dict()

    # Create fresh module and load
    module_load = MockQuantLinear3(64, 64)
    # Register buffers on target (simulating QuantLinear initialization)
    module_load.register_buffer("spinquant_r1_type", torch.tensor(0, dtype=torch.int32))
    module_load.register_buffer("spinquant_r1_size", torch.tensor(0, dtype=torch.int32))
    module_load.register_buffer("spinquant_r4_type", torch.tensor(0, dtype=torch.int32))
    module_load.register_buffer("spinquant_r4_size", torch.tensor(0, dtype=torch.int32))
    module_load.load_state_dict(sd)

    # Verify values survived
    assert int(module_load.spinquant_r1_type) == ROTATION_TYPE_HADAMARD
    assert int(module_load.spinquant_r1_size) == 64
    assert int(module_load.spinquant_r4_type) == ROTATION_TYPE_HADAMARD
    assert int(module_load.spinquant_r4_size) == 64

    print("  ✓ Buffers survive state_dict save/load")
    print(f"  ✓ R1: type={int(module_load.spinquant_r1_type)}, size={int(module_load.spinquant_r1_size)}")
    print(f"  ✓ R4: type={int(module_load.spinquant_r4_type)}, size={int(module_load.spinquant_r4_size)}")
    print()


def test_save_load_safetensors():
    """Test full save/load cycle using safetensors (simulated)."""
    print("=" * 60)
    print("Test 7: Save/Load via safetensors file")
    print("=" * 60)

    class MockQuantLinear4(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.bits = 4
            self.infeatures = in_f
            self.outfeatures = out_f
            self.register_buffer("weight", torch.randn(out_f, in_f))

        def forward(self, x):
            return x @ self.weight.T

    # Save module with rotation buffers
    module = MockQuantLinear4(128, 128)
    Q, _ = torch.linalg.qr(torch.randn(128, 128))
    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=128,
        random=False, is_trained=True, rotation_matrix=Q,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        path = os.path.join(tmpdir, "model.pt")
        torch.save(module.state_dict(), path)

        # Load into fresh module
        module_new = MockQuantLinear4(128, 128)
        # Pre-register buffers (in real usage, QuantLinear __init__ would do this
        # or we use strict=False)
        module_new.register_buffer("spinquant_r1_type", torch.tensor(0, dtype=torch.int32))
        module_new.register_buffer("spinquant_r1_size", torch.tensor(0, dtype=torch.int32))
        module_new.register_buffer("spinquant_r1_matrix", torch.zeros(128, 128, dtype=torch.float32))

        sd = torch.load(path, weights_only=True)
        module_new.load_state_dict(sd)

        # Verify matrix survived
        assert torch.allclose(module_new.spinquant_r1_matrix, Q.float(), atol=1e-6)

    print("  ✓ Full matrix survives save/load via file")
    print(f"  ✓ Matrix diff: {(module_new.spinquant_r1_matrix - Q.float()).abs().max():.2e}")
    print()


def test_config_json_save():
    """Test saving spinquant_config to config.json."""
    print("=" * 60)
    print("Test 8: Save spinquant_config to config.json")
    print("=" * 60)

    import json

    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

    config = SpinQuantConfig(
        r1=True, r2=True, r3=True, r4=True,
        online_r1_rotation=True,
        rotation_size=64,
        random_r1=False,
    )

    # Create a mock model with config
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
            })()

    model = MockModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a config.json
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"model_type": "test", "quantization_config": {"bits": 4}}, f)

        # Save spinquant config
        save_spinquant_config(model, tmpdir, config)

        # Read back
        with open(config_path, "r") as f:
            saved = json.load(f)

        sq_cfg = saved["quantization_config"]["spinquant_config"]
        assert sq_cfg["r1"] is True
        assert sq_cfg["r2"] is True
        assert sq_cfg["r3"] is True
        assert sq_cfg["r4"] is True
        assert sq_cfg["online_r1_rotation"] is True
        assert sq_cfg["rotation_size"] == 64
        assert sq_cfg["head_dim"] == 64  # 768 / 12
        assert sq_cfg["hidden_size"] == 768

    print("  ✓ spinquant_config saved to config.json")
    print(f"  ✓ Fields: r1={sq_cfg['r1']}, r3={sq_cfg['r3']}, head_dim={sq_cfg['head_dim']}")
    print()


def test_end_to_end_rotation_equivalence():
    """Test that rotation via buffer matches direct rotation."""
    print("=" * 60)
    print("Test 9: End-to-end rotation equivalence")
    print("=" * 60)

    # Simulate: during preprocessing, online R1 rotates weight and adds hook
    # After quantization, we inject buffer. At inference, buffer-based rotation
    # should produce the same result as hook-based rotation.

    dim = 128
    x = torch.randn(4, dim)

    # Simulate the rotated weight (W_new = matmul_hadU(W) along last dim)
    W_orig = torch.randn(256, dim)
    had_K, K = get_hadamard_K(dim)
    W_rotated = matmul_hadU(W_orig, hadamard_K=had_K, K=K)

    # Hook-based approach (preprocessing): y = matmul_hadU(x) @ W_rotated.T
    x_rotated = matmul_hadU(x, hadamard_K=had_K, K=K)
    y_hook = x_rotated @ W_rotated.T

    # Buffer-based approach (inference after load):
    class MockQL(nn.Module):
        def __init__(self):
            super().__init__()
            self.bits = 4
            self.infeatures = dim
            self.outfeatures = 256
            self.register_buffer("weight", W_rotated)

        def forward(self, x):
            return x @ self.weight.T

    module = MockQL()
    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=dim,
        random=False, is_trained=False, rotation_matrix=None,
    )
    _monkey_patch_forward(type(module))
    y_buffer = module(x)

    diff = (y_hook - y_buffer).abs().max().item()
    assert diff < 1e-4, f"Hook vs buffer mismatch: {diff}"

    # Also verify against original: y = x @ W_orig.T (no rotation)
    y_orig = x @ W_orig.T
    diff_orig = (y_hook - y_orig).abs().max().item()
    assert diff_orig < 1e-4, f"Rotation should be transparent: {diff_orig}"

    print(f"  ✓ Hook-based == Buffer-based: max diff = {diff:.2e}")
    print(f"  ✓ Rotated result == Original result: max diff = {diff_orig:.2e}")
    print(f"    (confirms rotation is mathematically transparent)")
    print()


def test_block_rotation_serialize():
    """Test serialization with block-wise rotation (rotation_size < in_features)."""
    print("=" * 60)
    print("Test 10: Block rotation serialization")
    print("=" * 60)

    dim = 256
    rot_size = 64  # Block rotation: 256/64 = 4 blocks

    class MockQL2(nn.Module):
        def __init__(self):
            super().__init__()
            self.bits = 4
            self.infeatures = dim
            self.outfeatures = dim
            self.register_buffer("weight", torch.eye(dim))

        def forward(self, x):
            return x @ self.weight.T

    module = MockQL2()

    # Inject block rotation
    _inject_rotation_buffers(
        module, _R1_PREFIX, rotation_size=rot_size,
        random=False, is_trained=False, rotation_matrix=None,
    )

    # Patch and test
    _monkey_patch_forward(type(module))

    x = torch.randn(2, dim)
    y = module(x)

    # Manual block rotation
    had_K, K = get_hadamard_K(rot_size)
    R = had_K.to(torch.float64)
    if R.shape[0] != rot_size:
        had_1, _ = get_hadamard_K(rot_size // K)
        R = torch.kron(had_K.to(torch.float64), had_1.to(torch.float64))
    R = (R / math.sqrt(rot_size)).float()

    x_block = x.reshape(2, -1, rot_size)
    x_rotated = (x_block @ R).reshape(2, dim)
    y_expected = x_rotated @ module.weight.T

    diff = (y - y_expected).abs().max().item()
    assert diff < 1e-4, f"Block rotation mismatch: {diff}"

    print(f"  ✓ Block rotation (size={rot_size}, {dim//rot_size} blocks) works correctly")
    print(f"  ✓ Max difference: {diff:.2e}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SpinQuant Serialization Tests")
    print("=" * 60 + "\n")

    test_inject_deterministic_hadamard()
    test_inject_random_hadamard()
    test_inject_trained_orthogonal()
    test_forward_patch_deterministic()
    test_forward_patch_trained()
    test_save_load_state_dict()
    test_save_load_safetensors()
    test_config_json_save()
    test_end_to_end_rotation_equivalence()
    test_block_rotation_serialize()

    print("=" * 60)
    print("  ALL 10 TESTS PASSED ✓")
    print("=" * 60)
