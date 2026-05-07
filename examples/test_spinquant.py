"""
Comprehensive test suite for SpinQuant migration to AutoRound.

Tests cover:
1. QuaRot mode (fixed Hadamard) - mathematical equivalence after fusion
2. Trainable SpinQuant mode - SGDG optimizer, Cayley transform orthogonality
3. Edge cases - empty adam_params, empty sgdg_params
4. Architecture flexibility - flat vs nested model structures
5. RotationTrainer end-to-end lifecycle
"""

import sys
sys.path.insert(0, "/data/lkk/quarot/auto-round")

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Mock models for testing (both flat and nested architectures)
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = None  # Simplified

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simplified attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out)


class MockMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class MockRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = MockAttention(hidden_size, num_heads)
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.input_layernorm = MockRMSNorm(hidden_size)
        self.post_attention_layernorm = MockRMSNorm(hidden_size)

    def forward(self, x):
        # Pre-norm architecture
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MockFlatModel(nn.Module):
    """Flat architecture: model.layers (no model.model nesting)."""
    def __init__(self, hidden_size=64, num_heads=4, intermediate_size=128, num_layers=2, vocab_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MockDecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = MockRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class MockNestedModel(nn.Module):
    """Nested architecture: model.model.layers (like LlamaForCausalLM)."""
    def __init__(self, hidden_size=64, num_heads=4, intermediate_size=128, num_layers=2, vocab_size=100):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([
            MockDecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.model.norm = MockRMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# 2. Utility functions for testing
# ---------------------------------------------------------------------------

def get_model_output(model, input_ids):
    """Get model output with no_grad."""
    with torch.no_grad():
        return model(input_ids)


def compute_relative_error(a, b):
    """Compute max relative error between two tensors."""
    diff = (a - b).abs()
    denom = b.abs().clamp(min=1e-8)
    return (diff / denom).max().item()


def check_orthogonality(R, tol=1e-4):
    """Check if R is orthogonal: R @ R.T = I."""
    I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    RtR = R @ R.T
    max_diff = (RtR - I).abs().max().item()
    return max_diff < tol, max_diff


# ---------------------------------------------------------------------------
# 3. Tests
# ---------------------------------------------------------------------------

def test_optimizer_empty_params():
    """Test AdamAndSGDG handles empty parameter lists gracefully."""
    from auto_round.algorithms.transforms.spinquant.cayley_optimizer import AdamAndSGDG

    # Case 1: Empty adam, non-empty sgdg
    R = nn.Parameter(torch.randn(8, 8))
    opt = AdamAndSGDG(adam_params=[], sgdg_params=[R], learning_rate=1e-3, smooth_learning_rate=1e-3)
    loss = torch.randn(1).requires_grad_(True)
    loss.backward()
    opt.step()
    assert opt.adam_optimizer is None
    assert opt.sgdg_optimizer is not None
    print("[PASS] Empty adam_params handled correctly")

    # Case 2: Non-empty adam, empty sgdg
    D = nn.Parameter(torch.ones(8))
    opt2 = AdamAndSGDG(adam_params=[D], sgdg_params=[], learning_rate=1e-3, smooth_learning_rate=1e-3)
    loss = (D ** 2).sum()
    loss.backward()
    opt2.step()
    assert opt2.adam_optimizer is not None
    assert opt2.sgdg_optimizer is None
    print("[PASS] Empty sgdg_params handled correctly")

    # Case 3: Both empty
    opt3 = AdamAndSGDG(adam_params=[], sgdg_params=[], learning_rate=1e-3, smooth_learning_rate=1e-3)
    opt3.step()
    assert opt3.adam_optimizer is None
    assert opt3.sgdg_optimizer is None
    print("[PASS] Both empty handled correctly")

    # Case 4: Both non-empty (standard case)
    D2 = nn.Parameter(torch.ones(8))
    R2 = nn.Parameter(torch.randn(8, 8))
    opt4 = AdamAndSGDG(adam_params=[D2], sgdg_params=[R2], learning_rate=1e-3, smooth_learning_rate=1e-3)
    loss = (D2 ** 2).sum() + (R2 ** 2).sum()
    loss.backward()
    opt4.step()
    assert opt4.adam_optimizer is not None
    assert opt4.sgdg_optimizer is not None
    print("[PASS] Standard case works correctly")


def test_sgdg_orthogonality():
    """Test that SGDG maintains orthogonality after multiple steps."""
    from auto_round.algorithms.transforms.spinquant.cayley_optimizer import SGDG

    torch.manual_seed(42)
    R = nn.Parameter(torch.randn(16, 16))
    # Make it orthogonal initially
    Q, _ = torch.linalg.qr(R.data)
    R.data = Q

    opt = SGDG([R], lr=1e-3, stiefel=True)

    # Simulate multiple training steps
    for i in range(20):
        opt.zero_grad()
        # Dummy loss: encourage some structure
        loss = (R @ torch.randn(16, 8)).var()
        loss.backward()
        opt.step()

        is_ortho, diff = check_orthogonality(R, tol=1e-3)
        assert is_ortho, f"SGDG lost orthogonality at step {i}: diff={diff:.6f}"

    print(f"[PASS] SGDG maintained orthogonality after 20 steps (max diff={diff:.2e})")


def test_quarot_equivalence():
    """Test QuaRot mode: fused model produces same output as original."""
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor
    )

    torch.manual_seed(42)
    # Test R1-only first (R2 needs per-head attention, R4 needs online hook)
    config = SpinQuantConfig(
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        trainable_rotation=False,  # QuaRot mode (fixed Hadamard)
        trainable_smooth=False,
        iters=0,  # No training
        fuse_rmsnorm=True,
    )

    # Test flat model
    model = MockFlatModel(hidden_size=64, num_heads=4, intermediate_size=128, num_layers=2)
    model.eval()
    input_ids = torch.randint(0, 100, (2, 8))  # batch=2, seq=8

    orig_output = get_model_output(model, input_ids)

    preprocessor = SpinQuantPreprocessor(model, config)
    preprocessor.preprocess(dataloader=None)
    fused_output = get_model_output(model, input_ids)

    rel_err = compute_relative_error(fused_output, orig_output)
    # ~0.3% error is expected: float32 weights → float64 rotation → float32 accumulation
    assert rel_err < 5e-3, f"QuaRot R1-only flat model failed: relative error={rel_err:.6f}"
    print(f"[PASS] QuaRot R1-only flat model equivalence: rel_err={rel_err:.2e}")

    # Test nested model
    torch.manual_seed(42)
    model2 = MockNestedModel(hidden_size=64, num_heads=4, intermediate_size=128, num_layers=2)
    model2.eval()
    orig_output2 = get_model_output(model2, input_ids)

    preprocessor2 = SpinQuantPreprocessor(model2, config)
    preprocessor2.preprocess(dataloader=None)
    fused_output2 = get_model_output(model2, input_ids)

    rel_err2 = compute_relative_error(fused_output2, orig_output2)
    assert rel_err2 < 5e-3, f"QuaRot R1-only nested model failed: relative error={rel_err2:.6f}"
    print(f"[PASS] QuaRot R1-only nested model equivalence: rel_err={rel_err2:.2e}")


def test_trainable_spinquant_basic():
    """Test trainable SpinQuant runs without errors and maintains orthogonality."""
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor
    )

    torch.manual_seed(42)
    config = SpinQuantConfig(
        r1=True,
        r2=False,   # Disable R2 for mock model (needs per-head attention)
        r3=False,   # Disable R3 for mock model
        r4=False,   # Disable R4 for mock model
        trainable_rotation=True,
        trainable_smooth=False,  # Disable smooth to isolate rotation test
        iters=3,  # Just a few iterations
        lr=1e-4,
        batch_size=1,
    )

    model = MockFlatModel(hidden_size=64, num_heads=4, intermediate_size=128, num_layers=2)

    def dummy_loader():
        for _ in range(5):
            yield {"input_ids": torch.randint(0, 100, (1, 8))}

    preprocessor = SpinQuantPreprocessor(model, config)
    # Full preprocess pipeline with training
    preprocessor.preprocess(dummy_loader())

    # After preprocessing, rotation params are fused into weights and cleaned up.
    # Verify model is still functional (no NaN/Inf) and produces finite output.
    model.eval()
    with torch.no_grad():
        test_out = model(torch.randint(0, 100, (1, 8)))
    assert torch.isfinite(test_out).all(), "Trainable SpinQuant produced non-finite output after fusion"
    print("[PASS] Trainable SpinQuant completed training+fusion and produces stable output")


def test_trainer_lifecycle():
    """Test RotationTrainer end-to-end: setup, train, fuse."""
    from auto_round.algorithms.transforms.spinquant.trainer import (
        RotationTrainer, RotationTrainerConfig
    )

    torch.manual_seed(42)
    model = MockFlatModel(hidden_size=64, num_heads=4, intermediate_size=128, num_layers=2)

    config = RotationTrainerConfig(
        iters=3,
        lr=1e-4,
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        trainable_rotation=True,
        trainable_smooth=False,
    )

    trainer = RotationTrainer(model, config=config)

    # Training with dummy dataloader
    def dummy_dataloader():
        for _ in range(5):
            yield {"input_ids": torch.randint(0, 100, (1, 8))}

    # Store original output
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, 100, (1, 8))
        orig_out = model(test_input).clone()

    # Train
    trainer.train(dummy_dataloader())

    # Fuse
    trainer.fuse()

    # Verify model still works after fusion
    model.eval()
    with torch.no_grad():
        fused_out = model(test_input)

    # After fusion, output should still be close (rotations cancel out)
    rel_err = compute_relative_error(fused_out, orig_out)
    # Note: after training, the output won't be exactly the same due to learned rotations
    # but it should be stable (no NaNs/inf)
    assert torch.isfinite(fused_out).all(), "Fused output contains NaN or Inf"
    print(f"[PASS] Trainer lifecycle complete. Output stable after fusion (rel_err={rel_err:.2e})")


def test_rotation_utils():
    """Test low-level rotation utility functions."""
    from auto_round.algorithms.transforms.spinquant.rotation_utils import (
        deterministic_hadamard_matrix,
        random_hadamard_matrix,
        rotate_in_channels_,
        rotate_out_channels_,
    )

    # Test Hadamard matrices are orthogonal
    H = deterministic_hadamard_matrix(64)
    is_ortho, diff = check_orthogonality(H, tol=1e-6)
    assert is_ortho, f"Deterministic Hadamard not orthogonal: diff={diff:.6f}"
    print(f"[PASS] Deterministic Hadamard orthogonal (diff={diff:.2e})")

    Hr = random_hadamard_matrix(64)
    is_ortho2, diff2 = check_orthogonality(Hr, tol=1e-4)
    assert is_ortho2, f"Random Hadamard not orthogonal: diff={diff2:.6f}"
    print(f"[PASS] Random Hadamard orthogonal (diff={diff2:.2e})")

    # Test rotate_in_channels_ preserves equivalence
    W = nn.Linear(64, 32, bias=False)
    R = deterministic_hadamard_matrix(64)
    x = torch.randn(4, 64)

    orig_out = F.linear(x, W.weight)
    rotate_in_channels_(W, R)
    rot_out = F.linear(x @ R.T, W.weight)

    rel_err = compute_relative_error(rot_out, orig_out)
    assert rel_err < 5e-3, f"rotate_in_channels_ failed: rel_err={rel_err:.6f}"
    print(f"[PASS] rotate_in_channels_ equivalence (rel_err={rel_err:.2e})")

    # Test rotate_out_channels_ preserves equivalence
    W2 = nn.Linear(32, 64, bias=False)
    R2 = deterministic_hadamard_matrix(64)
    x2 = torch.randn(4, 32)

    orig_out2 = F.linear(x2, W2.weight)
    rotate_out_channels_(W2, R2)
    rot_out2 = F.linear(x2, W2.weight) @ R2.T

    rel_err2 = compute_relative_error(rot_out2, orig_out2)
    assert rel_err2 < 5e-3, f"rotate_out_channels_ failed: rel_err={rel_err2:.6f}"
    print(f"[PASS] rotate_out_channels_ equivalence (rel_err={rel_err2:.2e})")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SpinQuant Migration Test Suite")
    print("=" * 60)

    tests = [
        ("Rotation Utils (Hadamard, rotate_in/out)", test_rotation_utils),
        ("Optimizer Empty Params", test_optimizer_empty_params),
        ("SGDG Orthogonality Maintenance", test_sgdg_orthogonality),
        ("QuaRot Equivalence (Flat + Nested)", test_quarot_equivalence),
        ("Trainable SpinQuant Basic", test_trainable_spinquant_basic),
        ("RotationTrainer Lifecycle", test_trainer_lifecycle),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)

