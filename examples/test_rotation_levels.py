"""
Comprehensive validation of SpinQuant rotation levels: R1, R1+R2, R1+R2+R3, R1+R2+R3+R4.

This test validates that the migrated implementation produces mathematically
equivalent outputs to the reference implementations in /data/lkk/quarot/.

Key differences from test_spinquant.py:
- Uses proper multi-head attention (required for R2 to cancel)
- Tests each rotation level incrementally
- Validates against the reference implementation's math
"""

import sys
sys.path.insert(0, "/data/lkk/quarot/auto-round")

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-head aware mock model (required for R2/R3 correctness)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Proper multi-head attention with per-head computation."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to heads: [batch, num_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # Reshape back: [batch, seq_len, hidden_size]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP matching Llama/Qwen architecture."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x):
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MockTransformerModel(nn.Module):
    """Mock model with proper multi-head attention (model.model.layers structure)."""

    def __init__(self, hidden_size=128, num_heads=4, intermediate_size=256,
                 num_layers=2, vocab_size=200):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.model.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Provide a config object for get_model_arch_info compatibility
        class _Config:
            pass
        self.config = _Config()
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_heads
        self.config.num_key_value_heads = num_heads
        self.config.head_dim = hidden_size // num_heads
        self.config.intermediate_size = intermediate_size
        self.config.model_type = "mock_transformer"

    def forward(self, input_ids=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# Reference implementation functions (from /data/lkk/quarot/)
# ---------------------------------------------------------------------------

def merge_rmsnorm_in_model_ref(model):
    """Reference RMSNorm merge (from inference_only_r1.py)."""
    for layer in model.model.layers:
        gamma_attn = layer.input_layernorm.weight.data
        layer.self_attn.q_proj.weight.data *= gamma_attn
        layer.self_attn.k_proj.weight.data *= gamma_attn
        layer.self_attn.v_proj.weight.data *= gamma_attn
        layer.input_layernorm.weight.data.fill_(1.0)

        gamma_mlp = layer.post_attention_layernorm.weight.data
        layer.mlp.gate_proj.weight.data *= gamma_mlp
        layer.mlp.up_proj.weight.data *= gamma_mlp
        layer.post_attention_layernorm.weight.data.fill_(1.0)

    model_norm_weight = model.model.norm.weight.data
    model.lm_head.weight.data = model.lm_head.weight.data * model_norm_weight
    model.model.norm.weight.data.fill_(1.0)


def fuse_layer_rotation_ref(layer, R_in=None, R_out=None):
    """Reference rotation fusion (from inference_only_r1.py)."""
    device, dtype = layer.weight.device, layer.weight.dtype
    W = layer.weight.data.to(torch.float32)
    W = W.T
    with torch.no_grad():
        if R_in is not None:
            R_in = R_in.to(W.device, torch.float32)
            W = R_in @ W
        if R_out is not None:
            R_out = R_out.to(W.device, torch.float32)
            W = W @ R_out
    W = W.T
    layer.weight.data = W.to(device=device, dtype=dtype)


def apply_had_to_linear_ref(module, had_dim=-1, output=False):
    """Reference Hadamard application to linear layer.

    CPU-compatible version of hadamard_utils2.apply_exact_had_to_linear.
    Uses deterministic Hadamard matrix multiplication.
    """
    in_features = module.in_features
    out_features = module.out_features
    W = module.weight.data.to(torch.float64)

    from auto_round.algorithms.transforms.spinquant.rotation_utils import (
        deterministic_hadamard_matrix,
    )

    if had_dim == -1:
        dim = out_features if output else in_features
        H = deterministic_hadamard_matrix(dim, dtype=torch.float64, device=W.device)
        if output:
            W = H @ W
        else:
            W = W @ H
    else:
        H = deterministic_hadamard_matrix(had_dim, dtype=torch.float64, device=W.device)
        if output:
            n_chunks = out_features // had_dim
            W_r = W.reshape(n_chunks, had_dim, in_features)
            W_r = torch.einsum('ij,kjl->kil', H, W_r)
            W = W_r.reshape(out_features, in_features)
        else:
            n_chunks = in_features // had_dim
            W_r = W.reshape(out_features, n_chunks, had_dim)
            W_r = torch.einsum('ijk,lk->ijl', W_r, H)
            W = W_r.reshape(out_features, in_features)

    module.weight.data = W.to(module.weight.dtype)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_output(model, input_ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids)


def compute_relative_error(a, b):
    diff = (a - b).abs()
    denom = b.abs().clamp(min=1e-8)
    return (diff / denom).max().item()


def compute_max_abs_error(a, b):
    return (a - b).abs().max().item()


def random_hadamard_matrix_ref(size, device="cpu"):
    """Reference random Hadamard from hadamard_utils2.py (CPU compatible)."""
    from auto_round.algorithms.transforms.spinquant.rotation_utils import (
        deterministic_hadamard_matrix,
    )
    H = deterministic_hadamard_matrix(size, dtype=torch.float64, device=device)
    D = torch.randint(0, 2, (size,), dtype=torch.float64, device=device) * 2 - 1
    return (H * D.unsqueeze(1)).to(torch.float32)


# ---------------------------------------------------------------------------
# Test: R1 only
# ---------------------------------------------------------------------------

def test_r1_only():
    """Validate R1 rotation: fused model ≈ original model output."""
    print("\n" + "=" * 60)
    print("TEST: R1 Only (hidden_size rotation)")
    print("=" * 60)

    torch.manual_seed(42)
    model = MockTransformerModel(hidden_size=128, num_heads=4, intermediate_size=256)
    model.eval()
    input_ids = torch.randint(0, 200, (2, 16))
    orig_output = get_model_output(model, input_ids)

    # --- Reference implementation ---
    torch.manual_seed(123)
    model_ref = copy.deepcopy(model)
    merge_rmsnorm_in_model_ref(model_ref)

    hidden_size = 128
    R1 = random_hadamard_matrix_ref(hidden_size)
    R1_inv = R1.T

    with torch.no_grad():
        W = model_ref.model.embed_tokens.weight.data.to(torch.float32)
        model_ref.model.embed_tokens.weight.data = (W @ R1).to(W.dtype)

    for layer in model_ref.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        fuse_layer_rotation_ref(attn.q_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(attn.k_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(attn.v_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(attn.o_proj, R_out=R1)
        fuse_layer_rotation_ref(mlp.gate_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(mlp.up_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(mlp.down_proj, R_out=R1)

    fuse_layer_rotation_ref(model_ref.lm_head, R_in=R1_inv)
    ref_output = get_model_output(model_ref, input_ids)

    # --- Our implementation ---
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor,
    )

    torch.manual_seed(42)
    model_ours = MockTransformerModel(hidden_size=128, num_heads=4, intermediate_size=256)
    model_ours.eval()
    config = SpinQuantConfig(
        r1=True, r2=False, r3=False, r4=False,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=True,
    )
    preprocessor = SpinQuantPreprocessor(model_ours, config)
    preprocessor.preprocess(dataloader=None)
    ours_output = get_model_output(model_ours, input_ids)

    # Validate reference equals original (should be near-zero error)
    ref_vs_orig = compute_relative_error(ref_output, orig_output)
    ours_vs_orig = compute_relative_error(ours_output, orig_output)

    print(f"  Reference vs Original:  rel_err = {ref_vs_orig:.2e}")
    print(f"  Ours vs Original:       rel_err = {ours_vs_orig:.2e}")

    assert ref_vs_orig < 5e-3, f"Reference R1 failed: {ref_vs_orig}"
    assert ours_vs_orig < 5e-3, f"Our R1 failed: {ours_vs_orig}"
    print("  [PASS] R1-only rotation preserves model equivalence")
    return True


# ---------------------------------------------------------------------------
# Test: R1 + R2
# ---------------------------------------------------------------------------

def test_r1_r2():
    """Validate R1+R2: per-head rotation on v_proj/o_proj."""
    print("\n" + "=" * 60)
    print("TEST: R1 + R2 (hidden_size + per-head rotation)")
    print("=" * 60)

    torch.manual_seed(42)
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads  # 32

    model = MockTransformerModel(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=256)
    model.eval()
    input_ids = torch.randint(0, 200, (2, 16))
    orig_output = get_model_output(model, input_ids)

    # --- Reference implementation (from inference_only_r1_r2.py) ---
    torch.manual_seed(123)
    model_ref = copy.deepcopy(model)
    merge_rmsnorm_in_model_ref(model_ref)

    R1 = random_hadamard_matrix_ref(hidden_size)
    R1_inv = R1.T

    with torch.no_grad():
        W = model_ref.model.embed_tokens.weight.data.to(torch.float32)
        model_ref.model.embed_tokens.weight.data = (W @ R1).to(W.dtype)

    for layer in model_ref.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        fuse_layer_rotation_ref(attn.q_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(attn.k_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(attn.v_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(attn.o_proj, R_out=R1)

        # R2: per-head Hadamard on v_proj (output) and o_proj (input)
        apply_had_to_linear_ref(attn.v_proj, had_dim=head_dim, output=True)
        apply_had_to_linear_ref(attn.o_proj, had_dim=head_dim, output=False)

        fuse_layer_rotation_ref(mlp.gate_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(mlp.up_proj, R_in=R1_inv)
        fuse_layer_rotation_ref(mlp.down_proj, R_out=R1)

    fuse_layer_rotation_ref(model_ref.lm_head, R_in=R1_inv)
    ref_output = get_model_output(model_ref, input_ids)

    # --- Our implementation ---
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor,
    )

    torch.manual_seed(42)
    model_ours = MockTransformerModel(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=256)
    model_ours.eval()
    config = SpinQuantConfig(
        r1=True, r2=True, r3=False, r4=False,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=True,
    )
    preprocessor = SpinQuantPreprocessor(model_ours, config)
    preprocessor.preprocess(dataloader=None)
    ours_output = get_model_output(model_ours, input_ids)

    ref_vs_orig = compute_relative_error(ref_output, orig_output)
    ours_vs_orig = compute_relative_error(ours_output, orig_output)

    print(f"  Reference vs Original:  rel_err = {ref_vs_orig:.2e}")
    print(f"  Ours vs Original:       rel_err = {ours_vs_orig:.2e}")

    # R1+R2 with per-head attention should cancel perfectly
    assert ref_vs_orig < 5e-3, f"Reference R1+R2 failed: {ref_vs_orig}"
    assert ours_vs_orig < 5e-3, f"Our R1+R2 failed: {ours_vs_orig}"
    print("  [PASS] R1+R2 rotation preserves model equivalence")
    return True


# ---------------------------------------------------------------------------
# Test: R1 + R2 + R3
# ---------------------------------------------------------------------------

def test_r1_r2_r3():
    """Validate R1+R2+R3: R3 is online Q/K rotation (cancels in attention scores)."""
    print("\n" + "=" * 60)
    print("TEST: R1 + R2 + R3 (+ online Q/K rotation)")
    print("=" * 60)

    torch.manual_seed(42)
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads

    model = MockTransformerModel(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=256)
    model.eval()
    input_ids = torch.randint(0, 200, (2, 16))
    orig_output = get_model_output(model, input_ids)

    # --- Reference: R1+R2 offline + R3 online (cancels in Q@K.T) ---
    # R3 applies Hadamard to Q and K outputs (per-head).
    # Since attention score = Q @ K.T, and both Q and K get same rotation:
    # (Q @ H) @ (K @ H).T = Q @ H @ H.T @ K.T = Q @ I @ K.T = Q @ K.T
    # So R3 has NO effect on the model output. It only helps quantization.
    # Therefore R1+R2+R3 output == R1+R2 output.

    # Our implementation with R3 enabled
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor,
    )

    torch.manual_seed(42)
    model_ours = MockTransformerModel(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=256)
    model_ours.eval()
    config = SpinQuantConfig(
        r1=True, r2=True, r3=True, r4=False,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=True,
    )
    preprocessor = SpinQuantPreprocessor(model_ours, config)
    preprocessor.preprocess(dataloader=None)
    ours_output = get_model_output(model_ours, input_ids)

    ours_vs_orig = compute_relative_error(ours_output, orig_output)
    print(f"  Ours (R1+R2+R3) vs Original: rel_err = {ours_vs_orig:.2e}")

    # R3 cancels in attention score, so output should still match original
    assert ours_vs_orig < 5e-3, f"Our R1+R2+R3 failed: {ours_vs_orig}"
    print("  [PASS] R1+R2+R3 rotation preserves model equivalence (R3 cancels in Q@K.T)")
    return True


# ---------------------------------------------------------------------------
# Test: R1 + R2 + R3 + R4
# ---------------------------------------------------------------------------

def test_r1_r2_r3_r4():
    """Validate R1+R2+R3+R4: R4 offline on down_proj + online on activation."""
    print("\n" + "=" * 60)
    print("TEST: R1 + R2 + R3 + R4 (full rotation)")
    print("=" * 60)

    torch.manual_seed(42)
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    intermediate_size = 256

    model = MockTransformerModel(
        hidden_size=hidden_size, num_heads=num_heads,
        intermediate_size=intermediate_size
    )
    model.eval()
    input_ids = torch.randint(0, 200, (2, 16))
    orig_output = get_model_output(model, input_ids)

    # --- Our implementation with R4 ---
    # R4 = offline Hadamard on down_proj input + online Hadamard on activation.
    # Combined: (x @ H) @ (W @ H).T = x @ H @ H @ W.T = x @ I @ W.T = x @ W.T
    # (because H is symmetric and H @ H = I for normalized Hadamard)

    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor,
    )

    torch.manual_seed(42)
    model_ours = MockTransformerModel(
        hidden_size=hidden_size, num_heads=num_heads,
        intermediate_size=intermediate_size
    )
    model_ours.eval()
    config = SpinQuantConfig(
        r1=True, r2=True, r3=True, r4=True,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=True,
    )
    preprocessor = SpinQuantPreprocessor(model_ours, config)
    preprocessor.preprocess(dataloader=None)
    ours_output = get_model_output(model_ours, input_ids)

    ours_vs_orig = compute_relative_error(ours_output, orig_output)
    print(f"  Ours (R1+R2+R3+R4) vs Original: rel_err = {ours_vs_orig:.2e}")

    # R4 offline + online should cancel perfectly
    assert ours_vs_orig < 5e-3, f"Our R1+R2+R3+R4 failed: {ours_vs_orig}"
    print("  [PASS] R1+R2+R3+R4 full rotation preserves model equivalence")
    return True


# ---------------------------------------------------------------------------
# Test: R2-only isolation (verify per-head Hadamard cancellation)
# ---------------------------------------------------------------------------

def test_r2_isolation():
    """Verify R2 per-head Hadamard cancels between v_proj and o_proj."""
    print("\n" + "=" * 60)
    print("TEST: R2 Isolation (per-head Hadamard cancellation)")
    print("=" * 60)

    torch.manual_seed(42)
    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads

    model = MockTransformerModel(hidden_size=hidden_size, num_heads=num_heads, intermediate_size=256)
    model.eval()
    input_ids = torch.randint(0, 200, (2, 16))
    orig_output = get_model_output(model, input_ids)

    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor,
    )

    model_r2 = copy.deepcopy(model)
    config = SpinQuantConfig(
        r1=False, r2=True, r3=False, r4=False,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=False,  # Don't fuse norm, just test R2
    )
    preprocessor = SpinQuantPreprocessor(model_r2, config)
    preprocessor.preprocess(dataloader=None)
    r2_output = get_model_output(model_r2, input_ids)

    rel_err = compute_relative_error(r2_output, orig_output)
    print(f"  R2-only vs Original: rel_err = {rel_err:.2e}")

    assert rel_err < 5e-4, f"R2 isolation test failed: {rel_err}"
    print("  [PASS] R2 per-head Hadamard cancels correctly in multi-head attention")
    return True


# ---------------------------------------------------------------------------
# Test: R4-only isolation (verify online+offline Hadamard cancellation)
# ---------------------------------------------------------------------------

def test_r4_isolation():
    """Verify R4 offline+online Hadamard cancels on MLP."""
    print("\n" + "=" * 60)
    print("TEST: R4 Isolation (MLP Hadamard cancellation)")
    print("=" * 60)

    torch.manual_seed(42)
    hidden_size = 128
    num_heads = 4
    intermediate_size = 256

    model = MockTransformerModel(
        hidden_size=hidden_size, num_heads=num_heads,
        intermediate_size=intermediate_size
    )
    model.eval()
    input_ids = torch.randint(0, 200, (2, 16))
    orig_output = get_model_output(model, input_ids)

    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig, SpinQuantPreprocessor,
    )

    model_r4 = copy.deepcopy(model)
    config = SpinQuantConfig(
        r1=False, r2=False, r3=False, r4=True,
        trainable_rotation=False,
        trainable_smooth=False,
        fuse_rmsnorm=False,
    )
    preprocessor = SpinQuantPreprocessor(model_r4, config)
    preprocessor.preprocess(dataloader=None)
    r4_output = get_model_output(model_r4, input_ids)

    rel_err = compute_relative_error(r4_output, orig_output)
    print(f"  R4-only vs Original: rel_err = {rel_err:.2e}")

    assert rel_err < 5e-4, f"R4 isolation test failed: {rel_err}"
    print("  [PASS] R4 offline+online Hadamard cancels correctly on MLP")
    return True


# ---------------------------------------------------------------------------
# Test: Hadamard involution property (H @ H = I)
# ---------------------------------------------------------------------------

def test_hadamard_involution():
    """Verify the key property: H @ H = I for our Hadamard implementation."""
    print("\n" + "=" * 60)
    print("TEST: Hadamard Involution (H @ H = I)")
    print("=" * 60)

    from auto_round.algorithms.transforms.spinquant.rotation_utils import (
        deterministic_hadamard_matrix,
    )

    for size in [32, 64, 128, 256]:
        H = deterministic_hadamard_matrix(size, dtype=torch.float64)
        I = torch.eye(size, dtype=torch.float64)
        product = H @ H
        err = (product - I).abs().max().item()
        assert err < 1e-10, f"H@H != I for size {size}: max err = {err}"
        print(f"  size={size}: H@H-I max_err = {err:.2e} ✓")

    print("  [PASS] Hadamard involution property verified")
    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SpinQuant Rotation Levels Validation Suite")
    print("=" * 60)

    tests = [
        ("Hadamard Involution (H@H=I)", test_hadamard_involution),
        ("R2 Isolation", test_r2_isolation),
        ("R4 Isolation", test_r4_isolation),
        ("R1 Only", test_r1_only),
        ("R1 + R2", test_r1_r2),
        ("R1 + R2 + R3", test_r1_r2_r3),
        ("R1 + R2 + R3 + R4", test_r1_r2_r3_r4),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
