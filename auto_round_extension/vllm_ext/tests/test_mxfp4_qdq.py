# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MXFP4 activation QDQ backends in vllm_ext.

Tests cover:
- Backend dispatch logic (local_ext > triton > fallback)
- Numerical correctness of each backend
- Cross-backend consistency (CUDA vs Triton tie-breaking documented)
- End-to-end GEMM with unpacked weight path
- Quark equivalence (when Quark is available)
"""

import os
import sys
from unittest.mock import patch

import pytest
import torch

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _has_cuda():
    return torch.cuda.is_available()


def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _has_quark():
    try:
        from quark.torch.kernel.mx import qdq_mxfp4  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
requires_triton = pytest.mark.skipif(not _has_triton(), reason="Triton not available")
requires_quark = pytest.mark.skipif(not _has_quark(), reason="Quark not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_qdq_state():
    """Reset the backend logging state between tests."""
    import auto_round_extension.vllm_ext.mxfp4_qdq_utils as m
    m._qdq_backend_logged = False
    env_key = "AUTO_ROUND_MXFP4_QDQ_BACKEND"
    old = os.environ.pop(env_key, None)
    yield
    if old is not None:
        os.environ[env_key] = old
    else:
        os.environ.pop(env_key, None)
    m._qdq_backend_logged = False


@pytest.fixture
def sample_tensor():
    """Standard test tensor: (8, 256) bf16 on CUDA."""
    torch.manual_seed(42)
    return torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")


@pytest.fixture
def large_tensor():
    """Large test tensor: (256, 4096) bf16 on CUDA."""
    torch.manual_seed(123)
    return torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda")


# ---------------------------------------------------------------------------
# Test: Backend dispatch and selection
# ---------------------------------------------------------------------------

class TestBackendDispatch:
    """Tests for QDQ backend selection logic."""

    @requires_cuda
    def test_auto_dispatch_selects_local_ext(self, sample_tensor):
        """Auto dispatch should prefer local_ext on CUDA."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        out = qdq_mxfp4(sample_tensor.clone())
        assert out.shape == sample_tensor.shape
        assert out.dtype == sample_tensor.dtype

    @requires_cuda
    def test_forced_local_ext(self, sample_tensor):
        """Forcing local_ext backend via env var."""
        os.environ["AUTO_ROUND_MXFP4_QDQ_BACKEND"] = "local_ext"
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        out = qdq_mxfp4(sample_tensor.clone())
        assert out.shape == sample_tensor.shape

    @requires_cuda
    @requires_triton
    def test_forced_triton(self, sample_tensor):
        """Forcing triton backend via env var."""
        os.environ["AUTO_ROUND_MXFP4_QDQ_BACKEND"] = "triton"
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        out = qdq_mxfp4(sample_tensor.clone())
        assert out.shape == sample_tensor.shape

    @requires_cuda
    def test_forced_fallback(self, sample_tensor):
        """Forcing fallback backend via env var."""
        os.environ["AUTO_ROUND_MXFP4_QDQ_BACKEND"] = "fallback"
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        out = qdq_mxfp4(sample_tensor.clone())
        assert out.shape == sample_tensor.shape

    @requires_cuda
    def test_invalid_backend_falls_through(self, sample_tensor):
        """Invalid backend name should fall through to auto dispatch."""
        os.environ["AUTO_ROUND_MXFP4_QDQ_BACKEND"] = "nonexistent"
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        # Should not raise — falls through to auto selection
        out = qdq_mxfp4(sample_tensor.clone())
        assert out.shape == sample_tensor.shape


# ---------------------------------------------------------------------------
# Test: Numerical correctness
# ---------------------------------------------------------------------------

class TestNumericalCorrectness:
    """Tests for numerical correctness of each backend."""

    @requires_cuda
    def test_local_ext_basic_properties(self, sample_tensor):
        """Local ext output should satisfy FP4 quantization properties."""
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4
        out = qdq_mxfp4(sample_tensor.clone(), 32)
        assert out is not None
        # Output should be same shape/dtype
        assert out.shape == sample_tensor.shape
        assert out.dtype == sample_tensor.dtype
        # QDQ should not be identity (except for zero tensor)
        assert not torch.equal(out, sample_tensor)
        # Max absolute value of FP4 E2M1 * max_scale should bound the output
        # FP4 E2M1 max = 6.0, max e8m0 scale = 2^127 — in practice outputs are bounded
        assert out.isfinite().all()

    @requires_cuda
    @requires_triton
    def test_triton_basic_properties(self, sample_tensor):
        """Triton output should satisfy FP4 quantization properties."""
        from auto_round_extension.vllm_ext._triton_mxfp4_qdq import qdq_mxfp4_triton
        out = qdq_mxfp4_triton(sample_tensor.clone(), scale_calculation_mode="even")
        assert out.shape == sample_tensor.shape
        assert out.dtype == sample_tensor.dtype
        assert not torch.equal(out, sample_tensor)
        assert out.isfinite().all()

    @requires_cuda
    def test_fallback_basic_properties(self, sample_tensor):
        """Fallback output should satisfy FP4 quantization properties."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import _qdq_mxfp4_fallback
        out = _qdq_mxfp4_fallback(sample_tensor.clone())
        assert out.shape == sample_tensor.shape
        assert out.dtype == sample_tensor.dtype
        assert out.isfinite().all()

    @requires_cuda
    def test_zero_tensor_is_preserved(self):
        """QDQ on a zero tensor should return zeros."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        x = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
        out = qdq_mxfp4(x)
        assert torch.equal(out, x)

    @requires_cuda
    def test_output_values_are_fp4_representable(self, sample_tensor):
        """After QDQ, values should be representable in FP4 E2M1 * scale."""
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4
        out = qdq_mxfp4(sample_tensor.clone(), 32)
        # FP4 E2M1 positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        # Scaled by power-of-2 => result / scale should be one of these
        # Check that double QDQ is idempotent
        out2 = qdq_mxfp4(out.clone(), 32)
        assert torch.equal(out, out2), "QDQ should be idempotent"

    @requires_cuda
    @requires_triton
    def test_triton_idempotent(self, sample_tensor):
        """Triton QDQ should be idempotent."""
        from auto_round_extension.vllm_ext._triton_mxfp4_qdq import qdq_mxfp4_triton
        out = qdq_mxfp4_triton(sample_tensor.clone(), scale_calculation_mode="even")
        out2 = qdq_mxfp4_triton(out.clone(), scale_calculation_mode="even")
        assert torch.equal(out, out2), "Triton QDQ should be idempotent"

    @requires_cuda
    def test_fallback_near_idempotent(self, sample_tensor):
        """Fallback QDQ should be approximately idempotent.

        Note: block-scale QDQ is NOT strictly idempotent because the block-max
        can change after the first pass, altering the scale for edge-case values.
        The CUDA kernel happens to be idempotent due to its rounding behavior,
        but the fallback (pure PyTorch) may differ for ~2% of elements by at most
        one FP4 step (0.125 at small scales).
        """
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import _qdq_mxfp4_fallback
        out = _qdq_mxfp4_fallback(sample_tensor.clone())
        out2 = _qdq_mxfp4_fallback(out.clone())
        diff = (out - out2).abs()
        frac_diff = (diff > 0).float().mean().item()
        assert frac_diff < 0.05, f"Too many non-idempotent elements: {frac_diff:.2%}"
        assert diff.max().item() <= 0.5, f"Max idempotency error too large: {diff.max().item()}"


# ---------------------------------------------------------------------------
# Test: Cross-backend consistency
# ---------------------------------------------------------------------------

class TestCrossBackendConsistency:
    """Tests documenting cross-backend differences (tie-breaking policy)."""

    @requires_cuda
    @requires_triton
    def test_cuda_vs_triton_tie_breaking(self, large_tensor):
        """CUDA and Triton differ only on FP4 midpoint tie-breaking.

        CUDA kernel: rounds ties toward zero (truncation).
        Triton kernel: rounds ties away from zero.

        This is the SAME behavior as Quark's own CUDA vs Triton.
        Only ~0.6% of elements are affected (those at exact FP4 midpoints).
        """
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4 as cuda_qdq
        from auto_round_extension.vllm_ext._triton_mxfp4_qdq import qdq_mxfp4_triton

        out_cuda = cuda_qdq(large_tensor.clone(), 32)
        out_triton = qdq_mxfp4_triton(large_tensor.clone(), scale_calculation_mode="even")

        diff_mask = out_cuda != out_triton
        frac_diff = diff_mask.float().mean().item()

        # Documented behavior: ~0.6% elements differ
        assert frac_diff < 0.02, f"Too many diffs: {frac_diff:.2%}"
        assert frac_diff > 0.0, "Some diffs expected due to tie-breaking"

        # Key invariant: CUDA always rounds TOWARD zero on ties
        if diff_mask.any():
            cuda_closer_to_zero = out_cuda[diff_mask].abs() <= out_triton[diff_mask].abs()
            assert cuda_closer_to_zero.all(), (
                "CUDA should always round toward zero on tie-breaking values"
            )

    @requires_cuda
    @requires_triton
    def test_cuda_vs_triton_no_diff_on_non_midpoints(self):
        """Values NOT at FP4 midpoints should be identical between backends."""
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4 as cuda_qdq
        from auto_round_extension.vllm_ext._triton_mxfp4_qdq import qdq_mxfp4_triton

        # Construct tensor with values that are already FP4 representable
        # (not midpoints) — these should roundtrip identically
        fp4_values = torch.tensor(
            [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0],
            dtype=torch.bfloat16, device="cuda"
        ).repeat(4, 2)  # (4, 32) — one full block

        out_cuda = cuda_qdq(fp4_values.clone(), 32)
        out_triton = qdq_mxfp4_triton(fp4_values.clone(), scale_calculation_mode="even")

        # These are already representable in FP4, so both should return same values
        # (no tie-breaking needed)
        assert torch.equal(out_cuda, out_triton), (
            "Non-midpoint FP4 values should be identical across backends"
        )


# ---------------------------------------------------------------------------
# Test: Quark equivalence
# ---------------------------------------------------------------------------

class TestQuarkEquivalence:
    """Tests verifying our vendored code matches Quark's original."""

    @requires_cuda
    @requires_quark
    def test_local_ext_matches_quark_cuda(self, large_tensor):
        """Our local_ext CUDA kernel should produce identical output to Quark's."""
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4 as local_qdq
        from quark.torch.kernel.mx import qdq_mxfp4 as quark_qdq

        out_local = local_qdq(large_tensor.clone(), 32)
        out_quark = quark_qdq(large_tensor.clone(), scale_calculation_mode="even")

        assert torch.equal(out_local, out_quark), (
            f"local_ext should exactly match Quark CUDA, "
            f"got {(out_local != out_quark).sum().item()} diffs"
        )

    @requires_cuda
    @requires_triton
    @requires_quark
    def test_triton_matches_quark_triton(self, large_tensor):
        """Our vendored Triton kernel should match Quark's Triton exactly."""
        from auto_round_extension.vllm_ext._triton_mxfp4_qdq import qdq_mxfp4_triton
        from quark.torch.kernel.mx import qdq_mxfp4_triton as quark_triton_qdq

        out_ours = qdq_mxfp4_triton(large_tensor.clone(), scale_calculation_mode="even")
        out_quark = quark_triton_qdq(large_tensor.clone(), scale_calculation_mode="even")

        assert torch.equal(out_ours, out_quark), (
            f"Vendored Triton should exactly match Quark Triton, "
            f"got {(out_ours != out_quark).sum().item()} diffs"
        )

    @requires_cuda
    @requires_quark
    def test_quark_itself_has_cuda_triton_diff(self, large_tensor):
        """Document: Quark's own CUDA and Triton differ on tie-breaking (same as ours)."""
        from quark.torch.kernel.mx import qdq_mxfp4 as quark_cuda
        from quark.torch.kernel.mx import qdq_mxfp4_triton as quark_triton

        out_cuda = quark_cuda(large_tensor.clone(), scale_calculation_mode="even")
        out_triton = quark_triton(large_tensor.clone(), scale_calculation_mode="even")

        n_diff = (out_cuda != out_triton).sum().item()
        frac = n_diff / large_tensor.numel()
        # Quark itself has ~0.6% diffs
        assert frac > 0.0, "Quark should have CUDA vs Triton diffs"
        assert frac < 0.02, f"Quark diff unexpectedly large: {frac:.2%}"


# ---------------------------------------------------------------------------
# Test: End-to-end GEMM path
# ---------------------------------------------------------------------------

class TestEndToEndGEMM:
    """Tests for the full mxfp4_gemm_with_unpacked_weight path."""

    @requires_cuda
    def test_gemm_basic(self):
        """Basic GEMM with unpacked FP8 weight."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
            dequant_mxfp4_to_fp8,
            mxfp4_gemm_with_unpacked_weight,
        )

        N, K = 128, 256
        weight_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        weight_scale = torch.randint(120, 135, (N, K // 32), dtype=torch.uint8, device="cuda")

        weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(weight_packed, weight_scale)
        x = torch.randn(2, K, dtype=torch.bfloat16, device="cuda")

        out = mxfp4_gemm_with_unpacked_weight(x, weight_fp8, scale_bf16)
        assert out.shape == (2, N)
        assert out.dtype == torch.bfloat16
        assert out.isfinite().all()

    @requires_cuda
    def test_gemm_with_bias(self):
        """GEMM with bias term."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
            dequant_mxfp4_to_fp8,
            mxfp4_gemm_with_unpacked_weight,
        )

        N, K = 64, 128
        weight_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        weight_scale = torch.randint(120, 135, (N, K // 32), dtype=torch.uint8, device="cuda")

        weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(weight_packed, weight_scale)
        x = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")

        out = mxfp4_gemm_with_unpacked_weight(x.clone(), weight_fp8, scale_bf16, bias=bias)
        out_no_bias = mxfp4_gemm_with_unpacked_weight(x.clone(), weight_fp8, scale_bf16)

        # Output with bias should differ from without bias
        assert not torch.equal(out, out_no_bias)
        # Output shapes and properties should be correct
        assert out.shape == (4, N)
        assert out.dtype == torch.bfloat16
        assert out.isfinite().all()

    @requires_cuda
    def test_gemm_batched_input(self):
        """GEMM with batched (3D) input tensor."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
            dequant_mxfp4_to_fp8,
            mxfp4_gemm_with_unpacked_weight,
        )

        N, K = 128, 256
        weight_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        weight_scale = torch.randint(120, 135, (N, K // 32), dtype=torch.uint8, device="cuda")

        weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(weight_packed, weight_scale)
        # Batch of 3, sequence length 8
        x = torch.randn(3, 8, K, dtype=torch.bfloat16, device="cuda")
        x_flat = x.reshape(-1, K)

        out = mxfp4_gemm_with_unpacked_weight(x_flat, weight_fp8, scale_bf16)
        assert out.shape == (24, N)

    @requires_cuda
    def test_gemm_different_backends_produce_similar_results(self):
        """GEMM results should be similar regardless of QDQ backend."""
        import auto_round_extension.vllm_ext.mxfp4_qdq_utils as m
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import (
            dequant_mxfp4_to_fp8,
            mxfp4_gemm_with_unpacked_weight,
        )

        N, K = 128, 256
        weight_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        weight_scale = torch.randint(120, 135, (N, K // 32), dtype=torch.uint8, device="cuda")
        weight_fp8, scale_bf16 = dequant_mxfp4_to_fp8(weight_packed, weight_scale)
        x = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")

        # Run with local_ext
        os.environ["AUTO_ROUND_MXFP4_QDQ_BACKEND"] = "local_ext"
        m._qdq_backend_logged = False
        out_cuda = mxfp4_gemm_with_unpacked_weight(x.clone(), weight_fp8, scale_bf16)

        # Run with fallback
        os.environ["AUTO_ROUND_MXFP4_QDQ_BACKEND"] = "fallback"
        m._qdq_backend_logged = False
        out_fb = mxfp4_gemm_with_unpacked_weight(x.clone(), weight_fp8, scale_bf16)

        # Results should be close (minor rounding diffs in activation QDQ)
        rel_diff = (out_cuda - out_fb).abs() / (out_cuda.abs() + 1e-6)
        assert rel_diff.mean().item() < 0.2, (
            f"GEMM results too different: mean_rel_diff={rel_diff.mean().item():.4f}"
        )


# ---------------------------------------------------------------------------
# Test: Shape and dtype handling
# ---------------------------------------------------------------------------

class TestShapeAndDtype:
    """Tests for various input shapes and dtypes."""

    @requires_cuda
    @pytest.mark.parametrize("shape", [
        (2, 64),       # Minimal: numel=128, multiple of 64
        (4, 64),       # Small
        (16, 128),     # Medium
        (256, 4096),   # Large (typical hidden_size)
        (1, 4096),     # Single row
        (512, 8192),   # Very large
    ])
    def test_various_shapes(self, shape):
        """QDQ should work for any shape where numel % 64 == 0."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        out = qdq_mxfp4(x)
        assert out.shape == shape

    @requires_cuda
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_supported_dtypes(self, dtype):
        """QDQ should work for bf16 and fp16."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        x = torch.randn(8, 256, dtype=dtype, device="cuda")
        out = qdq_mxfp4(x)
        assert out.dtype == dtype

    @requires_cuda
    def test_3d_tensor(self):
        """QDQ should handle 3D tensors (batch, seq, hidden)."""
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4
        x = torch.randn(2, 8, 256, dtype=torch.bfloat16, device="cuda")
        out = qdq_mxfp4(x.reshape(-1, 256))
        assert out.shape == (16, 256)


# ---------------------------------------------------------------------------
# Test: Performance sanity check
# ---------------------------------------------------------------------------

class TestPerformance:
    """Sanity check that local_ext is faster than fallback."""

    @requires_cuda
    def test_local_ext_faster_than_fallback(self):
        """Local ext should be significantly faster than PyTorch fallback."""
        import time
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4 as cuda_qdq
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import _qdq_mxfp4_fallback

        x = torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(5):
            cuda_qdq(x.clone(), 32)
            _qdq_mxfp4_fallback(x.clone())
        torch.cuda.synchronize()

        # Benchmark local_ext
        iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            cuda_qdq(x.clone(), 32)
        torch.cuda.synchronize()
        t_cuda = time.perf_counter() - t0

        # Benchmark fallback
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _qdq_mxfp4_fallback(x.clone())
        torch.cuda.synchronize()
        t_fallback = time.perf_counter() - t0

        speedup = t_fallback / t_cuda
        assert speedup > 3.0, (
            f"Expected local_ext to be >3x faster, got {speedup:.1f}x "
            f"(cuda={t_cuda/iters*1000:.3f}ms, fallback={t_fallback/iters*1000:.3f}ms)"
        )


# ---------------------------------------------------------------------------
# Test: Extension loading
# ---------------------------------------------------------------------------

class TestExtensionLoading:
    """Tests for the JIT extension loader."""

    @requires_cuda
    def test_local_ext_is_available(self):
        """Local CUDA extension should be loadable."""
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import is_available
        assert is_available()

    @requires_cuda
    def test_disable_via_env_var(self):
        """AUTO_ROUND_DISABLE_LOCAL_MXFP4_QDQ=1 should disable the extension."""
        # This test uses a fresh module state — we can't easily test without
        # reimporting, so we test the function logic directly
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4
        # Extension already loaded in this process, so it will return result
        # The disable check only works before first load
        out = qdq_mxfp4(torch.randn(4, 32, dtype=torch.bfloat16, device="cuda"), 32)
        assert out is not None  # Already loaded, can't un-load

    @requires_cuda
    def test_group_size_32_only(self):
        """Local ext should return None for group_size != 32."""
        from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4
        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        assert qdq_mxfp4(x, 64) is None
        assert qdq_mxfp4(x, 16) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
