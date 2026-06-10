#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Triton MXFP4 kernels: dequantization and fused GEMM.

Run:
    python test_triton_mxfp4_kernels.py
"""

import time
import sys

import torch

# Disable TF32 for accurate FP32 reference comparisons
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Ensure auto-round is importable
sys.path.insert(0, "/data/lkk/quarot/latest/new_commit/auto-round")


def create_test_data(M, N, group_size=32, device="cuda"):
    """Create test packed weights and scales for MXFP4 format."""
    # Random packed uint8 (simulates two FP4 values per byte)
    N_half = N // 2
    N_groups = N // group_size
    packed_weight = torch.randint(0, 256, (M, N_half), dtype=torch.uint8, device=device)
    # Random e8m0 scales (valid range: roughly 100-150 for reasonable scale values)
    weight_scale = torch.randint(120, 135, (M, N_groups), dtype=torch.uint8, device=device)
    return packed_weight, weight_scale


def test_mxfp4_dequant_correctness():
    """Test that Triton dequant matches PyTorch reference."""
    print("=" * 70)
    print("TEST: MXFP4 Dequant Correctness")
    print("=" * 70)

    from auto_round.triton_kernels.mxfp4_dequant import triton_mxfp4_dequant, triton_mxfp4_dequant_ref

    test_sizes = [(64, 128), (128, 256), (256, 1024), (512, 4096), (4096, 4096)]

    all_passed = True
    for M, N in test_sizes:
        packed_w, scale = create_test_data(M, N)

        # Triton kernel
        result_triton = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.float32)

        # PyTorch reference
        result_ref = triton_mxfp4_dequant_ref(packed_w, scale, output_dtype=torch.float32)

        # Compare
        max_diff = (result_triton - result_ref).abs().max().item()
        mean_diff = (result_triton - result_ref).abs().mean().item()
        passed = max_diff < 1e-4

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] Shape ({M:>4}, {N:>4}): max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

        if not passed:
            all_passed = False
            # Debug: print first few mismatches
            diff = (result_triton - result_ref).abs()
            idx = diff.argmax()
            row, col = idx // N, idx % N
            print(f"         Worst at [{row}, {col}]: triton={result_triton[row, col]:.6f}, ref={result_ref[row, col]:.6f}")

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    return all_passed


def test_mxfp4_gemm_correctness():
    """Test that fused Triton GEMM matches separate dequant + matmul."""
    print("=" * 70)
    print("TEST: Fused MXFP4 GEMM Correctness")
    print("=" * 70)

    from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm
    from auto_round.triton_kernels.mxfp4_dequant import triton_mxfp4_dequant

    test_sizes = [
        # (M_input, N_out, K_in)
        (1, 128, 128),      # single token
        (4, 256, 256),      # small batch
        (32, 512, 512),     # medium
        (64, 1024, 1024),   # larger
        (1, 4096, 4096),    # single token, large model dim
        (32, 4096, 4096),   # typical batch, large model dim
    ]

    all_passed = True
    for M, N, K in test_sizes:
        # Create test data
        input_tensor = torch.randn(M, K, dtype=torch.float32, device="cuda")
        packed_w, scale = create_test_data(N, K)  # Weight is [N, K/2]
        bias = torch.randn(N, dtype=torch.float32, device="cuda")

        # Fused Triton GEMM
        result_fused = triton_mxfp4_gemm(input_tensor, packed_w, scale, bias=bias)

        # Reference: separate dequant + matmul
        w_dequant = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.float32)
        result_ref = input_tensor @ w_dequant.T + bias

        # Compare (with TF32 disabled, both are true FP32; small diff from accumulation order)
        max_diff = (result_fused - result_ref).abs().max().item()
        rel_err = (result_fused - result_ref).abs() / (result_ref.abs() + 1e-8)
        mean_rel_err = rel_err.mean().item()

        # FP32 GEMM tolerance: grows with sqrt(K) due to accumulation rounding
        # For K=4096, expect ~0.1 max abs diff (Triton uses different block sizes than cuBLAS)
        tol = max(0.01, K * 3e-5)
        passed = max_diff < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] ({M:>3}×{K:>4}) @ ({N:>4}×{K:>4}).T: "
              f"max_diff={max_diff:.4e}, mean_rel_err={mean_rel_err:.4e}")

        if not passed:
            all_passed = False

    # Test without bias
    print("\n  Testing without bias...")
    input_tensor = torch.randn(16, 512, dtype=torch.float32, device="cuda")
    packed_w, scale = create_test_data(256, 512)
    result_no_bias = triton_mxfp4_gemm(input_tensor, packed_w, scale, bias=None)
    w_dequant = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.float32)
    ref_no_bias = input_tensor @ w_dequant.T
    max_diff = (result_no_bias - ref_no_bias).abs().max().item()
    passed = max_diff < 0.02
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] No-bias case: max_diff={max_diff:.4e}")
    if not passed:
        all_passed = False

    # Test batched input (3D)
    print("\n  Testing batched (3D) input...")
    input_3d = torch.randn(2, 16, 256, dtype=torch.float32, device="cuda")
    packed_w, scale = create_test_data(128, 256)
    result_3d = triton_mxfp4_gemm(input_3d, packed_w, scale, bias=None)
    w_dequant = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.float32)
    ref_3d = input_3d.reshape(-1, 256) @ w_dequant.T
    ref_3d = ref_3d.reshape(2, 16, 128)
    max_diff = (result_3d - ref_3d).abs().max().item()
    passed = max_diff < 0.01
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] 3D input [2,16,256]: max_diff={max_diff:.4e}")
    if not passed:
        all_passed = False

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    return all_passed


def test_mxfp4_gemm_performance():
    """Benchmark fused kernel vs separate dequant + matmul."""
    print("=" * 70)
    print("BENCHMARK: Fused GEMM vs Separate Dequant + Matmul")
    print("=" * 70)

    from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm
    from auto_round.triton_kernels.mxfp4_dequant import triton_mxfp4_dequant
    from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8

    # Typical LLM dimensions
    benchmarks = [
        # (batch, out_features, in_features, description)
        (1, 4096, 4096, "single token, hidden_dim=4096"),
        (32, 4096, 4096, "batch=32, hidden_dim=4096"),
        (1, 11008, 4096, "single token, up_proj (Llama-7B)"),
        (1, 14336, 4096, "single token, gate_proj (Llama2-13B)"),
    ]

    warmup_iters = 10
    bench_iters = 100

    for M, N, K, desc in benchmarks:
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        packed_w, scale = create_test_data(N, K)
        bias = torch.randn(N, dtype=torch.float32, device="cuda")

        # --- Fused Triton GEMM ---
        for _ in range(warmup_iters):
            triton_mxfp4_gemm(input_tensor, packed_w, scale, bias)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            triton_mxfp4_gemm(input_tensor, packed_w, scale, bias)
        torch.cuda.synchronize()
        time_fused = (time.perf_counter() - t0) / bench_iters * 1000  # ms

        # --- Separate: Triton dequant + torch.matmul ---
        for _ in range(warmup_iters):
            w = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.bfloat16)
            _ = input_tensor @ w.T + bias.bfloat16()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            w = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.bfloat16)
            _ = input_tensor @ w.T + bias.bfloat16()
        torch.cuda.synchronize()
        time_separate_triton = (time.perf_counter() - t0) / bench_iters * 1000

        # --- PyTorch reference (Python unpack + matmul) ---
        for _ in range(warmup_iters):
            unpacked = unpack_fp4_from_uint8(packed_w, N, K, dtype=torch.bfloat16)
            scale_f = torch.pow(2.0, scale.to(torch.int16).float() - 127.0)
            w_ref = (unpacked.reshape(N, -1, 32) * scale_f.unsqueeze(-1)).reshape(N, K).bfloat16()
            _ = input_tensor @ w_ref.T + bias.bfloat16()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            unpacked = unpack_fp4_from_uint8(packed_w, N, K, dtype=torch.bfloat16)
            scale_f = torch.pow(2.0, scale.to(torch.int16).float() - 127.0)
            w_ref = (unpacked.reshape(N, -1, 32) * scale_f.unsqueeze(-1)).reshape(N, K).bfloat16()
            _ = input_tensor @ w_ref.T + bias.bfloat16()
        torch.cuda.synchronize()
        time_pytorch = (time.perf_counter() - t0) / bench_iters * 1000

        speedup_vs_separate = time_separate_triton / time_fused
        speedup_vs_pytorch = time_pytorch / time_fused

        print(f"\n  {desc}")
        print(f"    Fused Triton GEMM:     {time_fused:>8.3f} ms")
        print(f"    Triton dequant + mm:   {time_separate_triton:>8.3f} ms  ({speedup_vs_separate:.2f}x slower)")
        print(f"    PyTorch unpack + mm:   {time_pytorch:>8.3f} ms  ({speedup_vs_pytorch:.2f}x slower)")

    print()


def test_rotated_linear():
    """Test RotatedMXFP4Linear module."""
    print("=" * 70)
    print("TEST: RotatedMXFP4Linear Module")
    print("=" * 70)

    from auto_round.triton_kernels.rotated_linear import RotatedMXFP4Linear
    from auto_round.triton_kernels.mxfp4_dequant import triton_mxfp4_dequant

    K, N = 256, 512
    M = 8  # batch size

    # Create random rotation matrix (orthogonal)
    Q, _ = torch.linalg.qr(torch.randn(K, K, device="cuda"))
    rotation_matrix = Q.to(torch.float32)

    # Create packed weights and scale
    packed_w, scale = create_test_data(N, K)
    bias = torch.randn(N, dtype=torch.float32, device="cuda")

    # Create RotatedMXFP4Linear
    rotated_linear = RotatedMXFP4Linear(
        packed_weight=packed_w,
        weight_scale=scale,
        rotation_matrix=rotation_matrix,
        bias=bias,
    ).cuda()

    # Forward pass
    input_tensor = torch.randn(M, K, dtype=torch.float32, device="cuda")
    output = rotated_linear(input_tensor)

    # Reference: manual rotation + dequant + matmul
    rotated_input = input_tensor @ rotation_matrix
    w_dequant = triton_mxfp4_dequant(packed_w, scale, output_dtype=torch.float32)
    ref_output = rotated_input @ w_dequant.T + bias

    max_diff = (output - ref_output).abs().max().item()
    passed = max_diff < 0.02  # K=256, expect small diff
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] With rotation: max_diff={max_diff:.4e}")

    # Test without rotation
    linear_no_rot = RotatedMXFP4Linear(
        packed_weight=packed_w,
        weight_scale=scale,
        rotation_matrix=None,
        bias=bias,
    ).cuda()
    output_no_rot = linear_no_rot(input_tensor)
    ref_no_rot = input_tensor @ w_dequant.T + bias
    max_diff_no_rot = (output_no_rot - ref_no_rot).abs().max().item()
    passed_no_rot = max_diff_no_rot < 0.02
    status = "✓ PASS" if passed_no_rot else "✗ FAIL"
    print(f"  [{status}] Without rotation: max_diff={max_diff_no_rot:.4e}")

    # Test repr
    print(f"  Module repr: {rotated_linear}")

    all_passed = passed and passed_no_rot
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
    return all_passed


def test_rotated_linear_performance():
    """Benchmark RotatedMXFP4Linear vs hook-based approach."""
    print("=" * 70)
    print("BENCHMARK: RotatedMXFP4Linear vs Hook-Based Rotation")
    print("=" * 70)

    from auto_round.triton_kernels.rotated_linear import RotatedMXFP4Linear
    from auto_round.triton_kernels.mxfp4_dequant import triton_mxfp4_dequant

    K, N = 4096, 4096

    # Create test data
    Q, _ = torch.linalg.qr(torch.randn(K, K, device="cuda"))
    rotation_matrix = Q.to(torch.bfloat16)
    packed_w, scale = create_test_data(N, K)
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")

    # Fused module
    fused_module = RotatedMXFP4Linear(
        packed_weight=packed_w,
        weight_scale=scale,
        rotation_matrix=rotation_matrix.float(),
        bias=bias.float(),
    ).cuda()

    # Simulate hook-based approach
    from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8

    def hook_based_forward(x, rot, packed_w, scale, bias, N, K):
        """Simulate: hook rotation + MXFP4QuantLinear.forward()"""
        # Hook: rotation
        x = x @ rot
        # MXFP4QuantLinear.dequant_weight_online()
        unpacked = unpack_fp4_from_uint8(packed_w, N, K, dtype=x.dtype)
        scale_f = torch.pow(2.0, scale.to(torch.int16).float() - 127.0).to(x.dtype)
        w = (unpacked.reshape(N, -1, 32) * scale_f.unsqueeze(-1)).reshape(N, K)
        # F.linear
        out = x @ w.T + bias.to(x.dtype)
        return out

    warmup_iters = 10
    bench_iters = 100

    for M, desc in [(1, "single token"), (32, "batch=32"), (128, "batch=128")]:
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # Fused
        for _ in range(warmup_iters):
            fused_module(input_tensor.float())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            fused_module(input_tensor.float())
        torch.cuda.synchronize()
        time_fused = (time.perf_counter() - t0) / bench_iters * 1000

        # Hook-based
        for _ in range(warmup_iters):
            hook_based_forward(input_tensor, rotation_matrix, packed_w, scale, bias, N, K)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            hook_based_forward(input_tensor, rotation_matrix, packed_w, scale, bias, N, K)
        torch.cuda.synchronize()
        time_hook = (time.perf_counter() - t0) / bench_iters * 1000

        speedup = time_hook / time_fused
        print(f"  {desc:>15}: fused={time_fused:.3f}ms, hook={time_hook:.3f}ms, speedup={speedup:.2f}x")

    print()


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Triton available: True")
    print()

    # Run tests
    results = []
    results.append(("Dequant Correctness", test_mxfp4_dequant_correctness()))
    results.append(("Fused GEMM Correctness", test_mxfp4_gemm_correctness()))
    results.append(("RotatedMXFP4Linear", test_rotated_linear()))

    # Run benchmarks
    test_mxfp4_gemm_performance()
    test_rotated_linear_performance()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_ok = False

    sys.exit(0 if all_ok else 1)
