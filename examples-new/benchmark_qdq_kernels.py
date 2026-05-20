#!/usr/bin/env python3
"""
Benchmark SpinQuant MXFP4 implementations for vLLM plugin.

Two benchmark modes:
  1. QDQ backends — activation quantize-dequantize kernel comparison:
     - local_ext: vendored CUDA kernel (from Quark)
     - quark_cuda: Quark default (hip/C++ extension)
     - quark_triton: Quark triton implementation
     - vllm_ext: auto_round_extension reference
     - fallback: pure PyTorch

  2. Runtime backends — full linear forward (rotation + QDQ + GEMM):
     - packed_fused: Triton fused kernel with packed MXFP4 weights
     - quark_like_dense: pre-dequant weights to dense, then GEMM
     - preunpack_fp8: pre-unpack to FP8, dynamic scaled GEMM

Usage:
    # QDQ backends only
    python examples/benchmark_qdq_kernels.py --mode qdq

    # Runtime backends only (full forward)
    python examples/benchmark_qdq_kernels.py --mode runtime

    # Both (default)
    python examples/benchmark_qdq_kernels.py

    # Custom shapes and options
    python examples/benchmark_qdq_kernels.py --shapes 256x4096,256x8192 --iters 200
    python examples/benchmark_qdq_kernels.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from auto_round.vllm_plugin._mxfp4_qdq_ext import qdq_mxfp4 as local_qdq_mxfp4
from auto_round.vllm_plugin.spinquant_mxfp4 import (
    MXFP4_BLOCK_SIZE,
    _mxfp4_act_qdq_fallback,
    _mxfp4_dequant_linear_fallback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SpinQuant MXFP4 implementations.")
    parser.add_argument(
        "--mode",
        choices=["qdq", "runtime", "all"],
        default="all",
        help="Benchmark mode: 'qdq' (activation QDQ only), 'runtime' (full forward), 'all'. Default: all",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Default: cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Activation dtype. Default: bfloat16",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=MXFP4_BLOCK_SIZE,
        help=f"MXFP4 block/group size. Default: {MXFP4_BLOCK_SIZE}",
    )
    parser.add_argument(
        "--shapes",
        default=None,
        help="Comma-separated shapes MxK. Default: mode-specific shapes.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations. Default: 10",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark iterations. Default: 100",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of table.",
    )
    return parser.parse_args()


def parse_shapes(spec: str) -> list[tuple[int, int]]:
    shapes = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        m_str, k_str = item.split("x", 1)
        shapes.append((int(m_str), int(k_str)))
    return shapes


def resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Activation QDQ Backends
# ═══════════════════════════════════════════════════════════════════════════════

def load_qdq_backends(group_size: int) -> tuple[list[tuple[str, Callable]], dict[str, str]]:
    """Load all available activation QDQ backends."""
    funcs: list[tuple[str, Callable]] = []
    avail: dict[str, str] = {}

    # 1. local_ext (vendored CUDA kernel)
    if torch.cuda.is_available():
        try:
            # Probe with 64 elements minimum (kernel requires numel % 64 == 0)
            sample = torch.randn(2, group_size, device="cuda", dtype=torch.bfloat16)
            result = local_qdq_mxfp4(sample, group_size)
            if result is not None:
                funcs.append(("local_ext", lambda x, gs=group_size: local_qdq_mxfp4(x, gs)))
                avail["local_ext"] = "available"
            else:
                avail["local_ext"] = "build failed"
        except Exception as exc:
            avail["local_ext"] = f"error: {exc}"
    else:
        avail["local_ext"] = "no CUDA"

    # 2. quark_cuda (default hip/C++ extension)
    try:
        orig_impl = os.environ.get("QUARK_MXFP4_IMPL", "")
        os.environ["QUARK_MXFP4_IMPL"] = "hip"
        # Force fresh import
        for key in list(sys.modules.keys()):
            if "quark.torch.kernel" in key:
                del sys.modules[key]
        from quark.torch.kernel import mx as quark_mx_cuda
        funcs.append(("quark_cuda", lambda x, m=quark_mx_cuda: m.qdq_mxfp4(x, scale_calculation_mode="even")))
        avail["quark_cuda"] = "available"
        if orig_impl:
            os.environ["QUARK_MXFP4_IMPL"] = orig_impl
        else:
            os.environ.pop("QUARK_MXFP4_IMPL", None)
    except Exception as exc:
        avail["quark_cuda"] = f"error: {exc}"

    # 3. quark_triton
    try:
        os.environ["QUARK_MXFP4_IMPL"] = "triton"
        for key in list(sys.modules.keys()):
            if "quark.torch.kernel" in key:
                del sys.modules[key]
        from quark.torch.kernel import mx as quark_mx_triton
        funcs.append(("quark_triton", lambda x, m=quark_mx_triton: m.qdq_mxfp4(x, scale_calculation_mode="even")))
        avail["quark_triton"] = "available"
        os.environ.pop("QUARK_MXFP4_IMPL", None)
    except Exception as exc:
        avail["quark_triton"] = f"error: {exc}"
        os.environ.pop("QUARK_MXFP4_IMPL", None)

    # 4. vllm_ext
    try:
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4 as vllm_ext_qdq
        funcs.append(("vllm_ext", vllm_ext_qdq))
        avail["vllm_ext"] = "available"
    except Exception as exc:
        avail["vllm_ext"] = f"error: {exc}"

    # 5. fallback (always available)
    funcs.append(("fallback", lambda x, gs=group_size: _mxfp4_act_qdq_fallback(x, gs)))
    avail["fallback"] = "available"

    return funcs, avail


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Runtime Backends (full forward simulation)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_fake_mxfp4_weights(
    N: int, K: int, group_size: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fake MXFP4 packed weights + scale for benchmarking.

    Returns (weight_packed [N, K//2] uint8, weight_scale [N, K//group_size] uint8, weight_dense [N,K]).
    """
    # Generate random float weight, quantize to MXFP4, pack
    weight_fp = torch.randn(N, K, device=device, dtype=dtype)

    # Pack: two 4-bit values per byte
    weight_packed = torch.randint(0, 256, (N, K // 2), device=device, dtype=torch.uint8)

    # Scale: random e8m0 exponents (valid range for testing)
    weight_scale = torch.randint(100, 140, (N, K // group_size), device=device, dtype=torch.uint8)

    # For quark_like_dense and preunpack_fp8, we need the pre-dequantized dense weight
    weight_dense = _mxfp4_dequant_linear_fallback(
        torch.randn(1, K, device=device, dtype=dtype),
        weight_packed, weight_scale, group_size
    )
    # Actually just get the dequanted weight directly
    # Use the fallback to dequant
    weight_dense = _dequant_mxfp4_to_dense(weight_packed, weight_scale, group_size, dtype)

    return weight_packed, weight_scale, weight_dense


def _dequant_mxfp4_to_dense(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP4 packed weights to dense float tensor."""
    N, K_half = weight_packed.shape
    K = K_half * 2

    # Unpack nibbles
    low = (weight_packed & 0x0F).to(torch.int8)
    high = ((weight_packed >> 4) & 0x0F).to(torch.int8)
    # Interleave
    unpacked = torch.stack([low, high], dim=-1).reshape(N, K)

    # Convert E2M1 nibble to float: sign(1) + exp(2) + mantissa(1)
    sign = ((unpacked >> 3) & 1).to(dtype) * (-2) + 1
    exp_bits = (unpacked >> 1) & 0x3
    mant_bit = (unpacked & 1).to(dtype)

    # E2M1: value = (-1)^s * 2^(exp-1) * (1 + mantissa*0.5) for exp>0
    #        value = (-1)^s * 0.5 * mantissa for exp==0 (subnormal)
    exp_f = exp_bits.to(dtype)
    normal_mask = exp_bits > 0
    val = torch.where(
        normal_mask,
        sign * torch.pow(2.0, exp_f - 1) * (1.0 + mant_bit * 0.5),
        sign * 0.5 * mant_bit,
    )

    # Apply scale: each group of `group_size` shares one e8m0 exponent
    scale_float = torch.pow(2.0, weight_scale.to(dtype) - 127.0)
    val = val.reshape(N, -1, group_size) * scale_float.unsqueeze(-1)
    return val.reshape(N, K)


def load_runtime_backends(
    N: int, K: int, group_size: int, device: torch.device, dtype: torch.dtype
) -> tuple[list[tuple[str, Callable]], dict[str, str], dict[str, torch.Tensor]]:
    """Load runtime backends for benchmarking full linear forward.

    Returns (backends, availability, shared_tensors).
    """
    funcs: list[tuple[str, Callable]] = []
    avail: dict[str, str] = {}

    weight_packed = torch.randint(0, 256, (N, K // 2), device=device, dtype=torch.uint8)
    weight_scale = torch.randint(100, 140, (N, K // group_size), device=device, dtype=torch.uint8)
    weight_dense = _dequant_mxfp4_to_dense(weight_packed, weight_scale, group_size, dtype)

    shared = {
        "weight_packed": weight_packed,
        "weight_scale": weight_scale,
        "weight_dense": weight_dense,
    }

    # 1. packed_fused (Triton MXFP4 GEMM)
    try:
        from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm
        wp = weight_packed
        ws = weight_scale

        def _packed_fused(x, _wp=wp, _ws=ws, _gs=group_size):
            out = triton_mxfp4_gemm(x.float(), _wp, _ws, bias=None, group_size=_gs, fp32_precision="tf32")
            return out.to(x.dtype)

        funcs.append(("packed_fused", _packed_fused))
        avail["packed_fused"] = "available"
    except Exception as exc:
        avail["packed_fused"] = f"error: {exc}"

    # 2. quark_like_dense (pre-dequant, dense GEMM)
    wd = weight_dense

    def _quark_like_dense(x, _wd=wd):
        return torch.nn.functional.linear(x, _wd)

    funcs.append(("quark_like_dense", _quark_like_dense))
    avail["quark_like_dense"] = "available"

    # 3. preunpack_fp8 (FP8 scaled GEMM)
    try:
        from vllm._custom_ops import scaled_fp8_quant, cutlass_scaled_mm

        wf8, w_scale_f8 = scaled_fp8_quant(weight_dense)
        shared["weight_fp8"] = wf8
        shared["weight_fp8_scale"] = w_scale_f8

        def _preunpack_fp8(x, _wf8=wf8, _ws=w_scale_f8):
            x_f8, x_scale = scaled_fp8_quant(x)
            out = cutlass_scaled_mm(x_f8, _wf8.t(), x_scale, _ws, out_dtype=x.dtype)
            return out

        funcs.append(("preunpack_fp8", _preunpack_fp8))
        avail["preunpack_fp8"] = "available"
    except Exception as exc:
        avail["preunpack_fp8"] = f"error: {exc}"

    return funcs, avail, shared


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def benchmark_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Measure GPU kernel time using CUDA events."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x)
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))

    return {
        "mean_ms": round(statistics.mean(elapsed), 4),
        "median_ms": round(statistics.median(elapsed), 4),
        "min_ms": round(min(elapsed), 4),
        "std_ms": round(statistics.stdev(elapsed), 4) if len(elapsed) > 1 else 0.0,
    }


def print_table(title: str, results: list[dict], avail: dict[str, str]) -> None:
    """Print a formatted benchmark table."""
    print(f"\n{'═' * 80}")
    print(f"  {title}")
    print(f"{'═' * 80}")

    for name, status in avail.items():
        marker = "✓" if status == "available" else "✗"
        print(f"  {marker} {name:>14}: {status}")
    print()

    for bench in results:
        shape = tuple(bench["shape"])
        print(f"  Shape: {shape[0]:>5} x {shape[1]:<6}")
        print(f"  {'Backend':<16} {'Mean':>8} {'Median':>8} {'Min':>8} {'Std':>8} {'Relative':>8}")
        print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, timing in bench["timings"].items():
            rel = bench["relative"][name]
            print(
                f"  {name:<16} {timing['mean_ms']:>7.3f}  {timing['median_ms']:>7.3f}  "
                f"{timing['min_ms']:>7.3f}  {timing['std_ms']:>7.3f}  {rel:>6.2f}x"
            )
        print()


def run_benchmark_suite(
    mode: str,
    funcs: list[tuple[str, Callable]],
    shapes: list[tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[dict]:
    """Run benchmark for all shapes/backends."""
    results = []
    for shape in shapes:
        x = torch.randn(*shape, device=device, dtype=dtype)
        bench = {"shape": list(shape), "timings": {}}
        for name, fn in funcs:
            bench["timings"][name] = benchmark_fn(fn, x, warmup, iters)
        means = {name: t["mean_ms"] for name, t in bench["timings"].items()}
        fastest = min(means.values()) if means else 1.0
        bench["relative"] = {name: round(m / fastest, 3) for name, m in means.items()}
        results.append(bench)
    return results


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    group_size = args.group_size

    output = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "group_size": group_size,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    # --- Part 1: QDQ backends ---
    if args.mode in ("qdq", "all"):
        qdq_shapes = parse_shapes(args.shapes) if args.shapes else [
            (1, 4096), (16, 4096), (64, 4096), (256, 4096),
            (1024, 4096), (256, 8192), (256, 25600),
        ]
        qdq_funcs, qdq_avail = load_qdq_backends(group_size)
        qdq_results = run_benchmark_suite("qdq", qdq_funcs, qdq_shapes, device, dtype, args.warmup, args.iters)
        output["qdq"] = {"availability": qdq_avail, "benchmarks": qdq_results}

        if not args.json:
            print(f"\nDevice: {output['device_name']} | dtype: {args.dtype} | group_size: {group_size}")
            print_table("Activation QDQ Backends (MXFP4 quantize-dequantize)", qdq_results, qdq_avail)

    # --- Part 2: Runtime backends ---
    if args.mode in ("runtime", "all"):
        rt_shapes = parse_shapes(args.shapes) if args.shapes else [
            (1, 4096), (16, 4096), (64, 4096), (256, 4096),
            (1024, 4096), (256, 8192),
        ]
        # For runtime, N (output features) is typically intermediate_size or hidden_size
        # We fix N = K for simplicity (square-ish GEMM) or use typical model dims
        N_out = 4096  # output dim (e.g., hidden_size for down_proj)

        rt_funcs, rt_avail, _ = load_runtime_backends(N_out, rt_shapes[0][1], group_size, device, dtype)

        rt_results = []
        for shape in rt_shapes:
            M, K = shape
            # Rebuild backends for each K (weight dimension may differ)
            if K != rt_shapes[0][1]:
                rt_funcs_k, _, _ = load_runtime_backends(N_out, K, group_size, device, dtype)
            else:
                rt_funcs_k = rt_funcs

            x = torch.randn(M, K, device=device, dtype=dtype)
            bench = {"shape": [M, K, N_out], "timings": {}}
            for name, fn in rt_funcs_k:
                bench["timings"][name] = benchmark_fn(fn, x, args.warmup, args.iters)
            means = {name: t["mean_ms"] for name, t in bench["timings"].items()}
            fastest = min(means.values()) if means else 1.0
            bench["relative"] = {name: round(m / fastest, 3) for name, m in means.items()}
            rt_results.append(bench)

        output["runtime"] = {"availability": rt_avail, "N_out": N_out, "benchmarks": rt_results}

        if not args.json:
            if args.mode == "all":
                print()
            print(f"  (N_out={N_out} for all runtime benchmarks)")
            print_table(
                f"Runtime Backends (full forward: act QDQ + GEMM, N_out={N_out})",
                rt_results, rt_avail
            )

    if args.json:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
