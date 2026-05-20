#!/usr/bin/env python3
"""
Benchmark SpinQuant activation MXFP4 QDQ backends.

Compares three implementations used by the vLLM SpinQuant plugin:
1. Quark kernel: quark.torch.kernel.mx.qdq_mxfp4
2. vllm_ext reference: auto_round_extension.vllm_ext.mxfp4_qdq_utils.qdq_mxfp4
3. Plugin fallback: auto_round.vllm_plugin.spinquant_mxfp4._mxfp4_act_qdq_fallback

Usage:
    python examples/benchmark_qdq_kernels.py
    python examples/benchmark_qdq_kernels.py --dtype float16
    python examples/benchmark_qdq_kernels.py --shapes 1x4096,16x4096,256x8192
    python examples/benchmark_qdq_kernels.py --iters 200 --warmup 20 --json
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

from auto_round.vllm_plugin.spinquant_mxfp4 import MXFP4_BLOCK_SIZE, _mxfp4_act_qdq_fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MXFP4 activation QDQ backends.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run on. Default: cuda",
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
        default="1x4096,16x4096,64x4096,256x4096,1024x4096,256x8192",
        help="Comma-separated shapes in the form MxK. Default: 1x4096,16x4096,64x4096,256x4096,1024x4096,256x8192",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per backend/shape. Default: 10",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark iterations per backend/shape. Default: 100",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def parse_shapes(spec: str) -> list[tuple[int, int]]:
    shapes = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            m_str, k_str = item.split("x", 1)
            shape = (int(m_str), int(k_str))
        except ValueError as exc:
            raise ValueError(f"Invalid shape '{item}', expected MxK") from exc
        shapes.append(shape)
    if not shapes:
        raise ValueError("No valid shapes provided")
    return shapes


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def load_backends(group_size: int) -> tuple[list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]], dict[str, str]]:
    funcs: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = []
    availability: dict[str, str] = {}

    try:
        from quark.torch.kernel import mx as quark_mx

        funcs.append(("quark", lambda x: quark_mx.qdq_mxfp4(x, scale_calculation_mode="even")))
        availability["quark"] = "available"
    except Exception as exc:  # pragma: no cover - environment dependent
        availability["quark"] = f"unavailable: {exc!r}"

    try:
        from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4 as vllm_ext_qdq

        funcs.append(("vllm_ext", vllm_ext_qdq))
        availability["vllm_ext"] = "available"
    except Exception as exc:  # pragma: no cover - environment dependent
        availability["vllm_ext"] = f"unavailable: {exc!r}"

    funcs.append(("fallback", lambda x: _mxfp4_act_qdq_fallback(x, group_size)))
    availability["fallback"] = "available"
    return funcs, availability


@torch.inference_mode()
def benchmark_backend(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
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
        "mean": round(statistics.mean(elapsed), 4),
        "median": round(statistics.median(elapsed), 4),
        "min": round(min(elapsed), 4),
    }


def print_report(results: dict) -> None:
    print(f"device: {results['device_name']}")
    print(f"dtype:  {results['dtype']}")
    print(f"group:  {results['group_size']}")
    print()
    for name, status in results["availability"].items():
        print(f"{name:>8}: {status}")

    print()
    for bench in results["benchmarks"]:
        print(f"shape={tuple(bench['shape'])}")
        for backend, timing in bench["timings_ms"].items():
            rel = bench["relative_to_fastest"][backend]
            print(
                f"  {backend:>8}: mean={timing['mean']:.4f} ms  "
                f"median={timing['median']:.4f} ms  min={timing['min']:.4f} ms  "
                f"rel={rel:.3f}x"
            )
        print()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not args.device.startswith("cuda"):
        raise ValueError(f"This benchmark expects a CUDA device, got {args.device}")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    shapes = parse_shapes(args.shapes)
    funcs, availability = load_backends(args.group_size)

    results = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "group_size": args.group_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "availability": availability,
        "benchmarks": [],
    }

    for shape in shapes:
        if shape[-1] % args.group_size != 0:
            raise ValueError(
                f"Shape {shape} has last dim not divisible by group_size={args.group_size}"
            )
        x = torch.randn(*shape, device=device, dtype=dtype)
        bench = {"shape": list(shape), "timings_ms": {}}
        for name, fn in funcs:
            bench["timings_ms"][name] = benchmark_backend(fn, x, args.warmup, args.iters)
        means = {name: stat["mean"] for name, stat in bench["timings_ms"].items()}
        fastest = min(means.values())
        bench["relative_to_fastest"] = {
            name: round(mean / fastest, 3) for name, mean in means.items()
        }
        results["benchmarks"].append(bench)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print_report(results)


if __name__ == "__main__":
    main()
