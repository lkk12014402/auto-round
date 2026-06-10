#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test: Load a SpinQuant MXFP4 model with vLLM using our plugin.

This validates the complete pipeline:
  1. Plugin registration
  2. Model config detection
  3. Weight loading (weight_packed, weight_scale, spinquant buffers)
  4. Inference with online R1 rotation
"""

import sys
import time

sys.path.insert(0, "/data/lkk/quarot/latest/new_commit/auto-round")

# Register plugin BEFORE importing vllm
import auto_round.vllm_plugin  # noqa: F401

MODEL_PATH = "/data/lkk/quarot/latest/new_commit/examples-new/roundtrip_Qwen3-0.6B_20260511_145203/Qwen3-0.6B-R1-MXFP4/Qwen3-0.6B-mxfp-w4g32"


def test_vllm_inference():
    """Test vLLM inference with SpinQuant MXFP4 model."""
    from vllm import LLM, SamplingParams

    print(f"Loading model: {MODEL_PATH}")
    print("Quantization: spinquant_mxfp4")
    print()

    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        quantization="spinquant_mxfp4",
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Test generation
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Einstein's theory of",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=32)

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    gen_time = time.time() - t0

    total_tokens = 0
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        total_tokens += len(output.outputs[0].token_ids)
        print(f"  Prompt: {prompt!r}")
        print(f"  Output: {generated!r}")
        print()

    throughput = total_tokens / gen_time
    print(f"Generated {total_tokens} tokens in {gen_time:.2f}s ({throughput:.1f} tok/s)")
    print("\n✅ vLLM inference with SpinQuant MXFP4 plugin successful!")


if __name__ == "__main__":
    test_vllm_inference()
