# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
vLLM out-of-tree plugin for SpinQuant/QuaRot online rotation + MXFP4 inference.

Registers a "spinquant_mxfp4" quantization method that applies:
  - R1: Hadamard rotation on activations before each linear layer
  - MXFP4 weight dequantization + GEMM (via Triton kernel or fallback)

Usage:
    # Register plugin before loading vLLM
    import auto_round.vllm_plugin  # noqa: F401

    # Then use vLLM normally
    from vllm import LLM
    llm = LLM(model="path/to/spinquant_mxfp4_model",
              quantization="spinquant_mxfp4")
"""

from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config
from auto_round.vllm_plugin._weight_loading_patch import apply_weight_loading_patch

# Apply patch so AutoWeightsLoader ignores top-level spinquant rotation matrices
# (e.g. spinquant_R2_head) that don't map to any model module/parameter.
apply_weight_loading_patch()

__all__ = ["SpinQuantMXFP4Config"]
