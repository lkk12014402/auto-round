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

"""
Triton kernels for accelerated MXFP4 inference with rotation support.

This package provides:
- ``mxfp4_dequant``: Standalone MXFP4 weight dequantization kernel
- ``mxfp4_gemm``: Fused dequant + GEMM kernel (avoids materializing full dequantized weights)
- ``rotated_linear``: ``RotatedMXFP4Linear`` module replacing hook-based rotation + MXFP4 inference
"""

from auto_round.triton_kernels.mxfp4_dequant import triton_mxfp4_dequant
from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm
from auto_round.triton_kernels.rotated_linear import (
    RotatedMXFP4Linear,
    patch_mxfp4_forward_triton,
)

__all__ = [
    "triton_mxfp4_dequant",
    "triton_mxfp4_gemm",
    "RotatedMXFP4Linear",
    "patch_mxfp4_forward_triton",
]
