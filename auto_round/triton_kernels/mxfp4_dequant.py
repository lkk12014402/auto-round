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
Triton kernel for MXFP4 weight dequantization.

Replaces the pure-PyTorch unpack_fp4_from_uint8 + scale multiplication with
a single GPU kernel that:
  1. Unpacks two FP4 values from each uint8 byte
  2. Converts FP4 E2M1 indices to float values
  3. Multiplies by per-group e8m0 shared exponent scale

Storage format:
  - Packed weights: [M, N/2] uint8 — each byte stores two FP4 values
    (low nibble = even column, high nibble = odd column)
  - Scale: [M, N/GROUP_SIZE] uint8 — e8m0 format (biased exponent, bias=127)
"""

import torch
import triton
import triton.language as tl


E8M0_BIAS = tl.constexpr(127)


@triton.jit
def _fp4_e2m1_to_float(idx):
    """Convert a 3-bit magnitude index (E2M1) to its float value.

    E2M1 mapping:
      0 -> 0.0, 1 -> 0.5, 2 -> 1.0, 3 -> 1.5,
      4 -> 2.0, 5 -> 3.0, 6 -> 4.0, 7 -> 6.0
    """
    exp_bits = (idx >> 1) & 0x3
    man_bit = (idx & 0x1).to(tl.float32)
    # Subnormal (exp=0): value = mantissa * 0.5
    # Normal (exp>0): value = (1 + mantissa*0.5) * 2^(exp-1)
    is_subnorm = exp_bits == 0
    normal_val = (1.0 + man_bit * 0.5) * tl.exp2((exp_bits - 1).to(tl.float32))
    subnorm_val = man_bit * 0.5
    return tl.where(is_subnorm, subnorm_val, normal_val)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def _mxfp4_dequant_kernel(
    packed_ptr,   # [M, N_half] uint8 (N_half = N // 2)
    scale_ptr,    # [M, N_groups] uint8 e8m0 (N_groups = N // GROUP_SIZE)
    output_ptr,   # [M, N] output dtype
    M,
    N,            # unpacked columns
    N_half,       # N // 2
    N_groups,     # N // GROUP_SIZE
    stride_pm,    # packed row stride
    stride_sm,    # scale row stride
    stride_om,    # output row stride
    GROUP_SIZE: tl.constexpr,  # 32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,     # must be multiple of GROUP_SIZE
):
    """Dequantize MXFP4 packed weights to float.

    Each program instance handles a BLOCK_M × BLOCK_N tile of the output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks
    mask_m = rm < M
    # cn is always < N because grid is computed with cdiv

    # --- Load packed bytes ---
    # Each packed byte holds two FP4 values: low nibble (even col), high nibble (odd col)
    # For output columns cn, the packed byte index is cn // 2
    packed_col = cn // 2  # [BLOCK_N]
    # Which nibble: 0=low (even col), 1=high (odd col)
    is_high_nibble = (cn % 2).to(tl.int32)

    # Load packed data: [BLOCK_M, BLOCK_N] via gather
    packed_offsets = rm[:, None] * stride_pm + packed_col[None, :]  # [BLOCK_M, BLOCK_N]
    mask_2d = mask_m[:, None] & (cn[None, :] < N)
    packed_bytes = tl.load(packed_ptr + packed_offsets, mask=mask_2d, other=0).to(tl.int32)

    # Extract nibble
    nibble = tl.where(
        is_high_nibble[None, :] == 1,
        (packed_bytes >> 4) & 0x0F,
        packed_bytes & 0x0F,
    )

    # --- Decode FP4 E2M1 ---
    sign_bit = (nibble >> 3) & 0x1  # bit 3 is sign
    mag_idx = nibble & 0x07         # bits 0-2 are magnitude index
    mag_float = _fp4_e2m1_to_float(mag_idx)
    fp4_val = tl.where(sign_bit == 1, -mag_float, mag_float)

    # --- Load and apply e8m0 scale ---
    # Scale is per group of GROUP_SIZE elements
    scale_col = cn // GROUP_SIZE  # [BLOCK_N]
    scale_offsets = rm[:, None] * stride_sm + scale_col[None, :]  # [BLOCK_M, BLOCK_N]
    scale_mask = mask_m[:, None] & (scale_col[None, :] < N_groups)
    scale_e8m0 = tl.load(scale_ptr + scale_offsets, mask=scale_mask, other=127).to(tl.int32)
    # e8m0 -> float: scale = 2^(e8m0 - 127)
    scale_float = tl.exp2((scale_e8m0 - E8M0_BIAS).to(tl.float32))

    # --- Dequantize ---
    result = fp4_val * scale_float

    # --- Store ---
    out_offsets = rm[:, None] * stride_om + cn[None, :]
    tl.store(output_ptr + out_offsets, result, mask=mask_2d)


@torch.compiler.disable()
def triton_mxfp4_dequant(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize MXFP4 packed weights using a Triton kernel.

    Args:
        packed_weight: [M, N/2] uint8 tensor — two FP4 values per byte.
        weight_scale: [M, N/GROUP_SIZE] uint8 tensor — e8m0 shared exponents.
        output_dtype: Desired output dtype (bfloat16 or float16).
        group_size: Number of elements sharing one scale (default 32).

    Returns:
        Dequantized weight tensor of shape [M, N] in output_dtype.
    """
    assert packed_weight.device.type == "cuda", "triton_mxfp4_dequant requires CUDA tensors"
    assert packed_weight.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8

    M, N_half = packed_weight.shape
    N = N_half * 2
    N_groups = N // group_size

    assert weight_scale.shape == (M, N_groups), (
        f"Scale shape mismatch: expected ({M}, {N_groups}), got {weight_scale.shape}"
    )

    # Ensure contiguous
    packed_weight = packed_weight.contiguous()
    weight_scale = weight_scale.contiguous()

    # Allocate output in float32 for kernel, then cast
    output = torch.empty((M, N), dtype=torch.float32, device=packed_weight.device)

    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _mxfp4_dequant_kernel[grid](
        packed_ptr=packed_weight,
        scale_ptr=weight_scale,
        output_ptr=output,
        M=M,
        N=N,
        N_half=N_half,
        N_groups=N_groups,
        stride_pm=packed_weight.stride(0),
        stride_sm=weight_scale.stride(0),
        stride_om=output.stride(0),
        GROUP_SIZE=group_size,
    )

    return output.to(output_dtype)


def triton_mxfp4_dequant_ref(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> torch.Tensor:
    """Reference (PyTorch) implementation for correctness validation."""
    from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8

    M, N_half = packed_weight.shape
    N = N_half * 2

    # Unpack FP4 values
    unpacked = unpack_fp4_from_uint8(packed_weight, M, N, dtype=torch.float32)

    # Apply e8m0 scale
    scale_e8m0 = weight_scale.view(torch.uint8).to(torch.int16)
    scale_float = torch.pow(2.0, (scale_e8m0.float() - 127.0))
    # Expand scale to match weight shape
    scale_expanded = scale_float.unsqueeze(-1).expand(-1, -1, group_size).reshape(M, N)
    result = unpacked * scale_expanded

    return result.to(output_dtype)
