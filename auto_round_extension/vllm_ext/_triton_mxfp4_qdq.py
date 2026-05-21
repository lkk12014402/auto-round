#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Vendored from AMD Quark (quark/torch/kernel/mx/triton.py) for MXFP4 activation QDQ.
# Only the subset needed for qdq_mxfp4_triton() is included.
#
# type: ignore

from enum import Enum

import torch
import triton
import triton.language as tl

# Adopted and modified from
# https://github.com/triton-lang/triton/blob/main/bench/triton_bench/numerics_details/mxfp.py
# -----------------------------------------------------------------------------
#                      Dequantization / Quantization Utilities
# -----------------------------------------------------------------------------


def get_max_quant_val(dtype: torch.dtype):
    d = {torch.uint8: 6.0, torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 448.0}
    assert dtype in d
    return d[dtype]


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _get_max_quant_exp(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 2
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _compute_quant_and_scale(
    src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr, DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0
):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    elif DEQUANT_SCALE_ROUNDING_MODE == 1:
        # DequantScaleRoundingMode.ROUND_DOWN
        dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    else:
        # DequantScaleRoundingMode.EVEN
        assert DEQUANT_SCALE_ROUNDING_MODE == 2
        max_val = max_val.to(tl.int32, bitcast=True)
        max_val = (max_val + 0x200000).to(tl.uint32, bitcast=True) & 0x7F800000
        max_val = max_val.to(tl.float32, bitcast=True)
        scale_e8m0_unbiased = tl.log2(max_val).floor() - _get_max_quant_exp(mx_tensor_dtype)
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
        dequant_scale_rounded = tl.exp2(scale_e8m0_unbiased)
        dequant_scale_exponent = dequant_scale_rounded.to(tl.uint32, bitcast=True)

    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    quant_tensor = f32_tensor * quant_scale

    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = quant_tensor & 0x7FFFFF

        E8_BIAS = 127
        E2_BIAS = 1
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas = tl.where(exponents < E8_BIAS, (0x400000 | (mantissas >> 1)) >> adjusted_exponents, mantissas)

        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_quant,
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16), f"{src_dtype=} must be bfloat16 or float16"
    )
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 1 if is_fp8 else 2
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant and mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant and mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, 32)
    full_scale_mask = scale_mask_k and mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(
        src_tensor, full_mask_src, mx_tensor_dtype, DEQUANT_SCALE_ROUNDING_MODE
    )

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


@triton.jit
def _upcast_from_mxfp(
    out_ptr,
    stride_o_outer,
    stride_o_quant: tl.constexpr,
    mx_scale_ptr,
    stride_scale_outer,
    stride_scale_quant,
    mx_tensor_ptr,
    stride_tensor_outer,
    stride_tensor_quant: tl.constexpr,
    outer_dim,
    quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
):
    tl.static_assert(stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % 32 == 0, "BLOCK_SIZE_K must be a multiple of 32")
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert(dst_dtype == tl.float16 or dst_dtype == tl.bfloat16)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
        "mx_tensor_ptr must be uint8",
    )
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")

    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    K_DIVISOR: tl.constexpr = 1 if is_fp8 else 2
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 32
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr += start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant and mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant and mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, 32)
    full_scale_mask = mask_scale and mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        tl.static_assert(dst_dtype == tl.float16)
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        dst_scale = dst_scale.to(tl.float16)

    if is_fp8:
        dst_tensor = tensor.to(dst_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if dst_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10

            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(dst_dtype, bitcast=True),
                dst_tensor,
            )
    else:
        dst_bias: tl.constexpr = 127 if dst_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if dst_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if dst_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(dst_dtype, bitcast=True)

    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)


# =============================================================================
# Python wrappers
# =============================================================================


class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1
    EVEN = 2


def axis_permute_order(ndim: int, axis: int, swizzle_axis: int | None = None) -> list[int]:
    permute_order = list(range(ndim))
    permute_order[axis], permute_order[-1] = permute_order[-1], permute_order[axis]

    scale_permute_order = permute_order.copy()
    if swizzle_axis is not None:
        axis = axis if axis >= 0 else axis + ndim
        swizzle_axis = swizzle_axis if swizzle_axis >= 0 else swizzle_axis + ndim
        if swizzle_axis == ndim - 1:
            swizzle_axis = axis
        scale_permute_order[swizzle_axis], scale_permute_order[-2] = (
            scale_permute_order[-2],
            scale_permute_order[swizzle_axis],
        )

    convert_order = [i for i, (a, b) in enumerate(zip(permute_order, scale_permute_order, strict=False)) if a != b]
    assert len(convert_order) == 0 or len(convert_order) == 2, (
        "Exactly 0 or 1 swap should be required to transform permute_order to scale_permute_order."
    )
    return permute_order, scale_permute_order, convert_order


def permute_shape(shape: tuple[int, ...], permute_order: list[int]) -> tuple[int, ...]:
    return tuple(shape[i] for i in permute_order)


def downcast_to_mxfp(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    swizzle_axis: int | None = None,
    out_quant_tensor: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
    BLOCK_OUT_DIM: int = 128,
    BLOCK_QUANT_DIM: int = 32,
):
    """Convert src tensor to MXFP format (quantization along axis dimension)."""
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    L = src_tensor.shape[axis]
    if out_quant_type == torch.uint8:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    is_fp8 = out_quant_type == torch.float8_e4m3fn or out_quant_type == torch.float8_e5m2
    divisor = 1 if is_fp8 else 2
    device = src_tensor.device

    packed_quant_dim = triton.cdiv(L, divisor)
    out_scale_dim = triton.cdiv(L, 32)

    permute_order, scale_permute_order, convert_order = axis_permute_order(ndim, axis, swizzle_axis)

    prmted_quant_tensor_shape = permute_shape(src_tensor.shape, permute_order)[:-1] + (packed_quant_dim,)
    prmted_scale_shape = permute_shape(src_tensor.shape, scale_permute_order)[:-1] + (out_scale_dim,)
    prmted_src_tensor = src_tensor.permute(permute_order)

    if out_quant_tensor is None:
        out_quant_tensor = torch.empty(prmted_quant_tensor_shape, dtype=out_quant_type, device=device)
    else:
        out_quant_tensor = out_quant_tensor.permute(permute_order)

    if out_scale is None:
        out_scale = torch.empty(prmted_scale_shape, dtype=torch.uint8, device=device)
    else:
        out_scale = out_scale.permute(scale_permute_order)

    unpadded_out_scale = out_scale

    reshaped_src_tensor = prmted_src_tensor.reshape(-1, L)
    blocks_quant_dim = triton.cdiv(reshaped_src_tensor.shape[-1], BLOCK_QUANT_DIM)
    blocks_out_dim = triton.cdiv(reshaped_src_tensor.shape[0], BLOCK_OUT_DIM)

    kernel_quant_tensor = out_quant_tensor.reshape(-1, packed_quant_dim)
    kernel_scale = unpadded_out_scale.reshape(-1, out_scale_dim)

    _downcast_to_mxfp[(blocks_out_dim, blocks_quant_dim)](
        kernel_quant_tensor,
        *kernel_quant_tensor.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        reshaped_src_tensor,
        *reshaped_src_tensor.stride(),
        *reshaped_src_tensor.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        DEQUANT_SCALE_ROUNDING_MODE.value,
        num_warps=8,
    )

    out_quant_tensor = out_quant_tensor.permute(permute_order)
    out_scale = out_scale.permute(permute_order).contiguous()
    return out_quant_tensor, out_scale, permute_shape(prmted_scale_shape, scale_permute_order)


def upcast_from_mxfp(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
    swizzle_axis: int | None = None,
    BLOCK_OUT_DIM: int = 128,
    BLOCK_QUANT_DIM: int = 32,
):
    """Upcast MXFP (packed) tensor back to float16 or bfloat16."""
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    multiplier = 1 if "float8" in str(tensor.dtype) else 2
    logical_quant_dim_shape = tensor.shape[axis] * multiplier
    assert tensor.ndim == scale.ndim
    assert tensor.dtype in {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}
    assert scale.dtype == torch.uint8
    assert dtype in {torch.float16, torch.bfloat16}

    permute_order, scale_permute_order, convert_order = axis_permute_order(ndim, axis, swizzle_axis)
    prmt_tensor = tensor.permute(permute_order).contiguous()
    prmt_scale = scale.permute(scale_permute_order).contiguous()

    quant_dim = prmt_tensor.shape[-1]
    reshaped_tensor = prmt_tensor.reshape(-1, quant_dim)
    reshaped_scale = prmt_scale.reshape(-1, prmt_scale.shape[-1])

    outer_dim = reshaped_tensor.shape[0]
    blocks_out_dim = triton.cdiv(outer_dim, BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(logical_quant_dim_shape, BLOCK_QUANT_DIM)

    out = torch.empty((outer_dim, logical_quant_dim_shape), dtype=dtype, device=tensor.device)
    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](
        out,
        out.stride(0),
        out.stride(1),
        reshaped_scale,
        reshaped_scale.stride(0),
        reshaped_scale.stride(1),
        reshaped_tensor,
        reshaped_tensor.stride(0),
        reshaped_tensor.stride(1),
        outer_dim,
        logical_quant_dim_shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        num_warps=8,
    )
    out = out.view(*prmt_tensor.shape[:-1], logical_quant_dim_shape)
    out = out.permute(permute_order)
    return out


def qdq_mxfp4_triton(x: torch.Tensor, scale_calculation_mode: str = "even") -> torch.Tensor:
    """MXFP4 activation quantize-dequantize using Triton kernels.

    This is equivalent to Quark's qdq_mxfp4 with scale_calculation_mode="even".
    """
    if scale_calculation_mode == "even":
        triton_scale_calculation_mode = DequantScaleRoundingMode.EVEN
    else:
        raise NotImplementedError(f"Unsupported scale calculation mode {scale_calculation_mode}")

    with torch.cuda.device(x.device):
        x_mxfp4, scale_e8m0, _ = downcast_to_mxfp(
            x,
            torch.uint8,
            axis=-1,
            swizzle_axis=None,
            out_quant_tensor=None,
            out_scale=None,
            DEQUANT_SCALE_ROUNDING_MODE=triton_scale_calculation_mode,
        )
        x_qdq = upcast_from_mxfp(x_mxfp4, scale_e8m0, x.dtype, axis=-1, swizzle_axis=None)
    return x_qdq
