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

import os
from typing import Union

import torch
from vllm.logger import init_logger

from auto_round_extension.vllm_ext.fp4_utils import cast_to_fp4, pack_fp4_to_uint8, unpack_fp4_from_uint8
from auto_round_extension.vllm_ext.utils import _to_mx_rceil, get_fp_scale
from auto_round_extension.vllm_ext._mxfp4_qdq_ext import qdq_mxfp4 as _local_qdq_mxfp4

logger = init_logger(__name__)

# Lazy import for Triton QDQ — avoids triggering Triton compilation at import time.
_triton_qdq = None
_triton_qdq_loaded = False


def _get_triton_qdq():
    """Lazily import the vendored Triton MXFP4 QDQ kernel on first use."""
    global _triton_qdq, _triton_qdq_loaded
    if _triton_qdq_loaded:
        return _triton_qdq
    _triton_qdq_loaded = True
    try:
        from auto_round_extension.vllm_ext._triton_mxfp4_qdq import qdq_mxfp4_triton
        _triton_qdq = qdq_mxfp4_triton
    except (ImportError, RuntimeError) as e:
        logger.debug("Triton MXFP4 QDQ not available: %s", e)
        _triton_qdq = None
    return _triton_qdq

F4_E2M1_MAX = 6.0
F32_EXP_BIAS = 127

F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)


def to_mxfp4_rceil(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
):

    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    # TODO(future PR): consider supporting padding
    assert data_hp.numel() % block_size == 0, f"data size must be multiple of block_size={block_size}"
    data_hp = data_hp.contiguous()
    # calculate the scale in e8m0 format

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), 1)

    # Add an epsilon to prevent the log2 function call for returning -inf
    # where the values are zero.
    eps = F32_MIN_NORMAL * (max_abs == 0).type(max_abs.dtype)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time

    max_pos = F4_E2M1_MAX
    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    data_lp = data_lp.reshape(orig_shape)
    orig_shape = [*orig_shape[:-1], orig_shape[-1] // 2]
    data_lp = cast_to_fp4(data_lp)
    data_lp = pack_fp4_to_uint8(data_lp)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.uint8)
    return scale_e8m0_biased, data_lp


def to_dtype(
    data_lp,
    scale_e8m0,
    elem_dtype,
    block_size,
    target_dtype,
    scale_dtype=None,
    return_scale=False,
):
    orig_shape = data_lp.shape
    last_dim = orig_shape[-1]
    data_lp = data_lp.reshape(-1, last_dim)
    result_shape = orig_shape[:-1] + (last_dim * 2,)
    assert data_lp.is_contiguous(), f"Data must be contiguous, got {data_lp.stride()}"

    assert elem_dtype == "fp4_e2m1", f"Expected 'fp4_e2m1', got {elem_dtype}"

    m, half_n = data_lp.shape
    n = half_n * 2
    data_hp = unpack_fp4_from_uint8(data_lp, m, n, dtype=target_dtype)

    data_hp = data_hp.reshape(-1, block_size)

    if scale_dtype is None:
        scale_dtype = target_dtype
    s_fp = get_fp_scale(scale_e8m0).reshape(-1, 1).to(scale_dtype)
    if return_scale:
        return data_hp.reshape(result_shape), s_fp

    data_hp = data_hp * s_fp
    data_hp = data_hp.reshape(result_shape)

    return data_hp


def run_mxfp4_emulations(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    # TODO: select the rounding mode based on config
    group_size = 32
    # quantize input to (FP4 and interleaved block scale)
    input_scale, x_q = to_mxfp4_rceil(
        data_hp=x,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
    )

    # dequantize input
    x_dq = to_dtype(
        data_lp=x_q,
        scale_e8m0=input_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
    )

    # dequantize weight
    w_dq = to_dtype(
        data_lp=weight,
        scale_e8m0=weight_scale,
        elem_dtype="fp4_e2m1",
        block_size=group_size,
        target_dtype=x.dtype,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    if bias is not None:
        out += bias
    return out


def dequant_mxfp4_to_fp8(data_lp, scale_e8m0):
    data_fp8, scale_float = to_dtype(
        data_lp=data_lp,
        scale_e8m0=scale_e8m0,
        elem_dtype="fp4_e2m1",
        block_size=32,
        target_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.bfloat16,
        return_scale=True,
    )
    return data_fp8, scale_float


def mxfp4_fp8_weight_to_bf16(weight_fp8, scale_bf16):

    origin_shape = weight_fp8.shape
    weight_fp8 = weight_fp8.reshape(-1, 32)
    scale_bf16 = scale_bf16.reshape(-1, 1)
    assert weight_fp8.shape[0] == scale_bf16.shape[0], f"shape mismatch: {weight_fp8.shape} vs {scale_bf16.shape}"
    dequant_weight_bf16 = weight_fp8.to(torch.bfloat16) * scale_bf16
    dequant_weight_bf16 = dequant_weight_bf16.reshape(origin_shape)
    return dequant_weight_bf16


def mxfp4_gemm_with_unpacked_weight(x, weight_fp8, weight_scale_bf16, bias=None):
    x = qdq_mxfp4(x)

    # dequantize weight
    w_dq = mxfp4_fp8_weight_to_bf16(weight_fp8, weight_scale_bf16)
    # matmul
    out = torch.matmul(x, w_dq.t())
    if bias is not None:
        out += bias
    return out


def fp4_121_positive(x: torch.Tensor) -> torch.Tensor:
    step1 = torch.round(2.0 * x) / 2.0
    step2 = torch.round(x)
    step3 = 2.0 * torch.round(x / 2.0)

    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def fp4_121_scaled_even_rounding(x: torch.Tensor) -> torch.Tensor:
    sign = x.sign()
    x_abs = x.abs()
    amax_x = x_abs.max(dim=-1, keepdim=True)[0]
    scale_tmp = torch.floor(torch.log2(amax_x)) - 2.0
    scale_clamp = torch.clamp(scale_tmp, min=-127, max=127)
    scale = torch.pow(2.0, scale_clamp)

    scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)

    x_fp4_abs = fp4_121_positive(x_abs / scale) * scale
    return sign * x_fp4_abs


def _qdq_mxfp4_fallback(
    x: torch.Tensor,
) -> torch.Tensor:
    """Pure-Python MXFP4 activation QDQ fallback (slow, device-agnostic)."""
    block_size = 32
    shape = x.shape
    x = x.reshape(-1, block_size)
    x = fp4_121_scaled_even_rounding(x)
    x = x.reshape(shape)
    return x


_qdq_backend_logged = False


def _log_qdq_backend(name: str) -> None:
    """Log QDQ backend selection once."""
    global _qdq_backend_logged
    if not _qdq_backend_logged:
        logger.info("[AutoRound] MXFP4 activation QDQ backend: %s", name)
        _qdq_backend_logged = True


# https://github.com/Anonymous1252022/fp4-all-the-way/blob/main/experimental/fp4.py
def qdq_mxfp4(
    x: torch.Tensor,
) -> torch.Tensor:
    """MXFP4 activation quantize-dequantize with smart backend dispatch.

    Backend selection priority (unless overridden by AUTO_ROUND_MXFP4_QDQ_BACKEND):
        local_ext (CUDA kernel) > triton > fallback (pure PyTorch)

    Environment variable AUTO_ROUND_MXFP4_QDQ_BACKEND can force a specific backend:
        local_ext | triton | fallback
    """
    block_size = 32

    forced_backend = os.environ.get("AUTO_ROUND_MXFP4_QDQ_BACKEND", "").lower().strip()
    is_cuda_fp16_bf16 = x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)

    # --- forced backend selection ---
    if forced_backend == "local_ext":
        if is_cuda_fp16_bf16:
            local_out = _local_qdq_mxfp4(x, block_size)
            if local_out is not None:
                _log_qdq_backend("local_ext (forced)")
                return local_out
        logger.warning("Forced QDQ backend 'local_ext' unavailable, using fallback")
        return _qdq_mxfp4_fallback(x)
    if forced_backend == "triton":
        triton_fn = _get_triton_qdq()
        if is_cuda_fp16_bf16 and triton_fn is not None:
            _log_qdq_backend("triton (forced)")
            return triton_fn(x, scale_calculation_mode="even")
        logger.warning("Forced QDQ backend 'triton' unavailable, using fallback")
        return _qdq_mxfp4_fallback(x)
    if forced_backend == "fallback":
        _log_qdq_backend("fallback (forced)")
        return _qdq_mxfp4_fallback(x)

    # --- auto selection (priority: local_ext > triton > fallback) ---
    if is_cuda_fp16_bf16:
        local_out = _local_qdq_mxfp4(x, block_size)
        if local_out is not None:
            _log_qdq_backend("local_ext")
            return local_out
    if is_cuda_fp16_bf16:
        triton_fn = _get_triton_qdq()
        if triton_fn is not None:
            _log_qdq_backend("triton")
            return triton_fn(x, scale_calculation_mode="even")
    _log_qdq_backend("fallback (pytorch)")
    return _qdq_mxfp4_fallback(x)
