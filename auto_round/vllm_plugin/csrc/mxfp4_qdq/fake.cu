// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Vendored from AMD Quark for the local auto-round MXFP4 activation QDQ kernel.

#include <torch/extension.h>

#include "common.h"

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__device__ float_type fp16_to_fp4_simulate(float_type* val) {
    uint16_t val_view = *(uint16_t*)val;

    uint16_t exp = val_view >> half_mantissa_bits;
    exp = exp & ((1 << half_exp_bits) - 1);

    bool sign = (val_view >> (half_mantissa_bits + half_exp_bits)) & 1;
    bool mantissa_last = (val_view >> (half_mantissa_bits - 1)) & 1;

    int16_t exp_unbias = exp - half_exp_bias;
    int16_t new_exp = exp_unbias + FLOAT4_EXP_BIAS;
    int16_t exp_shift = (new_exp <= 0) * (1 - new_exp);
    uint16_t tail_bits = min(16, half_mantissa_bits - FLOAT4_MANTISSA_BITS + exp_shift);

    uint16_t mantissa_plus_one = val_view & ((1 << (half_mantissa_bits + 1)) - 1);
    uint16_t half = 1 << (tail_bits - 1);
    uint16_t tail = mantissa_plus_one & ((1 << tail_bits) - 1);

    bool round_close = (tail < half);
    bool round_away = (tail > half);
    bool tie = tail == half;

    uint16_t new_mantissa;
    bool new_mantissa_close = 0;
    uint16_t new_exp_close = 0;
    bool new_mantissa_away = 0;
    uint16_t new_exp_away = 0;
    uint16_t new_exp_tie = 0;

    new_mantissa_close = (new_exp > 0) * mantissa_last;
    new_exp_close = exp;

    new_mantissa_away = (new_exp > 0) && (mantissa_last == 0);
    new_exp_away = exp + ((new_exp <= 0) || (mantissa_last == 1));

    new_exp_tie = (exp > (half_exp_bias - 2)) * (exp + (mantissa_last == 1));

    new_exp = round_away * new_exp_away + round_close * new_exp_close + tie * new_exp_tie;
    new_mantissa = round_away * new_mantissa_away + round_close * new_mantissa_close;

    new_mantissa = new_mantissa + (new_exp > (2 + half_exp_bias)) * (new_mantissa == 0);
    new_exp = (new_exp >= (half_exp_bias - 2)) * max((half_exp_bias - 2), min(new_exp, half_exp_bias + 2));

    uint16_t qdq_val = (sign << 15) + (new_exp << half_mantissa_bits) + (new_mantissa << (half_mantissa_bits - 1));
    float_type result = *(float_type*)(&qdq_val);
    return result;
}

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias, uint16_t val_to_add, uint16_t sign_exponent_mask>
__global__ void qdq_mxfp4_kernel(float_type* inp, float_type* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float_type elem = inp[idx];
    float_type block_max = habs_impl(elem);

    for (int i = 1; i < 32; i*=2) {
        block_max = hmax_impl(block_max, habs_impl(shfl_xor_bf16_or_half(block_max, i)));
    }

    uint16_t block_max_uint = (*(uint16_t*)(&block_max) + val_to_add) & sign_exponent_mask;
    block_max = *(float_type*)(&block_max_uint);

    uint8_t scale_exp = max(
        0,
        FLOAT8_E8M0_MAX_EXP + min(bf16_or_half2int_rn<float_type>(hfloor_impl(hlog2_impl(block_max))) - 2, FLOAT8_E8M0_MAX_EXP)
    );
    float_type scale = float_to_bf16_or_half<float_type>(powf(2.0, scale_exp - FLOAT8_E8M0_MAX_EXP));

    elem = hdiv_impl(elem, scale);
    float_type elem_fp4 = fp16_to_fp4_simulate<float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(&elem);
    out[idx] = hmul_impl(elem_fp4, scale);
}

void qdq_mxfp4_(torch::Tensor a, int group_size) {
    int block_size;

    at::DeviceGuard device_guard(a.device());
    int numel = a.numel();

    if (numel % 128 == 0) {
        block_size = 128;
    } else if (numel % 64 == 0) {
        block_size = 64;
    } else {
        TORCH_CHECK(1 == 0, "Expected qdq_mxfp4 input number of elements to be a multiple of 64, but it is not!");
    }

    dim3 dimGrid(numel / block_size, 1, 1);
    dim3 dimBlock(block_size, 1, 1);

    TORCH_CHECK(group_size == 32, "Expected group_size=32 in qdq_mxfp4_!");
    TORCH_CHECK(a.is_cuda(), "Expected qdq_mxfp4_ input to be on CUDA!");
    TORCH_CHECK(a.is_contiguous(), "Expected qdq_mxfp4_ input to be contiguous!");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Half) {
        qdq_mxfp4_kernel<__half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS, FLOAT16_VAL_TO_ADD, FLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__half*) a.data_ptr(), (__half*) a.data_ptr());
    }
    else if (a.scalar_type() == at::ScalarType::BFloat16) {
        qdq_mxfp4_kernel<__nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS, BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__nv_bfloat16*) a.data_ptr(), (__nv_bfloat16*) a.data_ptr());
    }
    else {
        TORCH_CHECK(false, "Wrong input dtype in qdq_mxfp4_!");
    }
}

torch::Tensor qdq_mxfp4(torch::Tensor a, int group_size) {
    int block_size;

    at::DeviceGuard device_guard(a.device());
    int numel = a.numel();

    if (numel % 128 == 0) {
        block_size = 128;
    } else if (numel % 64 == 0) {
        block_size = 64;
    } else {
        TORCH_CHECK(1 == 0, "Expected qdq_mxfp4 input number of elements to be a multiple of 64, but it is not!");
    }

    dim3 dimGrid(numel / block_size, 1, 1);
    dim3 dimBlock(block_size, 1, 1);

    TORCH_CHECK(group_size == 32, "Expected group_size=32 in qdq_mxfp4!");
    TORCH_CHECK(a.is_cuda(), "Expected qdq_mxfp4 input to be on CUDA!");
    TORCH_CHECK(a.is_contiguous(), "Expected qdq_mxfp4 input to be contiguous!");

    torch::Tensor out = at::empty(a.sizes(), a.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Half) {
        qdq_mxfp4_kernel<__half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS, FLOAT16_VAL_TO_ADD, FLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__half*) a.data_ptr(), (__half*) out.data_ptr());
    }
    else if (a.scalar_type() == at::ScalarType::BFloat16) {
        qdq_mxfp4_kernel<__nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS, BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__nv_bfloat16*) a.data_ptr(), (__nv_bfloat16*) out.data_ptr());
    }
    else {
        TORCH_CHECK(false, "Wrong input dtype in qdq_mxfp4!");
    }

    return out;
}
