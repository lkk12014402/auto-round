// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Vendored from AMD Quark for the local auto-round MXFP4 activation QDQ kernel.
// This file is only included from CUDA (.cu) translation units.

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <cstdio>

// ═══════════════════════════════════════════════════════════════════════
// Floating-point format constants
// ═══════════════════════════════════════════════════════════════════════

#define FLOAT16_MANTISSA_BITS 10
#define FLOAT16_EXP_BITS 5
#define FLOAT16_EXP_BIAS 15

#define FLOAT4_MANTISSA_BITS 1
#define FLOAT4_EXP_BITS 2
#define FLOAT4_EXP_BIAS 1

#define FLOAT8_E8M0_MAX_EXP 127

#define BFLOAT16_MANTISSA_BITS 7
#define BFLOAT16_EXP_BITS 8
#define BFLOAT16_EXP_BIAS 127

#define FLOAT16_VAL_TO_ADD \
  (1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define FLOAT16_SIGN_EXPONENT_MASK \
  (((1 << (FLOAT16_EXP_BITS + 1)) - 1) << FLOAT16_MANTISSA_BITS)

#define BFLOAT16_VAL_TO_ADD \
  (1 << (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define BFLOAT16_SIGN_EXPONENT_MASK \
  (((1 << (BFLOAT16_EXP_BITS + 1)) - 1) << BFLOAT16_MANTISSA_BITS)

// ═══════════════════════════════════════════════════════════════════════
// Device helper functions
// ═══════════════════════════════════════════════════════════════════════

template <typename T>
__device__ int bf16_or_half2int_rn(const T h);

template <typename T>
__device__ T float_to_bf16_or_half(const float x);

template <typename T>
__device__ T shfl_xor_bf16_or_half(T x, int laneMask);

template <>
inline __device__ int bf16_or_half2int_rn(const __half h) {
  return __half2int_rn(h);
}

template <>
inline __device__ int bf16_or_half2int_rn(const __nv_bfloat16 h) {
  return __float2int_rn(__bfloat162float(h));
}

template <>
inline __device__ __half float_to_bf16_or_half(const float x) {
  return __float2half(x);
}

template <>
inline __device__ __nv_bfloat16 float_to_bf16_or_half(const float x) {
  return __float2bfloat16(x);
}

template <>
inline __device__ __half shfl_xor_bf16_or_half(__half x, int laneMask) {
  return __shfl_xor_sync(0xffffffff, x, laneMask);
}

template <>
inline __device__ __nv_bfloat16 shfl_xor_bf16_or_half(__nv_bfloat16 x, int laneMask) {
  return __ushort_as_bfloat16(
    __shfl_xor_sync(0xffffffff, __bfloat16_as_ushort(x), laneMask)
  );
}

template <typename float_type>
__device__ float_type hmul_impl(float_type a, float_type b) {
  return __hmul(a, b);
}

template <typename float_type>
__device__ float_type hdiv_impl(float_type a, float_type b) {
  return __hdiv(a, b);
}

template <typename float_type>
__device__ float_type hmax_impl(float_type a, float_type b) {
  return __hmax(a, b);
}

template <typename float_type>
__device__ float_type habs_impl(float_type a) {
  return __habs(a);
}

template <typename float_type>
__device__ float_type hlog2_impl(float_type a) {
  return hlog2(a);
}

template <typename float_type>
__device__ float_type hfloor_impl(float_type a) {
  return hfloor(a);
}
