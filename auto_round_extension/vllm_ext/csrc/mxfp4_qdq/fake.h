// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Vendored from AMD Quark for the local auto-round MXFP4 activation QDQ kernel.
// Forward declarations only — definitions are in fake.cu.

#pragma once

#include <torch/extension.h>

void qdq_mxfp4_(torch::Tensor a, int group_size);
torch::Tensor qdq_mxfp4(torch::Tensor a, int group_size);
