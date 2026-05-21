// Copyright (c) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Derived from AMD Quark's MIT-licensed MXFP4 kernel binding.

#include <torch/extension.h>

#include "fake.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qdq_mxfp4", &qdq_mxfp4, "qdq_mxfp4");
  m.def("qdq_mxfp4_", &qdq_mxfp4_, "qdq_mxfp4_");
}
