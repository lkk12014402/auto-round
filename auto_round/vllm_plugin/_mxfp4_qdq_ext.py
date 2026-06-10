# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import _get_build_directory, load

logger = logging.getLogger("vllm")

_EXT_MODULE: Any | None = None
_LOAD_ATTEMPTED = False


def _load_extension() -> Any | None:
    global _EXT_MODULE, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _EXT_MODULE
    _LOAD_ATTEMPTED = True

    if not torch.cuda.is_available():
        return None
    if os.environ.get("AUTO_ROUND_DISABLE_LOCAL_MXFP4_QDQ", "0") == "1":
        logger.info("Disabled local MXFP4 QDQ extension via AUTO_ROUND_DISABLE_LOCAL_MXFP4_QDQ=1")
        return None

    source_dir = Path(__file__).resolve().parent / "csrc" / "mxfp4_qdq"
    sources = [
        str(source_dir / "binding.cpp"),
        str(source_dir / "fake.cu"),
    ]
    extra_cflags = ["-O2"]
    extra_cuda_cflags = ["-O2", "--extended-lambda"]
    name = "auto_round_mxfp4_qdq_ext"

    try:
        build_dir = _get_build_directory(name, verbose=False)
        logger.info(
            "Building local MXFP4 QDQ extension at %s (first build may take a while)",
            build_dir,
        )
        _EXT_MODULE = load(
            name=name,
            sources=sources,
            build_directory=build_dir,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=[str(source_dir)],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Failed to build local MXFP4 QDQ extension: %r", exc)
        _EXT_MODULE = None
    return _EXT_MODULE


def is_available() -> bool:
    return _load_extension() is not None


def qdq_mxfp4(x: torch.Tensor, group_size: int = 32) -> torch.Tensor | None:
    if group_size != 32:
        return None
    # CUDA kernel uses warp-level shuffles requiring numel % 64 == 0
    if x.numel() % 64 != 0:
        return None
    ext = _load_extension()
    if ext is None:
        return None
    return ext.qdq_mxfp4(x.contiguous(), group_size)
