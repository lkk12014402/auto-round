# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
vLLM plugin entry point for SpinQuant MXFP4.

This module is called when loaded as a vLLM general plugin.
It simply registers the SpinQuantMXFP4Config quantization method.

Usage (entry_points in pyproject.toml):
    [project.entry-points."vllm.general_plugins"]
    spinquant_mxfp4 = "auto_round.vllm_plugin.register:register"

Or via VLLM_PLUGINS env var:
    VLLM_PLUGINS=spinquant_mxfp4 python -m vllm.entrypoints.openai.api_server ...

Or simply import before using vLLM:
    import auto_round.vllm_plugin
"""


def register():
    """Register the spinquant_mxfp4 quantization config with vLLM."""
    import logging

    logger = logging.getLogger("vllm")
    # Importing this module triggers @register_quantization_config decorator
    from auto_round.vllm_plugin.spinquant_mxfp4 import SpinQuantMXFP4Config  # noqa: F401
    from auto_round.vllm_plugin._weight_loading_patch import apply_weight_loading_patch

    apply_weight_loading_patch()
    logger.info(
        "[AutoRound] SpinQuant MXFP4 plugin registered. "
        "Capabilities: R1/R4 online rotation, R2 offline (fused), MXFP4 quantization."
    )
