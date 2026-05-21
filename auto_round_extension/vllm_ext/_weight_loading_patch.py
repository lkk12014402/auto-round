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
Monkey-patch for vLLM's AutoWeightsLoader to handle SpinQuant rotation
matrices saved at the model root level (e.g., spinquant_R2_head).

Problem:
  auto-round serializes rotation matrices like `spinquant_R2_head` as
  top-level keys in the safetensors file. vLLM's AutoWeightsLoader
  cannot map these to any model module/parameter and raises ValueError.

Solution:
  Patch AutoWeightsLoader.__init__ to always include "spinquant_R" in
  ignore_unexpected_prefixes. This makes it silently skip unknown keys
  that start with "spinquant_R" instead of raising an error.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCH_APPLIED = False


def apply_weight_loading_patch():
    """Patch AutoWeightsLoader to ignore spinquant rotation matrix keys."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _PATCH_APPLIED = True

    try:
        from vllm.model_executor.models.utils import AutoWeightsLoader
    except ImportError:
        logger.warning(
            "Could not import AutoWeightsLoader from vllm. "
            "Weight loading patch not applied."
        )
        return

    _orig_init = AutoWeightsLoader.__init__

    def _patched_init(
        self,
        module,
        *,
        skip_prefixes=None,
        skip_substrs=None,
        ignore_unexpected_prefixes=None,
        ignore_unexpected_suffixes=None,
    ):
        # Ensure spinquant rotation matrix keys are ignored
        if ignore_unexpected_prefixes is None:
            ignore_unexpected_prefixes = []
        else:
            ignore_unexpected_prefixes = list(ignore_unexpected_prefixes)

        # Add spinquant rotation matrix prefix if not already present
        if not any(p.startswith("spinquant_R") for p in ignore_unexpected_prefixes):
            ignore_unexpected_prefixes.append("spinquant_R")

        _orig_init(
            self,
            module,
            skip_prefixes=skip_prefixes,
            skip_substrs=skip_substrs,
            ignore_unexpected_prefixes=ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=ignore_unexpected_suffixes,
        )

    AutoWeightsLoader.__init__ = _patched_init
    logger.info(
        "Applied SpinQuant weight loading patch: "
        "AutoWeightsLoader will ignore 'spinquant_R*' top-level keys."
    )

