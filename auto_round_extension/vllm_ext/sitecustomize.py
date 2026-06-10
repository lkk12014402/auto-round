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


def register_auto_round_vllm_plugin():
    """Entry point for vLLM general_plugins.

    Called automatically by vLLM's plugin loading mechanism when
    'auto_round_extension' is registered under the 'vllm.general_plugins'
    entry point group.
    """
    VLLM_ENABLE_AR_EXT = os.environ.get("VLLM_ENABLE_AR_EXT", "") in [
        "1",
        "true",
        "True",
    ]

    if not VLLM_ENABLE_AR_EXT:
        print("*****************************************************************************")
        print(
            "* Sitecustomize is loaded, but VLLM_ENABLE_AR_EXT is not set, skipping auto_round_vllm_extension *"
        )
        print("*****************************************************************************")
        return

    print("*****************************************************************************")
    print(f"* !!! VLLM_ENABLE_AR_EXT is set to {VLLM_ENABLE_AR_EXT}, applying auto_round_vllm_extension *")
    print("*****************************************************************************")

    # Register AutoRoundExtensionConfig via @register_quantization_config("auto-round")
    # This overrides vLLM's built-in INCConfig mapping for the "auto-round" quant method
    from auto_round_extension.vllm_ext.auto_round_ext import AutoRoundExtensionConfig  # noqa: F401

    from auto_round_extension.vllm_ext.envs_ext import extra_environment_variables  # noqa: F401


# Legacy: support direct import (e.g., `import auto_round_extension.vllm_ext.sitecustomize`)
VLLM_ENABLE_AR_EXT = os.environ.get("VLLM_ENABLE_AR_EXT", "") in [
    "1",
    "true",
    "True",
]

if VLLM_ENABLE_AR_EXT:
    register_auto_round_vllm_plugin()
