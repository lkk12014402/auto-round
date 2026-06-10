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

from copy import deepcopy
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config

# vLLM 0.19+: "auto-round" is mapped to INCConfig; older versions had a
# standalone AutoRoundConfig module. We try the old import first for compat.
try:
    from vllm.model_executor.layers.quantization.auto_round import AutoRoundConfig as _BaseConfig
except (ImportError, ModuleNotFoundError):
    from vllm.model_executor.layers.quantization.inc import INCConfig as _BaseConfig

from auto_round.schemes import QuantizationScheme
from auto_round_extension.vllm_ext.quant_method_linear import AutoRoundQuantLinearMethod
from auto_round_extension.vllm_ext.quant_method_moe import AutoRoundMoEMethod

logger = init_logger(__name__)


@register_quantization_config("auto-round")
class AutoRoundExtensionConfig(_BaseConfig):
    SUPPORTED_DTYPES = _BaseConfig.SUPPORTED_DTYPES.union({"mx_fp"})
    SUPPORTED_FORMATS = getattr(_BaseConfig, "SUPPORTED_FORMATS", {"auto_round:auto_gptq"}).union(
        {"auto_round:llm_compressor"}
    )

    @classmethod
    def get_name(cls) -> str:
        return "auto-round"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> "str | None":
        """Claim the 'auto-round' quant method before INCConfig can override it."""
        if hf_quant_cfg.get("quant_method", None) == "auto-round":
            return cls.get_name()
        return None

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
        act_bits: int = None,
        act_data_type: int = None,
        act_dynamic: bool = None,
        act_group_size: int = None,
        act_sym: bool = None,
    ) -> None:
        super().__init__(
            weight_bits=weight_bits,
            group_size=group_size,
            sym=sym,
            packing_format=packing_format,
            block_name_to_quantize=block_name_to_quantize,
            extra_config=extra_config,
            data_type=data_type,
            backend=backend,
        )
        self.act_bits = act_bits
        self.act_data_type = act_data_type
        self.act_dynamic = act_dynamic
        self.act_group_size = act_group_size
        self.act_sym = act_sym

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        # FIXME: (yi) make it compatible with `AutoRoundConfig`
        try:
            from vllm.attention.layer import Attention, MLAAttention
        except (ImportError, ModuleNotFoundError):
            from vllm.model_executor.layers.attention.attention import Attention
            from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        if isinstance(layer, (Attention, MLAAttention)):

            from auto_round_extension.vllm_ext.kv_cache import AutoRoundKVCacheMethod, AutoRoundKVCacheMethodForMLA

            if isinstance(layer, MLAAttention):
                return AutoRoundKVCacheMethodForMLA(self)

            return AutoRoundKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            quant_method = AutoRoundMoEMethod.get_moe_method(self, layer, prefix)
            return quant_method
        elif isinstance(layer, LinearBase):
            return AutoRoundQuantLinearMethod.get_method(self, layer, prefix)
        else:
            return None

    @staticmethod
    def _parse_quant_scheme(config: dict):
        quant_scheme_attrs = QuantizationScheme.get_attributes()
        filter_config = {key: value for key, value in config.items() if key in quant_scheme_attrs}
        quant_scheme = QuantizationScheme.from_dict(filter_config)
        return quant_scheme

    @classmethod
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundExtensionConfig":
        ar_config = cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(config, ["packing_format"], "auto_round:auto_gptq"),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"], "auto"),
            act_bits=cls.get_from_keys(config, ["act_bits"]),
            act_data_type=cls.get_from_keys(config, ["act_data_type"]),
            act_dynamic=cls.get_from_keys(config, ["act_dynamic"]),
            act_group_size=cls.get_from_keys(config, ["act_group_size"]),
            act_sym=cls.get_from_keys(config, ["act_sym"]),
        )
        # TODO: (yi) refine below implementation
        quant_scheme = AutoRoundExtensionConfig._parse_quant_scheme(config)
        layer_schemes = {}
        extra_config = getattr(ar_config, "extra_config", None)
        if extra_config is not None:
            for layer_name, layer_config in extra_config.items():
                layer_scheme = deepcopy(quant_scheme)
                for key, val in layer_config.items():
                    if key in layer_scheme.keys() and val != layer_scheme[key]:
                        layer_scheme[key] = val
                layer_schemes[layer_name] = layer_scheme
        ar_config.quant_scheme = quant_scheme
        ar_config.layer_schemes = layer_schemes
        return ar_config
