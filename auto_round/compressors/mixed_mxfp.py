# Copyright (c) 2026 Intel Corporation
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

from __future__ import annotations

import copy
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import torch

from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme
from auto_round.utils.common import to_standard_regex
from auto_round.utils import get_block_names, get_layer_names_in_block

MIXED_MXFP_POLICIES = ("dense_mxfp4", "moe_conservative", "moe_balanced")
MIXED_MXFP_POLICY_DEFAULT_SCHEMES = {
    "dense_mxfp4": "MXFP4",
    "moe_conservative": "MXFP8",
    "moe_balanced": "MXFP8",
}

_ATTENTION_ATTR_HINTS = (
    "num_heads",
    "num_attention_heads",
    "num_key_value_heads",
    "attention_head_size",
    "head_dim",
    "embed_dim",
    "all_head_size",
    "rotary_emb",
    "q_norm",
    "k_norm",
)
_ATTENTION_NAME_HINTS = ("attention", "attn")
_ROUTER_CLASS_HINTS = ("router", "switch", "dispatcher")
_ROUTER_NAME_HINTS = ("router", "switch", "dispatcher")
_MODULE_COLLECTION_TYPES = (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)


@dataclass
class MixedMXFPPolicyDefaults:
    policy: str
    default_scheme: str
    layer_config: dict[str, Union[str, dict]] = field(default_factory=dict)
    ignore_layers: list[str] = field(default_factory=list)
    expert_layers: set[str] = field(default_factory=set)
    attention_layers: set[str] = field(default_factory=set)
    router_layers: set[str] = field(default_factory=set)


def normalize_mixed_mxfp_policy(policy: Optional[str]) -> Optional[str]:
    if policy is None:
        return None
    policy = policy.strip().lower()
    if policy not in MIXED_MXFP_POLICIES:
        raise ValueError(f"Unsupported mixed_mxfp_policy '{policy}'. Expected one of {MIXED_MXFP_POLICIES}.")
    return policy


def resolve_mixed_mxfp_policy_scheme(
    policy: Optional[str],
    scheme: Union[str, QuantizationScheme, object],
) -> Union[str, QuantizationScheme, object]:
    """Return the effective base scheme for a mixed MXFP policy."""

    policy = normalize_mixed_mxfp_policy(policy)
    if policy is None:
        return scheme
    if not isinstance(scheme, (str, QuantizationScheme)):
        raise ValueError("mixed_mxfp_policy only supports preset-string or QuantizationScheme base schemes.")

    default_scheme = MIXED_MXFP_POLICY_DEFAULT_SCHEMES[policy]
    if isinstance(scheme, str) and scheme.upper() == "W4A16":
        logger.info("mixed_mxfp_policy=%s selected without an explicit MXFP scheme; using %s.", policy, default_scheme)
        return default_scheme

    resolved = asdict(preset_name_to_scheme(scheme.upper())) if isinstance(scheme, str) else asdict(scheme)
    bits = resolved.get("bits")
    group_size = resolved.get("group_size")
    data_type = (resolved.get("data_type") or "").lower()
    expected_bits = 4 if policy == "dense_mxfp4" else 8
    if bits == expected_bits and group_size == 32 and data_type == "mx_fp":
        return scheme

    raise ValueError(
        f"mixed_mxfp_policy={policy} requires an MXFP base scheme equivalent to "
        f"{default_scheme} (bits={expected_bits}, data_type='mx_fp', group_size=32)."
    )


def merge_mixed_mxfp_policy_defaults(
    policy_defaults: Optional[MixedMXFPPolicyDefaults],
    user_layer_config: Optional[dict],
    user_ignore_layers: str,
) -> tuple[dict, str]:
    """Merge policy defaults beneath explicit user overrides."""

    merged_layer_config = copy.deepcopy(policy_defaults.layer_config) if policy_defaults is not None else {}
    explicit_layer_config = copy.deepcopy(user_layer_config) or {}
    merged_layer_config.update(explicit_layer_config)
    explicit_layer_patterns = [re.compile(to_standard_regex(name)) for name in explicit_layer_config]

    ignore_items: list[str] = []
    seen = set()
    for raw_item in (user_ignore_layers or "").replace(" ", "").split(","):
        if raw_item and raw_item not in seen:
            ignore_items.append(raw_item)
            seen.add(raw_item)

    if policy_defaults is not None:
        for name in policy_defaults.ignore_layers:
            if any(pattern.search(name) for pattern in explicit_layer_patterns):
                continue
            if name not in seen:
                ignore_items.append(name)
                seen.add(name)
    return merged_layer_config, ",".join(ignore_items)


def build_mixed_mxfp_policy_defaults(
    model: torch.nn.Module,
    policy: Optional[str],
    *,
    supported_types: tuple,
    inner_supported_types: tuple,
    quant_block_list=None,
) -> Optional[MixedMXFPPolicyDefaults]:
    """Inspect the selected transformer blocks and build mixed MXFP defaults."""

    policy = normalize_mixed_mxfp_policy(policy)
    if policy is None:
        return None

    defaults = MixedMXFPPolicyDefaults(policy=policy, default_scheme=MIXED_MXFP_POLICY_DEFAULT_SCHEMES[policy])
    if policy == "dense_mxfp4":
        return defaults

    selected_layers = get_layer_names_in_block(
        model,
        supported_types=supported_types,
        quant_block_list=quant_block_list,
        class_names=inner_supported_types,
    )
    if not selected_layers:
        return defaults

    module_map = dict(model.named_modules())
    selected_layer_set = set(selected_layers)
    descendants_by_ancestor = _build_selected_descendants(selected_layers)
    block_names = [name for group in (quant_block_list or get_block_names(model)) for name in group]
    block_prefixes = tuple(block_names)

    expert_collections = set()
    for module_name, module in module_map.items():
        if not module_name or not _is_in_selected_blocks(module_name, block_prefixes):
            continue
        child_entries = []
        for child_name, child in module.named_children():
            full_name = _join_name(module_name, child_name)
            child_layers = descendants_by_ancestor.get(full_name, [])
            if child_layers:
                child_entries.append((full_name, child, child_layers))
        if len(child_entries) < 2 or not _children_look_like_experts(module, child_entries):
            continue

        parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        parent_module = module_map.get(parent_name)
        if parent_module is None:
            continue
        router_layers = set()
        for sibling_name, sibling in parent_module.named_children():
            full_name = _join_name(parent_name, sibling_name)
            if full_name == module_name:
                continue
            if _is_router_candidate(full_name, sibling, len(child_entries), descendants_by_ancestor):
                router_layers.update(_get_target_names(full_name, sibling, descendants_by_ancestor))
        if not router_layers:
            continue

        expert_collections.add(module_name)
        defaults.router_layers.update(router_layers)
        for _, _, child_layers in child_entries:
            defaults.expert_layers.update(child_layers)

    for layer_name in sorted(defaults.expert_layers):
        defaults.layer_config[layer_name] = "MXFP4"
    defaults.ignore_layers.extend(sorted(defaults.router_layers))

    if policy == "moe_conservative":
        expert_prefixes = tuple(name for name in expert_collections)
        for module_name, module in module_map.items():
            if not module_name or not _is_in_selected_blocks(module_name, block_prefixes):
                continue
            if expert_prefixes and any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in expert_prefixes):
                continue
            candidate_layers = [name for name in descendants_by_ancestor.get(module_name, []) if name in selected_layer_set]
            candidate_layers = [name for name in candidate_layers if name not in defaults.expert_layers]
            if len(candidate_layers) >= 2 and _is_attention_candidate(module_name, module):
                defaults.attention_layers.update(candidate_layers)
        defaults.ignore_layers.extend(sorted(defaults.attention_layers))

    defaults.ignore_layers = list(dict.fromkeys(defaults.ignore_layers))
    return defaults


def _join_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _build_selected_descendants(selected_layers: list[str]) -> dict[str, list[str]]:
    descendants = defaultdict(list)
    for layer_name in selected_layers:
        current = layer_name
        while True:
            descendants[current].append(layer_name)
            if "." not in current:
                break
            current = current.rsplit(".", 1)[0]
        descendants[""].append(layer_name)
    return descendants


def _is_in_selected_blocks(name: str, block_prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in block_prefixes)


def _children_look_like_experts(
    collection_module: torch.nn.Module,
    child_entries: list[tuple[str, torch.nn.Module, list[str]]],
) -> bool:
    """Return True when a module's children structurally resemble repeated experts.

    The detector prefers explicit collections (`ModuleList`/`Sequential`/`ModuleDict`).
    For unfused custom wrappers, it falls back to repeated non-leaf children with
    similar types or similar numbers of supported linear descendants.
    """

    if len(child_entries) < 2:
        return False
    if isinstance(collection_module, _MODULE_COLLECTION_TYPES):
        return True
    non_leaf_children = sum(1 for _, child, _ in child_entries if len(list(child.children())) > 0)
    if non_leaf_children < 2:
        return False
    child_types = [type(child).__name__ for _, child, _ in child_entries]
    if len(set(child_types)) == 1:
        return True
    layer_counts = [len(layers) for _, _, layers in child_entries]
    if len(set(layer_counts)) == 1 and min(layer_counts) >= 1:
        return True
    collection_like = any(isinstance(child, _MODULE_COLLECTION_TYPES) for _, child, _ in child_entries)
    return collection_like and min(len(layers) for _, _, layers in child_entries) >= 1


def _is_router_candidate(
    module_name: str,
    module: torch.nn.Module,
    num_experts: int,
    descendants_by_ancestor: dict[str, list[str]],
) -> bool:
    if _weight_matches_num_experts(module, num_experts):
        return True

    class_name = module.__class__.__name__.lower()
    leaf_name = module_name.rsplit(".", 1)[-1].lower()
    if any(token in class_name for token in _ROUTER_CLASS_HINTS):
        return True
    if leaf_name in _ROUTER_NAME_HINTS:
        return True
    if leaf_name == "gate":
        return True
    if _is_gate_router_fallback(leaf_name, module_name, descendants_by_ancestor):
        return True
    return False


def _weight_matches_num_experts(module: torch.nn.Module, num_experts: int) -> bool:
    shapes = []
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        shapes.append(tuple(weight.shape))
    for name, parameter in module.named_parameters(recurse=False):
        if name != "weight" and "weight" not in name:
            continue
        shapes.append(tuple(parameter.shape))
    return any(num_experts in shape for shape in shapes)


def _get_target_names(
    module_name: str,
    module: torch.nn.Module,
    descendants_by_ancestor: dict[str, list[str]],
) -> list[str]:
    if descendants_by_ancestor.get(module_name):
        return list(dict.fromkeys(descendants_by_ancestor[module_name]))
    if len(list(module.children())) == 0:
        return [module_name]
    return []


def _is_attention_candidate(module_name: str, module: torch.nn.Module) -> bool:
    if any(hasattr(module, attr) for attr in _ATTENTION_ATTR_HINTS):
        return True
    class_name = module.__class__.__name__.lower()
    if any(token in class_name for token in _ATTENTION_NAME_HINTS):
        return True
    leaf_name = module_name.rsplit(".", 1)[-1].lower()
    return any(token in leaf_name for token in _ATTENTION_NAME_HINTS)


def _is_gate_router_fallback(
    leaf_name: str,
    module_name: str,
    descendants_by_ancestor: dict[str, list[str]],
) -> bool:
    return "gate" in leaf_name and not leaf_name.endswith("gate_proj") and not descendants_by_ancestor.get(module_name)
