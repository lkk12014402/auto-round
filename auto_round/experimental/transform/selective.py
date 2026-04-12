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

from dataclasses import dataclass
from typing import Callable

import torch

from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.logger import logger
from auto_round.utils import check_to_quantized

__all__ = [
    "collect_linear_input_stats",
    "get_layer_hadamard_configs",
    "resolve_hadamard_layer_selection",
]

_FFN_UP_SUFFIXES = {
    "dense_h_to_4h",
    "dense_h_to_4h_1",
    "dense_h_to_4h_2",
    "fc1",
    "gate_proj",
    "up_proj",
    "w1",
    "w3",
}
_FFN_DOWN_SUFFIXES = {"dense_4h_to_h", "down_proj", "fc2", "w2"}
_Q_SUFFIXES = {"q_proj", "q_proj_linear", "q_lin", "query"}
_K_SUFFIXES = {"k_proj", "k_proj_linear", "k_lin", "key"}
_V_SUFFIXES = {"v_proj", "v_proj_linear", "v_lin", "value"}
_OUT_SUFFIXES = {"c_proj", "dense", "o_proj", "out_proj", "proj_out", "wo"}
_LM_HEAD_SUFFIXES = {"classifier", "embed_out", "lm_head", "output_projection"}


@dataclass
class _ActivationMoments:
    numel: int = 0
    row_count: int = 0
    sum1: float = 0.0
    sum2: float = 0.0
    sum3: float = 0.0
    sum4: float = 0.0
    energy_ratio_sum: float = 0.0

    def update(self, activation: torch.Tensor):
        if not isinstance(activation, torch.Tensor) or activation.ndim == 0:
            return
        if activation.shape[-1] <= 0:
            return

        flat_rows = activation.detach().to(torch.float32).reshape(-1, activation.shape[-1])
        flat = flat_rows.reshape(-1)
        self.numel += flat.numel()
        self.sum1 += flat.sum().item()
        self.sum2 += torch.square(flat).sum().item()
        self.sum3 += torch.pow(flat, 3).sum().item()
        self.sum4 += torch.pow(flat, 4).sum().item()

        row_energy = torch.square(flat_rows).sum(dim=-1).clamp_min(1e-12)
        max_sq = torch.square(flat_rows.abs().amax(dim=-1))
        self.energy_ratio_sum += (max_sq / row_energy).sum().item()
        self.row_count += flat_rows.shape[0]

    def finalize(self) -> dict[str, float]:
        if self.numel == 0:
            return {"kurtosis": 0.0, "energy_ratio": 0.0, "numel": 0, "rows": 0}

        n = float(self.numel)
        mean = self.sum1 / n
        raw2 = self.sum2 / n
        raw3 = self.sum3 / n
        raw4 = self.sum4 / n
        variance = max(raw2 - mean * mean, 0.0)
        if variance <= 1e-12:
            kurtosis = 0.0
        else:
            central4 = raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
            kurtosis = max(central4 / (variance * variance), 0.0)

        energy_ratio = self.energy_ratio_sum / max(self.row_count, 1)
        return {
            "kurtosis": float(kurtosis),
            "energy_ratio": float(energy_ratio),
            "numel": int(self.numel),
            "rows": int(self.row_count),
        }


def _is_attention_path(layer_name: str) -> bool:
    lowered = layer_name.lower()
    return "attn" in lowered or "attention" in lowered


def _detect_role(layer_name: str) -> str:
    suffix = layer_name.split(".")[-1].lower()
    if suffix in _LM_HEAD_SUFFIXES:
        return "lm_head"
    if suffix in _FFN_UP_SUFFIXES:
        return "ffn_up"
    if suffix in _FFN_DOWN_SUFFIXES:
        return "ffn_down"
    if suffix in _Q_SUFFIXES:
        return "attn_q"
    if suffix in _K_SUFFIXES:
        return "attn_k"
    if suffix in _V_SUFFIXES:
        return "attn_v"
    if suffix in _OUT_SUFFIXES and _is_attention_path(layer_name):
        return "attn_out"
    if "qkv" in suffix or "wqkv" in suffix:
        return "attn_qkv"
    return "unknown"


def _infer_head_dim(module: torch.nn.Module, modules_by_name: dict[str, torch.nn.Module], layer_name: str) -> int | None:
    parent_name = layer_name.rpartition(".")[0]
    parent = modules_by_name.get(parent_name)
    if parent is None:
        return None

    head_dim = getattr(parent, "head_dim", None)
    if isinstance(head_dim, int) and head_dim > 0:
        return head_dim

    for attr in ("num_heads", "num_attention_heads"):
        num_heads = getattr(parent, attr, None)
        if isinstance(num_heads, int) and num_heads > 0:
            out_features = getattr(module, "out_features", None)
            if isinstance(out_features, int) and out_features % num_heads == 0:
                return out_features // num_heads

    return None


def _build_structure_info(
    layer_name: str, module: torch.nn.Module, modules_by_name: dict[str, torch.nn.Module]
) -> dict[str, bool | int | str]:
    role = _detect_role(layer_name)
    in_features = getattr(module, "in_features", 0)
    out_features = getattr(module, "out_features", 0)
    is_expand = bool(out_features > in_features)
    is_compress = bool(out_features < in_features)
    has_nonlinearity_after = role == "ffn_up"
    is_residual_consumer = role in {"attn_out", "ffn_down", "lm_head"}
    head_dim = _infer_head_dim(module, modules_by_name, layer_name)
    return {
        "role": role,
        "is_expand": is_expand,
        "is_compress": is_compress,
        "has_nonlinearity_after": has_nonlinearity_after,
        "is_residual_consumer": is_residual_consumer,
        "head_dim": head_dim,
    }


def collect_linear_input_stats(
    model: torch.nn.Module,
    module_names: list[str],
    run_forward: Callable[[], None],
) -> dict[str, dict[str, float]]:
    modules_by_name = dict(model.named_modules())
    moments = {name: _ActivationMoments() for name in module_names if name in modules_by_name}
    handles = []

    for name, moment in moments.items():
        module = modules_by_name[name]

        def _hook(_, args, target_moment=moment):
            if not args:
                return
            activation = args[0]
            target_moment.update(activation)

        handles.append(module.register_forward_pre_hook(_hook, prepend=True))

    try:
        run_forward()
    finally:
        for handle in handles:
            handle.remove()

    return {name: moment.finalize() for name, moment in moments.items()}


def _collect_candidate_layer_names(
    model: torch.nn.Module,
    layer_config: dict,
    config: HadamardConfig,
) -> tuple[dict[str, torch.nn.Module], list[str]]:
    modules_by_name = dict(model.named_modules())
    candidate_names = []

    for layer_name, cfg in layer_config.items():
        module = modules_by_name.get(layer_name)
        if module is None or not check_to_quantized(cfg):
            cfg["hadamard_config"] = None
            continue
        if "lm_head" in layer_name:
            cfg["hadamard_config"] = None
            continue
        in_features = getattr(module, "in_features", None)
        if not isinstance(in_features, int) or in_features <= 0 or in_features % config.block_size != 0:
            cfg["hadamard_config"] = None
            continue
        candidate_names.append(layer_name)

    return modules_by_name, candidate_names


def _score_layer(stats: dict[str, float], structure: dict[str, bool | int | str], config: HadamardConfig) -> float:
    score = 0.0
    kurtosis = float(stats.get("kurtosis", 0.0))
    energy_ratio = float(stats.get("energy_ratio", 0.0))

    if kurtosis > config.kurtosis_threshold:
        score += 1.0
    if energy_ratio > config.energy_ratio_threshold:
        score += 1.0
    if structure["is_expand"]:
        score += 1.0
    if structure["has_nonlinearity_after"]:
        score += 0.5
    if structure["is_residual_consumer"]:
        score -= 1.5
    if structure["is_compress"]:
        score -= 1.0
    return score


def _build_layer_decision(
    layer_name: str,
    stats: dict[str, float],
    module: torch.nn.Module,
    modules_by_name: dict[str, torch.nn.Module],
    config: HadamardConfig,
) -> dict[str, bool | float | str | int | None]:
    structure = _build_structure_info(layer_name, module, modules_by_name)
    role = structure["role"]
    kurtosis = float(stats.get("kurtosis", 0.0))
    energy_ratio = float(stats.get("energy_ratio", 0.0))
    score = _score_layer(stats, structure, config)
    decision = {
        "enabled": False,
        "score": score,
        "kurtosis": kurtosis,
        "energy_ratio": energy_ratio,
        "role": role,
        "expand": bool(structure["is_expand"]),
        "compress": bool(structure["is_compress"]),
        "residual": bool(structure["is_residual_consumer"]),
        "has_nonlinearity_after": bool(structure["has_nonlinearity_after"]),
        "head_dim": structure["head_dim"],
        "reason": "score_below_threshold",
    }

    if role in {"lm_head", "attn_qkv"}:
        decision["reason"] = f"hard_skip:{role}"
    elif role in {"attn_q", "attn_k"}:
        decision["reason"] = "pending_qk_pair"
    elif score >= config.score_threshold:
        decision["enabled"] = True
        decision["reason"] = "score_threshold"

    return decision


def _finalize_qk_pair(q_name: str, k_name: str, decisions: dict[str, dict], config: HadamardConfig):
    q_head_dim = decisions[q_name]["head_dim"]
    k_head_dim = decisions[k_name]["head_dim"]
    if (isinstance(q_head_dim, int) and q_head_dim < config.min_qk_head_dim) or (
        isinstance(k_head_dim, int) and k_head_dim < config.min_qk_head_dim
    ):
        decisions[q_name]["reason"] = "head_dim_too_small"
        decisions[k_name]["reason"] = "head_dim_too_small"
        return

    pair_score = (float(decisions[q_name]["score"]) + float(decisions[k_name]["score"])) / 2.0
    if pair_score >= config.score_threshold:
        decisions[q_name]["enabled"] = True
        decisions[k_name]["enabled"] = True
        decisions[q_name]["reason"] = "paired_qk_score"
        decisions[k_name]["reason"] = "paired_qk_score"
    else:
        decisions[q_name]["reason"] = "paired_qk_score_below_threshold"
        decisions[k_name]["reason"] = "paired_qk_score_below_threshold"


def _compute_layer_decisions(
    candidate_names: list[str],
    stats_by_layer: dict[str, dict[str, float]],
    modules_by_name: dict[str, torch.nn.Module],
    config: HadamardConfig,
) -> dict[str, dict]:
    decisions = {}
    qk_groups = {}

    for layer_name in candidate_names:
        module = modules_by_name[layer_name]
        stats = stats_by_layer.get(layer_name, {})
        decision = _build_layer_decision(layer_name, stats, module, modules_by_name, config)
        if decision["role"] in {"attn_q", "attn_k"}:
            parent_name = layer_name.rpartition(".")[0]
            qk_groups.setdefault(parent_name, {})[decision["role"]] = layer_name
        decisions[layer_name] = decision

    for parent_name, group in qk_groups.items():
        q_name = group.get("attn_q")
        k_name = group.get("attn_k")
        if q_name is None or k_name is None:
            lone_name = q_name or k_name
            decisions[lone_name]["reason"] = "missing_qk_pair"
            decisions[lone_name]["enabled"] = False
            continue
        _finalize_qk_pair(q_name, k_name, decisions, config)

    return decisions


def _apply_layer_decisions(
    layer_config: dict,
    decisions: dict[str, dict],
    materialized_config: dict[str, int | str],
):
    for layer_name, cfg in layer_config.items():
        if decisions.get(layer_name, {}).get("enabled"):
            cfg["hadamard_config"] = dict(materialized_config)
        else:
            cfg["hadamard_config"] = None


def _log_layer_decision(layer_name: str, decision: dict):
    logger.info(
        "Layer: %s | role=%s | kurtosis=%.3f | ecr=%.4f | expand=%s | residual=%s | "
        "head_dim=%s | score=%.2f | hadamard=%s (%s)",
        layer_name,
        decision["role"],
        decision["kurtosis"],
        decision["energy_ratio"],
        decision["expand"],
        decision["residual"],
        decision["head_dim"],
        decision["score"],
        "enabled" if decision["enabled"] else "skipped",
        decision["reason"],
    )


def _build_selection_units(candidate_names: list[str], modules_by_name: dict[str, torch.nn.Module]) -> list[list[str]]:
    units = []
    visited = set()
    role_by_name = {name: _build_structure_info(name, modules_by_name[name], modules_by_name)["role"] for name in candidate_names}

    for layer_name in candidate_names:
        if layer_name in visited:
            continue

        role = role_by_name[layer_name]
        if role in {"attn_q", "attn_k"}:
            parent_name = layer_name.rpartition(".")[0]
            q_name = None
            k_name = None
            for candidate in candidate_names:
                if candidate.rpartition(".")[0] != parent_name:
                    continue
                if role_by_name[candidate] == "attn_q":
                    q_name = candidate
                elif role_by_name[candidate] == "attn_k":
                    k_name = candidate
            if q_name is not None and k_name is not None:
                units.append([q_name, k_name])
                visited.add(q_name)
                visited.add(k_name)
                continue

        units.append([layer_name])
        visited.add(layer_name)

    return units


def _resolve_streaming_hadamard_layer_selection(
    model: torch.nn.Module,
    layer_config: dict,
    config: HadamardConfig,
    modules_by_name: dict[str, torch.nn.Module],
    candidate_names: list[str],
    materialized_config: dict[str, int | str],
    run_forward: Callable[[], None],
) -> tuple[dict[str, dict], dict[str, dict]]:
    logger.warning(
        "Streaming Hadamard selection is enabled: calibration will be replayed once per layer "
        "or Q/K pair so decisions can be finalized and logged immediately."
    )
    decisions = {}
    selection_units = _build_selection_units(candidate_names, modules_by_name)

    for unit in selection_units:
        stats_by_layer = collect_linear_input_stats(model, unit, run_forward)
        unit_decisions = _compute_layer_decisions(unit, stats_by_layer, modules_by_name, config)
        decisions.update(unit_decisions)
        _apply_layer_decisions(layer_config, decisions, materialized_config)
        for layer_name in unit:
            _log_layer_decision(layer_name, unit_decisions[layer_name])

    return get_layer_hadamard_configs(layer_config), decisions


def get_layer_hadamard_configs(layer_config: dict) -> dict[str, dict]:
    selected_configs = {}
    for layer_name, cfg in layer_config.items():
        hadamard_config = cfg.get("hadamard_config")
        if not hadamard_config or not check_to_quantized(cfg):
            continue
        selected_configs[layer_name] = hadamard_config
    return selected_configs


def resolve_hadamard_layer_selection(
    model: torch.nn.Module,
    layer_config: dict,
    config: HadamardConfig,
    run_forward: Callable[[], None] | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    materialized_config = config.to_transform_dict()
    modules_by_name, candidate_names = _collect_candidate_layer_names(model, layer_config, config)

    if not candidate_names:
        return {}, {}

    if config.selector == "all":
        decisions = {}
        for layer_name in candidate_names:
            layer_config[layer_name]["hadamard_config"] = dict(materialized_config)
            decisions[layer_name] = {"enabled": True, "reason": "selector=all"}
        return get_layer_hadamard_configs(layer_config), decisions

    if run_forward is None:
        raise ValueError("run_forward must be provided when selector='heuristic'")

    if config.decision_timing == "streaming":
        return _resolve_streaming_hadamard_layer_selection(
            model,
            layer_config,
            config,
            modules_by_name,
            candidate_names,
            materialized_config,
            run_forward,
        )

    stats_by_layer = collect_linear_input_stats(model, candidate_names, run_forward)
    decisions = _compute_layer_decisions(candidate_names, stats_by_layer, modules_by_name, config)
    _apply_layer_decisions(layer_config, decisions, materialized_config)

    for layer_name in candidate_names:
        _log_layer_decision(layer_name, decisions[layer_name])

    return get_layer_hadamard_configs(layer_config), decisions
