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

import torch
import torch.nn as nn

from auto_round.experimental.qmodules.mx import MXFP4QuantLinear
from auto_round.experimental.transform.apply import apply_hadamard_transform
from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.patch_modules import INPUT_TRANSFORM_ATTR, WEIGHT_TRANSFORM_ATTR
from auto_round.experimental.transform.selective import resolve_hadamard_layer_selection
from auto_round.experimental.utils import normalize_hadamard_config
from auto_round.schemes import MXFP4


class _MiniAttention(nn.Module):
    def __init__(self, dim=128, head_dim=64):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


class _MiniMLP(nn.Module):
    def __init__(self, dim=128, hidden_dim=512):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)


class _MiniHadamardModel(nn.Module):
    def __init__(self, dim=128, hidden_dim=512):
        super().__init__()
        self.self_attn = _MiniAttention(dim=dim)
        self.mlp = _MiniMLP(dim=dim, hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(dim, dim * 2, bias=False)


class _QuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = MXFP4QuantLinear(32, 32, MXFP4)
        self.second = MXFP4QuantLinear(32, 32, MXFP4)


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(32, 32)
        self.second = nn.Linear(32, 32)


def _layer_config_for(model):
    return {
        name: {
            "bits": 4,
            "act_bits": 4,
            "data_type": "mx_fp",
            "act_data_type": "mx_fp",
            "group_size": 32,
            "act_group_size": 32,
            "sym": True,
            "act_sym": True,
            "act_dynamic": True,
            "hadamard_config": None,
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


def _heavy_tail(batch, dim, magnitude=32.0):
    tensor = torch.zeros(batch, dim, dtype=torch.float32)
    tensor[:, 0] = magnitude
    tensor[:, 1] = magnitude / 4
    return tensor


def test_normalize_selective_hadamard_string():
    config = normalize_hadamard_config("selective", "MXFP4")
    assert config["selector"] == "heuristic"
    assert config["block_size"] == 32
    assert config["hadamard_type"] == "hadamard"


def test_resolve_selective_hadamard_heuristic():
    model = _MiniHadamardModel()
    layer_config = _layer_config_for(model)
    config = HadamardConfig(selector="heuristic", block_size=32)

    def run_forward():
        heavy = _heavy_tail(8, 128)
        normal = torch.randn(8, 128)
        wide = _heavy_tail(8, 512)
        model.self_attn.q_proj(heavy)
        model.self_attn.k_proj(heavy)
        model.self_attn.v_proj(heavy)
        model.self_attn.o_proj(normal)
        model.mlp.up_proj(heavy)
        model.mlp.down_proj(wide)
        model.lm_head(normal)

    selected_configs, decisions = resolve_hadamard_layer_selection(model, layer_config, config, run_forward=run_forward)

    assert set(selected_configs) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "mlp.up_proj",
    }
    assert decisions["self_attn.q_proj"]["enabled"] is True
    assert decisions["self_attn.k_proj"]["enabled"] is True
    assert decisions["self_attn.v_proj"]["reason"] == "hard_skip:attn_v"
    assert decisions["self_attn.o_proj"]["reason"] == "hard_skip:attn_out"
    assert decisions["mlp.down_proj"]["reason"] == "hard_skip:ffn_down"
    assert layer_config["lm_head"]["hadamard_config"] is None


def test_apply_hadamard_transform_targets_only_selected_modules():
    model = _LinearModel()
    apply_hadamard_transform(
        model,
        config=None,
        module_configs={"first": {"block_size": 32}},
        location="weight",
        use_tqdm=False,
    )

    assert hasattr(model.first, WEIGHT_TRANSFORM_ATTR)
    assert not hasattr(model.second, WEIGHT_TRANSFORM_ATTR)


def test_apply_hadamard_input_hook_targets_only_selected_modules():
    model = _QuantModel()
    apply_hadamard_transform(
        model,
        config=None,
        module_configs={"first": {"block_size": 32}},
        location="input",
        use_tqdm=False,
    )

    assert len(model.first._forward_pre_hooks) > 0
    assert not hasattr(model.second, INPUT_TRANSFORM_ATTR)
    assert len(model.second._forward_pre_hooks) == 0
