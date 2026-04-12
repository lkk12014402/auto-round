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

import torch.nn as nn
import torch
from unittest.mock import patch

from auto_round.compressors.base import BaseCompressor
from auto_round.experimental.qmodules.mx import MXFP4QuantLinear
from auto_round.experimental.transform.apply import apply_hadamard_transform
from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.patch_modules import INPUT_TRANSFORM_ATTR, WEIGHT_TRANSFORM_ATTR
from auto_round.experimental.transform.selective import resolve_hadamard_layer_selection
from auto_round.experimental.utils import normalize_hadamard_config
from auto_round.logger import logger
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
    assert config["decision_timing"] == "batched"


def test_normalize_streaming_selective_hadamard_string():
    config = normalize_hadamard_config("selective_streaming", "MXFP4")
    assert config["selector"] == "heuristic"
    assert config["decision_timing"] == "streaming"
    assert config["block_size"] == 32


def test_normalize_lowmem_streaming_selective_hadamard_string():
    config = normalize_hadamard_config("selective_lowmem_streaming", "MXFP4")
    assert config["selector"] == "heuristic"
    assert config["decision_timing"] == "streaming"
    assert config["selection_execution"] == "blockwise"
    assert config["block_size"] == 32


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
        "self_attn.v_proj",
        "mlp.up_proj",
    }
    assert decisions["self_attn.q_proj"]["enabled"] is True
    assert decisions["self_attn.k_proj"]["enabled"] is True
    assert decisions["self_attn.v_proj"]["enabled"] is True
    assert decisions["self_attn.v_proj"]["reason"] == "score_threshold"
    assert decisions["self_attn.o_proj"]["reason"] == "score_below_threshold"
    assert decisions["mlp.down_proj"]["reason"] == "score_below_threshold"
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


def test_resolve_selective_hadamard_streaming_logs_immediately():
    model = _MiniHadamardModel()
    layer_config = _layer_config_for(model)
    config = HadamardConfig(selector="heuristic", block_size=32, decision_timing="streaming")
    run_count = {"value": 0}

    def run_forward():
        run_count["value"] += 1
        heavy = _heavy_tail(8, 128)
        normal = torch.randn(8, 128)
        wide = _heavy_tail(8, 512)
        model.self_attn.q_proj(heavy)
        model.self_attn.k_proj(heavy)
        model.self_attn.v_proj(heavy)
        model.self_attn.o_proj(normal)
        model.mlp.up_proj(heavy)
        model.mlp.down_proj(wide)

    with patch.object(logger, "info") as mock_info, patch.object(logger, "warning") as mock_warn:
        selected_configs, decisions = resolve_hadamard_layer_selection(model, layer_config, config, run_forward=run_forward)

    assert set(selected_configs) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj"}
    assert decisions["self_attn.q_proj"]["enabled"] is True
    assert decisions["self_attn.k_proj"]["enabled"] is True
    assert decisions["self_attn.v_proj"]["enabled"] is True
    assert run_count["value"] == 5
    log_messages = " ".join(str(call) for call in mock_info.call_args_list)
    assert "self_attn.q_proj" in log_messages
    assert "mlp.up_proj" in log_messages
    warning_messages = " ".join(str(call) for call in mock_warn.call_args_list)
    assert "Streaming Hadamard selection is enabled" in warning_messages


class _LowMemToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _MiniAttention(dim=128, head_dim=64)
        self.mlp = _MiniMLP(dim=128, hidden_dim=512)

    def forward(self, hidden_states):
        q = self.self_attn.q_proj(hidden_states)
        k = self.self_attn.k_proj(hidden_states)
        v = self.self_attn.v_proj(hidden_states)
        o = self.self_attn.o_proj(q + k + v)
        up = self.mlp.up_proj(hidden_states)
        return o + self.mlp.down_proj(torch.relu(up))


class _LowMemToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_LowMemToyBlock(), _LowMemToyBlock()])


def test_lowmem_hadamard_selection_uses_blockwise_replay():
    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.model = _LowMemToyModel()
    dummy.layer_config = {
        "layers.0.self_attn.q_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.0.self_attn.k_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.0.self_attn.v_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.0.self_attn.o_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.0.mlp.up_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.0.mlp.down_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.1.self_attn.q_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.1.self_attn.k_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.1.self_attn.v_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.1.self_attn.o_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.1.mlp.up_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
        "layers.1.mlp.down_proj": {"bits": 4, "act_bits": 4, "in_blocks": True, "hadamard_config": None},
    }
    dummy.quant_block_list = [["layers.0", "layers.1"]]
    dummy.nsamples = 2
    dummy.batch_size = 1
    dummy.infer_bs_coeff = 1
    dummy.device = "cpu"
    dummy.cache_device = "cpu"
    dummy.amp = False
    dummy.amp_dtype = torch.float32
    dummy.device_list = ["cpu"]
    dummy.low_gpu_mem_usage = True
    dummy.device_map = "cpu"
    dummy._run_hadamard_selection_calibration = lambda: None
    dummy._prepare_hadamard_block_inputs = BaseCompressor._prepare_hadamard_block_inputs.__get__(dummy, _Dummy)

    heavy = _heavy_tail(1, 128)
    dummy.try_cache_inter_data_gpucpu = lambda block_names, nsamples, layer_names=None: {
        "layers.0": {"hidden_states": [heavy.clone(), heavy.clone()]}
    }

    replay_calls = {"count": 0}

    def _get_block_outputs(block, input_ids, input_others, bs, device, cache_device, save_output=True):
        replay_calls["count"] += 1
        outputs = [block(inp).to(cache_device) for inp in input_ids]
        if save_output:
            return outputs
        return None

    dummy._get_block_outputs = _get_block_outputs
    dummy._resolve_hadamard_layer_selection_low_memory = BaseCompressor._resolve_hadamard_layer_selection_low_memory.__get__(
        dummy, _Dummy
    )

    config = HadamardConfig(selector="heuristic", selection_execution="blockwise", block_size=32)
    selected_configs, decisions = dummy._resolve_hadamard_layer_selection_low_memory(config)

    assert "layers.0.mlp.up_proj" in selected_configs
    assert "layers.1.mlp.up_proj" in selected_configs
    assert decisions["layers.0.mlp.down_proj"]["enabled"] is False
    assert replay_calls["count"] >= 4
