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

from auto_round.cli.parser import build_quantize_parser
from auto_round.compressors.mixed_mxfp import (
    build_mixed_mxfp_policy_defaults,
    merge_mixed_mxfp_policy_defaults,
    resolve_mixed_mxfp_policy_scheme,
)
from auto_round.compressors.utils import set_layer_config

_SUPPORTED_TYPES = (nn.Linear,)
_INNER_SUPPORTED_TYPES = ()
_SCALE_DTYPE = torch.float16


class _DenseMLP(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.gate_proj = nn.Linear(size, size)
        self.up_proj = nn.Linear(size, size)
        self.down_proj = nn.Linear(size, size)


class _AttentionMixer(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.num_heads = 4
        self.q_proj = nn.Linear(size, size)
        self.k_proj = nn.Linear(size, size)
        self.v_proj = nn.Linear(size, size)
        self.o_proj = nn.Linear(size, size)


class _LinearAttentionMixer(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.num_attention_heads = 4
        self.in_proj_q = nn.Linear(size, size)
        self.in_proj_k = nn.Linear(size, size)
        self.in_proj_v = nn.Linear(size, size)
        self.out_proj = nn.Linear(size, size)


class _Expert(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.gate_proj = nn.Linear(size, size)
        self.up_proj = nn.Linear(size, size)
        self.down_proj = nn.Linear(size, size)


class _SparseRouter(nn.Module):
    def __init__(self, size: int = 32, num_experts: int = 3):
        super().__init__()
        self.dispatch = nn.Linear(size, num_experts, bias=False)
        self.specialists = nn.ModuleList([_Expert(size) for _ in range(num_experts)])
        self.fallback = nn.Linear(size, size)


class _DenseBlock(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.attention = _AttentionMixer(size)
        self.feed_forward = _DenseMLP(size)


class _SparseBlock(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(size, size)
        self.sparse_branch = _SparseRouter(size)
        self.output_proj = nn.Linear(size, size)


class _HybridSparseBlock(nn.Module):
    def __init__(self, size: int = 32):
        super().__init__()
        self.full_mixer = _AttentionMixer(size)
        self.linear_mixer = _LinearAttentionMixer(size)
        self.sparse_branch = _SparseRouter(size)
        self.output_proj = nn.Linear(size, size)


class _TransformerFixture(nn.Module):
    def __init__(self, block_cls, num_layers: int = 2, size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([block_cls(size) for _ in range(num_layers)])
        self.outside_linear = nn.Linear(size, size)
        self.lm_head = nn.Linear(size, size, bias=False)


def _set_mixed_policy_config(model, policy, *, quant_block_list=None, layer_config=None, ignore_layers=""):
    scheme = resolve_mixed_mxfp_policy_scheme(policy, "W4A16")
    policy_defaults = build_mixed_mxfp_policy_defaults(
        model,
        policy,
        supported_types=_SUPPORTED_TYPES,
        inner_supported_types=_INNER_SUPPORTED_TYPES,
        quant_block_list=quant_block_list,
    )
    merged_layer_config, merged_ignore = merge_mixed_mxfp_policy_defaults(policy_defaults, layer_config, ignore_layers)
    return set_layer_config(
        model=model,
        layer_config=merged_layer_config,
        default_scheme=scheme,
        default_scale_dtype=_SCALE_DTYPE,
        supported_types=_SUPPORTED_TYPES,
        inner_supported_types=_INNER_SUPPORTED_TYPES,
        quant_block_list=quant_block_list,
        ignore_layers=merged_ignore,
        quant_lm_head=False,
    )


def test_dense_policy_quantizes_selected_dense_block_linears_to_mxfp4():
    model = _TransformerFixture(_DenseBlock)

    layer_config, _, _ = _set_mixed_policy_config(model, "dense_mxfp4", quant_block_list=[["layers.0"]])

    assert layer_config["layers.0.attention.q_proj"]["bits"] == 4
    assert layer_config["layers.0.feed_forward.gate_proj"]["bits"] == 4
    assert layer_config["layers.0.feed_forward.gate_proj"]["data_type"] == "mx_fp"
    assert "layers.1.attention.q_proj" not in layer_config
    assert "outside_linear" not in layer_config
    assert "lm_head" not in layer_config


def test_moe_balanced_policy_quantizes_experts_to_mxfp4_and_other_block_linears_to_mxfp8():
    model = _TransformerFixture(_SparseBlock, num_layers=1)

    layer_config, _, _ = _set_mixed_policy_config(model, "moe_balanced", quant_block_list=[["layers.0"]])

    assert layer_config["layers.0.sparse_branch.specialists.0.gate_proj"]["bits"] == 4
    assert layer_config["layers.0.sparse_branch.specialists.1.up_proj"]["bits"] == 4
    assert layer_config["layers.0.sparse_branch.fallback"]["bits"] == 8
    assert layer_config["layers.0.input_proj"]["bits"] == 8
    assert layer_config["layers.0.output_proj"]["bits"] == 8
    assert layer_config["layers.0.sparse_branch.dispatch"]["bits"] == 16


def test_moe_conservative_policy_keeps_attention_linears_in_fp():
    model = _TransformerFixture(_HybridSparseBlock, num_layers=1)

    layer_config, _, _ = _set_mixed_policy_config(model, "moe_conservative", quant_block_list=[["layers.0"]])

    assert layer_config["layers.0.sparse_branch.specialists.0.down_proj"]["bits"] == 4
    assert layer_config["layers.0.sparse_branch.dispatch"]["bits"] == 16
    assert layer_config["layers.0.full_mixer.q_proj"]["bits"] == 16
    assert layer_config["layers.0.linear_mixer.in_proj_q"]["bits"] == 16
    assert layer_config["layers.0.output_proj"]["bits"] == 8


def test_moe_balanced_policy_keeps_attention_linears_at_mxfp8():
    model = _TransformerFixture(_HybridSparseBlock, num_layers=1)

    layer_config, _, _ = _set_mixed_policy_config(model, "moe_balanced", quant_block_list=[["layers.0"]])

    assert layer_config["layers.0.sparse_branch.specialists.0.gate_proj"]["bits"] == 4
    assert layer_config["layers.0.sparse_branch.dispatch"]["bits"] == 16
    assert layer_config["layers.0.full_mixer.q_proj"]["bits"] == 8
    assert layer_config["layers.0.linear_mixer.in_proj_q"]["bits"] == 8


def test_dense_gated_mlp_gate_proj_is_not_treated_as_router():
    model = _TransformerFixture(_DenseBlock, num_layers=1)
    policy_defaults = build_mixed_mxfp_policy_defaults(
        model,
        "dense_mxfp4",
        supported_types=_SUPPORTED_TYPES,
        inner_supported_types=_INNER_SUPPORTED_TYPES,
        quant_block_list=[["layers.0"]],
    )

    assert "layers.0.feed_forward.gate_proj" not in policy_defaults.ignore_layers

    layer_config, _, _ = _set_mixed_policy_config(model, "dense_mxfp4", quant_block_list=[["layers.0"]])
    assert layer_config["layers.0.feed_forward.gate_proj"]["bits"] == 4


def test_explicit_overrides_take_precedence_over_policy():
    model = _TransformerFixture(_HybridSparseBlock, num_layers=1)

    layer_config, _, _ = _set_mixed_policy_config(
        model,
        "moe_conservative",
        quant_block_list=[["layers.0"]],
        layer_config={"layers.0.sparse_branch.dispatch": "MXFP8"},
        ignore_layers="layers.0.sparse_branch.specialists.0.up_proj",
    )

    assert layer_config["layers.0.sparse_branch.dispatch"]["bits"] == 8
    assert layer_config["layers.0.sparse_branch.specialists.0.up_proj"]["bits"] == 16
    assert layer_config["layers.0.full_mixer.q_proj"]["bits"] == 16


def test_cli_parser_accepts_mixed_mxfp_policy():
    args = build_quantize_parser().parse_args(["--model", "dummy-model", "--mixed_mxfp_policy", "moe_balanced"])

    assert args.mixed_mxfp_policy == "moe_balanced"
