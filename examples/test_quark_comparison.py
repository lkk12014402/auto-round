"""
公平对比实验：Quark vs 我们的实现

用完全相同的 mock 模型、完全相同的随机种子、完全相同的 R1 矩阵，
分别运行 Quark 的融合函数和我们的融合函数，对比输出误差。
"""

import sys
import math
import copy

sys.path.insert(0, "/data/lkk/quarot/auto-round")
sys.path.insert(0, "/data/lkk/quarot/Quark")

import torch
import torch.nn as nn

# ========== Mock 模型 ==========
class MockRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)

class MockAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, x):
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out)

class MockMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = MockAttention(hidden_size, num_heads)
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.input_layernorm = MockRMSNorm(hidden_size)
        self.post_attention_layernorm = MockRMSNorm(hidden_size)
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class MockFlatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 64)
        self.layers = nn.ModuleList([MockDecoderLayer(64, 4, 128) for _ in range(2)])
        self.norm = MockRMSNorm(64)
        self.lm_head = nn.Linear(64, 100, bias=False)
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

def rel_err(a, b):
    return ((a - b).abs() / b.abs().clamp(min=1e-8)).max().item()

# ========== 生成相同的模型和输入 ==========
torch.manual_seed(42)
model = MockFlatModel()
model.eval()
input_ids = torch.randint(0, 100, (2, 8))

with torch.no_grad():
    orig = model(input_ids)

print(f"原始输出: mean={orig.abs().mean():.4f}, max={orig.abs().max():.4f}")
print()

# ========== 实验 1: Quark 的融合函数 ==========
print("=== 实验 1: Quark 的融合函数 ===")
from quark.torch.algorithm.rotation.rotation_utils import (
    rotate_in_channels_ as quark_rotate_in,
    rotate_out_channels_ as quark_rotate_out,
    transform_rms_norm_and_linear as quark_fuse_rmsnorm,
)
from quark.torch.algorithm.rotation.hadamard import random_hadamard_matrix as quark_random_hadamard

torch.manual_seed(42)
model_quark = MockFlatModel()
model_quark.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
model_quark.eval()

with torch.no_grad():
    orig_quark = model_quark(input_ids)

print(f"原始输出: mean={orig_quark.abs().mean():.4f}, max={orig_quark.abs().max():.4f}")

# 生成相同的 R1（同样的随机种子）
R1_quark = quark_random_hadamard(64)

print("R1_quark", R1_quark)

# RMSNorm 融合（Quark 方式）
for layer in model_quark.layers:
    quark_fuse_rmsnorm(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
    quark_fuse_rmsnorm(layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj])

# model.norm 和 lm_head
quark_fuse_rmsnorm(model_quark.norm, [model_quark.lm_head])

# R1 融合（Quark 方式）
# embed_tokens (in-channel, dim=1)
quark_rotate_in(model_quark.embed_tokens, R1_quark)

for layer in model_quark.layers:
    attn = layer.self_attn; mlp = layer.mlp
    # prev_modules (out-channel)
    quark_rotate_out(attn.o_proj, R1_quark)
    quark_rotate_out(mlp.down_proj, R1_quark)
    # next_modules (in-channel)
    quark_rotate_in(attn.q_proj, R1_quark)
    quark_rotate_in(attn.k_proj, R1_quark)
    quark_rotate_in(attn.v_proj, R1_quark)
    quark_rotate_in(mlp.gate_proj, R1_quark)
    quark_rotate_in(mlp.up_proj, R1_quark)

# lm_head (in-channel)
quark_rotate_in(model_quark.lm_head, R1_quark)

with torch.no_grad():
    out_quark = model_quark(input_ids)

err_quark = rel_err(out_quark, orig)
print(f"Quark 融合: rel_err = {err_quark:.6f} ({err_quark*100:.4f}%)")

# ========== 实验 2: 我们的融合函数 ==========
print("\n=== 实验 2: 我们的融合函数 ===")
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    rotate_in_channels_, rotate_out_channels_, fuse_rmsnorm_in_model,
    random_hadamard_matrix as our_random_hadamard,
)

torch.manual_seed(42)
model_our = MockFlatModel()
model_our.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
model_our.eval()

with torch.no_grad():
    orig_our = model_our(input_ids)

print(f"原始输出: mean={orig_our.abs().mean():.4f}, max={orig_our.abs().max():.4f}")


# 生成相同的 R1
R1_our = our_random_hadamard(64)
R1_inv = R1_our.T

print("R1_our: ", R1_our)


# RMSNorm 融合（我们的方式）
fuse_rmsnorm_in_model(model_our)

# R1 融合（我们的方式）
with torch.no_grad():
    W = model_our.embed_tokens.weight.data
    model_our.embed_tokens.weight.data = (W.to(torch.float64) @ R1_our.to(torch.float64)).to(W.dtype)
    
    for layer in model_our.layers:
        attn = layer.self_attn; mlp = layer.mlp
        # in-channel: R_in=R1_inv (内部做 W @ R1_inv.T = W @ R1)
        rotate_in_channels_(attn.q_proj, R_in=R1_inv)
        rotate_in_channels_(attn.k_proj, R_in=R1_inv)
        rotate_in_channels_(attn.v_proj, R_in=R1_inv)
        rotate_in_channels_(mlp.gate_proj, R_in=R1_inv)
        rotate_in_channels_(mlp.up_proj, R_in=R1_inv)
        # out-channel: R_out=R1 (内部做 R1.T @ W = R1_inv @ W)
        rotate_out_channels_(attn.o_proj, R_out=R1_our)
        rotate_out_channels_(mlp.down_proj, R_out=R1_our)
    
    rotate_in_channels_(model_our.lm_head, R_in=R1_inv)

with torch.no_grad():
    out_our = model_our(input_ids)

err_our = rel_err(out_our, orig)
print(f"我们的融合: rel_err = {err_our:.6f} ({err_our*100:.4f}%)")

# ========== 交叉对比 ==========
print("\n=== 交叉对比 ===")
cross_err = rel_err(out_quark, out_our)
print(f"Quark vs 我们的 cross-diff = {cross_err:.2e}")

# 权重对比
print("\n=== 权重逐层对比 ===")
max_weight_diff = 0
for (n1, p1), (n2, p2) in zip(model_quark.named_parameters(), model_our.named_parameters()):
    diff = (p1 - p2).abs().max().item()
    max_weight_diff = max(max_weight_diff, diff)
    status = "✅" if diff < 1e-5 else "❌"
    print(f"  {n1}: diff={diff:.2e} {status}")

print(f"\n最大权重差异: {max_weight_diff:.2e}")

# ========== 实验 3: Identity 控制 ==========
print("\n=== 实验 3: Identity 矩阵控制 ===")
torch.manual_seed(42)
model_id = MockFlatModel()
model_id.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
model_id.eval()

I64 = torch.eye(64, dtype=torch.float64)

with torch.no_grad():
    W = model_id.embed_tokens.weight.data
    model_id.embed_tokens.weight.data = (W.to(torch.float64) @ I64).to(W.dtype)
    
    for layer in model_id.layers:
        attn = layer.self_attn; mlp = layer.mlp
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            w = proj.weight.data; proj.weight.data = (w.to(torch.float64) @ I64).to(w.dtype)
        for proj in [mlp.gate_proj, mlp.up_proj]:
            w = proj.weight.data; proj.weight.data = (w.to(torch.float64) @ I64).to(w.dtype)
        for proj in [attn.o_proj, mlp.down_proj]:
            w = proj.weight.data; proj.weight.data = (I64.T @ w.to(torch.float64)).to(w.dtype)
    
    w = model_id.lm_head.weight.data
    model_id.lm_head.weight.data = (w.to(torch.float64) @ I64).to(w.dtype)
    
    out_id = model_id(input_ids)

err_id = rel_err(out_id, orig)
print(f"Identity 控制: rel_err = {err_id:.2e}")

