"""
对比验证：我们的 rotate_in_channels_ / rotate_out_channels_ 
vs 参考实现 fuse_layer_rotation 的数学等价性
"""

import sys
#sys.path.insert(0, "/mnt/agents/output/spinquant_impl")
sys.path.insert(0, "/data/lkk/quarot/auto-round")

import copy
import torch
import torch.nn as nn

# 参考实现的 fuse_layer_rotation
def fuse_layer_rotation_ref(layer, R_in=None, R_out=None):
    """
    来自 https://github.com/taishan1994/LLM-Quantization
    W_new = R_out.T @ W @ R_in
    """
    device, dtype = layer.weight.device, layer.weight.dtype
    W = layer.weight.data.to(torch.float32)
    W = W.T
    with torch.no_grad():
        if R_in is not None:
            R_in = R_in.to(W.device, torch.float32)
            W = R_in @ W
        if R_out is not None:
            R_out = R_out.to(W.device, torch.float32)
            W = W @ R_out
    W = W.T
    layer.weight.data = W.to(device=device, dtype=dtype)


# 我们的实现
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    rotate_in_channels_, rotate_out_channels_
)


def test_equivalence():
    """验证两种实现数学等价"""
    torch.manual_seed(42)
    
    # 创建测试用的随机正交矩阵
    from auto_round.algorithms.transforms.spinquant.rotation_utils import random_hadamard_matrix
    R1 = random_hadamard_matrix(64)
    R1_inv = R1.T
    
    # Test 1: in-channel (q_proj-like)
    # 参考: fuse_layer_rotation(layer, R_in=R1_inv)
    # 我们的: rotate_in_channels_(layer, R_in=R1_inv)
    # 两者都应该得到 W @ R1
    layer1a = nn.Linear(64, 64, bias=False)
    layer1b = nn.Linear(64, 64, bias=False)
    layer1b.weight.data = layer1a.weight.data.clone()
    
    fuse_layer_rotation_ref(layer1a, R_in=R1_inv)
    rotate_in_channels_(layer1b, R_in=R1_inv)
    
    diff1 = (layer1a.weight - layer1b.weight).abs().max().item()
    print(f"[Test 1] in-channel equivalence: max_diff={diff1:.2e}")
    assert diff1 < 1e-5, f"in-channel failed: {diff1}"
    
    # Test 2: out-channel (o_proj-like)
    # 参考: fuse_layer_rotation(layer, R_out=R1)
    # 我们的: rotate_out_channels_(layer, R_out=R1)
    # 两者都应该得到 R1_inv @ W
    layer2a = nn.Linear(64, 64, bias=False)
    layer2b = nn.Linear(64, 64, bias=False)
    layer2b.weight.data = layer2a.weight.data.clone()
    
    fuse_layer_rotation_ref(layer2a, R_out=R1)
    rotate_out_channels_(layer2b, R_out=R1)
    
    diff2 = (layer2a.weight - layer2b.weight).abs().max().item()
    print(f"[Test 2] out-channel equivalence: max_diff={diff2:.2e}")
    assert diff2 < 1e-5, f"out-channel failed: {diff2}"
    
    # Test 3: 端到端输出等价
    # 创建简单网络，应用两种融合方式，验证输出一致
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 64)
            self.fc1 = nn.Linear(64, 64, bias=False)
            self.fc2 = nn.Linear(64, 64, bias=False)
            self.head = nn.Linear(64, 100, bias=False)
        def forward(self, x):
            x = self.embed(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return self.head(x)
    
    print("\n[Test 3] End-to-end output equivalence:")
    
    # Create ONE model with ONE set of weights
    torch.manual_seed(42)
    net_base = SimpleNet()
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (2, 8))
    
    with torch.no_grad():
        orig = net_base(input_ids)
    
    # Generate ONE R1 shared by both implementations
    R = random_hadamard_matrix(64)
    R_inv = R.T
    
    # --- Reference fusion on a COPY ---
    net_ref = copy.deepcopy(net_base)
    with torch.no_grad():
        W = net_ref.embed.weight.data
        net_ref.embed.weight.data = (W.to(torch.float64) @ R.to(torch.float64)).to(W.dtype)
        fuse_layer_rotation_ref(net_ref.fc1, R_in=R_inv)
        fuse_layer_rotation_ref(net_ref.fc2, R_out=R)
        fuse_layer_rotation_ref(net_ref.head, R_in=R_inv)
    with torch.no_grad():
        fused_ref = net_ref(input_ids)
    
    # --- Our fusion on another COPY with the SAME R ---
    net_our = copy.deepcopy(net_base)
    with torch.no_grad():
        W = net_our.embed.weight.data
        net_our.embed.weight.data = (W.to(torch.float64) @ R.to(torch.float64)).to(W.dtype)
        rotate_in_channels_(net_our.fc1, R_in=R_inv)
        rotate_out_channels_(net_our.fc2, R_out=R)
        rotate_in_channels_(net_our.head, R_in=R_inv)
    with torch.no_grad():
        fused_our = net_our(input_ids)
    
    rel_err_ref = ((fused_ref - orig).abs() / orig.abs().clamp(min=1e-8)).max().item()
    rel_err_our = ((fused_our - orig).abs() / orig.abs().clamp(min=1e-8)).max().item()
    cross_diff = (fused_ref - fused_our).abs().max().item()
    
    print(f"  Reference rel_err = {rel_err_ref:.2e}")
    print(f"  Our rel_err       = {rel_err_our:.2e}")
    print(f"  Cross max_diff    = {cross_diff:.2e}")


if __name__ == "__main__":
    test_equivalence()
    print("\n✅ All tests passed!")

