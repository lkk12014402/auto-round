# Quark Rotation (R1-R4) 完整分析

> 基于 Quark 代码库 `/data/lkk/quarot/Quark` 的实际实现分析

---

## 1. 核心概念：Residual Stream 的基变换

Transformer 的核心数据通路是 **residual stream**（残差流）。每一层对残差流进行读取（attention/MLP 输入）和写入（o_proj/down_proj 输出）。

**Offline R1 的本质**是对整个 residual stream 做一次**全局基变换（change of basis）**：

```
原始基:   x  →  [Layer 0]  →  x  →  [Layer 1]  →  ...  →  x  →  [lm_head]  →  logits
旋转基:   xR →  [Layer 0'] →  xR →  [Layer 1'] →  ...  →  xR →  [lm_head'] →  logits
```

关键：**logits 完全不变**，因为旋转在 lm_head 处被抵消。

---

## 2. Quark 的权重存储约定

Quark 使用 PyTorch 标准约定：`W.shape = [out_features, in_features]`

- **输入侧旋转** `rotate_in_channels_(layer, R)`：`W' = W @ R`
  - 对应 `rotation_utils.py:110-113`
- **输出侧旋转** `rotate_out_channels_(layer, R)`：`W' = R.T @ W`, `b' = b @ R`
  - 对应 `rotation_utils.py:116-125`

数学验证（行向量约定，x 是行向量）：
```
原始:     y = x @ W.T + b
输入旋转: y = (xR) @ (WR).T = xR @ R.T @ W.T = x @ W.T  ✓ （输入侧旋转抵消）
输出旋转: y = x @ (R.T @ W).T + b@R = x @ W.T @ R + b@R = (x@W.T + b) @ R  （输出加入旋转）
```

---

## 3. R1：Residual Stream 旋转

### 3.1 Offline R1（全局基变换，无需推理包装器）

**代码入口**: `rotation.py:659-877`（`fuse_normalization()` + `fuse_r1()`）

Offline R1 修改的层（以 Llama/Qwen 架构为例）：

#### 完整变换链

```
embed_tokens:  E' = E @ R          (输出旋转 → 残差流进入旋转基)
                                    ↓ 残差流: xR
┌─────────────── Layer i ───────────────┐
│                                       │
│  input_layernorm: γ 融入下游权重，设为1│
│                                       │
│  q_proj: W_q' = W_q @ R   (输入旋转)  │  ← 输入 xR，抵消得 x@W_q.T
│  k_proj: W_k' = W_k @ R   (输入旋转)  │
│  v_proj: W_v' = W_v @ R   (输入旋转)  │
│                                       │
│  [attention 计算，输出 attn_out]       │
│                                       │
│  o_proj: W_o' = R.T @ W_o (输出旋转)  │  ← 输出 attn_out @ W_o.T @ R
│         b_o' = b_o @ R                │     写回旋转基残差流
│                                       │
│  残差连接: xR + attn_out@W_o.T@R      │  ← 仍在旋转基 ✓
│                                       │
│  post_attn_layernorm: γ 融入下游，设为1│
│                                       │
│  gate_proj: W_g' = W_g @ R (输入旋转) │  ← 输入 xR，抵消
│  up_proj:   W_u' = W_u @ R (输入旋转) │
│                                       │
│  [MLP: act_fn(gate) * up → hidden]    │
│                                       │
│  down_proj: W_d' = R.T @ W_d (输出旋转)│  ← 输出写回旋转基
│             b_d' = b_d @ R             │
│                                       │
│  残差连接: 仍在旋转基 ✓               │
└───────────────────────────────────────┘
                    ↓
              (重复 N 层)
                    ↓
model.norm: γ 融入 lm_head，设为 1

lm_head: W_lm' = W_lm @ R   (输入旋转)
         输出: (hR) @ (W_lm @ R).T = h @ W_lm.T   ← logits 不变 ✓
```

#### RMSNorm 融合原理

RMSNorm 的计算：`y = (x / RMS(x)) * γ`

其中 RMS 归一化是**正交等变的**：`RMS(xR) = RMS(x)`（因为 R 是正交矩阵，不改变向量模长）

所以：
```
RMSNorm(xR) = (xR / RMS(xR)) * γ = (xR / RMS(x)) * γ
```

这里的 `γ` 是逐元素缩放（对角矩阵），不与旋转交换。因此需要把 `γ` **融入下游线性层的权重**中：

```python
# rotation_utils.py:158-175
# 对下游线性层: W_new = diag(γ) @ W_old
for linear in next_layers:
    linear.weight.data = (γ.unsqueeze(1) * linear.weight.data)
# RMSNorm 的 γ 设为全 1
norm.weight.data = torch.ones_like(norm.weight.data)
```

融合后 RMSNorm 变成纯 RMS 归一化（`γ=1`），与正交旋转完全交换。

#### 关键结论

> **Offline R1+R2 完全不需要推理时的 wrapper/hook。**
> embed_tokens 的旋转已经把输入"送入"旋转基，lm_head 的旋转把输出"带回"原始基。整条链路自洽。

**前提条件**：
1. 必须 untie `embed_tokens` 和 `lm_head`（它们需要不同的变换方向）
2. 必须融合所有 RMSNorm 的 γ

---

### 3.2 Online R1（局部旋转，需要推理包装器）

**代码入口**: `rotation.py:879-931`（`apply_online_r1()`）

Online R1 **不做全局基变换**，而是对每个 target module 单独做旋转：

```
残差流: x (原始基，不变！)
        ↓
[Wrapper: x → xR]  →  q_proj(xR): W_q' = W_q @ R
                                    输出: (xR) @ (W_q@R).T = x @ W_q.T  ✓
```

#### 修改的内容

| 组件 | 是否修改 |
|------|---------|
| embed_tokens | ❌ 不动 |
| RMSNorm | ❌ 不融合 |
| q/k/v_proj 权重 | ✅ `W @ R`（输入旋转）|
| q/k/v_proj 包装器 | ✅ `InputRotationWrapperHadamard` |
| o_proj | ❌ 不动 |
| gate/up_proj 权重 | ✅ `W @ R`（输入旋转）|
| gate/up_proj 包装器 | ✅ `InputRotationWrapperHadamard` |
| down_proj | ❌ 不动 |
| lm_head | ❌ 不动 |

#### `InputRotationWrapperHadamard` 的实现

```python
# rotation_utils.py:290-355
class InputRotationWrapperHadamard(InputRotationWrapper):
    """nn.Module 包装器，在 forward 中对输入做 Hadamard 变换"""

    def forward(self, x):
        # 1. 对输入做 Hadamard 旋转
        x_rotated = self.rotation_transform(x)  # x → xR
        # 2. 量化在这里发生（如果启用了 activation quantization）
        # 3. 调用原始线性层
        return self.original_module(x_rotated)
```

它是一个真正的 `nn.Module` 子类，可以被 `save_pretrained` / `load_pretrained` 正确序列化。

#### 为什么需要 Online R1？

**不是因为数学不等价**，而是因为**量化质量**：

| | Offline R1 | Online R1 |
|---|---|---|
| o_proj/down_proj 权重 | 被旋转 → 分布改变 → 量化质量可能下降 | 保持原始分布 → 量化质量不变 |
| RMSNorm | 融入权重 → 失去 per-channel rescaling | 保留 → 量化受益于 rescaling |
| embed/lm_head | 被旋转 | 不动 |

实验结果（Qwen3-0.6B + MXFP4）：
- Offline R1: hellaswag 精度下降 ~8-14%
- Online R1: hellaswag 精度基本不变

---

## 4. R2：Attention Head 内旋转

**代码入口**: `rotation.py:933-986`（`r2()`）

R2 在每个 attention head 内部做旋转，完全 offline，**无需推理包装器**。

```
v_proj:  W_v' = R2.T @ W_v     (per-head 输出旋转)
o_proj:  W_o' = W_o @ R2       (per-head 输入旋转)
```

数学验证：
```
V_head_new = V_head @ R2                    (V 输出被旋转)
attn_output_new = softmax(QK.T/√d) @ V_new  (attention 对 V 是线性的)
                = attn_output @ R2           (所以输出也被旋转)
o_proj_new(attn_output_new)
    = (attn_output @ R2) @ (W_o @ R2).T
    = attn_output @ R2 @ R2.T @ W_o.T
    = attn_output @ W_o.T                  ✓ 旋转抵消
```

#### R2 的 rotation_size

R2 的 rotation_size 等于 `head_dim`（如 Qwen3-0.6B 的 head_dim=128）。旋转是 per-head 的 block-diagonal 变换。

```python
# rotation.py:934-951
# 构建 block-diagonal 旋转矩阵
R2_block = create_block_diag_from_head_matrix(R2_head, num_heads)
rotate_out_channels_(v_proj, R2_block)   # V 输出旋转
rotate_in_channels_(o_proj, R2_block)    # O 输入旋转
```

---

## 5. R3：Q/K Post-RoPE 旋转

**代码入口**: `rotation.py:987-996` + `monkeypatch.py:37-65`

R3 通过 **monkey-patch attention forward** 实现，是 **online 的**（需要推理时的 monkeypatch）。

```
原始:  q, k = apply_rotary_pos_emb(q, k, cos, sin)
R3后:  q, k = apply_rotary_pos_emb(q, k, cos, sin)
       q = q @ R3    # per-head Hadamard
       k = k @ R3    # per-head Hadamard
```

数学：`(qR3)(kR3).T = q R3 R3.T k.T = q k.T` ✓

**不修改任何权重**，仅改变 attention score 计算过程中的中间 activation。

#### 为什么 R3 不能 offline fuse？

R3 必须在 **RoPE 之后** 应用。RoPE 是位置相关的旋转：
```
q_rope = q * cos + rotate_half(q) * sin
```
这不是简单的矩阵乘法，所以 R3 无法穿过 RoPE 融入 q_proj/k_proj 的权重。

#### 推理时需要

- monkeypatch（修改 attention forward 方法）
- Quark 注释：reload/export with `r3=True` 目前不支持 (`rotation.py:1274-1277`)

---

## 6. R4：MLP 中间层旋转

**代码入口**: `rotation.py:998-1055`

R4 在 MLP 的非线性激活之后、down_proj 之前插入旋转。

```
MLP 计算流:
gate_out = act_fn(gate_proj(x))     ← 非线性！
up_out   = up_proj(x)
hidden   = gate_out * up_out         ← 逐元素乘（非线性！）
           ↓
      [R4: hidden → hidden @ R4]     ← R4 在这里
           ↓
output   = down_proj(hidden @ R4)
```

#### R4 的融合方式

**down_proj 侧**可以离线融合：
```python
# rotation.py:1026-1035
# 非训练路径:
rotate_in_channels_(down_proj, R4)     # W_d' = W_d @ R4
# 然后包装 down_proj:
InputRotationWrapperHadamard(down_proj)  # 推理时对 input 做 R4 旋转
```

**gate_proj / up_proj 侧不能融合**，因为 `act_fn` 和逐元素乘法是非线性的：
```
act_fn(gate(x) @ R4) ≠ act_fn(gate(x)) @ R4   ← 非线性，不等价！
```

#### 推理时需要

- `InputRotationWrapperHadamard` 包装 `down_proj`
- 或 monkeypatch MLP forward

---

## 7. 总结：各旋转的融合能力

| 旋转 | 修改的权重 | RMSNorm | 推理需要 wrapper? | 能完全 offline fuse? |
|------|-----------|---------|-------------------|---------------------|
| **R1 offline** | embed, q/k/v, o_proj, gate/up, down_proj, lm_head | 融合 | ❌ 不需要 | ✅ 完全 offline |
| **R1 online** | q/k/v, gate/up | 不融合 | ✅ 需要 wrapper | ❌ |
| **R2** | v_proj(输出), o_proj(输入) | 不融合 | ❌ 不需要 | ✅ 完全 offline |
| **R3** | 无权重修改 | 不融合 | ✅ 需要 monkeypatch | ❌ |
| **R4** | down_proj(输入) | 不融合 | ✅ 需要 wrapper | ❌ |

### 推理部署场景

| 组合 | 推理时额外开销 |
|------|--------------|
| **Offline R1 only** | 无（完全 fused）|
| **Offline R1 + R2** | 无（完全 fused）|
| **Online R1 + R2** | R1 wrapper on q/k/v/gate/up |
| **R1 + R2 + R3** | R3 monkeypatch |
| **R1 + R2 + R3 + R4** | R3 monkeypatch + R4 wrapper on down_proj |

---

## 8. Quark 的完整流程：Rotation → Quantization → Save

```
1. 加载模型
   ↓
2. 应用旋转 (rotation preprocessing)
   - Offline R1: 修改所有权重 + 融合 RMSNorm
   - 或 Online R1: 修改 target 权重 + 注册 wrapper
   - R2: 修改 V/O 权重
   - R3: monkeypatch attention
   - R4: 修改 down_proj + 注册 wrapper
   ↓
3. 量化校准 (calibration)
   - 对旋转后的权重做量化（MXFP4/INT4/...）
   - Online wrapper 确保 activation 也在旋转后的空间被量化
   ↓
4. [可选] 训练旋转矩阵 (train_rotation)
   - 使用 RotationLinear 替换 nn.Linear
   - 训练 R2/R4 的旋转参数
   - post_process: 将学到的旋转 fuse 回权重
   ↓
5. 保存模型
   - Offline-fused 的权重直接保存
   - Online wrapper (nn.Module) 作为模型的一部分保存
   - R3 monkeypatch 需要 reload 时重新注册
   ↓
6. 推理加载
   - 加载权重（包含 fused 旋转）
   - Wrapper 自动恢复（因为是 nn.Module）
   - R3 需要重新 monkeypatch
```

---

## 9. 对 Auto-Round 迁移的启示

### 当前状态
- 我们实现了 Online R1（使用 `register_forward_pre_hook`，不可序列化）
- R2 是 offline fused（正确）
- R3/R4 使用 monkeypatch（正确）

### 需要改进
1. **Online R1 的 wrapper**：从 `register_forward_pre_hook` 改为 `nn.Module` wrapper（如 Quark 的 `InputRotationWrapperHadamard`），使其可序列化
2. **Offline R1 的量化质量**：当用户不需要 online wrapper 时，可以选择 offline R1 + 更好的量化方案
3. **R4 的 wrapper**：同样需要可序列化的 `nn.Module`

### 用户应该选择哪种 R1？

| 场景 | 推荐 |
|------|------|
| 精度优先 + 部署简单 | Offline R1（无 wrapper，但量化质量可能稍差）|
| 量化精度优先（MXFP4等） | Online R1（需要 wrapper，但量化质量更好）|
| 只做旋转不量化 | Offline R1（完全等价，更简洁）|
