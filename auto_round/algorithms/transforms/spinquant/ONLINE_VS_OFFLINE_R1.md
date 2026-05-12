# Online R1 vs Offline R1: Technical Deep Dive

## 1. Online R1 实际做了什么（纠正常见误解）

### ❌ 常见误解

> "Online R1 不修改 weight，只加 activation hook"

### ✅ 实际实现

**Online R1 同时修改 weight 和 activation**（见 `preprocessor.py: _apply_online_r1()`）：

```python
# 对每个 target module (q/k/v_proj, gate/up_proj):

# 1. 修改 weight: W_new = W @ R
module.weight.data = (W @ R).to(dtype)

# 2. 注册 activation pre-hook: x_rot = x @ R
def hook(module, args):
    x = args[0]
    return (x @ R,)
module.register_forward_pre_hook(hook)
```

### 数学等价性

`nn.Linear` 的计算是 `output = input @ weight.T`：

```
output = x_rot @ W_new.T
       = (x @ R) @ (W @ R).T
       = (x @ R) @ (R^T @ W^T)
       = x @ (R @ R^T) @ W^T
       = x @ I @ W^T
       = x @ W^T   ← 与原始完全相同
```

旋转在数学上完美抵消，模型行为不变（FP16 下精度损失 < 1e-4）。

### 那旋转的意义是什么？

关键：**量化发生在旋转之后、矩阵乘法之前**：

```
x → [hook: x@R] → quantize(x@R) → dequant → [W_new^T] → output
                                                ↑
                                        quantize(W@R) stored
```

- **Weight 量化目标**: `W @ R` — 旋转后 weight 分布更均匀 ✓
- **Activation 量化目标**: `x @ R` — 旋转后 activation 分布更均匀 ✓

虽然最终计算结果相同，但量化误差更小，因为双方都在更"友好"的分布中被量化。

---

## 2. Online R1 的作用范围

| 模块 | Weight 修改 | Activation Hook | 说明 |
|------|:-----------:|:---------------:|------|
| embed_tokens | ✗ | ✗ | 不修改 |
| q/k/v_proj | ✓ `W@R` | ✓ `x@R` | Target modules |
| gate/up_proj | ✓ `W@R` | ✓ `x@R` | Target modules |
| o_proj | ✗ | ✗ | 不修改 |
| down_proj | ✗ | ✗ | 不修改 |
| lm_head | ✗ | ✗ | 不修改 |
| RMSNorm γ | ✗ | - | **不需要吸收** |

**核心特征**：旋转的作用域是 **per-module 局部的**。每个 target module 独立完成
"rotate activation → compute with rotated weight → cancel"，不依赖其他层的状态。

---

## 3. Offline R1 的作用范围

Offline R1 让 hidden states 在**整个模型中**流转于旋转空间：

| 模块 | 修改方式 | 说明 |
|------|----------|------|
| embed_tokens | `W_embed @ R` | 输出到旋转空间 |
| RMSNorm γ | **吸收到下一层 weight** | ⚠️ 关键损失源 |
| q/k/v_proj | `R^T @ W` | 接收旋转输入 |
| o_proj | `W @ R` | 输出回旋转空间 |
| gate/up_proj | `R^T @ W` | 接收旋转输入 |
| down_proj | `W @ R` | 输出回旋转空间 |
| lm_head | `R^T @ W` | 接收旋转输入 |

Hidden states 从 embed 到 lm_head 始终在旋转空间中：
```
embed → [rotated space] → layer1 → [rotated space] → layer2 → ... → lm_head → [original space]
```

---

## 4. 为什么 Offline R1 效果比 Online 差

### 根本原因：RMSNorm γ 破坏旋转结构

**Online 的流程**（不涉及 RMSNorm）：
```
hidden → RMSNorm(hidden) * γ → x（含γ引入的per-channel outlier）
       → [hook: x @ R] → outlier 被均匀分散 ✓
       → quantize → 低误差
```

**Offline 的流程**（RMSNorm γ 是问题）：
```
hidden_rot → RMSNorm(hidden_rot) → 除以rms（标量，OK）
           → * γ（per-channel 缩放！）→ 引入新的 per-channel 不均匀 ✗
           → 没有新的旋转来修复 → outlier 重新出现
           → quantize → 高误差
```

### 详细分析

RMSNorm 的计算：`y = (x / rms(x)) * γ`

- `x / rms(x)`: rms 是标量，不改变旋转结构 — OK
- `* γ`: **per-channel 缩放**！这是致命的：

```python
# 旋转后的 hidden state 各维度分布均匀
h_rot = [2.1, 1.8, 2.0, 1.9, ...]  # 均匀 ✓

# 经过 γ 缩放后
y = h_rot * γ  # γ = [0.3, 2.5, 0.1, 1.8, ...]
y = [0.63, 4.5, 0.2, 3.42, ...]  # outlier 重新出现 ✗
```

γ 的 per-channel 缩放**破坏了旋转带来的均匀分布**，outlier 再次出现。

### 解决 γ 的尝试及其代价

Offline 模式必须"吸收" γ 到下一层权重中：

```python
# 吸收 γ: W_new = W @ diag(γ)，γ 设为全1
for weight in next_layer_weights:
    weight.data = weight.data * γ.unsqueeze(0)  # broadcast multiply
norm.weight.data.fill_(1.0)  # γ = 1
```

但这带来新问题：

| 问题 | 影响 |
|------|------|
| W 分布改变 | `W @ diag(γ)` 中大γ值放大某些列，weight outlier 重新出现 |
| FP16 精度损失 | 预乘 γ 在 FP16 下有累积误差 |
| 旋转与 γ 不可交换 | `R^T @ W @ diag(γ) ≠ R^T @ diag(γ) @ W`，优化方向矛盾 |

### 对比总结

| 维度 | Online R1 | Offline R1 |
|------|-----------|------------|
| RMSNorm γ | **不动** — hook 在 γ 之后旋转 | 必须吸收 — 引入精度损失 |
| Rotation 时机 | 在 per-channel outlier 出现后立即旋转 | 在模型最前端旋转，但 γ 再次破坏 |
| 量化看到的 activation | `x @ R`（post-γ，旋转后均匀） | `x_rot * γ`（γ 破坏均匀性） |
| 量化看到的 weight | `W @ R`（旋转后均匀） | `R^T @ W @ diag(γ)`（γ 引入不均匀） |
| 误差累积 | 无（per-module 局部） | 有（全模型链式传播） |
| embed/lm_head | 不修改 | 必须修改（边界处理复杂） |

---

## 5. 一句话总结

> **Online R1 在 RMSNorm γ 缩放之后旋转 activation，确保量化看到最均匀的分布。
> Offline R1 在 γ 缩放之前就做好旋转，但 γ 的 per-channel 缩放重新引入 outlier，
> 而吸收 γ 到权重中又破坏了 weight 的均匀性。这是 offline 效果差的根本原因。**

---

## 6. 为什么 Quark 选择 Online

1. **实现更简单**: 不需要处理 RMSNorm γ 吸收、embed/lm_head 边界
2. **数值更稳定**: 避免 FP16 下的累积误差
3. **效果更好**: 量化同时看到旋转后的 weight 和 activation
4. **灵活性**: 可随时切换 R 矩阵（random vs deterministic），无需重新 fuse
5. **兼容性**: 不改变模型的序列化格式（weight shape 不变）

---

## 7. 代码对应关系

```
# Online R1 (our implementation, matching Quark)
preprocessor.py: _apply_online_r1()
  ├── Weight rotation:  module.weight.data = (W @ R).to(dtype)     [line 737]
  ├── Activation hook:  module.register_forward_pre_hook(x @ R)    [line 779]
  └── RMSNorm:          NOT touched                                [line 264-268]

# Offline R1 (alternative, worse accuracy)
preprocessor.py: _fuse_offline_rotations()
  ├── embed_tokens:    W_embed @ R                                 [line 867]
  ├── Target modules:  R^T @ W                                     [line 854+]
  ├── prev modules:    W @ R                                       (o_proj, down_proj)
  ├── lm_head:         R^T @ W                                     (last layer)
  └── RMSNorm:         γ absorbed into weights (LOSSY)             [line 264-266]
```

---

## 8. 推理延迟影响

| 模式 | 额外计算 | 延迟增加 |
|------|----------|----------|
| Online R1 | 每个 target module 多一次 `x @ R` (或 butterfly) | ~3-5% |
| Offline R1 | 无额外计算（已 fuse 进权重） | 0% |

Online 的代价是推理时的额外矩阵乘法。但在量化场景下，精度提升远大于延迟增加。
对于 deterministic Hadamard，butterfly 算法复杂度为 O(n log n) 而非 O(n²)，
实际延迟增加可以接受。
