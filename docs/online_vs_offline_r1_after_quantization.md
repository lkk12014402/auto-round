# Online R1 vs Offline R1：量化后能否互换？

## 问题

Online R1 + 量化后保存模型时，能否把旋转 fuse 到 `embed_tokens`、`o_proj`、`down_proj` 等层中
（转成 offline R1），从而不再需要 `InputRotationWrapper`？

## 结论

**不能。** 量化后 online → offline 的转换会产生数值差异，精度会下降。原因有两个。

---

## 原因 1：RMSNorm gamma 不可交换

RMSNorm 的定义：

```
RMSNorm(x) = (x / rms(x)) ⊙ γ
```

其中 `⊙ γ` 是逐元素乘法（element-wise）。

**Offline R1** 需要残差流 (residual stream) 在旋转基下工作。设旋转矩阵为 H：

```
残差流: x_rot = x @ H
```

但 RMSNorm 中的 gamma 是逐元素操作，**不与旋转可交换**：

```
RMSNorm(x @ H) = (x @ H) / rms(x) ⊙ γ     ← gamma 作用在旋转后的维度上
H @ RMSNorm(x) = H @ ((x / rms(x)) ⊙ γ)     ← gamma 作用在原始维度上
```

这两者 **不相等**（除非 γ = 全1向量）。

因此 offline R1 必须在旋转之前先 **fuse gamma 到下一层的权重中**：

```python
# Offline R1 的步骤：
1. fuse_rmsnorm()    # γ 吸收到 q/k/v/gate/up 权重中，γ 置为 1
2. 旋转所有层        # embed→W@H, q/k/v→W@H^T, o_proj→H^T@W, ...
3. 量化
```

而 **online R1 不做 gamma fusion**。如果量化后想转 offline，就需要对 **已量化** 的权重做 gamma fusion，
这会改变量化后的数值。

---

## 原因 2：对已量化权重做旋转 ≠ 对旋转后权重做量化

量化是非线性操作。设 `Q()` 为量化函数：

```
Q(W @ H) ≠ Q(W) @ H
Q(H^T @ W) ≠ H^T @ Q(W)
```

**具体例子：** 量化后，`o_proj` 的权重是 `Q(W_o)`（原始分布上量化）。
如果保存时 fuse 旋转，变成 `H^T @ Q(W_o)`，这和 `Q(H^T @ W_o)` 不同。
后者是 offline R1 中先旋转再量化的结果。

---

## 两种方式的信号流对比

### Online R1 + 量化（当前实现）

```
embed_tokens: W_e（原始，未旋转）
    ↓ 输出: e(token)  [原始基]
RMSNorm: 有 gamma
    ↓ 输出: norm(x)  [原始基]
q_proj (wrapped):
    InputRotationWrapper: x_rot = H(norm(x))
    Linear: Q(W_q @ H) @ x_rot^T
    → q ≈ W_q @ norm(x)^T  [旋转抵消]
... 同理 k, v, gate, up ...

o_proj: Q(W_o)  输入=attn_out [原始基]
down_proj: Q(W_d)  输入=intermediate [原始基]
lm_head: W_lm（原始）输入=hidden_states [原始基]
```

**关键：** 残差流始终在 **原始基** 下，`o_proj`/`down_proj` 保持原始权重分布。

### Offline R1 + 量化

```
embed_tokens: W_e @ H（旋转后）
    ↓ 输出: e(token) @ H  [旋转基]
RMSNorm: gamma 已 fuse（γ=1）
    ↓ 输出: norm(x_rot)  [旋转基]
q_proj: Q(W_q @ H^T)  直接接收旋转基输入，不需要 wrapper
    → q ≈ W_q @ norm(x)^T  [旋转在 W 和输入中抵消]

o_proj: Q(H^T @ W_o)  输出回旋转基
down_proj: Q(H^T @ W_d)  输出回旋转基
lm_head: W_lm @ H^T  输入在旋转基
```

**关键：** 残差流在 **旋转基** 下，所有层的权重分布都被旋转改变。

### 量化后强行转 offline（错误做法）

```
embed_tokens: Q(W_e) → Q(W_e) @ H  ← ≠ Q(W_e @ H)
RMSNorm: gamma 未 fuse → 需要 fuse 到已量化权重中 ← 改变数值
o_proj: Q(W_o) → H^T @ Q(W_o)  ← ≠ Q(H^T @ W_o)
```

每一步都引入误差。

---

## 为什么 online R1 需要 wrapper 而不能"转成 offline"

| 操作 | 量化前可做？ | 量化后可做？ |
|------|------------|------------|
| fuse RMSNorm gamma | ✅ 精确 | ❌ 改变量化值 |
| 旋转 embed_tokens | ✅ 精确 | ❌ Q(W)@H ≠ Q(W@H) |
| 旋转 o_proj/down_proj | ✅ 精确 | ❌ H^T@Q(W) ≠ Q(H^T@W) |
| 移除 wrapper | ✅ (改成 offline) | ❌ 上述原因 |

**结论：量化前** online ↔ offline 可以自由转换（数学等价）。
**量化后** 必须保留 wrapper。

---

## 那保存模型时怎么办？

这正是 `InputRotationWrapperHadamard`（nn.Module 包装器）的作用：

1. **保存时**：wrapper 是 nn.Module，`save_pretrained()` 自动保存：
   - 旋转后的权重 `Q(W @ H)` → 正常保存
   - Hadamard 矩阵 → 作为 buffer 保存（`hadamard_K`）
   - `state_dict()` 重写 key 名，去掉 `.original_module.` 前缀

2. **加载时**：
   - `from_pretrained()` 加载权重（权重已经是旋转+量化后的）
   - 重新创建 `InputRotationWrapperHadamard` 包装目标模块
   - wrapper 从保存的 buffer 恢复 Hadamard 矩阵
   - 推理时自动对 input activation 做在线旋转

3. **推理时开销**：
   - 只有被 wrapper 包裹的模块（q/k/v/gate/up）需要在线旋转
   - `o_proj`、`down_proj`、`embed_tokens` 不需要任何额外操作
   - R2 完全离线 fuse，无推理开销

---

## Quark 的做法（参考）

Quark 的处理方式完全一致：

```python
# quark/torch/algorithm/rotation/rotation.py

# Online R1: 不 fuse RMSNorm，不动 embed/o_proj/down_proj
# 只旋转 target modules 的权重 + 用 InputRotationWrapperHadamard 包裹

# 保存: wrapper 作为 nn.Module 自动保存
# 加载: prepare_model_for_reloading_fake() 重建 wrapper

# Offline R1: fuse RMSNorm → 旋转所有层 → 无需 wrapper
```

---

## 总结

```
量化前:   online R1 ⟺ offline R1  （数学等价，可自由转换）
量化后:   online R1 ≠ offline R1  （量化是非线性的，不可交换）

Online R1 的优势: o_proj/down_proj 保持原始分布 → 量化质量更好
Online R1 的代价: 推理时需要 InputRotationWrapper 做在线旋转
```
