# Input Wrapper vs Embed Tokens Fusion: 推理时两种 Online R1 激活旋转方案的等价性分析

## 1. 问题背景

在 Online R1 rotation + quantization 流程中：

1. **训练 / 量化阶段**：target modules（q/k/v/gate/up_proj）的权重被旋转后量化，同时通过 `forward_pre_hook`（或 input wrapper）在线对输入激活做 Hadamard 旋转，保证数值正确。
2. **保存模型**：量化后的权重 $Q(W \cdot H)$ 被持久化，hook 信息不随模型保存。
3. **推理阶段**：需要恢复在线旋转，否则输入未旋转 + 权重已旋转 → 结果错误。

**核心问题**：推理时恢复在线旋转有两种思路：

| 方案 | 做法 |
|------|------|
| **A: Input Wrapper** | 在每个 target module 前加 wrapper / hook，对输入做 $x \mapsto x \cdot H$ |
| **B: Embed Tokens 融合** | 将 $H$ 融合进 `embed_tokens`（$W_e' = W_e \cdot H$），使 residual stream 从源头就处于旋转基底 |

直觉上，两者似乎等价——反正都是让 target module 收到 $x \cdot H$ 作为输入。但实际上 **它们不等价**，本文给出严格的数学推导。

---

## 2. 模型保存后的状态（两方案共用的前提）

Online R1 + quantization 完成后，保存的模型参数：

| 组件 | 状态 | 说明 |
|------|------|------|
| q/k/v_proj, gate/up_proj | $\hat{W} = Q(W \cdot H)$ | 权重已旋转 + 已量化 |
| o_proj | $\hat{W}_o = Q(W_o)$ | **未旋转**，正常量化 |
| down_proj | $\hat{W}_d = Q(W_d)$ | **未旋转**，正常量化 |
| embed_tokens | $W_e$ | **未修改** |
| lm_head | $W_h$ | **未修改** |
| RMSNorm | $\gamma$（逐元素缩放） | **未修改**（未 fuse） |

> 注意：这是 **online R1** 的特征——只修改 target modules 的权重，不动 embed_tokens / o_proj / down_proj / RMSNorm。

---

## 3. 方案 A：Input Wrapper（每个 target module 前加旋转）

### 3.1 单层 forward 完整追踪

以 transformer 第 $l$ 层为例，输入 residual 为 $x$（**原始基底**）：

```
1. Attention 分支
   ├─ x_norm = RMSNorm(x) = (x / rms(x)) ⊙ γ          ← γ 逐元素乘
   ├─ q = [x_norm · H] · Q(W_q · H)^T                  ← wrapper 先旋转输入
   ├─ k = [x_norm · H] · Q(W_k · H)^T
   ├─ v = [x_norm · H] · Q(W_v · H)^T
   ├─ attn_out = Attention(q, k, v)
   ├─ o_out = attn_out · Q(W_o)^T                       ← o_proj 无旋转
   └─ residual₁ = x + o_out                             ← 两项都在原始基底 ✓

2. MLP 分支
   ├─ x2_norm = RMSNorm(residual₁) = (residual₁ / rms(residual₁)) ⊙ γ
   ├─ gate = [x2_norm · H] · Q(W_gate · H)^T            ← wrapper 旋转
   ├─ up   = [x2_norm · H] · Q(W_up · H)^T              ← wrapper 旋转
   ├─ mlp_act = SiLU(gate) ⊙ up
   ├─ down_out = mlp_act · Q(W_d)^T                     ← down_proj 无旋转
   └─ residual₂ = residual₁ + down_out                  ← 两项都在原始基底 ✓
```

### 3.2 关键特性

- **residual stream 始终在原始基底**：$x$、$o\_out$、$down\_out$ 都在同一基底，可以安全相加。
- **旋转仅在 wrapper 内部发生**：每个 target module 的输入经过 $x_{\text{norm}} \cdot H$，与量化权重 $Q(W \cdot H)^T$ 做矩阵乘。
- **RMSNorm 的 γ 在旋转之前应用**：这一点至关重要。

q_proj 输入的精确表达式：

$$\boxed{x_{\text{rot}} = \left(\frac{x}{\text{rms}(x)} \odot \gamma\right) \cdot H}$$

逐分量展开：

$$[x_{\text{rot}}]_i = \sum_j \frac{x_j}{\text{rms}(x)} \cdot \gamma_j \cdot H_{ji}$$

矩阵形式：**先 diag(γ)，后 H**，即 $x_{\text{rot}} = x' \cdot \text{diag}(\gamma) \cdot H$，其中 $x' = x / \text{rms}(x)$。

---

## 4. 方案 B：Embed Tokens 融合（从源头旋转 residual stream）

### 4.1 思路

将 $H$ 融合进 `embed_tokens`：$W_e' = W_e \cdot H$。

这样 embed 输出就是 $e' = e \cdot H$（旋转基底），后续所有层都收到旋转后的 residual，target modules 不再需要 wrapper。

### 4.2 单层 forward 追踪

假设第 $l$ 层输入 residual 为 $x' = x \cdot H$（**旋转基底**）：

```
1. Attention 分支
   ├─ x'_norm = RMSNorm(x') = (x' / rms(x')) ⊙ γ
   │          = ((x · H) / rms(x)) ⊙ γ                  ← rms(x·H) = rms(x)
   ├─ q = x'_norm · Q(W_q · H)^T                        ← 无 wrapper，直接算
   ├─ ...
   ├─ o_out = attn_out · Q(W_o)^T                       ← o_proj 输出在原始基底！
   └─ residual₁ = x' + o_out
   │            = (x · H) + (attn_out · Q(W_o)^T)       ← ⚠️ 基底不一致！
```

### 4.3 问题一：RMSNorm 的 γ 与 H 不可交换

q_proj 输入在方案 B 中的精确表达式：

$$\boxed{x'_{\text{norm}} = \frac{x \cdot H}{\text{rms}(x)} \odot \gamma}$$

逐分量展开：

$$[x'_{\text{norm}}]_i = \left(\sum_j \frac{x_j}{\text{rms}(x)} \cdot H_{ji}\right) \cdot \gamma_i$$

矩阵形式：**先 H，后 diag(γ)**，即 $x'_{\text{norm}} = x' \cdot H \cdot \text{diag}(\gamma)$。

**对比方案 A 和 B：**

| | 方案 A (Input Wrapper) | 方案 B (Embed 融合) |
|---|---|---|
| 数学表达式 | $x' \cdot \text{diag}(\gamma) \cdot H$ | $x' \cdot H \cdot \text{diag}(\gamma)$ |
| 运算顺序 | **先 γ 后 H** | **先 H 后 γ** |
| 分量形式 | $\sum_j (x'_j \gamma_j) H_{ji}$ | $(\sum_j x'_j H_{ji}) \gamma_i$ |

两者相等的条件是：

$$\text{diag}(\gamma) \cdot H = H \cdot \text{diag}(\gamma)$$

即对角矩阵与 Hadamard 矩阵可交换。**这仅在 $\gamma$ 是常数向量（所有分量相同）时成立**。在实际模型中，RMSNorm 的 $\gamma$ 是通过训练学到的逐元素参数，各分量通常不同，因此：

$$\text{diag}(\gamma) \cdot H \neq H \cdot \text{diag}(\gamma) \quad \Rightarrow \quad \text{方案 A} \neq \text{方案 B}$$

#### 数值示例

取 $\gamma = [1.0, 0.5]$，$H = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1\\1 & -1\end{pmatrix}$，$x' = [1, 1]$：

**方案 A**（先 γ 后 H）：

$$x' \cdot \text{diag}(\gamma) = [1.0, 0.5] \quad \xrightarrow{\cdot H} \quad \frac{1}{\sqrt{2}}[1.5, 0.5]$$

**方案 B**（先 H 后 γ）：

$$x' \cdot H = \frac{1}{\sqrt{2}}[2, 0] \quad \xrightarrow{\odot \gamma} \quad \frac{1}{\sqrt{2}}[2.0, 0.0]$$

结果：$[1.5, 0.5] / \sqrt{2} \neq [2.0, 0.0] / \sqrt{2}$。**不相等。**

### 4.4 问题二：Residual Stream 基底不一致

即使忽略 RMSNorm γ 的问题，方案 B 还有更根本的错误——**residual stream 中混合了不同基底的向量**。

在方案 B 中：

$$\text{residual}_1 = \underbrace{x \cdot H}_{\text{旋转基底}} + \underbrace{\text{attn\_out} \cdot Q(W_o)^T}_{\text{原始基底}}$$

- $x \cdot H$：来自 embed_tokens 的旋转输出，处于**旋转基底**。
- $\text{attn\_out} \cdot Q(W_o)^T$：o_proj 的权重 $Q(W_o)$ **未经旋转**，其输出处于**原始基底**。

**两个不同基底的向量直接相加，数学上没有意义。** 这不是精度损失，而是 **计算错误**。

要修复这个问题，必须让 o_proj 的输出也处于旋转基底。这要求：

$$W_o' = H^T \cdot W_o \quad \Rightarrow \quad \text{需要} \; Q(H^T \cdot W_o)$$

但模型保存的是 $Q(W_o)$（未旋转的权重经过量化）。你无法从 $Q(W_o)$ 恢复出 $Q(H^T \cdot W_o)$，因为：

$$H^T \cdot Q(W_o) \neq Q(H^T \cdot W_o)$$

量化是非线性操作，不能与线性变换交换顺序。同理，down_proj 也有相同问题。

### 4.5 问题总结

方案 B（Embed Tokens 融合）存在两个独立的致命问题：

| 问题 | 原因 | 能否修复 |
|------|------|----------|
| RMSNorm γ 与 H 不可交换 | $\text{diag}(\gamma) \cdot H \neq H \cdot \text{diag}(\gamma)$ | 需要 fuse γ 到权重中（但权重已量化，无法 fuse） |
| Residual stream 基底不一致 | o_proj / down_proj 输出在原始基底，embed 输出在旋转基底 | 需要重新旋转 + 重新量化 o_proj / down_proj |

两个修复方案都要求**重新量化**模型，等于推翻了 online R1 + quantization 的全部结果。

---

## 5. 为什么 Offline R1 可以用 Embed Tokens 融合？

Offline R1 之所以可以将旋转融合进 embed_tokens 且不需要推理 wrapper，是因为它在量化之前就做了完整的全局基底变换：

| 操作 | Offline R1 做了什么 | Online R1 做了什么 |
|------|--------------------|--------------------|
| embed_tokens | $W_e' = W_e \cdot H$ | ❌ 不修改 |
| RMSNorm γ | **fuse γ 到相邻权重**（γ → 全 1） | ❌ 不修改 |
| q/k/v/gate/up_proj | $W' = \text{diag}(\gamma) \cdot W \cdot H$（吸收 γ） | $W' = W \cdot H$ |
| o_proj | $W_o' = H^T \cdot W_o$（输出旋转） | ❌ 不修改 |
| down_proj | $W_d' = H^T \cdot W_d$（输出旋转） | ❌ 不修改 |
| lm_head | $W_h' = \text{diag}(\gamma) \cdot W_h$（吸收最后一层 γ） | ❌ 不修改 |

关键差异：

1. **RMSNorm γ 被 fuse**：LayerNorm 变成 $x / \text{rms}(x)$（无逐元素缩放），纯标量缩放与正交旋转可交换。
2. **o_proj / down_proj 的输出被旋转**：保证 residual stream 始终在旋转基底。
3. **所有修改在量化之前完成**：量化看到的是已经变换后的权重分布。

这就是 offline R1 不需要推理 wrapper 的原因——整个模型在同一个基底下，端到端一致。

---

## 6. 完整对比

```
方案 A: Input Wrapper（正确✅）
═══════════════════════════════

embed_tokens ─→ x（原始基底）
                │
                ▼
         ┌─ RMSNorm(x) = x/rms(x) ⊙ γ ──→ x_norm（原始基底）
         │                                       │
         │                              ┌────────┘
         │                              ▼
         │                   Wrapper: x_norm · H ──→ q/k/v/gate/up_proj
         │                                              │
         │                              o_proj / down_proj（无旋转）
         │                                              │
         │                                              ▼
         └──────────── + ──────────────────── output（原始基底）✓
                   (residual)


方案 B: Embed Tokens 融合（错误❌）
════════════════════════════════════

embed_tokens·H ─→ x·H（旋转基底）
                    │
                    ▼
         ┌─ RMSNorm(x·H) = (x·H)/rms(x) ⊙ γ     ← γ 在旋转基底上逐元素乘（错！）
         │                                              │
         │                              ┌───────────────┘
         │                              ▼
         │                   直接进 q/k/v/gate/up_proj（无 wrapper）
         │                                              │
         │                              o_proj / down_proj（输出在原始基底）
         │                                              │
         │                                              ▼
         └──────────── + ──────────────────── 旋转基底 + 原始基底 = 💥
                   (residual 基底混乱)
```

---

## 7. 结论

> **Online R1 + quantization 后保存的模型，推理时只能使用 Input Wrapper，不能用 embed_tokens 融合。**

原因有两个，且每个都是独立的致命问题：

1. **RMSNorm 的逐元素 γ 与 Hadamard 旋转 H 不可交换**：wrapper 保证「先 γ 后 H」，与训练时一致；embed 融合变成「先 H 后 γ」，破坏了数值等价性。

2. **Residual stream 基底不一致**：embed 融合使 residual 处于旋转基底，但 o_proj / down_proj 的输出仍在原始基底（它们的权重未旋转），两者相加后 residual 毫无意义。

这两个问题的根源相同：**online R1 是局部变换**（只动 target modules），不是全局基底变换（offline R1）。局部变换不能通过在源头（embed_tokens）做一次全局旋转来等价替代。

---

## 8. 实践指导

| 场景 | 推荐做法 |
|------|----------|
| Online R1 + quant → 推理 | **必须**用 Input Wrapper（hook 或 nn.Module wrapper） |
| Offline R1 + quant → 推理 | 无需 wrapper（旋转已完全 fuse） |
| 保存/加载 online R1 量化模型 | 保存旋转配置（rotation_size, hadamard_K），加载时重新注册 hook |
| 想要无 wrapper 推理 | 使用 offline R1（但量化精度可能略低，因为权重分布变化更大） |
