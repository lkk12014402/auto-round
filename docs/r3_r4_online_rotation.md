# R3 & R4 在线旋转机制详解

## 概述

在 SpinQuant/QuaRot 的旋转框架中，R1 和 R2 是 **离线（offline）** 旋转——它们在推理前直接融合进模型权重，推理时不需要额外计算。而 R3 和 R4 是 **在线（online）** 旋转——它们必须在推理时实时应用到激活值（activation）上。

| 旋转 | 位置 | 方式 | 融合目标 |
|------|------|------|----------|
| R1 | 残差流 (residual stream) | 离线融合进权重 | embed, q/k/v/o_proj, gate/up/down_proj, lm_head |
| R2 | 注意力头内 (per-head) | 离线融合进权重 | v_proj (输出通道), o_proj (输入通道) |
| R3 | Q/K 在 RoPE 之后 | **在线** — monkeypatch hook | 无法融合 (因在RoPE之后) |
| R4 | MLP down_proj 输入 | **在线** hook + 离线半融合 | down_proj 输入通道（半融合） |

---

## R3：Q/K 在 RoPE 后的 Hadamard 旋转

### 为什么 R3 必须在线？

R3 应用于 Q 和 K，发生在 RoPE 之后。由于 RoPE 是位置相关的（position-dependent），它会改变 Q/K 的数值，因此无法把 R3 提前融合到 q_proj/k_proj 的权重中。

```
推理数据流:
  hidden → q_proj → Q → apply_rotary_pos_emb(Q) → Q_rope → [R3: Q_rope @ H] → attention
  hidden → k_proj → K → apply_rotary_pos_emb(K) → K_rope → [R3: K_rope @ H] → attention
```

### 数学原理

R3 对 Q 和 K 同时乘以正交 Hadamard 矩阵 H：

```
Attention(Q, K) = softmax(Q @ K^T / √d)
R3后:  softmax((Q@H) @ (K@H)^T / √d) = softmax(Q @ H @ H^T @ K^T / √d) = softmax(Q @ K^T / √d)
```

因为 H 是正交矩阵（H @ H^T = I），R3 不改变注意力分数。但它改变了 Q 和 K 的数值分布，使量化更友好。

### 实现机制：Monkeypatch

我们使用 **函数级 globals 替换** 来注入 R3，而不是重写整个 attention forward：

```python
# 原始的 attention forward 代码（HuggingFace transformers 中）:
def forward(self, hidden_states, ...):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    ...
    q, k = apply_rotary_pos_emb(q, k, cos, sin)  # ← 我们替换这个函数
    ...
```

注入步骤：
1. 获取 attention forward 方法的 `__globals__` 字典
2. 将 `apply_rotary_pos_emb` 替换为 `QKRotationWrapper(original_apply_rotary_pos_emb)`
3. 用新的 globals 创建 forward 方法的副本
4. `QKRotationWrapper.forward()` 调用原始 RoPE 后，立即对 Q/K 执行 Hadamard

```python
class QKRotationWrapper(nn.Module):
    def forward(self, *args, **kwargs):
        q, k = self.original_func(*args, **kwargs)  # 先正常执行 RoPE
        q = matmul_hadU(q.float()).to(orig_dtype)    # 再对 Q 做 Hadamard
        k = matmul_hadU(k.float()).to(orig_dtype)    # 再对 K 做 Hadamard
        return q, k
```

### 架构通用性

这种方法适用于所有在 attention forward 中调用 `apply_rotary_pos_emb` 的 HuggingFace 模型：
- Llama, Llama2, Llama3
- Qwen2, Qwen3
- Mistral
- Phi
- Gemma
- 等等

---

## R4：MLP 激活的 Block Hadamard 旋转

### 为什么 R4 需要在线？

R4 应用于 MLP 的中间激活（intermediate activation），即 `gate_proj * up_proj` 的输出，作为 `down_proj` 的输入。由于激活函数（SiLU/GELU）是非线性的，Hadamard 无法穿过激活函数融合到权重中。

```
MLP 数据流:
  hidden → gate_proj → SiLU(·) ─┐
                                 ├─→ element-wise multiply → [R4: x @ H_block] → down_proj → output
  hidden → up_proj ─────────────┘
```

### 半融合策略

R4 使用"半融合"策略（对称变换）：
- **离线**：将 Hadamard 融合进 `down_proj` 的权重输入通道
- **在线**：hook 对激活做相同的 Hadamard 变换

```
数学等价性:
  原始:    output = activation @ W_down^T
  R4变换:  output = (activation @ H_block) @ (W_down @ H_block)^T
         = activation @ H_block @ H_block^T @ W_down^T
         = activation @ W_down^T  (因 H_block @ H_block^T = I)
```

两边都做 Hadamard，相互抵消，数学上等价。但量化时：
- 权重 `W_down @ H_block` 的分布更均匀 → 权重量化更准
- 激活 `activation @ H_block` 的分布更均匀 → 激活量化更准（如果用 W4A4/W4A8）

### Block Hadamard

当 `intermediate_size` 不是 2 的幂次时（如 Qwen3-0.6B 的 3072 = 1024 × 3），使用 block Hadamard：

```python
# 找到最大的 2^n 因子 K
K = largest_pow2_dividing(intermediate_size)  # 3072 → K=1024

# Block Hadamard: 将向量分成 M 个 K 维块，对每块做 Hadamard
x = activation.reshape(-1, M, K)  # M = intermediate_size / K = 3
x = x @ H_K.T                     # H_K 是 K×K 的归一化 Hadamard 矩阵
x = x.reshape(original_shape)
```

### 实现机制：forward_pre_hook

R4 使用 PyTorch 的 `register_forward_pre_hook` 注册在 `down_proj` 上：

```python
def r4_hook(module, input):
    act = input[0]
    # Block Hadamard transform
    x = act.reshape(-1, M, K)
    x = x @ H_K.T
    return (x.reshape(act.shape),)

model.layers[i].mlp.down_proj.register_forward_pre_hook(r4_hook)
```

---

## 量化时的行为

### W4A16（仅权重量化）

在 W4A16 scheme 中，只量化权重，不量化激活：

```
量化流程:
  1. 离线: R1 融合权重, R2 融合权重, R4 半融合 down_proj 权重
  2. 量化: 对融合后的权重做 INT4 量化
  3. 推理: R3 hook 对 Q/K 做在线 Hadamard, R4 hook 对 down_proj 输入做在线 Hadamard

推理路径 (带量化):
  Q: hidden → q_proj(INT4) → dequant → RoPE → R3_Hadamard → attention
  MLP: hidden → gate_proj(INT4)→SiLU ⊗ up_proj(INT4) → R4_Hadamard → down_proj(INT4) → out
```

R3 和 R4 的在线 hook 作用于 **反量化后的激活**（float16），因此不受权重量化精度影响。

### W4A4 / W4A8（权重+激活量化）

当量化激活时，R3/R4 的作用更加关键：

```
不带旋转:  activation 可能有大 outlier → 量化损失大
带旋转:    activation @ H 分散 outlier → 量化更准确
```

此时 R4 的 block Hadamard 必须发生在激活量化 **之前**：

```
MLP 推理:
  gate ⊗ up → [R4 Hadamard] → [量化激活到 INT4/INT8] → down_proj(INT4) → output
```

---

## 在线 hook 的持久性

### 推理时

hook 在模型生命周期内持久存在：
- R3 monkeypatch: 替换了 attention 的 forward 方法（持久）
- R4 pre_hook: 注册在 down_proj 上（持久）

只要模型对象存在，hook 就在。调用 `model.eval()` 和 `torch.no_grad()` 不影响 hook。

### 序列化/保存时

目前的实现中，hook 是 **运行时注入** 的：
- 保存模型时（`model.save_pretrained()`），hook 不会被保存
- 加载保存的模型后，需要重新注册 R3/R4 hook
- 但权重中已经融合了 R4 的离线部分（`W_down @ H_block`），因此**必须**重新注册在线 hook，否则推理结果错误

### 与 auto-round 量化的集成

```python
# 正确的 rotation + quantization 流程:
model = load_model()

# 1. 应用旋转（R1/R2 融合权重, R3/R4 注册 hook）
apply_rotation(model, r1=True, r2=True, r3=True, r4=True)

# 2. 量化（auto-round 的 calibration 会经过 R3/R4 hook）
ar = AutoRound(model, scheme="W4A16", iters=0)
ar.quantize()

# 3. 评估（推理时 R3/R4 hook 自动生效）
evaluate(ar.model)
```

---

## 计算开销

| 旋转 | 离线开销 | 在线推理开销 |
|------|----------|-------------|
| R1 | O(d² × L) — 一次性矩阵乘法 | 0 |
| R2 | O(h × d_head² × L) — 一次性 | 0 |
| R3 | 0 | O(n_heads × seq_len × d_head × log(d_head)) per layer |
| R4 | O(inter × K) — 一次性 | O(seq_len × inter × log(K)) per layer |

其中 `matmul_hadU` 使用蝶形算法（butterfly），复杂度为 O(n log n) 而非 O(n²)。

对于 Qwen3-0.6B (d_head=128, inter=3072, K=1024):
- R3 每层额外: ~16 × seq_len × 128 × 7 = ~14K × seq_len FLOPs
- R4 每层额外: ~seq_len × 3072 × 10 = ~30K × seq_len FLOPs
- 相比 attention (O(seq² × d)) 和 MLP (O(seq × d × inter))，开销很小（<1%）
