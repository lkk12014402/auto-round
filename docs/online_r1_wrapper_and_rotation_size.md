# Online R1 Wrapper 与 Rotation Size 对比文档

## 1. Auto-Round 当前实现 vs Quark 的 InputRotationWrapper

### Quark: `InputRotationWrapperHadamard`（nn.Module）

```python
class InputRotationWrapperHadamard(InputRotationWrapper):
    """nn.Module 包装器，替换原始 Linear 层"""
    
    def __init__(self, original_module: nn.Linear, rotation_size: int):
        self.original_module = original_module   # 原始 Linear (权重已旋转)
        self.transform = HadamardTransform(...)  # 做 x @ R
        self.register_buffer("input_rotation", rotation_int8)  # 可序列化
    
    def forward(self, x):
        x = self.transform(x)         # 对 activation 做 Hadamard
        # 量化发生在这里 (activation quantization)
        x = self.original_module(x)    # 旋转后的 Linear
        return x
```

**关键特性：**
- 是 `nn.Module` 子类 → `model.state_dict()` / `save_pretrained()` 自动保存
- `state_dict()` 重写：去掉 `.original_module.` 前缀，保持与原始权重名一致
- `input_rotation` 存为 `int8` buffer（±1 值），推理时恢复 float
- 代理 `weight`/`bias`/`in_features`/`out_features` 到 `original_module`

### Auto-Round: `register_forward_pre_hook`（当前实现）

```python
def _make_online_r1_hook(hadamard_K, K, rotation_size, in_features):
    """返回一个 hook closure"""
    def hook(module, args):
        x = args[0]
        x_rotated = matmul_hadU(x, hadamard_K, K)  # 或 block rotation
        return (x_rotated,) + args[1:]
    return hook

# 注册
module.register_forward_pre_hook(hook)
```

**局限性：**
- ❌ `forward_pre_hook` **不可序列化** — `save_pretrained()` 不会保存 hook
- ❌ 加载模型后 hook 丢失 → 推理结果错误
- ✅ 对评估（evaluation）没问题 — 因为我们在同一进程中先 rotation 再 eval

### 差距与改进方向

| 特性 | Quark | Auto-Round (当前) | 需要改进？ |
|------|-------|------------------|-----------|
| 推理正确性 (same process) | ✅ | ✅ | - |
| 模型保存/加载 | ✅ (nn.Module) | ❌ (hook 丢失) | 是 |
| Activation 量化支持 | ✅ (hook 在量化层之前) | N/A | 未来 |
| 权重名兼容性 | ✅ (state_dict 重写) | ✅ (无额外层) | - |

**TODO**: 实现 `InputRotationWrapperHadamard` 等效的 nn.Module，使模型可正确保存/加载。

---

## 2. Rotation Size 配置对比

### Quark

Quark 使用**单一** `rotation_size` 参数（在 `RotationConfig` 中），影响 R1、R3、R4：

```python
# RotationConfig
rotation_size: int | None = None  # None = use model dimension

# 在 RotationProcessor 中的实际使用:
# R1:
if rotation_size is not None:
    r1_rotation_size = rotation_size      # 使用配置值
else:
    r1_rotation_size = model.config.hidden_size   # 默认 full hidden_size

# R2: 始终用 head_dim（由模型架构决定）

# R3: 
if rotation_size is not None:
    raise NotImplementedError("R3 does not support custom rotation_size")
# R3 始终用 head_dim

# R4:
if rotation_size is not None:
    r4_rotation_size = rotation_size      # 使用配置值
else:
    r4_rotation_size = model.config.intermediate_size  # 默认 full intermediate
```

**Quark 默认值**: `rotation_size=128`（在 MXFP4 脚本中）

### Auto-Round

Auto-Round 同样使用单一 `rotation_size` 参数：

```python
# SpinQuantConfig
rotation_size: Optional[int] = None   # None = use model dimension

# 内部使用:
self.r1_rotation_size = config.rotation_size or self.hidden_size      # R1
self.r2_rotation_size = self.head_dim                                  # R2: 始终 head_dim
self.r3_rotation_size = self.head_dim                                  # R3: 始终 head_dim
self.r4_rotation_size = config.rotation_size or self.intermediate_size # R4
```

### Normalization 差异（已修复）

| | Quark `_get_hadamard_K()` | Auto-Round `get_hadamard_K()` |
|---|---|---|
| 返回值 | **未归一化** (±1 值) | **已归一化** (H/√N) |
| 归一化位置 | 调用者做 (如 `/sqrt(rotation_size)`) | 函数内部已做 |
| `matmul_hadU` | butterfly 后 `/sqrt(n)` | butterfly 后 `/sqrt(n/K)` |

**注意**: 我们之前的 bug 就是因为 `get_hadamard_K` 返回已归一化矩阵，代码又除了一次 `√N`。

### Qwen3-0.6B 各旋转的实际 size

| 旋转 | 维度 | rotation_size=128 | rotation_size=None |
|------|------|-------------------|-------------------|
| R1 | hidden_size=1024 | 128 (8 blocks) | 1024 (full) |
| R2 | head_dim=128 | 128 (full) | 128 (full) |
| R3 | head_dim=128 | 128 (full) | 128 (full) |
| R4 | intermediate=3072 | 128 (24 blocks) | 3072 (需要 12×12 Hadamard) |

---

## 3. Quark 的模型保存流程 (Online R1 + R2 + Quantization)

### 完整流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Apply Rotations                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  R1 (online):                                            │
│    q_proj.weight = W_q @ R1        (永久修改)            │
│    q_proj → InputRotationWrapperHadamard(q_proj)         │
│    (同样: k_proj, v_proj, gate_proj, up_proj)            │
│                                                          │
│  R2 (offline):                                           │
│    v_proj.weight = R2.T @ W_v      (per-head 输出旋转)   │
│    o_proj.weight = W_o @ R2        (per-head 输入旋转)   │
│    (无 wrapper，完全 fused)                              │
│                                                          │
│  此时模型结构:                                           │
│    model.layers.0.self_attn.q_proj = Wrapper(Linear)     │
│    model.layers.0.self_attn.k_proj = Wrapper(Linear)     │
│    model.layers.0.self_attn.v_proj = Wrapper(Linear)     │ ← R2 fused in weight
│    model.layers.0.self_attn.o_proj = Linear              │ ← R2 fused in weight
│    model.layers.0.mlp.gate_proj    = Wrapper(Linear)     │
│    model.layers.0.mlp.up_proj      = Wrapper(Linear)     │
│    model.layers.0.mlp.down_proj    = Linear              │
│                                                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Quantization (calibration + freeze)              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Wrapper.forward():                                      │
│    x → HadamardTransform(x)  → [act_quant] → Linear(x)  │
│                                                          │
│  Quark 的 quantizer 发现 wrapper.original_module 是      │
│  nn.Linear，对其做权重量化 (MXFP4):                      │
│    original_module.weight → quantized weight              │
│                                                          │
│  freeze: 将 FakeQuantize → 真实量化格式                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Export / Save                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  fake_quantized 格式 (用于 evaluation/reload):           │
│    model.save_pretrained(path)                           │
│    - Wrapper 是 nn.Module → 自动保存                     │
│    - input_rotation buffer (int8) 被保存                 │
│    - state_dict 重写确保权重名不变                        │
│                                                          │
│  real_quantized 格式 (用于部署):                         │
│    使用 QParamsLinearWithRotation:                        │
│    - 保存量化参数 + input_rotation                       │
│    - 加载时重建 wrapper                                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Reload / Inference                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  fake_quantized:                                         │
│    1. 加载模型 (普通 Linear)                              │
│    2. prepare_model_for_reloading_fake():                 │
│       - 查找 online rotation layers (from config)        │
│       - 为这些层 register_buffer("input_rotation", ...)   │
│    3. load_state_dict() 填入 input_rotation 值           │
│    4. 重建 InputRotationWrapper                          │
│                                                          │
│  real_quantized:                                         │
│    QParamsLinearWithRotation.forward() 内部处理           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 关键问答

**Q: R1 需要 offline fuse 才能保存吗？**

**A: 不需要。** Online R1 的 weight 侧已经是 "fused" 的（`W' = W @ R`），这就是保存的权重。
Activation 侧通过 `InputRotationWrapperHadamard`（nn.Module）或 `input_rotation` buffer 保存。
加载时重建 wrapper → 推理正确。

**Q: R2 保存需要额外处理吗？**

**A: 不需要。** R2 完全 offline fused（V output + O input），没有 wrapper，权重直接保存即可。

**Q: 如果不保存 wrapper，直接加载会怎样？**

**A:** 权重是 `W @ R`（已旋转），但 activation 没有对应的 `x @ R`，计算变为：
```
y = x @ (W @ R).T = x @ R.T @ W.T ≠ x @ W.T
```
结果错误，精度会非常差（类似我们之前 double-normalization bug 的效果）。

---

## 4. Auto-Round 当前状态与改进路径

### 当前状态（评估可用，保存不可用）

```
rotation_size=128, online_r1=True:
  - R1: hook (forward_pre_hook) ← 不可序列化
  - R2: offline fused ← 保存正常
  - R3: monkeypatch ← 不可序列化
  - R4: hook + weight fused ← hook 不可序列化
```

**当前工作流：**
1. 加载模型 → rotation → quantize → evaluate（同一进程，OK）
2. 如果需要保存 → ❌ rotation hooks 丢失

### 改进路径

**Phase 1**（当前）: Evaluation-only — hook 方案足够

**Phase 2**（未来）: 模型保存/部署
1. 实现 `InputRotationWrapperHadamard` (nn.Module) 替代 hook
2. 实现 `state_dict` 重写（去掉 `.original_module.` 前缀）
3. 实现 reload 逻辑（从 config/buffer 重建 wrapper）

**Phase 3**（未来）: 与 auto-round 量化管道集成
- Wrapper 在量化层之前 → activation quantization 看到旋转后的分布
- 与 auto-round 的 `AutoRound.quantize()` 流程兼容
