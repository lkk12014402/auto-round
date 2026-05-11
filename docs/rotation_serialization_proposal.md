# SpinQuant/QuaRot 模型序列化方案

## 1. 问题陈述

当前 auto-round 中 SpinQuant/QuaRot 的 rotation 后模型无法正确 save/load：

- **Offline rotations (R1 fused, R2 fused)**: weights 已经融合，可以正常保存 ✅
- **Online R1 (hook-based)**: `forward_pre_hook` 不会被 `save_pretrained()` 保存 ❌
- **Online R3 (monkeypatch)**: 替换的 forward 方法不会被保存 ❌
- **Online R4 (hook-based)**: `forward_pre_hook` 不会被保存 ❌

**之前尝试过的方案**: 采用 Quark 的 `InputRotationWrapper` 子模块策略，但由于 auto-round
的量化流水线（`WrapperLinear` / `WrapperWALayer`）在 `named_modules()` 遍历时会发现并尝试
二次包装内部 Linear，导致失败。

---

## 2. 三个框架的序列化策略对比

| 框架 | Offline Rotation | Online Rotation | 保存格式 | 加载方式 |
|------|-----------------|-----------------|----------|----------|
| **Quark** | 融合到 weight | `InputRotationWrapper` 子模块 + `register_buffer` | safetensors (state_dict) | 先创建空 buffer，load 后 `post_process_after_loading()` 重建 Transform |
| **llm-compressor** | 融合到 weight，然后删除子模块 | 子模块 (Parameter) + hook | safetensors + config.json 中 `transform_config` | HFQuantizer 接口 + vLLM 重建 hook |
| **auto-round (现有 hadamard)** | 融合到 weight | `patch_quantlinear()` 在 pack 时 `register_buffer("hadamard_matrix")` | safetensors | QuantLinear forward 使用 buffer |

### 关键观察

1. **Quark**: 采用子模块替换，不会遇到 auto-round 的二次包装问题（因为 Quark 自己控制量化流水线）
2. **llm-compressor**: offline 融合后删除子模块（不保存）；online 保留子模块但 vLLM 需要特殊 PR 支持
3. **auto-round (现有)**: 最优雅——`patch_quantlinear()` 在 **pack 阶段**（量化之后）将 rotation 写入 QuantLinear 的 buffer，避免了量化过程中的冲突

---

## 3. 方案设计

### 3.1 核心思路：分两阶段处理

```
[训练/校准阶段]              [打包/保存阶段]
Rotation → Quantization → Pack（写入 rotation 信息到 QuantLinear）→ Save
                                                                      ↓
[加载/推理阶段]
Load QuantLinear → 检测 rotation buffers → 重建 online rotation → Inference
```

**关键设计原则：**
- 量化阶段，rotation 用 hook 方式工作（兼容 `WrapperLinear`）
- pack 阶段，将 rotation 信息写入 QuantLinear 的 buffer（序列化）
- 加载阶段，从 buffer 重建 online rotation

### 3.2 各 Rotation Level 的序列化策略

| Level | 量化阶段 | 保存什么 | 加载后如何恢复 |
|-------|---------|---------|--------------|
| **R1 (offline fused)** | 直接融合到 weight，无需 hook | 无需额外保存（weight 已变） | 无需恢复 |
| **R1 (online)** | hook + weight 已 rotated | rotation_size + hadamard_K (如果非 pow2) 写入 QuantLinear buffer | 从 buffer 重建 hook 或用 `InputRotationWrapperHadamard` |
| **R2 (offline fused)** | 直接融合到 weight | 无需额外保存 | 无需恢复 |
| **R3 (online)** | monkeypatch forward | head_dim（确定性 Hadamard，可从 config 重建） | 重新 monkeypatch |
| **R4 (online)** | forward_pre_hook on down_proj | rotation_size + had_K/K 写入 QuantLinear buffer | 从 buffer 重建 hook |

### 3.3 具体实现方案

#### 方案 A：扩展 `patch_quantlinear` 模式（推荐）

**优势**: 复用 auto-round 已有的、被验证过的模式。

**原理**: auto-round 的 `patch_quantlinear()` 已经证明可以在 pack 阶段将 rotation matrix
写入 QuantLinear 的 buffer，并且 QuantLinear 的 forward 可以使用该 buffer。我们对 SpinQuant
的 online rotations 做同样的事。

```python
# 保存阶段: patch_quantlinear_spinquant()
def patch_quantlinear_spinquant(model, spinquant_config):
    """在 pack 阶段，将 SpinQuant online rotation 信息注入 QuantLinear。"""
    
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        
        # Online R1: target modules (q/k/v/gate/up_proj)
        if is_online_r1_target(name, spinquant_config):
            rotation_size = spinquant_config.r1_rotation_size or module.infeatures
            had_K, K = get_hadamard_K(rotation_size)
            module.register_buffer("spinquant_r1_had_K", had_K)
            module.register_buffer("spinquant_r1_K", torch.tensor(K))
            module.register_buffer("spinquant_r1_rotation_size", 
                                   torch.tensor(rotation_size))
        
        # Online R4: down_proj
        if is_r4_target(name, spinquant_config):
            r4_size = spinquant_config.r4_rotation_size
            had_K, K = get_hadamard_K(r4_size)
            module.register_buffer("spinquant_r4_had_K", had_K)
            module.register_buffer("spinquant_r4_K", torch.tensor(K))
            module.register_buffer("spinquant_r4_rotation_size",
                                   torch.tensor(r4_size))
```

```python
# 加载阶段: 扩展 QuantLinear.forward()
class QuantLinear:
    def forward(self, x):
        # 在原始 forward 前应用 online rotation
        if hasattr(self, "spinquant_r1_had_K"):
            x = matmul_hadU(x, self.spinquant_r1_had_K, int(self.spinquant_r1_K))
        
        if hasattr(self, "spinquant_r4_had_K"):
            x = matmul_hadU(x, self.spinquant_r4_had_K, int(self.spinquant_r4_K))
        
        # ... 原始量化 forward ...
```

**R3 处理**: R3 是确定性 Hadamard（取决于 head_dim），可以只保存 config 信息，加载时重建 monkeypatch。

```python
# 在 config.json 的 quantization_config 中保存
"spinquant_config": {
    "r1": true, "r2": true, "r3": true, "r4": true,
    "online_r1_rotation": true,
    "r1_rotation_size": null,  # null = hidden_size
    "r4_rotation_size": null,  # null = intermediate_size
    "head_dim": 128
}
```

#### 方案 B：`InputRotationWrapperHadamard` + 延迟替换

**思路**: 不在量化阶段替换模块，而是在 **pack 之后、save 之前** 替换。

```
Quantization(用 hook) → Pack(生成 QuantLinear) → Replace QuantLinear with Wrapper → Save
```

```python
# Save 阶段
def wrap_for_serialization(model, spinquant_config):
    """在 pack 完成后，将需要 online rotation 的 QuantLinear 包装。"""
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear) and is_online_r1_target(name):
            wrapper = QuantLinearWithRotation(module, spinquant_config)
            set_module(model, name, wrapper)
```

**问题**: 需要新建 `QuantLinearWithRotation` 类，改动较大，且需要处理 `save_pretrained` 时的
key 重命名。

#### 方案 C：纯 config + hook 重建（最轻量）

**思路**: 只保存 config 信息到 `config.json`，加载时从 config 重建所有 online hooks。

```python
# Save: 把 spinquant config 写入 config.json
model.config.spinquant_config = spinquant_config.to_dict()
model.save_pretrained(...)

# Load: 检测 config 并重建
model = AutoModelForCausalLM.from_pretrained(...)
if hasattr(model.config, "spinquant_config"):
    rebuild_spinquant_hooks(model, SpinQuantConfig(**model.config.spinquant_config))
```

**优势**: 最简单，无需改 QuantLinear  
**劣势**: 
- 确定性 Hadamard 可以重建，但 random/trained rotation 不行
- 用户需要手动调用 rebuild（或注册 HFQuantizer）
- 与 auto-round 现有的 rotation 保存模式不一致

---

## 4. 推荐方案：A + C 混合

### 4.1 设计决策

| 场景 | 策略 |
|------|------|
| R1 offline + R2 offline（默认 quarot） | 融合到 weight，零额外开销 |
| R1 online（确定性 Hadamard） | **方案 A**: QuantLinear buffer + patched forward |
| R3 online | **方案 C**: config 保存 + 加载时重建 monkeypatch |
| R4 online | **方案 A**: QuantLinear buffer + patched forward |
| R1/R2 trained (非确定性) | **方案 A** + 实际矩阵存入 buffer |

### 4.2 为什么不全用方案 C？

- R1 online 对应的 weight 已经被 rotated（`W_new = matmul_hadU(W)`），QuantLinear pack 的
  是 rotated weight。推理时 **必须** 对 activation 做相应 rotation，否则精度错误。
- 如果 rotation 信息只在 config 里，用户忘了调 rebuild 就会产生错误结果且没有报错。
- 写入 QuantLinear buffer 后，forward 自动应用，用户无需额外操作。

### 4.3 为什么 R3 用方案 C 而不是 A？

- R3 作用在 attention 的 Q/K **输出**（RoPE 之后），不是 Linear 的输入
- R3 不对应任何特定 QuantLinear 的 pack（它夹在两个 Linear 之间）
- R3 是确定性 Hadamard（由 head_dim 完全决定），无需存储矩阵
- monkeypatch 重建逻辑简单、成熟

### 4.4 实现步骤

```
Step 1: patch_quantlinear_spinquant()
        - 在 BaseCompressor._apply_rotations() 之后、量化之前调用
        - 但实际 buffer 写入在 pack 阶段（lazy）
        - 或者：在 save 之前统一注入

Step 2: 扩展 QuantLinear.forward() / QuantLinear.dequant_and_forward()
        - 检测 spinquant_r1_* / spinquant_r4_* buffer
        - 若存在，在 forward 前应用 rotation

Step 3: spinquant_config 写入 config.json
        - 在 save_pretrained 时注入
        - 包含 r3 信息、rotation_size 等

Step 4: 加载时自动重建 R3
        - 注册 post_init hook 或扩展 from_pretrained
        - 检测 config.spinquant_config，调用 register_spinquant_hooks(r3_only)
```

---

## 5. 详细实现设计

### 5.1 数据流图

```
┌─────────────── 训练/校准阶段 ─────────────────┐
│                                                │
│  Model (fp16/bf16)                             │
│     ↓                                          │
│  SpinQuantPreprocessor.preprocess()            │
│     ├── R1 online: rotate weight + hook        │
│     ├── R2 offline: fuse into v/o weight       │
│     ├── R3 online: monkeypatch attention       │
│     └── R4 online: hook on down_proj           │
│     ↓                                          │
│  Quantization (RTN/GPTQ/AutoRound)            │
│     ↓                                          │
│  Pack → QuantLinear                            │
│                                                │
└────────────────────────────────────────────────┘
                        ↓
┌─────────────── 保存阶段 ──────────────────────┐
│                                                │
│  inject_spinquant_buffers(model, config)       │
│     ├── R1 target QuantLinear:                 │
│     │      .register_buffer("spinquant_r1_*") │
│     ├── R4 target QuantLinear (down_proj):     │
│     │      .register_buffer("spinquant_r4_*") │
│     └── config.json: spinquant_config          │
│     ↓                                          │
│  model.save_pretrained(save_dir)               │
│     → safetensors 含 spinquant buffers         │
│     → config.json 含 spinquant_config          │
│                                                │
└────────────────────────────────────────────────┘
                        ↓
┌─────────────── 加载/推理阶段 ─────────────────┐
│                                                │
│  model = AutoModelForCausalLM.from_pretrained()│
│     → QuantLinear 自动恢复 spinquant buffers   │
│     ↓                                          │
│  rebuild_spinquant_online(model)               │
│     ├── QuantLinear.forward 自动用 R1/R4 buffer│
│     └── R3: 从 config 重建 monkeypatch         │
│     ↓                                          │
│  model.generate() / lm_eval                    │
│                                                │
└────────────────────────────────────────────────┘
```

### 5.2 文件改动列表

| 文件 | 改动 |
|------|------|
| `auto_round/algorithms/transforms/spinquant/serialize.py` | **新建** - `inject_spinquant_buffers()`, `rebuild_spinquant_online()` |
| `auto_round/algorithms/transforms/spinquant/__init__.py` | 导出新函数 |
| `auto_round/quantizer.py` 或 compressors | 在 save 前调用 `inject_spinquant_buffers()` |
| `auto_round/data_type/QuantLinear` (或 monkey-patch) | forward 检测 spinquant buffer 并应用 |
| `config.json` 输出 | 增加 `spinquant_config` 字段 |

### 5.3 Buffer 命名规范

```
QuantLinear state_dict keys:
  model.layers.0.self_attn.q_proj.weight          # 量化后 weight
  model.layers.0.self_attn.q_proj.weight_scale    # 量化 scale
  model.layers.0.self_attn.q_proj.spinquant_r1_had_K      # Hadamard matrix
  model.layers.0.self_attn.q_proj.spinquant_r1_K          # K value (scalar)
  model.layers.0.self_attn.q_proj.spinquant_r1_size       # rotation_size
  model.layers.0.mlp.down_proj.spinquant_r4_had_K         # R4 Hadamard
  model.layers.0.mlp.down_proj.spinquant_r4_K             # R4 K value
  model.layers.0.mlp.down_proj.spinquant_r4_size          # R4 rotation_size
```

### 5.4 config.json 结构

```json
{
  "model_type": "qwen3",
  "quantization_config": {
    "quant_method": "intel/auto-round",
    "bits": 4,
    "group_size": 128,
    "spinquant_config": {
      "r1": true,
      "r2": true,
      "r3": true,
      "r4": true,
      "online_r1_rotation": true,
      "r1_rotation_size": null,
      "r4_rotation_size": null,
      "head_dim": 128,
      "random_r1": false,
      "random_r2": false
    }
  }
}
```

---

## 6. 替代方案：纯 Offline 模式（最简单）

如果短期内不想处理 online rotation 的序列化复杂度：

```python
# 默认 QuaRot config: 全部 offline fuse
SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    online_r1_rotation=False,  # ← 关键：offline R1
    trainable_rotation=False,
)
```

**这种模式下**：
- R1: 融合到所有相关 weight（embed、q/k/v、gate/up、o_proj、down_proj、lm_head）
- R2: 融合到 v_proj output, o_proj input
- R3/R4: 关闭
- **无需任何额外序列化逻辑** — `save_pretrained()` 直接保存 rotated+quantized weights

**精度影响**：
- R1+R2 offline 在大多数场景下精度足够好
- R3/R4 主要用于极端量化场景（如 W4A4 或 FP4）

---

## 7. 实现优先级建议

| 优先级 | 任务 | 复杂度 | 价值 |
|--------|------|--------|------|
| P0 | 纯 Offline 模式验证（R1+R2 offline → save → load → inference） | 低 | 高 |
| P1 | Online R1 序列化（QuantLinear buffer + forward patch） | 中 | 中 |
| P2 | Online R4 序列化（同 R1 方式） | 中 | 中 |
| P3 | R3 config 重建 | 中 | 低（R3 主要用于 A4 场景） |
| P4 | trained/random rotation 矩阵持久化 | 高 | 低（实验性功能） |

---

## 8. 开放问题

1. **QuantLinear forward 修改**: 直接修改 QuantLinear 源码还是用 monkey-patch？
   - 修改源码更清晰，但影响面大
   - monkey-patch 类似现有 `patch_quantlinear()`，但 forward patch 更侵入
   
2. **加载时 R3 重建的触发点**: 
   - 选项 A: 用户手动调用 `rebuild_spinquant_online(model)`
   - 选项 B: 注册 `AutoRound.from_pretrained()` 的 post-load hook
   - 选项 C: 写 HFQuantizer 扩展（最正规，但工程量大）

3. **与现有 hadamard rotation 的兼容**: 
   - 现有 `patch_quantlinear()` 已经在 QuantLinear 上注册了 `hadamard_matrix` buffer
   - SpinQuant 的 R1 online rotation 在 **相同的 QuantLinear** 上再加 buffer 是否冲突？
   - 逻辑上不冲突（不同的 buffer 名），但 forward 的 rotation 顺序需要明确

4. **Offline R1 是否需要修改 embed_tokens 和 lm_head？**
   - 是的，offline R1 需要 rotate 所有 residual stream 的入口和出口
   - `untie_embeddings` + `fuse_rmsnorm` 是前置条件
   - 这些操作改变了模型结构（RMSNorm 被 fuse），可能影响 HuggingFace 的模型加载

---

## 9. 建议下一步

1. **立即可做**: 验证 R1+R2 offline 模式下 save → load → inference 的完整流程
2. **讨论确认**: 选择 online rotation 的序列化方案（方案 A vs B vs C）
3. **原型实现**: 基于确认的方案，实现 `serialize.py`
