# Rotation + Quantization Scheme 兼容性分析

## 1. 结论

**Rotation（QuaRot/SpinQuant）是完全 scheme-agnostic 的。** 无论 Quark 还是 Auto-round，rotation 都作为 pre-quantization 步骤独立执行，不关心后续使用什么量化方式。

---

## 2. Quark 支持的量化 Scheme（17种）

来源: `quark/torch/quantization/config/template.py`

| 类别 | Scheme | 说明 |
|------|--------|------|
| **INT4 weight-only** | `int4_wo_32` | group_size=32 |
| | `int4_wo_64` | group_size=64 |
| | `int4_wo_128` | group_size=128 |
| | `int4_wo_per_channel` | per-channel |
| **UINT4 weight-only** | `uint4_wo_32/64/128/per_channel` | 同上，无符号 |
| **INT8** | `int8` | 全精度整型量化 |
| **FP8** | `fp8` | E4M3 / E5M2 |
| | `ptpc_fp8` | Per-Token Per-Channel FP8 |
| **MX (Microscaling)** | `mxfp4` | block_size=32, E2M1 |
| | `mxfp6_e3m2` | block_size=32, E3M2 |
| | `mxfp6_e2m3` | block_size=32, E2M3 |
| | `mxfp4_mxfp6_e2m3` | 混合精度 |
| | `mxfp4_fp8` | 混合精度 |
| | `mx6` | MX6 格式 |
| **BF16** | `bfp16` | Block FP16 |

### Quark 中 Rotation + Scheme 的测试证据

| Scheme | 测试文件 | 状态 |
|--------|---------|------|
| `int4_wo_128` | `test/test_for_torch/test_llm_template.py` — `test_rotation_algorithm()` | ✅ 通过 |
| `uint4` (weight-only) | `test/test_for_torch/test_smoke.py` — `test_smoke_rotation()` | ✅ 通过 |
| `mxfp4` | `examples/torch/language_modeling/rotation/run_mxfp4_spinquant.sh` | ✅ 示例 |

### Quark 的设计：Rotation 与 Scheme 完全独立

```python
# template.py — QuantConfig 构造
config = QuantConfig(
    scheme="mxfp4",                    # 量化 scheme（独立）
    algorithm=[RotationConfig(...)],   # rotation 算法（独立）
)
```

- `RotationConfig` 是独立的 `AlgoConfig`，与 `scheme` 参数完全正交
- 执行顺序：`rotation.apply()` → `quantizer.quantize()` → `export()`
- rotation 代码中无任何对 scheme/data_type 的检查

---

## 3. Auto-round SpinQuant 实现的 Scheme 兼容性

### 3.1 架构设计

```python
# compressors_new/base.py — 执行顺序
Phase 1: resolve_scheme         # 确定量化 scheme
Phase 2: resolve_formats        # 确定导出格式
Phase 3: patch_model            # 模型结构修改
Phase 4: build_layer_config     # 构建量化层配置
Phase 4.5: _apply_rotations()   # ← Rotation 在这里执行
Phase 5: quantize               # 量化（RTN/SignRound/Calibration）
Phase 6: export                 # 导出保存
```

Rotation 在 Phase 4.5 执行，只做两件事：
1. 修改权重（offline fuse R1/R2/R4 weight side）
2. 注册 `forward_pre_hook`（online R1/R3/R4 input side）

**完全不感知后续的量化 scheme。**

### 3.2 各 Scheme 的兼容性分析

| Scheme | 兼容性 | Hook 执行路径 | 验证状态 |
|--------|--------|--------------|---------|
| **MXFP4 (RTN)** | ✅ | `WrapperWALayer.forward()` 执行 stolen hooks | ✅ hellaswag=0.4550 |
| **W4A16 INT4 (RTN)** | ✅ | `WrapperLinear.forward()` 执行 orig_layer._forward_pre_hooks | ❌ 未测 |
| **W4A16 INT4 (SignRound)** | ✅ | 同上，多轮优化中 hook 持续生效 | ❌ 未测 |
| **NVFP4** | ✅ | calibration forward 中 hook 自动生效 | ❌ 未测 |
| **FP8** | ✅ | 同 INT4 路径 | ❌ 未测 |
| **W4A8 (weight+activation)** | ✅ | hook 在 activation quantization 之前执行 | ❌ 未测 |

### 3.3 为什么所有 Scheme 都兼容

关键代码路径分析：

**RTN 量化（weight-only）：**
```python
# WrapperLinear.forward() — wrapper.py line 503-518
def forward(self, x):
    # 1. 执行 pre_hooks（包括 rotation hook）
    for hook in self.orig_layer._forward_pre_hooks.values():
        result = hook(self.orig_layer, (x,))
        if result is not None:
            x = result[0]
    
    # 2. 量化 weight
    weight_q = self.quant_weight(self.orig_layer.weight)
    
    # 3. F.linear(x_rotated, weight_q, bias)
    return F.linear(x, weight_q, self.orig_layer.bias)
```

**MXFP4/NVFP4 量化（weight+activation）：**
```python
# WrapperWALayer.forward() — wrapper.py line 572-607
def forward(self, x):
    # 1. 执行 stolen hooks（包括 rotation hook）
    for hook_fn in self._stolen_pre_hooks:
        result = hook_fn(self, (x,))
        if result is not None:
            x = result[0]
    
    # 2. 量化 activation
    x_q = self.quant_activation(x)
    
    # 3. 量化 weight + matmul
    return self.orig_layer.forward(x_q)  # orig_layer 内部做 weight quant
```

**两种路径都在量化之前执行 rotation hook**，这正是预期行为：
- 先旋转 input: `x_rot = x @ H`
- 再量化 weight: `w_q = Quantize(W @ H)`（weight 已经 offline fuse 了旋转）
- 最终: `output = w_q @ x_rot = Q(W@H) @ (x@H)`

### 3.4 特殊考虑

#### NVFP4 的 Global Scale Calibration

NVFP4 需要 calibration 计算 input 的 global scale：
```
calibration forward: input → hook(rotation) → rotated_input → 统计 max/min → 计算 global_scale
inference forward:   input → hook(rotation) → rotated_input → scale_quantize → matmul
```

Rotation hook 在 calibration 和 inference 中都会执行，calibration 看到的 input 分布已经是旋转后的——这正是我们想要的（旋转使分布更均匀，有利于量化）。

#### SignRound 的多轮优化

SignRound 通过多轮 forward 优化 rounding：
```
for step in range(num_steps):
    output = WrapperLinear.forward(x)  # hook 每轮都执行
    loss = compute_loss(output, teacher_output)
    optimize_rounding(loss)
```

Rotation hook 在每轮优化中都正确执行，确保优化看到的是旋转后的 input。

#### Activation Quantization (W4A8, W8A8)

```
forward: input → rotation_hook → x_rot → activation_quant → x_rot_q → weight_quant_matmul
```

Rotation 在 activation quantization 之前执行。旋转后的 activation 分布更均匀，有利于 activation 量化。这是 rotation 的核心价值之一。

---

## 4. 推荐测试计划

### 优先级 1：已验证
- [x] R1 + MXFP4 RTN (hellaswag=0.4550)

### 优先级 2：建议验证
- [ ] R1 + W4A16 INT4 RTN (group_size=128)
- [ ] R1+R2 + MXFP4 RTN
- [ ] R1 + NVFP4 RTN (验证 calibration 是否正常)

### 优先级 3：扩展验证
- [ ] R1 + W4A16 INT4 SignRound (验证多轮优化)
- [ ] R1+R2+R3+R4 + MXFP4 RTN
- [ ] R1 + W4A8 (验证 activation quantization 交互)

---

## 5. 总结

| 维度 | 状态 |
|------|------|
| **Rotation 是否 scheme-agnostic** | ✅ 完全独立 |
| **Hook 能否在所有量化路径中执行** | ✅ WrapperLinear 和 WrapperWALayer 都执行 hooks |
| **Calibration 能否正确看到旋转后的 input** | ✅ hook 在 calibration forward 中自动生效 |
| **SignRound 优化是否正确** | ✅ hook 在每轮优化中执行 |
| **需要额外代码适配不同 scheme** | ❌ 不需要 |
| **Quark 是否对不同 scheme 有特殊处理** | ❌ 没有，同样 scheme-agnostic |
