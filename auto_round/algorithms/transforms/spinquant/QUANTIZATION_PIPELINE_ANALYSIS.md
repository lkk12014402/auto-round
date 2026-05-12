# Quantization Pipeline Analysis: Rotation + GPTQ/AWQ

> 深入分析 Quark 和 llm-compressor 中 Rotation (QuaRot/SpinQuant) 与 GPTQ/AWQ/SmoothQuant
> 的组合方式、执行顺序和算法原理。

---

## 目录

1. [Pipeline 架构对比](#1-pipeline-架构对比)
2. [Online R1 vs Offline R1 原理](#2-online-r1-vs-offline-r1-原理)
3. [GPTQ 算法原理与实现](#3-gptq-算法原理与实现)
4. [AWQ 算法原理与实现](#4-awq-算法原理与实现)
5. [SmoothQuant 算法原理与实现](#5-smoothquant-算法原理与实现)
6. [Rotation + 量化的组合流程](#6-rotation--量化的组合流程)
7. [三框架对比表](#7-三框架对比表)

---

## 1. Pipeline 架构对比

### 1.1 Quark 的 Pipeline

Quark 使用 `ModelQuantizer` + `algo_config` 列表实现顺序执行：

```python
# quark/torch/quantization/api.py - ModelQuantizer.quantize_model()

def quantize_model(self, model, dataloader):
    # Step 0: 设备检查
    self._check_model_device(model)

    # Step 1: 替换 nn.Linear → QuantLinear (挂载量化器)
    model = self._prepare_model(model)

    # Step 2: 按 algo_config 列表顺序执行算法 (rotation → gptq → awq ...)
    model = self._apply_advanced_quant_algo(model, dataloader)

    # Step 3: 校准 (收集 scale/zero-point)
    model = self._do_calibration(model, dataloader)

    # Step 4: 后校准优化
    model = self._do_post_calib_optimazation(model)
```

算法通过 `PROCESSOR_MAP` 注册和查找：

```python
# quark/torch/algorithm/api.py

PROCESSOR_MAP = {
    "rotation": RotationProcessor,    # QuaRot/SpinQuant
    "quarot":   RotationProcessor,    # 别名
    "smooth":   SmoothQuantProcessor,
    "awq":      AwqProcessor,
    "gptq":     GptqProcessor,
    "gptaq":    GptaqProcessor,
    "qronos":   QronosProcessor,
}

def apply_advanced_quant_algo(model, config, is_accelerate, dataloader):
    # 禁用所有量化器 (保持 FP 精度做预处理)
    for module in model.modules():
        if isinstance(module, ScaledFakeQuantize):
            module.disable_fake_quant()
            module.disable_observer()

    # 按列表顺序逐个执行
    for i in range(len(config.algo_config)):
        processor = PROCESSOR_MAP[config.algo_config[i].name](model, config.algo_config[i], dataloader)
        processor.apply()  # 每个算法独立 apply
```

**用户配置示例**：

```bash
# 命令行: 指定多个算法
python quantize_quark.py \
    --quant_algo rotation,gptq \
    --quant_algo_config_file rotation ./rotation_config.json \
    --quant_algo_config_file gptq ./gptq_config.json
```

```python
# Python API
quant_config = template.get_config(
    scheme="mxfp4",
    algorithm=["rotation", "gptq"],   # 列表形式, 按顺序执行
    algo_configs={
        "rotation": rotation_config,
        "gptq": gptq_config,
    },
)
quantizer = ModelQuantizer(quant_config)
model = quantizer.quantize_model(model, dataloader)
```

---

### 1.2 llm-compressor 的 Pipeline

llm-compressor 使用 Modifier 系统 + Recipe 列表：

```python
# llmcompressor/entrypoints/oneshot.py

class Oneshot:
    """
    生命周期:
    1. Preprocessing: 加载模型, untie embeddings
    2. Calibration: 按 recipe 顺序执行 Modifier 生命周期
    3. Postprocessing: 保存模型
    """
```

**Modifier 生命周期**：

```python
# llmcompressor/modifiers/modifier.py

class Modifier(ModifierInterface, HooksMixin):
    def on_initialize(self, state) -> bool   # 准备配置
    def on_start(self, state, event)         # 应用 transforms
    def on_event(self, state, event)         # 处理校准事件
    def on_end(self, state, event)           # 清理
    def on_finalize(self, state) -> bool     # 最终收尾
```

**Pipeline 执行流程**：

```
oneshot(model, recipe, pipeline)
  → session.initialize(recipe, model)
    → FOR EACH modifier: modifier.on_initialize(state)
  → CalibrationPipeline (sequential / datafree / independent)
    → FOR EACH modifier:
        modifier.on_event(CALIBRATION_EPOCH_START)
        modifier.on_event(SEQUENTIAL_EPOCH_END)   # 逐层校准
        modifier.on_event(CALIBRATION_EPOCH_END)
  → session.finalize()
    → FOR EACH modifier: modifier.on_finalize(state)
```

**用户配置示例**：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.gptq import GPTQModifier

# Recipe = Modifier 列表, 按顺序执行
recipe = [
    SpinQuantModifier(                         # 第一步: Rotation
        rotations=["R1", "R2", "R4"],
        transform_block_size=128,
        transform_type="hadamard",
    ),
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),  # 第二步: GPTQ
]

oneshot(model=model, recipe=recipe, pipeline="sequential")
```

**SpinQuantModifier 的 on_start**：

```python
# llmcompressor/modifiers/transform/spinquant/base.py

class SpinQuantModifier(Modifier):
    def on_start(self, state, event, **kwargs):
        model = state.model
        untie_word_embeddings(model)         # 解绑 embedding
        self._center_embeddings(model)       # 中心化 embedding
        self._fuse_norms(model)              # 吸收 RMSNorm γ
        apply_transform_config(model, self.transform_config)  # 应用 R1/R2/R3/R4
```

---

### 1.3 架构对比总结

| 维度 | Quark | llm-compressor |
|------|-------|----------------|
| **组合方式** | `algo_config` 列表，顺序执行 | `recipe` Modifier 列表，生命周期驱动 |
| **注册机制** | `PROCESSOR_MAP` 字典 | Modifier 类自动发现 |
| **执行控制** | 简单 for 循环 | 事件驱动 (on_start/on_event/on_end) |
| **Pipeline 选择** | 固定流程 | 智能选择 (sequential/datafree/independent) |
| **多算法支持** | ✅ rotation + gptq/awq | ✅ SpinQuant + GPTQ/AWQ |
| **Rotation + GPTQ** | ✅ 已验证 | ✅ 已验证 |
| **执行顺序** | Rotation **必须在列表第一个** | SpinQuantModifier **放在 recipe 前面** |

---

## 2. Online R1 vs Offline R1 原理

### 2.1 Online R1 的实际实现

**核心：同时修改 weight 和 activation**（两侧都旋转）

```python
# preprocessor.py: _apply_online_r1()

for module in target_modules:  # q/k/v_proj, gate/up_proj
    # 1️⃣ 修改 weight: W_new = W @ R
    module.weight.data = (W @ R).to(dtype)

    # 2️⃣ 注册 activation pre-hook: x_rot = x @ R
    def hook(module, args):
        x = args[0]
        return (x @ R,)
    module.register_forward_pre_hook(hook)
```

**数学等价性**（`nn.Linear`: `output = input @ weight.T`）：

```
output = x_rot @ W_new.T
       = (x @ R) @ (W @ R).T
       = (x @ R) @ (R^T @ W^T)
       = x @ (R @ R^T) @ W^T
       = x @ I @ W^T
       = x @ W^T   ← 与原始完全相同 ✓
```

### 2.2 Online 对量化的意义

虽然最终计算结果相同，但**量化发生在旋转后的空间中**：

```
x → [hook: x@R] → quantize_act(x@R) → dequant → [(W@R)^T] → output
                   ↑ activation 量化目标            ↑ weight 量化目标
                   分布均匀, 误差小                  分布均匀, 误差小
```

### 2.3 为什么 Offline 效果比 Online 差

**根本原因：RMSNorm γ 的 per-channel 缩放破坏旋转均匀性**

| | Online R1 | Offline R1 |
|---|---|---|
| **流程** | hidden → RMSNorm*γ → [hook: x@R] → quant | hidden_rot → RMSNorm → *γ → quant (无新旋转) |
| **RMSNorm γ** | **不动**，hook 在 γ 之后旋转 | 必须吸收到权重 (`W@diag(γ)`) |
| **量化看到的 act** | `x @ R` (post-γ, 均匀) | `x_rot * γ` (γ 破坏均匀性) |
| **量化看到的 weight** | `W @ R` (均匀) | `R^T @ W @ diag(γ)` (γ 引入不均匀) |
| **误差累积** | 无 (per-module 局部) | 有 (全模型链式传播) |
| **推理额外开销** | 每层一次 `x@R` | 无 |

**Quark 默认使用 Online R1**（`online_r1_rotation=True`），原因：
1. 实现更简单（不需要 γ 吸收、embed/lm_head 边界处理）
2. 数值更稳定（避免 FP16 累积误差）
3. 量化效果更好（activation 和 weight 都在最优分布中被量化）
4. 灵活性好（可随时切换 R 矩阵）

---

## 3. GPTQ 算法原理与实现

### 3.1 核心思想

GPTQ (Gradient-based Post-Training Quantization) 基于 **Optimal Brain Quantization (OBQ)** 框架：
逐列量化权重，利用 Hessian 矩阵信息将量化误差最优地分配到未量化的列上。

### 3.2 数学原理

**目标**：最小化量化后的输出误差

```
min_Q || W·X - Q·X ||² = min_Q || (W - Q) · X ||²
```

其中 `H = 2·X·X^T` 是 Hessian 矩阵（输入的外积）。

**逐列量化 + 误差传播**：

```
对第 i 列:
  1. 量化: q_i = quantize(w_i)
  2. 误差: δ_i = (w_i - q_i) / H_ii
  3. 传播: W[:, i+1:] -= δ_i · H[i, i+1:]^T / H[i, i]
```

用 Cholesky 分解加速 Hessian 逆的计算。

### 3.3 Quark 实现

```python
# quark/torch/algorithm/gptq/gptq.py

class GPTQ(BaseHessianAlgorithm):
    """核心量化算法"""

    def add_batch_quantized(self, inp, out, name):
        """累积 Hessian: H += X^T @ X"""
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, ...):
        """分块量化 + 误差传播"""
        # Hessian 逆 (Cholesky)
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        # 按 block_size 分块处理
        for i1 in range(0, columns, blocksize):
            W1 = W[:, i1:i2].clone()
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]           # Hessian 对角元素

                # 量化
                q = quantize(w, scale, zero_point)
                Q1[:, i] = q

                # 误差传播到块内剩余列
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)

            # 块间误差传播
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
```

**关键参数**：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `block_size` | 分块大小 | 128 |
| `damp_percent` | Hessian 对角线阻尼 | 0.01 |
| `act_order` | 按 activation 重要性排序列 | True |
| `static_groups` | 静态 group 量化 | True |
| `group_size` | per-group 量化粒度 | 128 |

### 3.4 llm-compressor 实现

```python
# llmcompressor/modifiers/gptq/gptq_quantize.py

def accumulate_hessian(inp, module, H, num_samples):
    """累积 Hessian"""
    if isinstance(module, torch.nn.Linear):
        inp = inp.reshape((-1, inp.shape[-1])).t()
    inp = math.sqrt(2) * inp.to(GPTQ_PRECISION)
    H += inp.matmul(inp.t())
    return H, num_samples

def quantize_weight(W, H, quant_args, blocksize=128, percdamp=0.01):
    """核心量化循环"""
    # 阻尼 + Cholesky 逆
    damp = percdamp * torch.mean(torch.diag(H))
    H[diag, diag] += damp
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    # 逐块量化
    for i1 in range(0, num_columns, blocksize):
        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = fake_quantize(w, scale, zero_point, ...)
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)

        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
```

### 3.5 Activation Ordering

```python
# 按 Hessian 对角线 (= 输入方差) 降序排列列
perm = torch.argsort(torch.diag(H), descending=True)
W = W[:, perm]     # 重排列
H = H[perm][:, perm]  # 重排 Hessian
```

**意义**：先量化"重要"的列（方差大 = 对输出影响大），误差传播到不重要的列，最小化整体损失。

---

## 4. AWQ 算法原理与实现

### 4.1 核心思想

AWQ (Activation-Aware Weight Quantization) 的核心观察：

> **只有 1% 的"显著"权重通道对模型精度影响巨大**，这些通道对应于 activation 中的大值通道。

解决方案：**通过 per-channel 缩放保护这些显著通道**。

### 4.2 数学原理

对于 `y = W @ x`，引入对角缩放矩阵 `s`：

```
y = W @ x = (W · diag(s)) @ (diag(1/s) · x)
    ↑ 等价变换 (恒等式)

量化: Q(W · diag(s)) @ (x / s) ≈ W @ x
```

- `s` 大的通道：weight 缩小 → 量化误差减小
- `s` 小的通道：weight 放大 → 量化误差增大（但这些通道不重要）

**最优 s 的搜索**：

```
s* = argmin_s || Q(W·diag(s)) @ (x/s) - W @ x ||²
```

Quark 用 grid search 在 `s = x_max^α` 空间搜索最优 α：

```python
for α in linspace(0, 1, n_grid):
    s = x_max.pow(α).clamp(min=1e-4)
    s = s / sqrt(s.max() * s.min())  # 归一化
    loss = MSE(Q(W*s) @ (x/s), W @ x)
    if loss < best:
        best_s = s
```

### 4.3 Quark 实现

```python
# quark/torch/algorithm/awq/awq.py

class AwqProcessor(BaseAlgoProcessor):
    def apply(self):
        for layer in decoder_layers:
            # 1. 收集 activation 特征 (forward hook)
            input_feat = self._get_input_feat(layer, dataloader)

            # 2. 对每组 (prev_op, [target_layers]):
            for prev_op, layers in scaling_groups:
                # 3. 搜索最优 scale
                scales = self._search_best_scale(layers, input_feat)

                # 4. 应用 scale
                apply_scale(prev_op, layers, scales)

            # 5. Weight clipping (可选)
            if self.auto_clip:
                self._apply_clip(layer, input_feat)

    def _search_best_scale(self, layers, input_feat, n_grid=20):
        weight = torch.cat([m.weight for m in layers], dim=0)
        w_max = weight.abs().mean(dim=0)  # per-channel weight 重要性
        x_max = input_feat.abs().mean(dim=0)  # per-channel activation 重要性

        best_scales = None
        best_error = float('inf')

        for ratio in linspace(0, 1, n_grid):
            scales = x_max.pow(ratio).clamp(min=1e-4)
            scales /= sqrt(scales.max() * scales.min())

            # 模拟量化: Q(W * s) @ (x / s)
            scaled_weight = weight * scales
            q_weight = quantize(scaled_weight)
            q_output = (q_weight / scales) @ input_feat.T
            fp_output = weight @ input_feat.T

            loss = MSE(q_output, fp_output)
            if loss < best_error:
                best_error = loss
                best_scales = scales

        return best_scales
```

### 4.4 llm-compressor 实现

```python
# llmcompressor/modifiers/transform/awq/base.py

class AWQModifier(Modifier):
    """AWQ 通过 Modifier 生命周期集成"""

    def on_start(self, state, event):
        self._set_resolved_mappings(state.model)
        self._setup_activation_cache_hooks()   # 注册 hook 收集 activation

    def on_sequential_epoch_end(self, state, event):
        # 每层校准完成后, 计算并应用 scales
        for mapping in self.resolved_mappings:
            smooth_scales = self._compute_scales(mapping)
            self._apply_smoothing(mapping, smooth_scales)

    def _compute_scales(self, mapping):
        act_scales = self.activation_cache[mapping.smooth_name]
        weight_scales = torch.cat([l.weight.abs().max(dim=0) for l in mapping.balance_layers])

        # AWQ 的 duo_scaling: s = act^α / weight^(1-α)
        scales = act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha)
        return scales.clamp(min=1e-5)
```

### 4.5 Weight Clipping

AWQ 还支持**权重裁剪**来进一步降低量化误差：

```python
def _compute_best_clip(self, linear_layer, input_feat, n_grid=20, max_shrink=0.5):
    w = linear_layer.weight
    org_max = w.abs().amax(dim=-1, keepdim=True)

    best_max = org_max
    best_error = float('inf')

    for i in range(int(max_shrink * n_grid)):
        max_val = org_max * (1 - i / n_grid)  # 逐步缩小裁剪阈值
        clipped_w = torch.clamp(w, -max_val, max_val)
        q_w = quantize(clipped_w)

        error = MSE(q_w @ input_feat.T, w @ input_feat.T)
        if error < best_error:
            best_error = error
            best_max = max_val

    # 应用最优裁剪
    linear_layer.weight.data = torch.clamp(w, -best_max, best_max)
```

---

## 5. SmoothQuant 算法原理与实现

### 5.1 核心思想

> **Activation 的 outlier 很难量化，但 weight 很"平滑"。
> 把 activation 的困难"转移"给 weight。**

通过 per-channel 缩放，将 activation 的动态范围转移到 weight 上：

```
Y = (X · diag(s⁻¹)) @ (diag(s) · W^T) = X̃ @ W̃^T
```

- `X̃ = X / s`：activation 变平滑（除以大值通道的 scale）
- `W̃ = s · W`：weight 吸收 scale（变得不那么平滑，但仍可量化）

### 5.2 最优 Scale 的计算

```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
```

- `α = 0.5`：平均分配困难
- `α → 1`：更多转移给 weight（activation 友好）
- `α → 0`：更多保留在 activation（weight 友好）

### 5.3 Quark 实现

```python
# quark/torch/algorithm/awq/smooth.py

class SmoothQuantProcessor(BaseAlgoProcessor):
    def __init__(self, model, config, dataloader):
        self.alpha = config.alpha  # 默认 0.5

    def apply(self):
        for layer in self.modules:
            # 1. 收集 activation 统计
            input_feat, act_scales = self._get_act_scale_and_input_feat(layer)

            # 2. 计算每组 scale
            for prev_op, layers in scaling_config:
                scales = self._search_best_scale(act_scales, prev_op, layers)
                scales_list.append(scales)

            # 3. 应用 scales
            apply_scale(layer, scales_list)

    def _search_best_scale(self, act_scales, prev_op, layers):
        layer_act_scales = act_scales.float()
        weight_scales = torch.cat([fc.weight.abs().max(dim=0) for fc in layers])

        # SmoothQuant 公式: s = act^α / weight^(1-α)
        best_scales = (layer_act_scales.pow(self.alpha) /
                       weight_scales.pow(1 - self.alpha)).clamp(min=1e-5)
        return best_scales
```

### 5.4 llm-compressor 实现

```python
# llmcompressor/modifiers/transform/smoothquant/base.py

class SmoothQuantModifier(Modifier):
    smoothing_strength: float = 0.5  # α
    algorithm: Literal["smoothquant", "log_equalization"] = "smoothquant"

    def on_sequential_epoch_end(self, state, event):
        for mapping in self.resolved_mappings:
            act_scales = self.activation_stats[mapping.smooth_name]
            weight_scales = self._get_weight_scales(mapping.balance_layers)

            # 计算 smooth scale
            scales = act_scales.pow(self.smoothing_strength) / \
                     weight_scales.pow(1 - self.smoothing_strength)

            # 应用: activation /= scales, weight *= scales
            self._apply_smoothing(mapping.smooth_layer, mapping.balance_layers, scales)
```

### 5.5 SmoothQuant vs AWQ 的区别

| 维度 | SmoothQuant | AWQ |
|------|-------------|-----|
| **目标** | 平衡 act/weight 量化难度 | 保护显著 weight 通道 |
| **Scale 来源** | `act^α / weight^(1-α)` (闭式) | Grid search 最小化 MSE |
| **应用位置** | 前一层 output → 下一层 input | 同上 |
| **Weight clipping** | ❌ | ✅ |
| **计算开销** | 低 (一次计算) | 高 (grid search) |
| **适用场景** | W8A8 | W4A16, W4A8 |

---

## 6. Rotation + 量化的组合流程

### 6.1 完整 Pipeline (以 Quark 为例)

```
                    ┌─────────────────────────────────────────────┐
                    │         Quark quantize_model()              │
                    └─────────────────────────────────────────────┘
                                      │
                    ┌─────────────────────────────────────────────┐
Step 1:             │  _prepare_model()                           │
nn.Linear →         │  替换 nn.Linear → QuantLinear              │
QuantLinear         │  挂载 FakeQuantize (weight + activation)    │
                    └─────────────────────────────────────────────┘
                                      │
                    ┌─────────────────────────────────────────────┐
Step 2a:            │  RotationProcessor.apply()                  │
Rotation            │  ┌───────────────────────────────────────┐  │
(第一个算法)         │  │ Online R1:                            │  │
                    │  │   - target weight: W_new = W @ R      │  │
                    │  │   - activation hook: x_rot = x @ R    │  │
                    │  │   (RMSNorm γ 不动)                     │  │
                    │  ├───────────────────────────────────────┤  │
                    │  │ Offline R2:                            │  │
                    │  │   - V_proj: W_v = W_v @ R2^T          │  │
                    │  │   - O_proj: W_o = R2 @ W_o            │  │
                    │  ├───────────────────────────────────────┤  │
                    │  │ Online R3/R4:                          │  │
                    │  │   - Register forward hooks             │  │
                    │  └───────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘
                                      │
                    ┌─────────────────────────────────────────────┐
Step 2b:            │  GptqProcessor.apply() 或 AwqProcessor     │
GPTQ/AWQ            │  ┌───────────────────────────────────────┐  │
(第二个算法)         │  │ GPTQ:                                 │  │
                    │  │   - 收集 Hessian (此时 act 已被 R1    │  │
                    │  │     hook 旋转!)                         │  │
                    │  │   - 对 W_new (已旋转) 做逐列量化       │  │
                    │  │   - Hessian 反映旋转后的分布           │  │
                    │  ├───────────────────────────────────────┤  │
                    │  │ AWQ:                                   │  │
                    │  │   - 收集 activation scales (旋转后)    │  │
                    │  │   - 对已旋转的 weight 做 per-channel   │  │
                    │  │     缩放                               │  │
                    │  └───────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘
                                      │
                    ┌─────────────────────────────────────────────┐
Step 3:             │  _do_calibration()                          │
校准                 │  - 启用 FakeQuantize observer              │
                    │  - Forward pass 收集 scale/zero-point       │
                    │  - (此时 weight 和 activation 都在旋转空间) │
                    └─────────────────────────────────────────────┘
                                      │
                    ┌─────────────────────────────────────────────┐
Step 4:             │  Freeze & Export                            │
导出                 │  - FakeQuantize → FrozenFakeQuantize        │
                    │  - 导出量化模型                             │
                    └─────────────────────────────────────────────┘
```

### 6.2 关键洞察：GPTQ/AWQ 看到的是什么

| 算法 | 看到的 Weight | 看到的 Activation | 说明 |
|------|--------------|------------------|------|
| **GPTQ** | `W @ R` (已旋转) | `x @ R` (hook 旋转后) | Hessian 反映旋转后分布 |
| **AWQ** | `W @ R` (已旋转) | `x @ R` (hook 旋转后) | Scale 基于旋转后统计 |
| **SmoothQuant** | `W @ R` (已旋转) | `x @ R` (hook 旋转后) | α 平衡旋转后空间 |

**核心**：因为 Rotation 在前面执行，GPTQ/AWQ 的 Hessian 和 activation scales 都是在**旋转后的均匀空间**中计算的。这意味着：
- Hessian 更"均匀"→ 误差传播更稳定
- Activation scales 更均匀 → AWQ 的 per-channel 缩放更小
- 整体量化误差更低

### 6.3 Quark train_rotation.py 的两阶段流程

Quark 的训练脚本展示了更完整的生产流程：

```python
# 阶段 1: 训练 Rotation (使用动态量化做量化感知训练)
quant_config_rotation.algo_config = [rotation_config]       # 只有 rotation
quant_config_rotation.global_quant_config.weight.is_dynamic = True  # 动态量化

quantizer = ModelQuantizer(quant_config_rotation)
model = quantizer.quantize_model(model)   # 挂载量化器 + 执行 rotation

# ... Cayley SGD 训练 rotation 矩阵 ...
trainer.train()

# 后处理: 固定 rotation
model = RotationProcessor.post_process_trained_rotation(model, quant_config_no_rotation)

# 阶段 2: 正式量化 (rotation 已固定, 执行 GPTQ)
quant_config_no_rotation.algo_config = [gptq_config]       # 只有 GPTQ
quantizer = ModelQuantizer(quant_config_no_rotation)
model = quantizer.quantize_model(model, calibration_dataloader)  # 执行 GPTQ

# 导出
quantizer.freeze(model)
export_safetensors(model, output_dir)
```

---

## 7. 三框架对比表

### 7.1 算法支持

| 算法 | Quark | llm-compressor | auto-round |
|------|:-----:|:--------------:|:----------:|
| **Rotation (R1-R4)** | ✅ | ✅ (R1,R2,R4; R3 WIP) | ✅ |
| **GPTQ** | ✅ | ✅ | ✅ (SignRound) |
| **AWQ** | ✅ | ✅ | ❌ (不同方法) |
| **SmoothQuant** | ✅ | ✅ | ❌ |
| **Rotation + GPTQ** | ✅ | ✅ | ✅ (目标) |
| **Rotation + AWQ** | ✅ | ✅ | 🔜 |
| **SpinQuant training** | ✅ Cayley SGD | ❌ NotImplemented | ⚠️ 实验 |
| **Online R1** | ✅ (默认) | ❌ (仅 offline fuse) | ✅ (默认) |
| **Random Hadamard** | ✅ | ✅ (random-hadamard) | ✅ |
| **Rotation size (block)** | ✅ | ✅ (transform_block_size) | ✅ |

### 7.2 Pipeline 对比

| 维度 | Quark | llm-compressor | auto-round |
|------|-------|----------------|------------|
| **入口** | `ModelQuantizer.quantize_model()` | `oneshot(recipe=...)` | `AutoRound(...)` |
| **算法组合** | `algo_config` 列表 | `recipe` Modifier 列表 | 内置 pipeline |
| **执行顺序** | 顺序 for 循环 | 事件驱动生命周期 | 内置顺序 |
| **Pipeline 选择** | 固定 | 智能 (sequential/datafree) | 固定 |
| **模型格式** | 自定义 QuantLinear | compressed-tensors | AutoGPTQ 兼容 |
| **保存格式** | safetensors + config | HuggingFace 标准 | HuggingFace 标准 |

### 7.3 Rotation 实现对比

| 维度 | Quark | llm-compressor | auto-round |
|------|-------|----------------|------------|
| **R1 默认模式** | Online | Offline (fuse) | Online |
| **R1 offline 支持** | ✅ | ✅ (唯一模式) | ✅ |
| **RMSNorm 处理** | Online: 不动; Offline: 吸收 | 吸收 | Online: 不动; Offline: 吸收 |
| **R3 实现** | MonkeyPatch RoPE | 🔜 WIP | MonkeyPatch RoPE |
| **R4 实现** | forward_pre_hook on down_proj | TransformScheme hook | forward_pre_hook on down_proj |
| **序列化** | RotationLinear 参数 | TransformConfig 元数据 | QuantLinear buffer |
| **加载恢复** | RotationLinear.forward | apply_transform_config | patch_quantlinear_forward |

### 7.4 算法在 Pipeline 中的位置

```
┌─────────────────────────────────────────────────────────────────────┐
│                    量化 Pipeline 执行顺序                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ① Rotation (QuaRot/SpinQuant)                                     │
│     ├─ R1: Online weight rotation + activation hook                 │
│     ├─ R2: Offline fuse into V_proj/O_proj                         │
│     ├─ R3: Online RoPE monkeypatch                                 │
│     └─ R4: Online MLP down_proj hook                               │
│                                                                     │
│  ② SmoothQuant (可选)                                              │
│     └─ Per-channel scale: act^α / weight^(1-α)                     │
│                                                                     │
│  ③ AWQ (可选, 与 SmoothQuant 互斥或叠加)                           │
│     └─ Grid search optimal scale per channel                       │
│                                                                     │
│  ④ GPTQ (可选)                                                     │
│     └─ Hessian-based column-wise quantization + error propagation  │
│                                                                     │
│  ⑤ Calibration (校准 quantizer scale/zero-point)                   │
│     └─ Forward pass with FakeQuantize observers                    │
│                                                                     │
│  ⑥ Export (导出)                                                    │
│     └─ Pack weights, save config, save rotation metadata           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附录 A: 算法一句话总结

| 算法 | 一句话 |
|------|--------|
| **Rotation** | 正交变换使 weight/activation 分布更均匀，降低量化误差 |
| **GPTQ** | 利用 Hessian 矩阵逐列量化，将误差最优分配到未量化列 |
| **AWQ** | 通过 per-channel 缩放保护 activation 中的重要通道 |
| **SmoothQuant** | 将 activation outlier 的困难转移给 weight |
| **Rotation + GPTQ** | 先旋转使分布均匀，再用 GPTQ 精细量化旋转后的权重 |

---

## 附录 B: 为什么 Rotation + GPTQ 效果 > 单独使用

```
单独 GPTQ:
  Weight (有 outlier) → GPTQ → 量化误差较大 (outlier 列误差无法充分传播)

Rotation + GPTQ:
  Weight (有 outlier) → R1 旋转 → Weight (均匀) → GPTQ → 量化误差很小
                                                    ↑
                                    Hessian 也更均匀, 误差传播更有效
```

Rotation 让 GPTQ 的 Hessian 更接近"等方性"（各方向方差相近），
使得 Cholesky 分解更稳定，误差传播更均匀——本质上是让 GPTQ 工作在最优条件下。
