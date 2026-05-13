# Block Lifecycle Hooks 架构分析

## 1. 核心概念：Block vs Layer vs Linear

在 auto-round 的语境中，这三个术语有明确的层级关系：

```
Model
├── model.layers[0]          ← "Block" (Decoder Layer)
│   ├── self_attn
│   │   ├── q_proj           ← "Linear Layer"
│   │   ├── k_proj           ← "Linear Layer"
│   │   ├── v_proj           ← "Linear Layer"
│   │   └── o_proj           ← "Linear Layer"
│   └── mlp
│       ├── gate_proj        ← "Linear Layer"
│       ├── up_proj          ← "Linear Layer"
│       └── down_proj        ← "Linear Layer"
├── model.layers[1]          ← "Block" (Decoder Layer)
│   └── ... (同构)
└── model.layers[N-1]
```

- **Block** = 整个 Decoder Layer（`model.layers[i]`），是 Compressor 循环的粒度
- **Layer** = 通常也指 Decoder Layer（在 rotation 语境中）
- **Linear Layer** = 具体的线性层（`q_proj`, `down_proj` 等），是量化的最小单元

---

## 2. Compressor 的 Block 循环

### 2.1 `block_names` 是什么？

`get_block_names(model)` 通过搜索模型中的 `ModuleList` 自动发现所有 decoder layers：

```python
# 对于 Qwen3-0.6B，返回:
[
    ["model.layers.0", "model.layers.1", ..., "model.layers.27"]
]
```

### 2.2 循环结构 (`_quantize_blocks`)

```python
for i in range(0, len(block_names), nblocks):
    n = block_names[i]                    # "model.layers.0"
    m = get_module(model, n)              # 获取整个 DecoderLayer 模块

    # 1. 基础设施：物化、设备放置
    materialize_model_(m)
    m = m.to(device)

    # 2. Rotation Hook（新增）
    self._on_block_ready(m, n, i)         # ← layer-wise rotation 在此应用

    # 3. 收集参考输出
    reference_output = quantizer._get_block_outputs(m, input_ids, ...)

    # 4. 量化
    quantizer.quantize_block(m, input_ids, input_others, reference_output, ...)

    # 5. 后处理 Hook
    self._on_block_quantized(m, n, i)

    # 6. 更新输入（用量化后的输出作为下一个 block 的输入）
    input_ids = ...  # 更新
```

**关键点**：循环中传递的 `m` 始终是**整个 DecoderLayer**，永远不会是单独的 `v_proj` 或 `q_proj`。

---

## 3. Layer-wise Rotation 的工作方式

### 3.1 初始化阶段 (`prepare_layerwise`)

初始化是**全局一次性**的，基于 `model.config` 的结构参数决定所有 rotation 的维度：

```python
def prepare(self):
    # 从 model.config 读取全局参数
    hidden_size = model.config.hidden_size          # e.g. 4096
    num_heads = model.config.num_attention_heads    # e.g. 32
    head_dim = hidden_size // num_heads             # e.g. 128
    intermediate_size = model.config.intermediate_size  # e.g. 14336

    # 1. 验证维度兼容性
    self._validate_dimensions()

    # 2. 初始化 R 矩阵（所有 layer 共享）
    self._init_rotation_matrices()

    # 3. 预计算 R1 状态
    self._prepare_r1_state()
```

### 3.2 R 矩阵的维度由什么决定？

| 旋转 | 矩阵尺寸 | 取决于 | 可用户覆盖 |
|------|----------|--------|-----------|
| R1 | `rotation_size × rotation_size` | `hidden_size`（或 `config.rotation_size`） | ✅ |
| R2 | `head_dim × head_dim` | `hidden_size / num_attention_heads` | ❌ 固定 |
| R3 | `head_dim × head_dim` | 同上 | ❌ 固定 |
| R4 | `r4_rotation_size × r4_rotation_size` | `intermediate_size`（或 `config.rotation_size`） | ✅ |

### 3.3 维度验证与自动降级 (`_validate_dimensions`)

如果模型维度不兼容某种 rotation，会**自动禁用**并给出 warning：

```python
def _validate_dimensions(self):
    # R1: 必须是 2 的幂，且能整除 hidden_size
    if not is_pow2(r1_rotation_size):
        logger.warning("Disabling R1")
        self.config.r1 = False
    elif hidden_size % r1_rotation_size != 0:
        logger.warning("Disabling R1")
        self.config.r1 = False

    # R2/R3: head_dim 必须是 2 的幂
    if not is_pow2(head_dim):
        logger.warning("Disabling R2/R3")
        self.config.r2 = False
        self.config.r3 = False

    # R4: 需要能找到 Hadamard 分解，且能整除 intermediate_size
    try:
        get_hadamard_K(r4_rotation_size)
    except ValueError:
        logger.warning("Disabling R4")
        self.config.r4 = False
```

### 3.4 R 矩阵初始化的决策树 (`_init_rotation_matrices`)

```
R1 初始化:
├── online + deterministic + not trainable
│   └── 不存矩阵，运行时 butterfly 算法计算（最高效）
├── trainable
│   └── R1 = eye(r1_size)，可训练参数
├── random
│   └── R1 = random_hadamard(r1_size)，需持久化
└── deterministic (offline)
    └── R1 = deterministic_hadamard(r1_size)，固定

R2 初始化:
├── trainable → eye(head_dim)
├── random   → random_hadamard(head_dim)
└── deterministic → deterministic_hadamard(head_dim)

R3/R4: 类似逻辑
```

### 3.5 所有 Layer 共享同一组 R 矩阵

```
R1 矩阵 (64×64) ──→ layer[0].q_proj, layer[0].k_proj, ..., layer[27].up_proj
R2 矩阵 (128×128) ─→ layer[0].v_proj/o_proj, ..., layer[27].v_proj/o_proj
R4 分解 (had_K, K) ─→ layer[0].down_proj, ..., layer[27].down_proj
```

这是因为 Transformer 的所有 decoder layers 是**同构**的——相同维度、相同结构。

### 3.6 `rotate_layer` 如何知道对每个子模块做什么？

**通过结构属性路径直接访问**，不是通过 name string 匹配：

```python
def rotate_layer(self, layer, layer_idx):
    attn = layer.self_attn        # ← 直接访问属性
    mlp = layer.mlp

    # R1: 旋转 q/k/v/gate/up_proj 的 weight + 注册 hook
    if self.config.r1:
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            module = getattr(attn, proj_name)      # ← hasattr + getattr
            module.weight.data = W @ R1
            # 注册 forward_pre_hook: x → x @ R1

        for proj_name in ("gate_proj", "up_proj"):
            module = getattr(mlp, proj_name)
            module.weight.data = W @ R1

    # R2: fuse 到 v_proj 输出列 + o_proj 输入行 (per head)
    if self.config.r2:
        attn.v_proj.weight.data = ...  # 列旋转
        attn.o_proj.weight.data = ...  # 行旋转

    # R4: fuse 到 down_proj 输入行
    if self.config.r4:
        mlp.down_proj.weight.data = ...

    # R3: monkeypatch apply_rotary_pos_emb
    # R4: 注册 forward_pre_hook on down_proj
```

每个 layer 执行**完全相同**的旋转组合，`layer_idx` 只用于 logging。

---

## 4. Auto-Round 的 Quantizer 分派与 `quantize_block` 实现

### 4.1 Quantizer 分派机制

Auto-Round 根据 `iters` 参数选择不同的 Quantizer 和 Compressor 组合：

```
用户输入: iters=?
    │
    ├── iters > 0 (e.g. iters=200)
    │   ├── Config:     SignRoundConfig
    │   ├── Quantizer:  SignRoundQuantizer
    │   ├── Compressor: CalibCompressor
    │   └── 特点:       需要校准数据，有训练循环，block-wise reconstruction
    │
    └── iters == 0
        ├── Config:     RTNConfig
        ├── 需要 imatrix 或 act calibration?
        │   ├── 是 → Quantizer:  OptimizedRTNQuantizer
        │   │       Compressor: CalibratedRTNCompressor
        │   │       特点:       需要校准数据收集 act_max/imatrix，但无训练循环
        │   │
        │   └── 否 → Quantizer:  RTNQuantizer
        │           Compressor: ZeroShotCompressor
        │           特点:       完全不需要校准数据，逐层直接量化
        │
```

代码中的分派逻辑（`entry.py`）:
```python
if iters == 0:
    config = RTNConfig(bits=bits, group_size=group_size, ...)
else:
    config = SignRoundConfig(iters=iters, batch_size=batch_size, ...)
```

```python
# RTNConfig 进一步分派:
if enable_imatrix or needs_act_calib:
    quant_config._alg_cls = "OptimizedRTNQuantizer"
    return CalibratedRTNCompressor(...)
else:
    quant_config._alg_cls = "RTNQuantizer"
    return ZeroShotCompressor(...)
```

### 4.2 场景 A: `iters > 0` — SignRoundQuantizer (AutoRound Tuning)

**核心思路：Block-wise Reconstruction**

Auto-Round 不是逐个 linear 独立量化（如 GPTQ），而是把整个 block 内的所有 linear 层**联合优化**，通过最小化整个 block 的输出重建误差。

**Compressor**: `CalibCompressor`
**Quantizer**: `SignRoundQuantizer`（定义在 `algorithms/quantization/sign_round/quantizer.py`）

```python
def quantize_block(self, block, input_ids, input_others, reference_output, ...):
    # ═══ Phase 1: Wrapping ═══
    # 遍历 block 内所有 linear，包装为可训练量化层
    quantized_layers, unquantized_layers = self.wrapper_block(block, ...)

    # wrapper_block 做了什么:
    #   for n, m in block.named_modules():
    #       if type(m) in [Linear, Conv1d]:
    #           new_m = WrapperLinear(m, ...)  # 包含量化参数 (scale, zero_point, v)
    #           set_module(block, n, new_m)    # 替换原模块

    # 此时 block 内部变成:
    #   self_attn.q_proj → WrapperLinear(q_proj, params={v, min, max})
    #   self_attn.k_proj → WrapperLinear(k_proj, params={v, min, max})
    #   ...
    #   mlp.down_proj    → WrapperLinear(down_proj, params={v, min, max})

    # ═══ Phase 2: 收集可训练参数 ═══
    round_params = []   # rounding 参数 (v)
    minmax_params = []  # min/max scale 参数
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            for key in m.params:
                if "min" in key or "max" in key:
                    minmax_params.append(m.params[key])
                else:
                    round_params.append(m.params[key])

    optimizer = SGD([
        {"params": round_params, "lr": lr},
        {"params": minmax_params, "lr": minmax_lr},
    ])

    # ═══ Phase 3: 训练循环（iters=200）═══
    for i in range(self.iters):
        indices = sample_batch(nsamples, batch_size)

        # 整个 block 前向传播（带量化噪声）
        output_q = block(input_ids[indices], **input_others)

        # 与 FP 参考输出计算 MSE loss
        loss = MSE(output_q, reference_output[indices])

        # 反向传播 + 更新所有 linear 的量化参数
        loss.backward()
        optimizer.step()

        # 记录最优参数
        if loss < best_loss:
            best_params = save_params(block)
            best_loss = loss

    # ═══ Phase 4: Unwrap ═══
    # 用最优参数恢复原始 linear（带量化后的权重）
    unwrapper_block(block, best_params)
```

**关键特征**:
- 需要 `input_ids`（校准数据）和 `reference_output`（FP 参考输出）
- 有 `iters` 轮训练循环，优化 rounding 决策和 min/max scale
- 所有 linear 层**联合优化**，最小化整个 block 的输出重建误差
- CalibCompressor 在调用前负责收集 reference_output

### 4.3 场景 B: `iters == 0` — RTN Quantizer (Round-To-Nearest)

#### B1: ZeroShotCompressor + RTNQuantizer（不需要校准数据）

**最简单的路径**：直接对每个 linear 层做 Round-To-Nearest 量化，无需校准数据，无需训练。

**Compressor**: `ZeroShotCompressor`
**Quantizer**: `RTNQuantizer`（定义在 `algorithms/quantization/rtn/quantizer.py`）

```python
# RTNQuantizer.quantize_block:
def quantize_block(self, block, input_ids=None, input_others=None,
                   reference_output=None, **kwargs):
    # 注意: input_ids, input_others, reference_output 全部不使用!

    # 直接遍历 block 内所有 linear，逐个量化
    for _name, m in block.named_modules():
        if hasattr(m, "global_name") and check_to_quantized(m):
            self.quantize_layer(m.global_name)    # ← 逐层独立量化

    return {}   # RTN 没有可训练参数

# RTNQuantizer.quantize_layer:
def quantize_layer(self, name):
    m = get_module(self.model, name)              # 获取单个 linear

    m = WrapperLinear(
        m,
        enable_minmax_tuning=False,               # 不调优
        enable_round_tuning=False,                 # 不调优
        iters=0,                                   # 零迭代
    )
    m = m.unwrapper({})                           # 直接 unwrap（RTN 取整）

    set_module(self.model, name, m)               # 写回
```

**ZeroShotCompressor 的循环**也更简单:
```python
# ZeroShotCompressor.quantize:
all_blocks = get_block_names(model)
for block_names in all_blocks:
    for block_name in block_names:
        block = get_module(model, block_name)
        materialize_model_(block)
        quantizer.quantize_block(block)           # 直接量化，无需收集数据
        mv_module_from_gpu(block)
```

**关键特征**:
- **不需要**校准数据、reference_output
- **不需要**训练循环（iters=0）
- 每个 linear 层**独立**量化（RTN 取整）
- 速度最快，精度最低

#### B2: CalibratedRTNCompressor + OptimizedRTNQuantizer（需要校准数据）

当启用 imatrix 或需要 activation calibration 时使用。

**Compressor**: `CalibratedRTNCompressor`（继承自 `CalibCompressor`）
**Quantizer**: `OptimizedRTNQuantizer`（继承自 `RTNQuantizer`）

```python
# OptimizedRTNQuantizer.quantize_block:
def quantize_block(self, block, input_ids=None, input_others=None,
                   reference_output=None, **kwargs):

    # 1. 利用 imatrix (importance matrix) 归一化
    for name, m in block.named_modules():
        if hasattr(m, "imatrix"):
            m.imatrix /= m.imatrix_cnt           # 归一化重要性

    # 2. 逐层量化（利用 imatrix 做更优的 RTN）
    for name, m in block.named_modules():
        if hasattr(m, "global_name") and check_to_quantized(m):
            self.quantize_layer_outside_block(m.global_name)
```

**关键特征**:
- 需要校准数据（用于收集 activation statistics / imatrix）
- **不需要**训练循环（仍然是 RTN，但有更好的 scale 估计）
- 精度优于纯 RTN，但不如 SignRound (iters>0)

### 4.4 三种 Quantizer 对比总表

| 维度 | SignRoundQuantizer | OptimizedRTNQuantizer | RTNQuantizer |
|------|-------------------|----------------------|--------------|
| 触发条件 | `iters > 0` | `iters == 0` + imatrix/act_calib | `iters == 0` 纯量化 |
| Config | `SignRoundConfig` | `RTNConfig` | `RTNConfig` |
| Compressor | `CalibCompressor` | `CalibratedRTNCompressor` | `ZeroShotCompressor` |
| 需要校准数据 | ✅ | ✅ | ❌ |
| 需要 reference_output | ✅ | ❌ | ❌ |
| 训练循环 | ✅ (iters 轮) | ❌ | ❌ |
| 优化方式 | block-wise 联合优化 | imatrix 辅助 RTN | 纯 RTN 取整 |
| 量化粒度 | 参数 per-linear, 优化 per-block | per-linear 独立 | per-linear 独立 |
| 精度 | ★★★★★ | ★★★ | ★★ |
| 速度 | ★★ (慢，需训练) | ★★★★ | ★★★★★ (最快) |

### 4.5 与 GPTQ 的对比

```
GPTQ (per-layer reconstruction):
  q_proj → 单独量化，最小化 q_proj 输出误差
  k_proj → 单独量化，最小化 k_proj 输出误差
  ...
  缺点：各层独立优化，忽略层间交互

Auto-Round SignRound (per-block reconstruction):
  [q_proj + k_proj + v_proj + o_proj + gate + up + down]
  → 联合优化，最小化整个 block 的输出误差
  优点：考虑层间交互，误差补偿更充分

Auto-Round RTN (per-linear, no optimization):
  q_proj → WrapperLinear → unwrap (RTN 取整)
  k_proj → WrapperLinear → unwrap
  ...
  特点：速度最快，无需校准数据，精度最低
```

### 4.6 粒度对比

| 维度 | SignRound (iters>0) | RTN (iters=0) |
|------|---------------------|---------------|
| Compressor 循环 | per-block | per-block |
| Rotation 应用 | per-block | per-block |
| 量化参数 | per-linear（每个 linear 独立的 scale/v） | per-linear（每个 linear 独立的 scale） |
| 优化目标 | per-block（联合最小化 block 输出误差） | per-linear（独立 RTN，无优化目标） |
| 内存占用 | per-block + 优化器状态 | per-block（更小，无优化器） |

---

## 5. Block Lifecycle Hooks 覆盖范围

### 5.1 Hook 集成点总览

Block Lifecycle Hooks（`_on_block_ready` / `_on_block_quantized` / `finalize_layerwise`）已集成到**所有三种** Compressor 的 block 循环中：

| Compressor | 场景 | 集成位置 | Hook 调用点 |
|-----------|------|---------|------------|
| `CalibCompressor` | iters>0, SignRound | `_quantize_blocks()` | ✅ `_on_block_ready` + `_on_block_quantized` + `finalize_layerwise` |
| `CalibCompressor` | iters>0, public API | `quantize_block()` | ✅ `_on_block_ready` + `_on_block_quantized` |
| `CalibratedRTNCompressor` | iters=0, OptimizedRTN | `_quantize_via_rtn_blockwise()` | ✅ `_on_block_ready` + `_on_block_quantized` + `finalize_layerwise` |
| `ZeroShotCompressor` | iters=0, 纯 RTN | `quantize()` blockwise 循环 | ✅ `_on_block_ready` + `_on_block_quantized` + `finalize_layerwise` |
| `ZeroShotCompressor` | iters=0, public API | `quantize_block()` | ✅ `_on_block_ready` + `_on_block_quantized` |

### 5.2 各场景的详细 Hook 位置

#### 场景 A: CalibCompressor（iters>0, SignRound Tuning）

```python
# CalibCompressor._quantize_blocks():
for i in range(0, len(block_names), nblocks):
    m = get_module(model, block_names[i])
    materialize_model_(m)
    m = m.to(device)
    
    self._on_block_ready(m, block_name, i)          # ← HOOK: rotation
    
    reference_output = quantizer._get_block_outputs(m, ...)
    quantizer.quantize_block(m, input_ids, ..., reference_output)
    
    self._on_block_quantized(m, block_name, i)      # ← HOOK: post-quant

# After loop:
for t in self._rotation_transforms:
    t.finalize_layerwise(model)                     # ← HOOK: finalize
```

#### 场景 B: CalibratedRTNCompressor（iters=0, OptimizedRTN + imatrix）

```python
# CalibratedRTNCompressor._quantize_via_rtn_blockwise():
block_idx = 0
for block_names in all_blocks:
    for block_name in block_names:
        block = get_module(model, block_name)
        materialize_model_(block)
        block = block.to(device)
        
        self._on_block_ready(block, block_name, block_idx)    # ← HOOK: rotation
        
        # collect act_max via forward hooks
        hook_handles = quantizer._register_act_max_hook(block)
        input_ids = quantizer._get_block_outputs(block, ...)
        
        quantizer.quantize_block(block)             # RTN with imatrix
        
        self._on_block_quantized(block, block_name, block_idx) # ← HOOK: post-quant
        block_idx += 1

# After loop:
for t in self._rotation_transforms:
    t.finalize_layerwise(model)                     # ← HOOK: finalize
```

#### 场景 C: ZeroShotCompressor（iters=0, 纯 RTN）

```python
# ZeroShotCompressor.quantize():
block_idx = 0
for block_names in all_blocks:
    for block_name in block_names:
        block = get_module(model, block_name)
        materialize_model_(block)
        
        self._on_block_ready(block, block_name, block_idx)    # ← HOOK: rotation
        
        quantizer.quantize_block(block)             # 纯 RTN
        
        self._on_block_quantized(block, block_name, block_idx) # ← HOOK: post-quant
        block_idx += 1

# After loop:
for t in self._rotation_transforms:
    t.finalize_layerwise(model)                     # ← HOOK: finalize
```

### 5.3 Hook 在不同场景下的行为

由于 `_on_block_ready` 定义在 `BaseCompressor` 中，所有 Compressor 共享同一个实现：

```python
def _on_block_ready(self, block, block_name, block_idx):
    transforms = getattr(self, "_rotation_transforms", None)
    if not transforms:
        return  # 无 rotation config 时，hook 是 no-op
    
    for t in transforms:
        t.rotate_layer(block, layer_idx=block_idx)
```

- **无 rotation**（默认）: `_rotation_transforms` 不存在或为空 → hook 立即返回，**零开销**
- **有 rotation**: 遍历 rotation transforms，对当前 block 施加旋转

这意味着 hook 的存在**不会影响**没有启用 rotation 的正常量化流程。

---

## 6. Rotation + Quantization 完整流程

### 6.1 Full-Model Rotation（传统模式，需要全模型在 GPU）

```
1. rotation.apply_to_model(model)     # 全模型旋转（需要全模型 on GPU）
2. for each block:
     block → GPU
     reference_output = block(inputs)
     quantize_block(block, ...)
     block → CPU
```

### 6.2 Layer-wise Rotation（新模式，省显存）

```
1. rotation.prepare_layerwise(model)  # 只初始化 R 矩阵（几 KB）
2. for each block:
     block → GPU
     _on_block_ready(block, idx):
       rotation.rotate_layer(block, idx)  # 仅旋转当前 block
     reference_output = block(inputs)
     quantize_block(block, ...)
     _on_block_quantized(block, idx)
     block → CPU
3. rotation.finalize_layerwise(model)
```

### 6.3 Layer-wise 模式的约束

| 约束 | 原因 |
|------|------|
| 必须 `online_r1_rotation=True` | Offline R1 修改 embed_tokens/lm_head，改变 inter-layer hidden state space，与 pre-cached inputs 不兼容 |
| R3/R4 需要 hook 机制 | 不能 offline fuse 到权重（需要运行时计算） |
| 训练模式需要额外内存 | `trainable_rotation=True` 仍需全模型 forward/backward |

---

## 7. 关键设计决策总结

### 7.1 为什么 Rotation 不放在 Quantizer 内部？

- Quantizer（`SignRoundQuantizer`）是**纯算法**：wrapper → 优化 → unwrap
- Rotation 是**模型变换**：修改权重 + 注册 hook
- 两者正交：rotation 不需要知道量化细节，量化不需要知道 rotation 细节
- **Template Method + Mediator Pattern**：Compressor 作为协调者，分别调用两者

### 7.2 为什么所有 layer 用同一个 R 矩阵？

- 数学保证：SpinQuant/QuaRot 的理论证明中，R 是全局正交矩阵
- 同构结构：所有 decoder layer 维度相同
- 效率：不需要为每层存储/计算独立的 R

### 7.3 为什么用 `hasattr`/`getattr` 而不是 name 匹配？

- 更 robust：不依赖 name string 的格式
- 更 Pythonic：直接访问对象属性
- 自动容错：`if hasattr(attn, "q_proj")` 对缺失属性自动跳过

---

## 8. 优化方案：Context Manager 消除 Hook 调用重复

### 8.1 当前问题

`_on_block_ready` 和 `_on_block_quantized` 虽然**定义在 `BaseCompressor` 一处**，但**调用点散落在三个 Compressor 各自的 block 循环中**：

```
CalibCompressor._quantize_blocks()           → 手动调用 _on_block_ready + _on_block_quantized
CalibratedRTNCompressor._quantize_via_rtn_blockwise() → 手动调用 _on_block_ready + _on_block_quantized
ZeroShotCompressor.quantize()                → 手动调用 _on_block_ready + _on_block_quantized
```

每个循环都需要：
1. 在正确位置调用 `self._on_block_ready(block, name, idx)`
2. 在正确位置调用 `self._on_block_quantized(block, name, idx)`
3. 循环结束后调用 `finalize_layerwise`

这导致 **3×3 = 9 个手动调用点**，容易遗漏、不一致。

### 8.2 方案 A：Context Manager + `_finalize_block_processing`

在 `BaseCompressor` 中新增两个方法：

```python
from contextlib import contextmanager

class BaseCompressor:
    
    @contextmanager
    def _block_lifecycle(self, block, block_name, block_idx):
        """Context manager wrapping a single block's lifecycle hooks.

        Usage::

            with self._block_lifecycle(block, name, idx):
                reference_output = ...          # compressor-specific
                quantizer.quantize_block(...)   # compressor-specific

        ``_on_block_ready`` fires on entry (after device placement, before
        any forward pass or quantization).
        ``_on_block_quantized`` fires on exit (after quantization, before
        cleanup/offload).
        """
        self._on_block_ready(block, block_name, block_idx)
        try:
            yield
        finally:
            self._on_block_quantized(block, block_name, block_idx)

    def _finalize_block_processing(self, model):
        """Finalize layer-wise rotation after all blocks are processed.

        Call this once after the block loop completes. Safe to call even
        when no rotation transforms are active (no-op).
        """
        transforms = getattr(self, "_rotation_transforms", None)
        if transforms:
            for t in transforms:
                t.finalize_layerwise(model)
```

### 8.3 各 Compressor 改造后的代码

#### CalibCompressor._quantize_blocks()

```python
# 改造前（3 个调用点）:
self._on_block_ready(m, block_name_or_names, i)
reference_output = quantizer._get_block_outputs(m, ...)
quantizer.quantize_block(m, ..., reference_output)
self._on_block_quantized(m, block_name_or_names, i)
# ... 循环结束后 ...
for t in self._rotation_transforms:
    t.finalize_layerwise(model)

# 改造后（1 个 with + 1 个 finalize）:
with self._block_lifecycle(m, block_name_or_names, i):
    reference_output = quantizer._get_block_outputs(m, ...)
    quantizer.quantize_block(m, ..., reference_output)
# ... 循环结束后 ...
self._finalize_block_processing(model)
```

#### CalibratedRTNCompressor._quantize_via_rtn_blockwise()

```python
# 改造前:
self._on_block_ready(block, block_name, block_idx)
hook_handles = quantizer._register_act_max_hook(block)
input_ids = quantizer._get_block_outputs(block, ...)
quantizer.quantize_block(block)
self._on_block_quantized(block, block_name, block_idx)
# ... 循环结束后 ...
for t in self._rotation_transforms:
    t.finalize_layerwise(model)

# 改造后:
with self._block_lifecycle(block, block_name, block_idx):
    hook_handles = quantizer._register_act_max_hook(block)
    input_ids = quantizer._get_block_outputs(block, ...)
    quantizer.quantize_block(block)
# ... 循环结束后 ...
self._finalize_block_processing(model)
```

#### ZeroShotCompressor.quantize()

```python
# 改造前:
self._on_block_ready(block, block_name, block_idx)
quantizer.quantize_block(block)
self._on_block_quantized(block, block_name, block_idx)
# ... 循环结束后 ...
for t in self._rotation_transforms:
    t.finalize_layerwise(model)

# 改造后:
with self._block_lifecycle(block, block_name, block_idx):
    quantizer.quantize_block(block)
# ... 循环结束后 ...
self._finalize_block_processing(model)
```

### 8.4 对比

| 维度 | 改造前 | 改造后 (Context Manager) |
|------|--------|------------------------|
| 手动调用点 | 9 个（3 循环 × 3 调用） | 6 个（3 循环 × (1 with + 1 finalize)） |
| 遗漏风险 | 高（容易忘记 `_on_block_quantized`） | 低（`with` 自动保证 enter/exit 配对） |
| 异常安全 | 需手动 try/finally | `with` 自动保证（`finally` 语义） |
| 代码可读性 | `_on_block_ready` 和 `_on_block_quantized` 散落在循环各处 | `with` 清晰标记 lifecycle 边界 |
| 新增 Compressor | 需记住加 3 个调用 | 只需 1 个 `with` + 1 个 `finalize` |
| 侵入性 | — | 极低（只加 2 个方法到 BaseCompressor） |

### 8.5 为什么不进一步统一 Block 循环？（方案 B）

理论上可以把三个 Compressor 的 block 循环统一到 `BaseCompressor._for_each_block()`，子类只注入差异逻辑。但实际上三个循环差异很大：

| Compressor | 独有逻辑 |
|-----------|---------|
| `CalibCompressor` | 收集 reference_output、处理 q_input 替换、DDP setup、multi-GPU AlignDevicesHook、nblocks>1 的 WrapperMultiblock |
| `CalibratedRTNCompressor` | imatrix 收集、act_max hook 注册、low_gpu_mem_usage 中间 offload |
| `ZeroShotCompressor` | immediate_saving shard write、tied_weights 处理、无 calibration 数据 |

强行统一会引入大量 `if/else` 分支或过多的 hook 点，反而降低可读性。**Context Manager 方案是当前架构下的最佳平衡点**。

---

## 9. Context Manager 实现完成状态

### 9.1 修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `auto_round/compressors_new/base.py` | 新增 `from contextlib import contextmanager`；新增 `_block_lifecycle()` context manager（~15行）；新增 `_finalize_block_processing()` helper（~10行） |
| `auto_round/compressors_new/calib.py` | `CalibCompressor._quantize_blocks()` 改用 `with self._block_lifecycle(...)`；`CalibCompressor.quantize_block()` 公共 API 改用 context manager；`CalibratedRTNCompressor._quantize_via_rtn_blockwise()` 新增 `block_idx` 计数器 + context manager + finalize |
| `auto_round/compressors_new/zero_shot.py` | `ZeroShotCompressor.quantize_block()` 公共 API 改用 context manager；`ZeroShotCompressor.quantize()` blockwise 循环新增 `block_idx` + context manager + finalize |

### 9.2 验证结果

```
✓ BaseCompressor has _block_lifecycle and _finalize_block_processing
✓ CalibCompressor._quantize_blocks uses context manager (直接调用已移除)
✓ CalibratedRTNCompressor._quantize_via_rtn_blockwise uses context manager
✓ ZeroShotCompressor.quantize uses context manager
✓ ZeroShotCompressor.quantize_block uses context manager
✓ CalibCompressor.quantize_block uses context manager
✓ Context manager calls hooks in correct order: ready → yield → quantized
✓ Context manager is exception-safe (_on_block_quantized fires on error)
✓ py_compile: base.py, calib.py, zero_shot.py 全部通过
```

### 9.3 Context Manager 行为保证

1. **调用顺序保证**: `_on_block_ready` → yield (业务逻辑) → `_on_block_quantized`
2. **异常安全**: 即使 yield 内部抛出异常，`_on_block_quantized` 仍会执行（`finally` 语义）
3. **向后兼容**: 如果没有配置 `_rotation_transforms`，hooks 是 no-op，零开销
4. **WrapperMultiblock 兼容**: `_on_block_ready` 内部处理 `nblocks > 1` 时 `block.layers` 的迭代

---

## 10. Save/Load 兼容性分析

### 10.1 核心结论

**Block-wise rotation 与 Full-model rotation 产生完全相同的最终模型状态，Save/Load 100% 兼容。**

### 10.2 分析依据

#### 10.2.1 两种 Rotation 模式的最终状态等价性

| 操作 | Full-model (一次性) | Block-wise (逐 block) |
|------|--------------------|-----------------------|
| R1: 权重吸收 | `W_new = R1 @ W` 一次性改所有 layer | 每个 block 单独 `W_new = R1 @ W`，结果相同 |
| R1: 激活 hook | 全部 layer 注册 hook | 逐个 layer 注册 hook，finalize 后效果相同 |
| R2: Head rotation | `W_q_new = W_q @ R2.T` 全部 layer | 逐 layer 做，矩阵运算结果相同 |
| R4: Hadamard fuse | `W_down_new = Had @ W_down` 全部 layer | 逐 layer 做，结果相同 |
| R3: Monkeypatch | 全部 layer 替换 forward | 逐 layer 替换，最终效果相同 |

**关键点**: 所有 decoder layers 是同构的（相同 hidden_size, num_heads, intermediate_size），使用相同的 R 矩阵。对于每个 layer，rotation 是独立的纯数学运算（矩阵乘法），不依赖其他 layer 的状态。因此 block-wise 和 full-model 产生完全相同的权重和 hook 配置。

#### 10.2.2 Save 路径分析

```python
# serialize.py: inject_spinquant_buffers()
def inject_spinquant_buffers(model, rotation_config):
    """将 model-level 的 R 矩阵注入为 per-module buffers"""
    # 读取 model.spinquant_R1, model.spinquant_R4_had_K 等
    # 写入每个 linear module 的 buffer: linear.spinquant_R1, linear.spinquant_R4_had_K
    # 写入 config.json: rotation_config 参数
```

- **不关心 R 矩阵是如何 apply 的**（一次性 vs 逐 block）
- **只读取最终的 model-level R 矩阵** → 两种方式初始化的是同一组 R 矩阵
- **只看最终的权重状态** → 两种方式处理后权重完全相同

#### 10.2.3 Load 路径分析

```python
# serialize.py: 加载流程
preregister_spinquant_buffers(model, config)  # 1. 预注册 buffer 占位
model.load_state_dict(state_dict)              # 2. 加载权重 + buffer
rebuild_spinquant_online(model, config)        # 3. 重建运行时 hook
```

- `preregister_spinquant_buffers`: 从 config.json 读取 rotation 参数，预注册 buffer 形状
- `load_state_dict`: 填充实际的 R 矩阵值和已旋转的权重
- `rebuild_spinquant_online`: 从 per-module buffer 重建 online hook（R1 激活旋转、R4 激活 Hadamard）

**所有步骤都是 transport-agnostic**：只依赖最终的 state_dict 内容和 config.json，不依赖 rotation 是如何 apply 的。

#### 10.2.4 R3 的特殊处理

R3（Hadamard on MLP residual）**不存储在 buffer 中**。它是确定性的（Hadamard 矩阵由维度唯一确定），在 load 时从 config 参数重建：

```python
# rebuild_spinquant_online() 中：
if config.r3_enabled:
    # 根据 intermediate_size 生成确定性 Hadamard 矩阵
    # monkeypatch layer.forward 插入 Hadamard 变换
```

### 10.3 兼容性矩阵

| 场景 | Full-model Save → Full-model Load | Full-model Save → Block-wise Load | Block-wise Save → Full-model Load | Block-wise Save → Block-wise Load |
|------|:---:|:---:|:---:|:---:|
| 可行? | ✅ | ✅ | ✅ | ✅ |

**原因**: Save 的产物（state_dict + config.json）完全相同，Load 不区分来源。

### 10.4 `finalize_layerwise()` 的作用

```python
def finalize_layerwise(self, model):
    """在所有 block 处理完毕后调用"""
    # 1. 清理临时状态（如 layer 计数器）
    # 2. 验证所有 layer 都已处理
    # 3. 可选：将 model-level R 矩阵提升为 buffer（与 full-model 路径对齐）
```

调用 `finalize_layerwise()` 后，模型状态与 full-model rotation 后的状态**完全一致**，后续的 save/load 流程无需任何修改。

### 10.5 总结

```
┌─────────────────────────────────────────────────────────────┐
│                    Rotation Apply                             │
│                                                              │
│   Full-model path ──┐                                        │
│                     ├──→ 相同的最终模型状态                    │
│   Block-wise path ──┘         │                              │
│                               ▼                              │
│                    inject_spinquant_buffers()                 │
│                               │                              │
│                               ▼                              │
│                    state_dict + config.json                   │
│                               │                              │
│                               ▼                              │
│                    preregister → load → rebuild               │
│                               │                              │
│                               ▼                              │
│                    完整的推理模型（带 online hooks）            │
└─────────────────────────────────────────────────────────────┘
```

**Block Lifecycle Hooks 的 Context Manager 实现与现有的 save/load 机制完全兼容，无需任何额外修改。**

---

## 11. Rotation + Quantization 场景适配总结

### 11.1 Block Lifecycle Hooks 覆盖的三种量化场景

| 场景 | Compressor | Quantizer | iters | 描述 |
|------|-----------|-----------|-------|------|
| SignRound Tuning | `CalibCompressor` | `SignRoundQuantizer` | >0 (e.g. 200) | 带校准数据的 block-wise 联合优化 |
| RTN + imatrix | `CalibratedRTNCompressor` | `OptimizedRTNQuantizer` | 0 | 带 imatrix 辅助的 RTN |
| Pure RTN | `ZeroShotCompressor` | `RTNQuantizer` | 0 | 无校准数据的纯 RTN |

**所有三种场景都已集成 Context Manager，rotation 在每个 block 量化前自动 apply。**

### 11.2 典型使用方式：Rotation + Quantization (iters=200)

```python
from auto_round import AutoRound

# rotation + quantization 联合使用
quantizer = AutoRound(
    model=model,
    tokenizer=tokenizer,
    bits=4,
    group_size=128,
    iters=200,                          # SignRound tuning
    rotation={
        "rotation_method": "spinquant",
        "rotation_mode": "r1+r2+r4",    # 选择 rotation 级别
        "online_r1_rotation": True,      # layer-wise 必须为 True
        "layerwise_rotation": True,      # 启用 block-wise rotation
    },
)
quantizer.quantize()
quantizer.save_quantized(output_dir)
```

**执行流程**:
```
1. prepare_layerwise(model)         → 初始化 R 矩阵（几 KB，CPU 即可）
2. for each block in model.layers:
   │  with _block_lifecycle(block, name, idx):
   │    ├── _on_block_ready:        → rotate_layer(block) → apply R1/R2/R4
   │    ├── reference_output = ...  → 收集未量化的参考输出
   │    ├── quantize_block(...)     → 200 iters SignRound 优化
   │    └── _on_block_quantized     → (no-op / 可扩展)
3. _finalize_block_processing(model) → finalize_layerwise() 清理
4. save_quantized(output_dir)        → inject_buffers + save state_dict
```

### 11.3 典型使用方式：Rotation + RTN (iters=0)

```python
quantizer = AutoRound(
    model=model,
    tokenizer=tokenizer,
    bits=4,
    group_size=128,
    iters=0,                            # RTN, 无训练
    rotation={
        "rotation_method": "spinquant",
        "rotation_mode": "r1+r2+r4",
        "online_r1_rotation": True,
        "layerwise_rotation": True,
    },
)
quantizer.quantize()
quantizer.save_quantized(output_dir)
```

**与 iters=200 的区别仅在于 quantize_block 内部**：
- iters=200: `SignRoundQuantizer` → 200 轮 AdamW 优化，逐步逼近最优量化参数
- iters=0: `RTNQuantizer` → 直接 round-to-nearest，无迭代优化

Rotation 部分（`_on_block_ready` 中的 `rotate_layer`）在两种情况下**完全相同**。

### 11.4 Layer-wise 约束

- **`online_r1_rotation` 必须为 `True`**: Offline R1 会修改 `embed_tokens` 和 `lm_head`，改变 inter-layer hidden state 的空间表示，与 pre-cached inputs 不兼容
- **`layerwise_rotation` 必须为 `True`**: 否则走 full-model 路径（`_apply_rotations` 在 compressor 初始化时一次性 apply）
- **所有 decoder layers 同构**: 同一组 R 矩阵对所有 layer 生效，不支持 per-layer 不同 rotation config
