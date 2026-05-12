# Layer-wise Rotation Proposal

> **目标**: 将 QuaRot/SpinQuant 的 rotation 操作从全模型一次性执行改为逐层（layer-wise / block-wise）执行，与 auto-round 的 block-wise 量化流程融合，实现大模型（70B+）场景下的显存可控。

---

## 1. 问题描述

### 1.1 当前架构

目前 rotation 和 quantization 是两个独立的阶段，串行执行：

```
Stage 1: 全模型 Rotation（整个模型必须在 GPU 上）
    BaseCompressor.post_init() → _apply_rotations()
    └── apply_rotation(model, rotation_cfg)
        └── SpinQuantRotation.apply_to_model()
            └── SpinQuantPreprocessor.preprocess()
                ├── 初始化 R 矩阵（R1/R2/R3/R4）
                ├── [可选] 训练 rotation 矩阵
                ├── 遍历所有 decoder layers，修改 weight + 注册 hook
                └── embed_tokens / lm_head 处理（仅 offline R1）

Stage 2: Block-wise 量化（逐层 GPU，显存可控）
    CalibCompressor.quantize()
    ├── cache block inputs（forward 一次整个模型，模型此时已 rotated）
    ├── model 移到 CPU
    └── for each block:
        ├── move block to GPU
        ├── materialize + convert dtype
        ├── collect reference output + act_max
        ├── SignRoundQuantizer.quantize_block()
        │   ├── wrapper_block() → nn.Linear 替换为 WrapperLinear
        │   ├── 优化迭代（WrapperLinear 执行 _forward_pre_hooks）
        │   └── unwrapper_block() → 还原为 nn.Linear
        ├── collect q_outputs
        └── move block to CPU
```

### 1.2 显存瓶颈

| 模型规模 | 模型参数量 (FP16) | Stage 1 显存需求 | Stage 2 显存需求 |
|----------|-------------------|------------------|------------------|
| 7B       | ~14 GB            | ~14 GB           | ~1-2 GB/block    |
| 13B      | ~26 GB            | ~26 GB           | ~2-3 GB/block    |
| 70B      | ~140 GB           | ~140 GB ❌ OOM   | ~4-6 GB/block ✅  |
| 405B     | ~810 GB           | ~810 GB ❌ OOM   | ~10-15 GB/block ✅|

**核心矛盾**: Stage 2 已经做到了 block-wise 显存可控，但 Stage 1（rotation）仍需要**整个模型同时在 GPU 上**，成为 70B+ 模型的瓶颈。

### 1.3 Quark 和 llm-compressor 也有同样的问题

- **Quark**: `RotationProcessor.process()` 遍历整个模型做 rotation，然后才进入 GPTQ/AWQ 的 layer-wise 量化
- **llm-compressor**: `SpinQuantModifier.on_start()` 一次性处理整个模型的 rotation，再进入 `GPTQModifier` 的逐层量化

三个框架都是"全模型 rotation → 逐层量化"的架构，都存在相同的显存瓶颈。

### 1.4 现有代码的关键调用链

```python
# 入口：BaseCompressor.post_init() [base.py:667]
def post_init(self):
    self._resolve_scheme()        # Phase 1: 解析 scheme、构造 quantizer
    self._resolve_formats()       # Phase 2
    self._patch_model()           # Phase 3: MoE 等模型修补
    self._build_layer_config()    # Phase 4: 构建 layer_config
    self._apply_rotations()       # Phase 4.5: ← 全模型 rotation（在此执行）
    self._hardware_setup()        # Phase 5

# _apply_rotations [base.py:847]
def _apply_rotations(self):
    for rotation_cfg in self.rotation_configs:
        self.model_context.model = apply_rotation(
            self.model_context.model, rotation_cfg, ...)

# apply_rotation [transforms/__init__.py:143]
def apply_rotation(model, config, ...):
    rotation = BaseRotation.from_config(config)          # → SpinQuantRotation
    return rotation.apply_to_model(model, ...)

# SpinQuantRotation.apply_to_model [algorithm.py:80]
def apply_to_model(self, model, ...):
    preprocessor = SpinQuantPreprocessor(model, self.config)
    return preprocessor.preprocess(dataloader)            # ← 全模型一次性处理

# 然后是 quantize 流程 [calib.py:1069]
def quantize(self):
    self.post_init()                                      # ← rotation 已完成
    all_inputs = self.try_cache_inter_data_gpucpu(...)     # cache（模型已 rotated）
    self.model = mv_module_from_gpu(self.model)
    for block_names in all_blocks:
        self._quantize_blocks(...)                        # 逐 block 量化
```

---

## 2. 可行性分析

### 2.1 Rotation 的层局部性

关键发现：**Online 模式下，所有 R1-R4 操作都是 per-decoder-layer 的，没有跨层依赖。**

| Rotation | 作用对象 | 操作类型 | 跨层依赖？ |
|----------|---------|---------|-----------|
| R1 (Online) | q/k/v_proj, gate/up_proj 的 weight + hook | per-module weight 修改 + hook 注册 | ❌ 无 |
| R2 (Offline fuse) | v_proj 的 output + o_proj 的 input | per-head weight 修改 | ❌ 无 |
| R3 (Hook) | attention 的 apply_rotary_pos_emb | monkeypatch + hook | ❌ 无 |
| R4 (Hook) | down_proj 的 forward_pre_hook | hook 注册 | ❌ 无 |

同一个 R 矩阵被**所有层共享**，但每层的操作是独立的。R 矩阵本身很小（R1: hidden_size × hidden_size，R2: head_dim × head_dim），不构成显存瓶颈。

### 2.2 Online R1 为什么 cached inputs 无需额外处理

Online R1 的关键特性：**hook 注册在 target module（q/k/v_proj）上，不在 decoder layer 的入口**。

```
cached_input（block 的原始输入，未 rotate）
    ↓
[decoder_layer_i]                                # block-level forward
    ├── input_layernorm(x)               → h     # 对 x 做 LayerNorm
    ├── q_proj.hook: h @ R               → h_rot # ← R1 hook 自动生效
    ├── q_proj.linear(h_rot, W@R)        → q     # (h@R) @ (W@R)^T = h @ W^T
    ├── ...同理 k_proj, v_proj...
    ├── attention → output
    ├── residual_add(x, output)          → x'    # 输出 ≡ 原始 FP 输出
    ├── post_layernorm(x')               → h2
    ├── gate_proj.hook: h2 @ R           → h2_rot
    ├── gate_proj.linear(h2_rot, W@R)    → gate
    ├── ...
    └── output = x' + mlp_out                    # 输出 ≡ 原始 FP 输出
```

因为 `(h@R) @ (W@R)^T = h @ (R@R^T) @ W^T = h @ W^T`，每一层的**输入和输出都与未 rotate 的 FP 计算等价**。所以 cached inputs 和逐层传递的 reference_output 都不需要变换。

### 2.3 Offline R1 的特殊性

Offline R1 会修改 `embed_tokens`、`o_proj`、`down_proj` 和 `lm_head`，改变了隐状态的坐标空间：

```
embed_tokens  ──→  [decoder_layer_0] ──→ ... ──→ [decoder_layer_N] ──→  lm_head
      ↑                                                                    ↑
  offline R1 修改                                                    offline R1 修改
  online R1 不修改                                                   online R1 不修改
```

- **Online R1**: embed_tokens 和 lm_head **不需要修改**，完全 per-layer 可行
- **Offline R1**: 需要额外处理 embed_tokens 和 lm_head，且需要重新 cache inputs

### 2.4 R 矩阵的显存开销

| 矩阵 | 大小（70B 模型, hidden=8192, head_dim=128） | 显存 |
|------|----------------------------------------------|------|
| R1   | 8192 × 8192, float64                         | ~512 MB |
| R2   | 128 × 128, float64                           | ~0.1 MB |
| R3   | 128 × 128, float64                           | ~0.1 MB |
| R4   | intermediate_size（28672） or rotation_size   | 视 rotation_size 而定 |

即使 R1 矩阵存储在 GPU 上，也只占 ~512 MB，远小于单个 decoder layer 的参数量（70B 模型单层 ~1.8 GB FP16）。

### 2.5 结论：完全可行

Online 模式下的 R1/R2/R3/R4 具有完美的层局部性，可以安全地拆分为逐层执行。

---

## 3. 设计方案

### 3.1 深入分析现有架构

深入研究 auto-round 的代码后，发现了更多的架构细节：

**关键发现 1：WrapperLinear 已经支持 hook 转发**

```python
# wrapper.py:489-519
class WrapperLinear(nn.Module):
    def forward(self, x):
        x = x.to(self.device)
        weight_q, *_ = self._qdq_weight(...)

        if self.enable_act_quant:
            # ★ 主动执行 orig_layer 的 _forward_pre_hooks
            for hook in self.orig_layer._forward_pre_hooks.values():
                result = hook(self.orig_layer, (x,))
                if result is not None:
                    x = result[0] if isinstance(result, tuple) else result
            x, _, _ = self._qdq_act(x, ...)
        elif len(self.orig_layer._forward_pre_hooks) > 0:
            # ★ 即使没有 act_quant，也执行 pre-hooks
            for hook in self.orig_layer._forward_pre_hooks.values():
                result = hook(self.orig_layer, (x,))
                ...
```

这意味着：**rotation hook 注册在 orig_layer（nn.Linear）上，WrapperLinear 在 forward 中会主动执行它们。** Hook 的执行顺序是：pre-hooks（rotation）→ activation quantization → weight quantization → linear。这正好是 Online R1 需要的顺序。

**关键发现 2：WrapperWALayer 会"偷走"hook**

```python
# wrapper.py:554-558
class WrapperWALayer(nn.Module):
    def __init__(self, orig_layer, ...):
        # 把 orig_layer 的 pre-hooks 偷过来，自己执行
        self._stolen_pre_hooks = list(orig_layer._forward_pre_hooks.values())
        orig_layer._forward_pre_hooks.clear()
```

推理路径（WrapperWALayer）也兼容 rotation hooks。

**关键发现 3：`_apply_rotations` 在 `post_init` 中执行**

```python
# base.py:667
def post_init(self):
    ...
    self._apply_rotations()  # Phase 4.5
    ...
```

rotation 发生在 `post_init()` 中，而 `quantize()` 调用 `self.post_init()` 作为第一步。这意味着当 `quantize()` 开始 cache inputs 时，模型已经被 rotated 了。

**关键发现 4：`_quantize_blocks` 的 block 处理流程**

```python
# calib.py:917-1054
for i in range(0, len(block_names), nblocks):
    m = get_module(model, n)          # 从 CPU 上的 model 获取 block

    materialize_model_(m)             # 从 meta 实例化权重
    convert_module_to_hp_if_necessary(m, amp_dtype, device)
    m = m.to(device)                  # → GPU

    # ── 收集 reference output + act_max ──
    reference_output = _get_block_outputs(m, input_ids, ...)

    # ── SignRound 量化 ──
    quantizer.quantize_block(m, input_ids, ...)
    # └── 内部调用 wrapper_block() → WrapperLinear 包装
    # └── 训练迭代
    # └── unwrapper_block() → 还原

    # ── 收集 q_outputs ──
    q_input = _get_block_outputs(m, input_ids, ...)

    # ── 清理 ──
    mv_module_from_gpu(m)             # → CPU
    input_ids = reference_output      # 下一层的输入 = 当前层 FP 输出
```

### 3.2 核心设计思路

基于以上分析，最佳融合方案是：

**在 `_apply_rotations()` 中不执行全模型 rotation，而是保存 preprocessor 状态；在 `_quantize_blocks()` 的循环中，每个 block 上 GPU 后、reference_output 收集前，执行该 block 的 rotation。**

```
改造后的流程：

post_init():
  _apply_rotations()
  ├── layerwise=False (默认): 全模型 rotation（向后兼容）
  └── layerwise=True:
      └── preprocessor.prepare()  # 只初始化 R 矩阵，不修改 weight

quantize():
  cache block inputs（模型未 rotated，原始 FP16）
  for each block:
    ├── move block to GPU
    ├── ★ preprocessor.rotate_layer(block, idx)  # layer-wise rotation
    ├── collect reference_output    # rotation 后的 FP 输出 ≡ 原始 FP 输出
    ├── quantize_block()            # WrapperLinear 自动执行 rotation hooks
    ├── collect q_outputs
    └── move block to CPU
```

### 3.3 SpinQuantPreprocessor API 扩展

```python
class SpinQuantPreprocessor:
    # ── 原有 API（保留向后兼容）──
    def preprocess(self, dataloader=None) -> nn.Module:
        """全模型一次性 rotation。"""
        ...

    # ── 新增 API：分阶段 rotation ──
    def prepare(self, dataloader=None) -> None:
        """全局准备阶段：验证维度、初始化 R 矩阵、[可选]训练。
        不修改任何模型 weight。显存需求极低（仅 R 矩阵 buffer）。"""
        ...

    def rotate_layer(self, layer: nn.Module, layer_idx: int) -> None:
        """对单个 decoder layer 执行所有 rotation 操作。
        layer 须已在 GPU 上。"""
        ...

    def rotate_embedding(self) -> None:
        """处理 embed_tokens（仅 offline R1 需要）。"""
        ...

    def rotate_lm_head(self) -> None:
        """处理 lm_head（仅 offline R1 需要）。"""
        ...

    def finalize(self) -> None:
        """清理训练工件，存储 config 到模型。"""
        ...
```

### 3.4 BaseRotation 接口扩展

在 `BaseRotation` 中添加 layer-wise 支持：

```python
class BaseRotation(ABC):
    @abstractmethod
    def apply_to_model(self, model, data_type="mx_fp", **kwargs):
        """全模型一次性 rotation（现有接口）。"""
        ...

    def prepare_layerwise(self, model, data_type="mx_fp", **kwargs):
        """准备 layer-wise rotation（可选实现）。
        返回一个可调用对象或 preprocessor。
        默认 raise NotImplementedError。"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support layer-wise rotation"
        )

    def rotate_layer(self, layer, layer_idx, **kwargs):
        """对单层执行 rotation（可选实现）。"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support layer-wise rotation"
        )

    @property
    def supports_layerwise(self) -> bool:
        """是否支持 layer-wise rotation。"""
        return False
```

SpinQuantRotation 覆盖：

```python
@BaseRotation.register("spinquant")
class SpinQuantRotation(BaseRotation, RotationSerializer):
    @property
    def supports_layerwise(self) -> bool:
        return True

    def prepare_layerwise(self, model, data_type="mx_fp", **kwargs):
        preprocessor = SpinQuantPreprocessor(model, self.config)
        preprocessor.prepare(kwargs.get("dataloader"))
        self._preprocessor = preprocessor
        return preprocessor

    def rotate_layer(self, layer, layer_idx, **kwargs):
        self._preprocessor.rotate_layer(layer, layer_idx)
```

### 3.5 BaseCompressor 集成

```python
# base.py: _apply_rotations() 改造
def _apply_rotations(self) -> None:
    if not self.rotation_configs:
        return

    self._rotation_preprocessors = []

    for rotation_cfg in self.rotation_configs:
        rotation = BaseRotation.from_config(rotation_cfg)

        if self.layerwise_rotation and rotation.supports_layerwise:
            # Layer-wise: 只初始化，不修改 weight
            logger.info("Preparing layer-wise rotation (R matrices only).")
            preprocessor = rotation.prepare_layerwise(
                self.model_context.model,
                data_type=self.quantize_config.data_type,
            )
            self._rotation_preprocessors.append(preprocessor)
        else:
            # 全模型 rotation（现有行为）
            logger.info("Applying full-model rotation.")
            self.model_context.model = rotation.apply_to_model(
                self.model_context.model,
                data_type=self.quantize_config.data_type,
            )
```

```python
# calib.py: _quantize_blocks() 中集成
for i in range(0, len(block_names), nblocks):
    m = get_module(model, n)
    materialize_model_(m)
    convert_module_to_hp_if_necessary(m, amp_dtype, device)
    m = m.to(device)

    # ═══ NEW: Layer-wise rotation ═══
    # 必须在 reference_output 收集前、wrapper_block 前执行
    for preprocessor in self._rotation_preprocessors:
        preprocessor.rotate_layer(m, layer_idx=i)
    # ════════════════════════════════

    # ── 后续不变 ──
    reference_output = _get_block_outputs(m, input_ids, ...)
    quantizer.quantize_block(m, input_ids, input_others, reference_output, ...)
    ...
```

---

## 4. Cached Inputs 的正确性分析

### 4.1 Online R1 — 无需额外处理

这是整个方案最微妙也最重要的部分。

**场景**：layer-wise 模式下，cache inputs 时模型尚未 rotated。那用未 rotated 的 inputs 喂给 rotated 的 block，结果正确吗？

**答案：正确。** 因为 Online R1 hook 注册在子模块（q/k/v_proj）上，不在 block 入口：

```
cached_input (原始 activation, 未 rotate)
    ↓
block.forward(cached_input)
    ├── input_layernorm(x)                → h      # 不变
    ├── q_proj._forward_pre_hooks: h @ R  → h_rot  # hook 自动触发
    ├── q_proj.linear(h_rot, W@R)                   # (h@R) @ (W@R)^T = h @ W^T ← 等价
    ├── ...
    ├── o_proj(attn_out)                             # o_proj 未修改（Online R1），输出不变
    ├── residual: x + attn_out                       # = 原始 FP 输出
    ├── ...
    └── output = x + mlp_out                         # = 原始 FP 输出
```

**数学证明**：

对于 Online R1，每个 target module 的计算是：
```
output = (x @ R) @ (W @ R)^T
       = x @ R @ R^T @ W^T
       = x @ I @ W^T
       = x @ W^T
```
所以 block 的 output **完全等价于未 rotated 的 FP 输出**。

这意味着：
1. **reference_output** 不受影响（数值等价）
2. **下一层的 input_ids = reference_output** 也不受影响
3. 量化时 WrapperLinear 通过转发 `orig_layer._forward_pre_hooks` 自动执行 rotation

### 4.2 R2 / R3 / R4 — 同样无需额外处理

- **R2**：fuse 到 v_proj output + o_proj input，两侧抵消，不改变 block 输出
- **R3**：monkeypatch attention 内部的 RoPE，不改变 block 输入/输出接口
- **R4**：hook on down_proj，`(x@R) @ (W@R)^T = x@W^T`，不改变 block 输出

### 4.3 Offline R1 — 需要重新 cache

Offline R1 修改了 `o_proj` 和 `down_proj` 的 output 维度（`R1_inv @ W`），使得 block 输出处于 rotated 坐标空间。cached inputs 不再适用。

**解决方案**：在 layer-wise rotation 模式下，强制要求 `online_r1_rotation=True`。Offline R1 仍使用全模型模式。

---

## 5. WrapperLinear 兼容性详解

### 5.1 训练路径（WrapperLinear）

```python
# wrapper.py:489-537
class WrapperLinear(nn.Module):
    def forward(self, x):
        weight_q, *_ = self._qdq_weight(...)     # 量化 weight

        # ★ 执行 orig_layer 的 pre-hooks（rotation hooks）
        for hook in self.orig_layer._forward_pre_hooks.values():
            result = hook(self.orig_layer, (x,))
            if result is not None:
                x = result[0]

        # 执行 activation 量化
        x, _, _ = self._qdq_act(x, ...)

        # 执行 linear: output = x @ weight_q.T
        output = self.orig_forward(x, weight_q, bias)

        # ★ 执行 orig_layer 的 post-hooks
        for hook in self.orig_layer._forward_hooks.values():
            hook_result = hook(self.orig_layer, (x,), output)
            ...
```

执行顺序：`rotation hook(x→x@R) → act_quant(x@R) → weight_quant(W@R) → linear((x@R) @ Q(W@R)^T)`

这是正确的：量化在 rotated space 中进行，rotation hook 保证 activation 和 weight 在同一个坐标空间中。

### 5.2 推理路径（WrapperWALayer）

```python
# wrapper.py:554-558
class WrapperWALayer(nn.Module):
    def __init__(self, orig_layer):
        # ★ "偷走" hook
        self._stolen_pre_hooks = list(orig_layer._forward_pre_hooks.values())
        orig_layer._forward_pre_hooks.clear()

    def forward(self, x):
        # ★ 执行被偷走的 pre-hooks
        for hook in self._stolen_pre_hooks:
            result = hook(self.orig_layer, (x,))
            ...
        x, _, _ = self._qdq_act(x, ...)
        return self.orig_layer(x)
```

也正确兼容。

### 5.3 Layer-wise rotation 的时序

```
1. block to GPU
2. rotate_layer(block)        # 修改 nn.Linear weight + 注册 hook
3. reference_output = forward  # hook 生效，输出等价
4. quantizer.quantize_block()
   4a. wrapper_block()        # nn.Linear → WrapperLinear(orig_layer=nn.Linear)
   4b. 训练迭代              # WrapperLinear.forward 主动执行 orig_layer._forward_pre_hooks ✅
   4c. unwrapper_block()     # WrapperLinear → nn.Linear（hook 仍在 nn.Linear 上）
5. q_output = forward         # hook 仍在，正确
6. block to CPU
```

**结论：WrapperLinear 已经完全兼容 rotation hooks，不需要任何修改。**

---

## 6. 显存分析

### 6.1 Layer-wise 模式下的峰值显存

以 70B 模型（hidden_size=8192, 80 layers）为例：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| R1 矩阵 | ~512 MB | 8192×8192 float64 |
| R2 矩阵 | ~0.1 MB | 128×128 float64 |
| 单层参数 (FP16) | ~1.8 GB | q/k/v/o_proj + gate/up/down_proj |
| Cached inputs | ~2-4 GB | nsamples × seq_len × hidden_size |
| Reference output | ~2-4 GB | 同上 |
| SignRound 优化 | ~2-4 GB | 梯度 + 优化器状态 |
| **总计** | **~8-14 GB** | **单卡 L20 (48GB) 轻松放下** |

### 6.2 与全模型 Rotation 的对比

| 指标 | 全模型 Rotation | Layer-wise Rotation |
|------|----------------|---------------------|
| 峰值显存 (70B) | ~140 GB ❌ | ~8-14 GB ✅ |
| 峰值显存 (405B) | ~810 GB ❌ | ~15-25 GB ✅ |
| 需要多卡 tensor parallel | 是 | 否 |
| 实现复杂度 | 低 | 中 |
| rotation 正确性 | 基准 | 数学等价（Online R1） |

---

## 7. 完整的集成点详解

### 7.1 `_apply_rotations()` 改造

```python
# base.py
def _apply_rotations(self) -> None:
    """Phase 4.5 – Apply Hadamard / rotation transforms.

    When layerwise_rotation=True, only prepare() is called here
    (initializes R matrices as model buffers). The actual per-layer
    rotation is deferred to _quantize_blocks().
    """
    if not self.rotation_configs:
        return

    self._rotation_preprocessors = []  # 用于 layer-wise 模式

    for rotation_cfg in self.rotation_configs:
        rotation = BaseRotation.from_config(rotation_cfg)

        if getattr(self, 'layerwise_rotation', False) and rotation.supports_layerwise:
            logger.info("[Rotation] Layer-wise mode: preparing R matrices only.")
            prep = rotation.prepare_layerwise(
                self.model_context.model,
                data_type=self.quantize_config.data_type,
            )
            self._rotation_preprocessors.append(prep)
        else:
            logger.info("[Rotation] Full-model mode: applying rotation now.")
            self.model_context.model = rotation.apply_to_model(
                self.model_context.model,
                data_type=self.quantize_config.data_type,
            )
```

### 7.2 `_quantize_blocks()` 中的集成

```python
# calib.py: _quantize_blocks() 核心循环

for i in range(0, len(block_names), nblocks):
    if nblocks == 1:
        n = block_names[i]
        m = get_module(model, n)
    else:
        names = block_names[i : min(i + nblocks, len(block_names))]
        modules = [get_module(model, n) for n in names]
        m = WrapperMultiblock(modules)

    # ── Infrastructure: materialize, dtype, device ──
    materialize_model_(m)
    convert_module_to_hp_if_necessary(m, amp_dtype, device)
    m = m.to(device)

    # ══════════════════════════════════════════════════
    # ★ NEW: Layer-wise rotation
    # 在 reference_output 收集之前执行
    # ══════════════════════════════════════════════════
    rotation_preprocessors = getattr(self, '_rotation_preprocessors', [])
    if rotation_preprocessors:
        if nblocks == 1:
            for prep in rotation_preprocessors:
                prep.rotate_layer(m, layer_idx=i)
        else:
            # Multi-block: rotate each layer in the WrapperMultiblock
            for j, sub_module in enumerate(modules):
                for prep in rotation_preprocessors:
                    prep.rotate_layer(sub_module, layer_idx=i + j)
    # ══════════════════════════════════════════════════

    # ── Infrastructure: reference output, act_max ──
    reference_output = _get_block_outputs(m, input_ids, ...)

    # ── Algorithm: quantize (SignRound/RTN) ──
    quantizer.quantize_block(m, input_ids, ...)
    # └── wrapper_block() → WrapperLinear 自动转发 rotation hooks ✅

    # ── q_outputs, cleanup ──
    ...
    mv_module_from_gpu(m)
    input_ids = reference_output  # 下一层的输入
```

### 7.3 `quantize()` 中的 finalize

```python
# calib.py: quantize() 尾部

# 在 block 循环完成后
for prep in getattr(self, '_rotation_preprocessors', []):
    # Offline R1: 处理 embed_tokens / lm_head（如果需要）
    if hasattr(prep, 'rotate_embedding'):
        prep.rotate_embedding()
    if hasattr(prep, 'rotate_lm_head'):
        prep.rotate_lm_head()
    prep.finalize()

self._quantize_layers(layer_names, all_inputs)  # 处理 block 外的层（如 lm_head）
```

---

## 8. 用户 API

### 8.1 最简用法

```python
from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

# 方式 1: 全模型 rotation（默认，向后兼容）
model = AutoRound(
    model=model, tokenizer=tokenizer,
    rotation_config=SpinQuantConfig(r1=True, r2=True, online_r1_rotation=True),
    ...
)
model, _ = model.quantize()

# 方式 2: Layer-wise rotation（适用于大模型）
model = AutoRound(
    model=model, tokenizer=tokenizer,
    rotation_config=SpinQuantConfig(r1=True, r2=True, online_r1_rotation=True),
    layerwise_rotation=True,  # ← 唯一新增参数
    ...
)
model, _ = model.quantize()
```

### 8.2 自动检测

可以考虑根据模型大小自动启用 layer-wise rotation：

```python
# 自动判断：模型参数 > 单 GPU 显存时自动启用
if model_size_gb > available_gpu_memory_gb * 0.7:
    layerwise_rotation = True
    logger.info("Auto-enabled layer-wise rotation (model too large for full-model mode)")
```

---

## 9. 风险和注意事项

### 9.1 Hook 的 device 一致性

rotation hook 闭包中持有 R 矩阵或 hadamard_K 的引用。当 block 在 GPU 和 CPU 之间移动时，hook 中的 tensor 需要跟随移动。

**现有代码已处理**：hook 中使用 `.to(x.device)` 动态适配设备：
```python
def hook(module, args):
    x = args[0]
    R = R_block_f32.to(x.device, dtype=x.dtype)  # ← 自动适配
    x = (x @ R).reshape(shape).to(dtype)
    return (x,) + args[1:]
```

### 9.2 torch.compile 兼容性

`_resolve_block_forward()` 中已有处理：当存在 act_quant hooks 时自动退回 plain forward。rotation hooks 也属于 forward_pre_hooks，torch.compile 通常能内联处理。但需要测试验证。

### 9.3 Multi-block quantization (nblocks > 1)

方案中已考虑：当 nblocks > 1 时，对 `WrapperMultiblock` 内的每个子 module 分别执行 `rotate_layer()`。

### 9.4 low_cpu_mem_usage 兼容性

`low_cpu_mem_usage=True` 时，block 可能被 offload 到 disk。rotation 必须在 `materialize_model_()` 之后执行，确保 weight 已经从 disk 加载到内存。

### 9.5 Trainable rotation + layer-wise

训练阶段需要完整的 forward/backward pass，不能逐层执行。解决方案：
- 如果 `trainable_rotation=True`，`prepare()` 阶段仍需加载整个模型做训练
- 或者：让用户先单独训练 R 矩阵（保存为 checkpoint），再加载 R 矩阵做 layer-wise quantization
- 或者：训练用 gradient checkpointing + 低 batch_size 降低显存

### 9.6 Offline R1 的限制

Layer-wise 模式下，**只支持 Online R1**。原因：Offline R1 改变了层间隐状态的坐标空间，cached inputs 不再有效。强制限制：

```python
def prepare(self, dataloader=None):
    if self.config.r1 and not self.config.online_r1_rotation:
        raise ValueError(
            "Layer-wise rotation requires online_r1_rotation=True. "
            "Offline R1 changes inter-layer hidden state space, "
            "which is incompatible with pre-cached block inputs."
        )
    ...
```

---

## 10. 实施计划

### Phase 1: 核心框架

1. `BaseRotation` 添加 `prepare_layerwise()` / `rotate_layer()` / `supports_layerwise` 接口
2. `SpinQuantPreprocessor` 添加 `prepare()` / `rotate_layer()` / `finalize()` 方法
3. 从现有 `_apply_online_r1()` / `_fuse_r2_rotation()` / `_fuse_r4_rotation()` 提取单层逻辑
4. `BaseCompressor._apply_rotations()` 支持 layer-wise 分支
5. `CalibCompressor._quantize_blocks()` 集成 `rotate_layer()` 调用

### Phase 2: 验证

6. 单元测试：全模型 rotation vs layer-wise rotation 输出一致性
7. 集成测试：Qwen3-0.6B + Online R1+R2 的 lm_eval 结果对比
8. R3+R4 的 layer-wise 支持和验证

### Phase 3: 大模型

9. 70B 模型 layer-wise rotation + quantization 端到端
10. 显存 profiling
11. 自动检测逻辑

---

## 11. 总结

| 维度 | 全模型 Rotation | Layer-wise Rotation |
|------|----------------|---------------------|
| 显存 | O(模型参数量) | O(单层参数量 + R矩阵 + cache) |
| 正确性 | 基准 | 数学等价（Online R1）|
| WrapperLinear 兼容 | ✅ hooks 已支持 | ✅ hooks 已支持 |
| 实现改动量 | N/A | 中（~5个文件，~200行新代码）|
| 适用场景 | 小模型 (≤13B) | 大模型 (70B+) |
| 限制 | 无 | 仅 Online R1（Offline R1 不兼容 cached inputs）|
| 向后兼容 | N/A | 完全兼容（默认仍用全模型模式）|

**核心结论**：

1. **WrapperLinear 已经完美兼容 rotation hooks** — 它主动执行 `orig_layer._forward_pre_hooks`，不需要任何修改
2. **Online R1 的 `R·R^T=I` 抵消特性** 使得 block 输出等价于原始 FP 输出，cached inputs 无需重新计算
3. **集成点在 `_quantize_blocks()` 中，在 `materialize` 之后、`_get_block_outputs` 之前** — 这是唯一正确的位置
4. **改动量可控**：约 5 个文件、200 行新代码，不影响现有行为
5. **Offline R1 不支持 layer-wise** — 因为它改变了层间隐状态空间，与 pre-cached inputs 不兼容
