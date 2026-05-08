# Auto-Round 支持 InputRotationWrapper 的重构分析

## 1. 问题回顾

`InputRotationWrapperHadamard` 是一个 `nn.Module`，包装了 `nn.Linear`，在 `forward` 中先对输入做 Hadamard 旋转，再调用内部 Linear。但它与 auto-round 的量化 pipeline **不兼容**：

```
wrapper_block() 通过 block.named_modules() 递归遍历
    → 发现 "q_proj.original_module" 是 nn.Linear
    → 用 WrapperLinear 包装了内部的 nn.Linear（而不是外层 wrapper）
    → 量化 / unwrap 后内部 Linear 变成 WrapperWALayer
    → WrapperWALayer.forward 调 orig_layer.forward(x)
    → orig_layer.weight 已被压缩为 shape [0]
    → 💥 size mismatch
```

**核心矛盾**：auto-round 通过 `type(m) in SUPPORTED_LAYER_TYPES`（精确类型匹配）发现量化目标，再通过 `named_modules()` 递归遍历。任何把 `nn.Linear` 存为子模块的 wrapper，其内部 Linear 都会被发现并被错误包装。

---

## 2. auto-round 量化 pipeline 关键路径分析

### 2.1 模块发现与包装（wrapper.py）

```python
# wrapper.py:768-769 — 精确类型匹配
def wrapper_block(block, ...):
    for n, m in block.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:   # ← 只认 nn.Linear, Conv1D
            new_m = WrapperLinear(m, ...)
            set_module(block, n, new_m)
```

**关键约束**：
- 使用 `type(m)`（精确类型）而非 `isinstance(m)`（多态）
- `SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)`
- 全局常量，在 `auto_round/utils/common.py:607` 定义

### 2.2 WrapperLinear 的 forward_pre_hook 传播

```python
# wrapper.py:503-518 — 显式运行 orig_layer 的 hooks
def forward(self, x):
    weight_q, *_ = self._qdq_weight(...)
    # 在量化激活之前，先运行 orig_layer 的 forward_pre_hooks
    for hook in self.orig_layer._forward_pre_hooks.values():
        result = hook(self.orig_layer, (x,))
        if result is not None:
            x = result[0]
    # 然后做激活量化
    x, _, _ = self._qdq_act(x, ...)
    # 最后做线性计算
    output = self.orig_forward(x, weight_q, bias)  # → F.linear(x, w, b)
```

### 2.3 WrapperWALayer 的 hook 偷取

```python
# wrapper.py:554-607 — 从 orig_layer 偷取 hooks
class WrapperWALayer:
    def __init__(self, orig_layer, ...):
        self._stolen_pre_hooks = list(orig_layer._forward_pre_hooks.values())
        orig_layer._forward_pre_hooks.clear()  # 防止重复执行

    def forward(self, x):
        # 1) 先运行偷取的 hooks（如 Hadamard 旋转）
        for hook in self._stolen_pre_hooks:
            result = hook(self.orig_layer, (x,))
            ...
        # 2) 激活量化
        x, _, _ = self.orig_layer.act_quant_func(x, ...)
        # 3) 原始 linear forward
        return self.orig_layer.forward(x)
```

### 2.4 Export/Save 路径（export_to_nvfp_mx.py）

```python
# export_to_nvfp_mx.py:56-62
def pack_layer(name, model, backend, device=None):
    layer = get_module(model, name)
    if type(layer) not in SUPPORTED_LAYER_TYPES and not isinstance(layer, WrapperWALayer):
        return  # 跳过非标准类型

    if isinstance(layer, WrapperWALayer):
        layer = layer.orig_layer   # 拆掉 WrapperWALayer
        set_module(model, name, layer)

    # 然后打包为 QuantLinear...
```

### 2.5 Pipeline 总结

```
                    发现目标                包装                  校准量化
named_modules() ──→ type check ──→ WrapperLinear(orig) ──→ 训练 v/scale ──→
                                                                            │
    推理                  保存                  解包装                      │
WrapperWALayer ←── pack_layer() ←── unwrapper_block() ←──── best_params ←──┘
```

**所有环节都假设量化目标是 `nn.Linear`**。任何非 `nn.Linear` 的 wrapper 类型在这条链上的每个环节都可能出问题。

---

## 3. compressors_new 架构概览

`compressors_new/` 是 auto-round 正在进行的重构：

```
entry.py         ← AutoRound.__new__() 路由到具体 Compressor
    ↓
base.py          ← BaseCompressor：40+ 方法，7 阶段 pipeline
    ↓
calib.py         ← CalibCompressor / CalibratedRTNCompressor
    ↓
spinquant_mixin.py ← SpinQuantMixin（rotation 预处理）
```

### 3.1 新 pipeline 的阶段

```python
# base.py: post_init() 中的 7 个阶段
Phase 1: _resolve_scheme()         # 解析量化方案
Phase 2: _resolve_formats()        # 解析输出格式
Phase 3: _patch_model()            # 模型结构修改（MoE 合并等）
Phase 4: _build_layer_config()     # 构建逐层配置
Phase 4.5: _apply_rotations()      # 应用 Hadamard / rotation 变换
Phase 5: _hardware_setup()         # 硬件 / 设备设置
Phase 6: calibrate & quantize      # 校准 + 量化（在 calib.py）
```

### 3.2 Rotation 集成点

```python
# base.py:845-876
def _apply_rotations(self):
    """Phase 4.5 – rotation 在 layer_config 之后、量化之前"""
    for rotation_cfg in self.rotation_configs:
        self.model_context.model = apply_rotation(
            self.model_context.model, rotation_cfg, data_type=...
        )
    # 此时 model 已带旋转权重 + online hooks
    # 接下来 calib.py 的 wrapper_block() 会发现这些模块
```

**问题在 Phase 4.5 和 Phase 6 的衔接**：Phase 4.5 注册了 hooks（或安装了 wrapper），Phase 6 的 `wrapper_block()` 递归遍历模块树时会撞到 wrapper 内部的 Linear。

---

## 4. 支持 InputRotationWrapper 的方案分析

### 方案 A：修改 `wrapper_block()` 跳过 wrapper 内部模块

**思路**：在 `wrapper_block()` 遍历时，如果发现某个 `nn.Linear` 的祖先是 `InputRotationWrapperHadamard`，则跳过。

```python
# wrapper.py 修改
from auto_round.algorithms.transforms.spinquant.rotation_utils import InputRotationWrapperHadamard

def wrapper_block(block, ...):
    # 收集所有 rotation wrapper 的路径前缀
    wrapper_prefixes = set()
    for n, m in block.named_modules():
        if isinstance(m, InputRotationWrapperHadamard):
            wrapper_prefixes.add(n + ".")

    for n, m in block.named_modules():
        # 跳过 wrapper 内部的子模块
        if any(n.startswith(p) for p in wrapper_prefixes):
            continue
        if type(m) in SUPPORTED_LAYER_TYPES:
            ...  # 正常包装
```

| 优点 | 缺点 |
|------|------|
| 改动小（~10行） | 需要 import spinquant 模块，引入耦合 |
| 不改变量化逻辑 | wrapper 内部的权重不会被量化 ❌ |
| - | export/save 也需要同步修改 |

**致命问题**：wrapper 内部的 `nn.Linear` 被跳过 = 不量化。target modules 的权重就不会被量化，失去了量化的意义。

### 方案 B：将 InputRotationWrapperHadamard 加入 SUPPORTED_LAYER_TYPES

**思路**：让 auto-round 直接量化 wrapper 本身。

```python
# utils/common.py
from auto_round.algorithms.transforms.spinquant.rotation_utils import InputRotationWrapperHadamard
SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D, InputRotationWrapperHadamard)
```

同时需要：
1. wrapper 拥有 `weight`、`bias`、`in_features`、`out_features` 属性 ✅（已实现）
2. `WrapperLinear.__init__` 能正确处理 wrapper 类型
3. `WrapperLinear.forward` 的 `orig_forward` 选择正确的 forward 路径

```python
# wrapper.py:127 — 当前的类型判断
self.orig_forward = self.linear_forward if type(self.orig_layer) == torch.nn.Linear else self.conv1d_forward
# 需要改为：
self.orig_forward = self.linear_forward if type(self.orig_layer) != transformers.pytorch_utils.Conv1D else self.conv1d_forward
```

**但最关键的问题**：`WrapperLinear.linear_forward` 是 `F.linear(x, weight, bias)`，直接绕过了 wrapper 的旋转逻辑：

```python
# 量化训练时的 forward 路径：
WrapperLinear.forward(x):
    weight_q = self._qdq_weight(...)      # 量化权重
    output = self.orig_forward(x, weight_q, bias)  # → F.linear(x, weight_q, bias)
    #                          ↑ 注意：x 没有经过旋转！
```

旋转发生在 `InputRotationWrapperHadamard.forward()` 里，但 `WrapperLinear` 完全绕过了它。

**修复思路**：改 `WrapperLinear.forward` 让它调用 `self.orig_layer.forward(x)` 而非 `F.linear`... 但这又会破坏现有的量化逻辑（`_qdq_weight` 产生的 `weight_q` 没有被使用）。

| 优点 | 缺点 |
|------|------|
| wrapper 会被量化 | 量化训练时旋转逻辑被绕过 ❌ |
| 不需要隐藏子模块 | 需要大量修改 WrapperLinear |
| - | 耦合：SUPPORTED_LAYER_TYPES 依赖 spinquant |
| - | export 路径全部需要适配 |

### 方案 C：Wrapper 自持 weight + 注册为 SUPPORTED_LAYER_TYPES + 修改 WrapperLinear

**思路**：结合方案 B，但让 wrapper 只持有 weight/bias（不存 original_module），然后在 `WrapperLinear` 中特殊处理旋转。

```python
class InputRotationWrapperHadamard(nn.Module):
    def __init__(self, original_module, rotation_size, ...):
        super().__init__()
        self.weight = original_module.weight    # 自持 weight
        self.bias = original_module.bias
        self._in_features = original_module.in_features
        self._out_features = original_module.out_features
        # 旋转参数...

    def forward(self, x):
        x = self._rotate(x)
        return F.linear(x, self.weight, self.bias)
```

然后在 `WrapperLinear` 中：

```python
def __init__(self, orig_layer, ...):
    # 判断是否有旋转
    self._has_rotation = isinstance(orig_layer, InputRotationWrapperHadamard)
    if self._has_rotation:
        self._rotate_fn = orig_layer._rotate  # 保存旋转函数
    self.orig_forward = self.rotated_linear_forward if self._has_rotation else self.linear_forward

def rotated_linear_forward(self, x, weight, bias):
    x = self._rotate_fn(x)   # 先旋转
    return F.linear(x, weight, bias)  # 再做量化后的线性计算
```

| 优点 | 缺点 |
|------|------|
| 旋转逻辑在量化训练中保留 ✅ | 需要修改 WrapperLinear（中等侵入性） |
| weight 正常被量化 ✅ | 需要修改 WrapperWALayer 保留旋转 |
| named_modules 不会发现内部 Linear ✅ | export 路径需要适配 |
| state_dict 键名兼容（无前缀） ✅ | SUPPORTED_LAYER_TYPES 需要扩展 |

### 方案 D：使用 forward_pre_hook（当前方案）

**思路**：不用 wrapper，直接在 target module 的 `nn.Linear` 上注册 `forward_pre_hook`。

```python
def _apply_online_r1(self):
    for module in target_modules:
        module.weight.data = matmul_hadU(module.weight.data, ...)
        hook = self._make_online_r1_hook(r1_size, ...)
        module.register_forward_pre_hook(hook)
```

**为什么 hook 能工作**：
- `nn.Linear` 不变，`wrapper_block()` 正常发现、包装
- `WrapperLinear.forward()` 显式运行 `orig_layer._forward_pre_hooks`（line 505-508）
- `WrapperWALayer.__init__()` 偷取 hooks（line 557），在 `forward()` 中先运行（line 572-575）
- export 路径不受影响（`pack_layer` 只看 `nn.Linear` / `WrapperWALayer`）

| 优点 | 缺点 |
|------|------|
| 零侵入 auto-round 代码 ✅ | hook 不随 `save_pretrained` 保存 ❌ |
| 完美兼容量化 pipeline ✅ | 加载模型后需要重新注册 hook |
| 完美兼容 export ✅ | 需要额外的旋转配置保存/加载机制 |
| WrapperLinear 已内置 hook 传播 ✅ | |
| WrapperWALayer 已内置 hook 偷取 ✅ | |

---

## 5. 方案对比总结

| 维度 | A: 跳过内部 | B: 加入类型 | C: 自持weight+改WrapperLinear | D: Hook |
|------|:-----------:|:-----------:|:-----------------------------:|:-------:|
| 量化正确性 | ❌ 不量化 | ❌ 旋转绕过 | ✅ | ✅ |
| auto-round 改动量 | 小 | 大 | 中 | **零** |
| export 兼容 | 需改 | 需改 | 需改 | ✅ 原生 |
| 模型保存/加载 | ✅ | ✅ | ✅ | ❌ 需额外机制 |
| 推荐度 | ❌ | ❌ | ⚠️ 可行但复杂 | ✅ 当前最佳 |

---

## 6. 推荐策略：Hook + 旋转配置持久化

### 6.1 短期方案（当前已实现）

使用 **方案 D（Hook）** 作为主路径：
- 量化训练：hook 被 WrapperLinear 和 WrapperWALayer 自动传播
- 推理：WrapperWALayer 偷取 hook 并在 forward 中执行
- 零侵入 auto-round 核心代码

### 6.2 模型保存 / 加载的解决方案

Hook 不随模型保存，但可以通过以下机制恢复：

```python
# 保存时：在 config.json 或单独文件中记录旋转配置
{
    "rotation_config": {
        "type": "spinquant",
        "r1": true,
        "r2": true,
        "online_r1": true,
        "rotation_size": 1024,
        "r3": false,
        "r4": false
    }
}

# 加载时：根据配置重新注册 hook
from auto_round.algorithms.transforms.spinquant import register_rotation_hooks
model = AutoModelForCausalLM.from_pretrained("path/to/quantized_model")
register_rotation_hooks(model, rotation_config)
```

这与 Quark 的 `prepare_model_for_reloading_fake()` 模式一致。

### 6.3 长期方案（如果 auto-round 架构演进）

如果 auto-round 的 `compressors_new` 重构引入了更灵活的模块发现机制（比如可插拔的 `layer_filter` 或 `module_visitor`），可以重新考虑 **方案 C**：

```python
# 假设未来的 compressors_new 支持自定义模块发现
class BaseCompressor:
    def get_quantizable_modules(self, block):
        """可被子类覆盖的模块发现方法"""
        for n, m in block.named_modules():
            if type(m) in self.supported_types:
                yield n, m

# 在 spinquant_mixin.py 中覆盖
class SpinQuantMixin:
    def get_quantizable_modules(self, block):
        for n, m in block.named_modules():
            if isinstance(m, InputRotationWrapperHadamard):
                yield n, m  # 量化 wrapper 本身
            elif type(m) in self.supported_types:
                # 跳过 wrapper 内部的模块
                if not self._is_inside_rotation_wrapper(n, block):
                    yield n, m
```

但这需要 `WrapperLinear` 也支持自定义 forward 路径（旋转 + F.linear），改动面较大。

### 6.4 对 compressors_new 的建议

基于分析，建议在 `compressors_new` 重构中考虑以下改进：

1. **模块发现可配置化**：将 `wrapper_block()` 中的模块发现逻辑抽象为可覆盖的方法，而非硬编码 `type(m) in SUPPORTED_LAYER_TYPES`。

2. **WrapperLinear 支持自定义 forward**：当前 `orig_forward` 只有 `linear_forward` 和 `conv1d_forward` 两种选择。可以增加一个通用的 `module_forward` 路径，调用 `self.orig_layer.forward(x_with_quantized_weight)` —— 但需要仔细处理量化权重替换的时机。

3. **Hook 注册表**：在 `BaseCompressor` 中维护一个 hook 注册表，记录所有在 Phase 4.5 注册的 hooks。保存时将 hook 配置序列化，加载时自动恢复。

4. **Transform 感知的包装**：让 `wrapper_block()` 感知 Phase 4.5 的 rotation transform，在包装时保留 hook 的语义。当前 `WrapperLinear` 已经做到了（显式传播 `_forward_pre_hooks`），但更显式的 API 设计会更清晰。

---

## 7. 结论

**当前最优解是 Hook 方案（方案 D）**，原因是：

1. **零侵入**：不修改 auto-round 的任何核心代码
2. **已内置支持**：`WrapperLinear` 和 `WrapperWALayer` 已经显式处理了 `_forward_pre_hooks` 的传播和偷取
3. **量化正确**：hook 在量化训练和推理时都被正确执行
4. **export 兼容**：不影响 MXFP4/NVFP8 的打包和保存

唯一缺点是 hook 不随模型保存，需要额外的配置持久化机制。这可以通过在 `config.json` 中添加 `rotation_config` 字段来解决。

如果未来 `compressors_new` 的重构引入了可扩展的模块发现和包装机制，可以重新评估 wrapper 方案。但在当前架构下，hook 是唯一不需要修改 auto-round 核心代码就能正确工作的方案。
