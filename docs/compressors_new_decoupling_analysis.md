# compressors_new 能否解耦量化流程以支持 InputRotationWrapper

## 1. 当前耦合点分析

auto-round 的量化 pipeline 中，模块发现 → 包装 → 量化 → 解包装 → 导出，形成了一条紧密耦合链。wrapper 方案之所以失败，是因为这条链上的**每个环节**都硬编码了对 `nn.Linear` 的假设。

### 1.1 耦合链路全景

```
                    ┌─────────────────────────────────────────────────┐
                    │  compressors_new/base.py                        │
                    │  Phase 4.5: _apply_rotations()                  │
                    │  → 在 nn.Linear 上注册 hook / 安装 wrapper      │
                    └──────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────────┐
                    │  quantizer (sign_round/quantizer.py)            │
                    │  quantize_block():                              │
                    │    self.wrapper_block(block, ...)  ←────────────│──── 硬编码①
                    │    → wrapper.py:wrapper_block()                 │
                    └──────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────────┐
                    │  wrapper.py:wrapper_block()                     │
                    │  for n, m in block.named_modules():             │
                    │      if type(m) in SUPPORTED_LAYER_TYPES: ←────│──── 硬编码②
                    │          WrapperLinear(m, ...)                  │
                    └──────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────────┐
                    │  wrapper.py:WrapperLinear.__init__()            │
                    │  type(self.orig_layer) == torch.nn.Linear ←────│──── 硬编码③
                    │  → 选择 linear_forward 或 conv1d_forward       │
                    └──────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────────┐
                    │  wrapper.py:WrapperLinear.forward()             │
                    │  weight_q = self._qdq_weight(...)               │
                    │  output = self.orig_forward(x, weight_q, bias)  │
                    │  → F.linear(x, weight_q, bias)  ←──────────────│──── 硬编码④
                    │    绕过了任何 wrapper 的自定义 forward           │
                    └──────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────────┐
                    │  wrapper.py:WrapperLinear.unwrapper()           │
                    │  → WrapperWALayer(self.orig_layer, ...)         │
                    │  WrapperWALayer.forward():                      │
                    │    self.orig_layer.forward(x) ←────────────────│──── 硬编码⑤
                    └──────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────────┐
                    │  export_to_nvfp_mx.py:pack_layer()             │
                    │  type(layer) not in SUPPORTED_LAYER_TYPES ←────│──── 硬编码⑥
                    │  isinstance(layer, WrapperWALayer)             │
                    │  layer = layer.orig_layer  → nn.Linear         │
                    └────────────────────────────────────────────────┘
```

**6 个硬编码点**，分布在 4 个不同文件中。这就是为什么加一个 wrapper 需要改动面如此之大。

### 1.2 compressors_new 是否解耦了？

```
compressors_new 架构：

entry.py ──→ BaseCompressor ──→ CalibCompressor ──→ Quantizer
                │                      │
          spinquant_mixin.py      wrapper_block() ← 仍然硬编码
                │                      │
        _apply_rotations()        WrapperLinear ← 仍然硬编码
```

**答案：没有。** `compressors_new` 的重构集中在：
- 将 monolithic 的 `LLMCompressor` 拆分为 `BaseCompressor` + mixin（MLLM、Diffusion、SpinQuant）
- 入口路由（`entry.py` 根据 config 选择 Compressor 类型）
- Phase 化的 `post_init()`（7 个阶段有清晰的前置/后置条件）
- 配置系统重构（`ExtraConfig` 层级化）

但**核心量化路径没有变**：`wrapper_block()` → `WrapperLinear` → `F.linear` 这条链路完全不变，`SUPPORTED_LAYER_TYPES` 仍然是全局硬编码。

---

## 2. 需要解耦的核心抽象

要让 auto-round 原生支持 InputRotationWrapper（或任何自定义 layer wrapper），需要在以下 3 层引入抽象：

### 2.1 层级一：模块发现（Module Discovery）

**当前**：`type(m) in SUPPORTED_LAYER_TYPES`（硬编码全局常量）

**需要**：可扩展的模块发现策略

```python
# 方案：在 Quantizer 上定义可覆盖的方法
class BaseQuantizer:
    def is_quantizable(self, name: str, module: nn.Module) -> bool:
        """判断一个模块是否应该被量化。子类可覆盖。"""
        return type(module) in self.supported_types

    def get_quantizable_modules(self, block: nn.Module) -> list[tuple[str, nn.Module]]:
        """从 block 中发现所有可量化模块。子类可覆盖。"""
        return [(n, m) for n, m in block.named_modules()
                if self.is_quantizable(n, m)]
```

SpinQuant quantizer 可以覆盖：

```python
class SpinQuantQuantizer(BaseQuantizer):
    def is_quantizable(self, name, module):
        # 跳过 rotation wrapper 内部的子模块
        if ".original_module" in name:
            return False
        # rotation wrapper 本身可量化
        if isinstance(module, InputRotationWrapperHadamard):
            return True
        return super().is_quantizable(name, module)
```

### 2.2 层级二：Forward 策略（Forward Strategy）

**当前**：`WrapperLinear.forward()` 硬编码为 `F.linear(x, weight_q, bias)`

**需要**：可组合的 forward 路径，让 pre-transform（如旋转）能插入到量化 forward 中

```python
class WrapperLinear:
    def __init__(self, orig_layer, ...):
        # 选择 forward 策略
        self.orig_forward = self._resolve_forward_strategy(orig_layer)

    def _resolve_forward_strategy(self, orig_layer):
        """可被子类或策略模式覆盖"""
        if type(orig_layer) == torch.nn.Linear:
            return self.linear_forward
        elif type(orig_layer) == Conv1D:
            return self.conv1d_forward
        elif hasattr(orig_layer, 'pre_transform'):
            # 支持带 pre_transform 的自定义 layer
            return self._make_transformed_forward(orig_layer.pre_transform)
        else:
            return self.linear_forward

    def _make_transformed_forward(self, pre_transform):
        def forward(x, weight, bias):
            x = pre_transform(x)
            return F.linear(x, weight, bias)
        return forward
```

### 2.3 层级三：导出适配（Export Adapter）

**当前**：`pack_layer()` 硬编码检查 `type(layer) in SUPPORTED_LAYER_TYPES`

**需要**：可扩展的导出适配器

```python
class ExportAdapter:
    def can_pack(self, name: str, layer: nn.Module) -> bool:
        return type(layer) in SUPPORTED_LAYER_TYPES or isinstance(layer, WrapperWALayer)

    def unwrap_for_export(self, layer: nn.Module) -> nn.Linear:
        """将任何包装层拆解为 nn.Linear 用于导出"""
        if isinstance(layer, WrapperWALayer):
            return layer.orig_layer
        if isinstance(layer, InputRotationWrapperHadamard):
            # 创建一个纯 nn.Linear 用于导出
            linear = nn.Linear(layer.in_features, layer.out_features,
                             bias=layer.bias is not None)
            linear.weight = layer.weight
            linear.bias = layer.bias
            return linear
        return layer
```

---

## 3. 最小侵入性的重构方案

如果不想大改 auto-round，可以只在关键点引入 **策略注入**：

### 3.1 在 BaseCompressor 上暴露 wrapper_block 自定义点

**当前** (`base.py:729`)：
```python
self.wrapper_block = wrapper_block  # 直接绑定全局函数
```

**改为**：
```python
self.wrapper_block = self._get_wrapper_block_fn()

def _get_wrapper_block_fn(self):
    """返回包装函数。子类/mixin 可覆盖以支持自定义 layer 类型。"""
    return wrapper_block
```

SpinQuantMixin 覆盖：
```python
class SpinQuantMixin:
    def _get_wrapper_block_fn(self):
        base_fn = wrapper_block
        def rotation_aware_wrapper_block(block, *args, **kwargs):
            # 在包装前，把 rotation wrapper 内部的 Linear 标记为不可量化
            # 或者把 wrapper 本身标记为可量化
            ...
            return base_fn(block, *args, **kwargs)
        return rotation_aware_wrapper_block
```

### 3.2 在 WrapperLinear 上支持 pre_transform

**改动量最小的方式**：利用已有的 `_forward_pre_hooks` 机制。

当前 `WrapperLinear.forward` 已经支持运行 `orig_layer._forward_pre_hooks`。这就是 hook 方案能工作的原因。所以最小侵入方案就是 **保持 hook**，不需要改 WrapperLinear。

### 3.3 在导出路径上支持 rotation metadata

在 `save_quantized_as_fp` 中，保存旋转配置：

```python
def save_quantized_as_fp(output_dir, model, ...):
    # ... 现有保存逻辑 ...

    # 保存旋转配置（如果有）
    rotation_config = getattr(model, '_rotation_config', None)
    if rotation_config:
        import json
        with open(os.path.join(output_dir, 'rotation_config.json'), 'w') as f:
            json.dump(rotation_config, f, indent=2)
```

---

## 4. 方案对比：大重构 vs 最小改动

| 维度 | 大重构（抽象化 3 层） | 最小改动（hook + 配置持久化） |
|------|---------------------|---------------------------|
| 改动文件数 | 5-8 个 | 1-2 个 |
| 改动行数 | 200-400 行 | 30-50 行 |
| 破坏性风险 | 中（改了核心接口） | 极低（只加代码，不改接口） |
| 对其他功能影响 | 需要全面回归测试 | 基本不影响 |
| 可扩展性 | 高（未来任何 transform 都能接入） | 低（只解决 rotation 场景） |
| 维护成本 | 高（多了抽象层） | 低 |
| 对上游贡献的接受度 | 低（改动面太大） | 高 |

---

## 5. 建议

### 短期（当前迭代）

**使用 hook 方案，零侵入 auto-round**：
- 量化阶段：hook 被 WrapperLinear/WrapperWALayer 自动传播 ✅
- 推理阶段：WrapperWALayer 偷取 hook 并正确执行 ✅
- 保存/加载：在 `config.json` 中添加 `rotation_config` 字段，加载时重新注册 hook

### 中期（贡献到 auto-round）

**在 compressors_new 中引入最小抽象**：
1. `BaseCompressor._get_wrapper_block_fn()` — 可覆盖的包装策略
2. `WrapperLinear._resolve_forward_strategy()` — 可扩展的 forward 路径
3. `BaseQuantizer.is_quantizable()` — 可覆盖的模块发现

这 3 个改动加起来约 50 行代码，不破坏现有接口，但为 rotation wrapper 和其他自定义 layer 类型打开了扩展点。

### 长期（架构演进）

如果 auto-round 需要支持更多 transform（不只是 rotation，还有 SmoothQuant、GPTQ 等的融合），可以引入 **Transform Pipeline** 抽象：

```python
class TransformPipeline:
    """可组合的模型变换管道"""
    transforms: list[Transform]  # 按顺序应用

class Transform(ABC):
    @abstractmethod
    def apply(self, model) -> model:
        """修改模型权重/结构"""
    @abstractmethod
    def get_forward_hooks(self) -> dict[str, Callable]:
        """返回需要注册的运行时 hooks"""
    @abstractmethod
    def serialize_config(self) -> dict:
        """序列化配置用于保存/加载"""
```

但这是更大的架构变更，建议在 auto-round 社区充分讨论后再推进。
