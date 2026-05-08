# Quark vs Auto-round: 为什么 Quark 能用 InputRotationWrapper

## 1. 问题描述

在将 SpinQuant/QuaRot 的旋转实现从 Quark 迁移到 Auto-round 的过程中，我们发现：

- **Quark** 可以正常使用 `InputRotationWrapperHadamard`（一个 `nn.Module` wrapper 包裹 `nn.Linear`）进行 rotation + quantization
- **Auto-round** 使用同样的 wrapper 后量化管线会崩溃

本文详细分析两个框架的架构差异，解释这个不兼容的根本原因。

---

## 2. 核心结论

| 维度 | Quark | Auto-round |
|------|-------|-----------|
| **模块发现机制** | torch.fx 图追踪（操作级） | `named_modules()` + `type()` 检查（类型级） |
| **看到什么** | `ops.aten.linear.default` 算子节点 | `nn.Module` 实例的 Python 类型 |
| **Wrapper 透明性** | ✅ 追踪自动穿透 wrapper 的 `forward()` | ❌ 只看到 wrapper 类型，不在白名单 |
| **量化与旋转的时序** | 量化先 → 旋转后（wrapper 包裹已量化模块） | 旋转先 → 量化后（量化需要识别 wrapper） |
| **量化目标表示** | FX 图中的算子节点 | 模块树中的 `nn.Module` 实例 |

**本质区别：Quark 操作计算图（what actually runs），Auto-round 操作模块树（what structurally exists）。**

---

## 3. Quark 的架构设计

### 3.1 torch.fx Graph Tracing

Quark 使用 `torch.export.export_for_training()` 将模型转换为 `torch.fx.GraphModule`：

```python
# quark/torch/quantization/graph/processor/model_importer.py (lines 78-81)
graph_module = torch.export.export_for_training(
    mod=model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
).module()  # 转换为 torch.fx.GraphModule
```

**关键特性：** `torch.export` 会**执行**模型的 `forward()` 方法进行追踪（tracing），将所有实际执行的运算记录为图中的节点。它追踪的是"实际发生了什么计算"，而不是"模块树长什么样"。

当模型包含 `InputRotationWrapperHadamard` 时，追踪过程如下：

```
追踪 forward() 执行路径：
  InputRotationWrapperHadamard.forward(x)
    → self.transform(x)              → 记录为 matmul / Hadamard 节点
    → self.original_module(x)         → 记录为 ops.aten.linear.default 节点
```

追踪结果：FX 图中直接出现 `ops.aten.linear.default` 节点，**无论外面套了多少层 wrapper**。

### 3.2 基于图节点的量化替换

Quark 在 FX 图上操作，查找 `ops.aten.linear.default` 节点并替换为 `QuantLinear`：

```python
# quark/torch/quantization/graph/optimization/pre_quant/replace_linear_to_qtlinear.py
def replace_linear_qtlinear(m: GraphModule) -> GraphModule:
    for n in m.graph.nodes:
        if not is_linear_node(n):  # 检查 node.target == ops.aten.linear.default
            continue
        
        # 提取 weight 和 bias
        linear_weight = _get_tensor_constant_from_node(weight_node, m)
        linear_bias = _get_tensor_constant_from_node(bias_node, m)
        
        # 创建 QuantLinear
        quantized_linear = QuantLinear(in_features, out_features, device, bias, empty_config)
        quantized_linear.weight.data = linear_weight.data.clone()
        
        # 在图中替换节点
        quant_linear_node = m.graph.create_node(
            "call_module", quant_linear_name, (input_activation_node,), {}
        )
        linear_node.replace_all_uses_with(quant_linear_node)
```

**核心要点：** 量化代码操作的是图节点（算子），不是模块实例。不需要关心 `nn.Module` 的类型，也不需要 `SUPPORTED_LAYER_TYPES` 白名单。

### 3.3 量化先于旋转的执行时序

Quark 的执行顺序保证了 wrapper 不会干扰量化：

```
Step 1: 原始模型
  model.layers[i].self_attn.q_proj = nn.Linear(1024, 1024)

Step 2: quantize_model() — 量化替换
  model.layers[i].self_attn.q_proj = QuantLinear(1024, 1024)
  # 此时模型中全是 QuantLinear，没有 wrapper

Step 3: apply rotation — 旋转包装
  model.layers[i].self_attn.q_proj = InputRotationWrapperHadamard(
      original_module=QuantLinear(1024, 1024),  # 包裹的是已量化的模块
      transform=HadamardTransform
  )
```

由于量化发生在旋转之前，量化代码永远不会遇到 `InputRotationWrapperHadamard` 类型。wrapper 只在量化完成后才被添加。

### 3.4 InputRotationWrapper 的透明代理设计

```python
# quark/torch/algorithm/rotation/rotation_utils.py (lines 243-354)
class InputRotationWrapper(nn.Module):
    """
    透明代理 wrapper：对外部代码表现得像是被包裹的模块本身。
    """
    
    def __getattr__(self, name: str) -> Any:
        # 代理属性访问到内部模块
        if name in {"weight", "bias", "in_features", "out_features"}:
            return getattr(self.original_module, name)
        else:
            return super().__getattr__(name)
    
    def __setattr__(self, key: str, value: Any) -> None:
        # 代理属性设置到内部模块
        if key in {"weight", "bias", "in_features", "out_features"}:
            setattr(self.original_module, key, value)
        else:
            super().__setattr__(key, value)
    
    def state_dict(self, *args, prefix: str = "", **kwargs):
        # 展平 state_dict，去掉 ".original_module." 前缀
        destination_local = super().state_dict(*args, prefix=prefix, **kwargs)
        for param_name in list(destination_local):
            if ".original_module." in param_name:
                new_name = param_name.replace(".original_module.", ".")
                destination_local[new_name] = destination_local.pop(param_name)
        return destination_local
    
    def forward(self, x: torch.Tensor) -> Any:
        x = self.transform(x)           # 对 input 做 Hadamard 旋转
        x = self.original_module(x)      # 调用 QuantLinear.forward()（含量化）
        return x
```

**三个透明性保证：**
1. **属性代理**：`wrapper.weight` → `wrapper.original_module.weight`，外部代码无感知
2. **State dict 展平**：保存时去掉 `.original_module.` 前缀，checkpoint 格式与原模型一致
3. **Forward 委托**：先旋转 input，再调用内部模块的 forward（含量化逻辑）

---

## 4. Auto-round 的架构设计

### 4.1 基于模块树的类型检查发现

```python
# auto_round/wrapper.py (lines 768-782)
def wrapper_block(block, ...):
    for n, m in block.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:    # ← 硬编码类型检查
            new_m = WrapperLinear(m, ...)
            set_module(block, n, new_m)
```

```python
# auto_round/utils/common.py (line 607)
SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)
```

**问题：** `type(InputRotationWrapperHadamard)` 不在 `SUPPORTED_LAYER_TYPES` 中，所以 wrapper 不会被量化处理。

### 4.2 六个硬编码耦合点

Auto-round 的量化管线中有 6 个地方假设目标模块是 `nn.Linear`：

| # | 位置 | 代码 | 问题 |
|---|------|------|------|
| 1 | `wrapper_block()` | `type(m) in SUPPORTED_LAYER_TYPES` | Wrapper 类型不匹配，被跳过 |
| 2 | `named_modules()` 递归 | 遍历进 wrapper 内部 | 找到内部的 `nn.Linear`，但路径变成 `xxx.original_module`，破坏模块路径假设 |
| 3 | `WrapperLinear.__init__` | `type(orig_layer) == nn.Linear` | 选择 `linear_forward` or `conv1d_forward`，不认识 wrapper |
| 4 | `WrapperLinear.forward` | `F.linear(x, weight_q, bias)` | 绕过 wrapper 的旋转逻辑，直接做线性运算 |
| 5 | `WrapperWALayer.forward` | `self.orig_layer.forward(x)` | MXFP4 量化后 weight shape 变成 `[0]`，调用 forward 崩溃 |
| 6 | `pack_layer()` | `type(layer) not in SUPPORTED_LAYER_TYPES` | 导出检查失败 |

### 4.3 named_modules() 递归问题

当模型包含 wrapper 时，`named_modules()` 会递归进入 wrapper 内部：

```python
# 假设模型结构:
# model.layers[0].self_attn.q_proj = InputRotationWrapperHadamard(
#     original_module=nn.Linear(1024, 1024)
# )

for name, module in model.named_modules():
    print(name, type(module))
    
# 输出:
# "layers.0.self_attn.q_proj"                    → InputRotationWrapperHadamard  ❌ 不在白名单
# "layers.0.self_attn.q_proj.original_module"    → nn.Linear                   ✓ 在白名单
#   但是路径多了 ".original_module"，后续 set_module() 时路径错乱
```

---

## 5. 对比图解

### 5.1 Quark 的数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│ Quark Pipeline                                                      │
│                                                                     │
│  原始模型                                                            │
│  ┌──────────┐                                                       │
│  │ nn.Linear │ ──── torch.export ────→ FX Graph                     │
│  └──────────┘         追踪                ┌───────────────────────┐ │
│                                           │ ops.aten.linear.default│ │
│                                           └───────────┬───────────┘ │
│                                                       │             │
│                                          replace_linear_qtlinear()  │
│                                                       │             │
│                                           ┌───────────▼───────────┐ │
│                                           │     QuantLinear       │ │
│                                           └───────────┬───────────┘ │
│                                                       │             │
│                                          apply rotation wrapper     │
│                                                       │             │
│                              ┌────────────────────────▼──────────┐  │
│                              │ InputRotationWrapperHadamard      │  │
│                              │   ├── transform: Hadamard旋转    │  │
│                              │   └── original_module: QuantLinear│  │
│                              └───────────────────────────────────┘  │
│                                                                     │
│  量化和旋转互不干扰：量化操作图节点，旋转包装已量化模块                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Auto-round 的数据流（失败路径）

```
┌─────────────────────────────────────────────────────────────────────┐
│ Auto-round Pipeline (失败)                                          │
│                                                                     │
│  原始模型 + 旋转（旋转先于量化）                                         │
│  ┌───────────────────────────────────┐                              │
│  │ InputRotationWrapperHadamard      │                              │
│  │   ├── transform: Hadamard旋转    │                              │
│  │   └── original_module: nn.Linear  │                              │
│  └───────────────────┬───────────────┘                              │
│                      │                                              │
│        wrapper_block() 遍历 named_modules()                         │
│                      │                                              │
│           type(m) in SUPPORTED_LAYER_TYPES?                         │
│                      │                                              │
│    ┌─────────────────┼─────────────────┐                            │
│    │                 │                 │                             │
│  Wrapper 本身     原始 Linear          │                             │
│  type = Wrapper   (路径带 .original_module)                          │
│  ❌ 不在白名单     ✓ 在白名单，但路径错                                 │
│                   → WrapperLinear 包裹内部 Linear                    │
│                   → 绕过旋转逻辑                                     │
│                   → MXFP4 压缩后 weight=[0]                         │
│                   → 推理崩溃 💥                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 为什么 Auto-round 不能简单模仿 Quark

### 6.1 不能直接采用 FX 追踪

Auto-round 的整个量化流程（校准、SignRound 优化、逐块量化）都建立在模块树操作之上。切换到 FX 图追踪意味着重写核心量化管线，工程量巨大。

### 6.2 不能简单调换量化/旋转顺序

Quark 的"量化先、旋转后"模式依赖于 FX 图追踪来发现量化目标。即使 Auto-round 先量化再旋转，后续的 unwrap、导出等步骤仍然会因为不认识 wrapper 类型而失败。

### 6.3 不能只添加类型到白名单

将 `InputRotationWrapperHadamard` 加入 `SUPPORTED_LAYER_TYPES` 会引发连锁问题：
- `WrapperLinear.__init__` 不知道如何为 wrapper 选择 forward 策略
- `WrapperLinear.forward` 会用 `F.linear` 绕过旋转
- `pack_layer` 不知道如何导出 wrapper 类型

---

## 7. Auto-round 的解决方案：Hook 机制

Auto-round 采用 `forward_pre_hook` 替代 wrapper，完美解决了兼容性问题：

```python
# 不修改模块树结构，只在 nn.Linear 上注册 hook
def _make_online_r1_hook(hadamard_K, rotation_size):
    def hook(module, args):
        x = args[0]
        x_rotated = matmul_hadU(x, hadamard_K, rotation_size)
        return (x_rotated,) + args[1:]
    return hook

# 注册 hook
for name, module in model.named_modules():
    if should_rotate(name):
        module.register_forward_pre_hook(_make_online_r1_hook(H, size))
```

**为什么 hook 能工作：**

1. **模块树不变**：所有目标模块仍然是 `nn.Linear`，`wrapper_block()` 正常发现和包装
2. **Hook 被保留**：`WrapperLinear.forward()` 显式执行 `self.orig_layer._forward_pre_hooks`（wrapper.py line 503-518）
3. **Hook 被继承**：`WrapperWALayer.__init__()` 从 `orig_layer` 偷取 hooks（line 557），在 `forward()` 中执行（line 572-575）
4. **零侵入**：不需要修改 auto-round 的任何核心代码

---

## 8. 总结

| 方案 | 原理 | 适用框架 | 优势 | 限制 |
|------|------|---------|------|------|
| **FX 图追踪** | 追踪计算图，自动穿透 wrapper | Quark | 架构级支持，无类型限制 | 需要 FX 基础设施 |
| **Module wrapper** | 用 nn.Module 包裹目标模块 | Quark | 自然的 PyTorch 模式，支持 save/load | 依赖框架不做类型检查 |
| **forward_pre_hook** | 在 forward 前注入旋转 | Auto-round | 零侵入，模块树不变 | 需要额外机制持久化 hook 配置 |

**根本教训：** 量化框架的可扩展性取决于模块发现机制。基于 FX 图的发现天然支持任意模块组合，基于类型检查的发现则对模块类型有严格假设。Auto-round 未来如果要原生支持 wrapper 模式，需要在模块发现层引入可覆盖的抽象（如 `is_quantizable()` 方法），或迁移到基于图的量化管线。
