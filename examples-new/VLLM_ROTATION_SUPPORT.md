# vLLM Rotation (QuaRot/SpinQuant) 支持分析

## 概述

本文档分析 vLLM 对 llm-compressor 导出的 QuaRot/SpinQuant 旋转模型的推理支持情况。

重点回答：**哪些 R1/R2/R3/R4 组合可以被 vLLM 正确加载和推理？**

## 结论速查表

| 组合 | llm-compressor 导出 | vLLM 推理 | 说明 |
|------|---------------------|-----------|------|
| R1 | ✅ offline (weight fused) | ✅ | 权重已融合，对 vLLM 透明 |
| R1+R2 | ✅ offline (weight fused) | ✅ | 同上 |
| R1+R2+R4 | ✅ offline+online | ✅ | R4 online 部分通过 LinearTransformMethod 处理 |
| R1+R2+R3 | ✅ 代码已实现 (online) | ❌ | vLLM 缺少 attention transform 支持 |
| R1+R2+R3+R4 | ✅ 代码已实现 (online) | ❌ | 同上 |

## 背景：R1-R4 旋转是什么

```
R1: 全局 Hadamard 旋转，作用于隐藏状态维度 (hidden_size)
    - 影响: embedding, all linear layers, lm_head
    - 类型: offline（融合到权重中）

R2: 注意力头内旋转，作用于 head_dim 维度
    - 影响: v_proj (output), o_proj (input)
    - 类型: offline（融合到权重中）

R3: RoPE 后 Q/K 旋转，作用于 head_dim 维度
    - 影响: attention 模块的 Q 和 K（RoPE 之后）
    - 类型: online（运行时 hook）

R4: MLP 下投影输入旋转，作用于 intermediate_size 维度
    - 影响: down_proj / gate_down_proj
    - 类型: 混合（online activation + offline weight fusion）
```

## 详细分析

### 1. vLLM 如何加载 compressed_tensors 模型

vLLM 通过 `CompressedTensorsConfig` 加载模型：

```python
# compressed_tensors.py, line 105-108
transform_config = ModelCompressor.parse_transform_config(config)
if transform_config is not None:
    self.transform_config = transform_config
```

对每一层调用 `get_quant_method()` 分配处理方法：

```python
# compressed_tensors.py, line 156-198
def get_quant_method(self, layer, prefix):
    if isinstance(layer, LinearBase):
        # 处理线性层的量化 + transform
        input_tfms, output_tfms = get_linear_transform_schemes(...)
        if any((input_tfms, output_tfms)):
            return CompressedTensorsLinearTransformMethod(...)
        ...
    if isinstance(layer, Attention):
        return CompressedTensorsKVCacheMethod(self)  # 仅 KV cache 量化，不处理 R3
```

### 2. Transform 位置类型与处理方式

`compressed_tensors` 库定义了 6 种 `TransformLocation`：

| Location | Runtime | 作用对象 | vLLM 支持 |
|----------|---------|----------|-----------|
| `weight_input` | offline | 权重 | ✅ 已融合，无需处理 |
| `weight_output` | offline | 权重 | ✅ 已融合，无需处理 |
| `input` | online | 激活值 | ✅ `CompressedTensorsLinearTransformMethod` |
| `output` | online | 激活值 | ✅ `CompressedTensorsLinearTransformMethod` |
| `q_attn` | online | Q 注意力值 | ❌ 未实现 |
| `k_cache` | online | K cache 值 | ❌ 未实现 |

`get_linear_transform_schemes()` 的关键过滤逻辑：

```python
# linear.py, line 207-222
for scheme_name, scheme, args in get_schemes_args(transform_config):
    if is_match(part_name, layer, args.targets, args.ignore) and args.is_online():
        if args.location == TransformLocation.INPUT:
            input_tfms[part_index] = ...
        elif args.location == TransformLocation.OUTPUT:
            output_tfms[part_index] = ...
        else:
            raise ValueError(f"Cannot apply `{args.location}` to `{layer_name}`")
```

> 如果 R3 的 `q_attn`/`k_cache` 进入此逻辑，会直接 **raise ValueError**。

### 3. 各 Rotation 的具体处理路径

#### R1 — ✅ 完全支持

llm-compressor 定义：
```python
# R1: 全部 offline
TransformArgs(targets=[embedding, attn_o, mlp_out], location="weight_output")
TransformArgs(targets=[attn_q, attn_k, attn_v, mlp_in, lm_head], location="weight_input", inverse=True)
```

- 所有位置都是 `weight_input` / `weight_output` → `is_online()` 返回 `False`
- llm-compressor 在导出时将旋转矩阵融合到权重中
- vLLM 加载时视为普通权重，**无需任何 transform 支持**

#### R2 — ✅ 完全支持

llm-compressor 定义：
```python
# R2: 全部 offline
TransformArgs(targets=[attn_v], location="weight_output")
TransformArgs(targets=[attn_o], location="weight_input", inverse=True)
```

- 与 R1 相同，全部 offline，融合到权重中

#### R4 — ✅ 完全支持

llm-compressor 定义：
```python
# R4: 混合 (online + offline)
TransformArgs(targets=[mlp_out], location="input")         # online: 激活值旋转
TransformArgs(targets=[mlp_out], location="weight_input", inverse=True)  # offline: 权重融合
```

- `weight_input` 部分：offline，已融合到 down_proj 权重
- `input` 部分：online，通过 `CompressedTensorsLinearTransformMethod` 处理
  - `HadamardTransform` 在 down_proj 前端对激活值执行 Hadamard 变换
  - 使用 `hadacore` CUDA 算子（如可用）或 dense matrix multiply

R4 的 vLLM 执行流程：
```
activation → HadamardTransform.forward(x) → R4·x → down_proj(R4·x)
                                                    (weight 已融合 R4⁻¹)
```

#### R3 — ❌ 不支持

llm-compressor 定义：
```python
# R3: 全部 online
TransformArgs(targets=[attn], location="q_attn")   # Q 旋转（RoPE 之后）
TransformArgs(targets=[attn], location="k_cache")   # K 旋转（RoPE 之后）
```

不支持的原因有**两层**：

**原因 1：llm-compressor R3 状态**

llm-compressor 的 `_create_r3_scheme()` 已完整实现，代码上支持 R3 导出
（使用 `q_attn` 和 `k_cache` 位置）。示例文件中的 "R3 will be added in future
release" 注释可能已过时。但 R3 是 online transform，需要推理端支持。

**原因 2：vLLM 缺少注意力级 transform 支持**
```python
# vllm/transform/module.py, line 29
# "and attention transforms method (not implemented yet)"
```

- `get_quant_method()` 只对 `LinearBase` 层分配 transform 方法
- `Attention` 层只分配 `CompressedTensorsKVCacheMethod`（仅 KV cache 量化）
- 没有 `CompressedTensorsAttentionTransformMethod` 或类似实现
- `q_attn`/`k_cache` 位置若进入 `get_linear_transform_schemes()` 会触发 ValueError

### 4. 限制条件

#### Tensor Parallelism 不支持 online transform

```python
# module.py, line 49-51
if get_tensor_model_parallel_world_size() > 1:
    raise NotImplementedError(
        "Online transforms with tensor parallelism is not supported"
    )
```

这意味着：
- R4 的 online 部分在 TP>1 时不工作
- 多 GPU 推理 R1+R2+R4 模型时，不能使用 tensor parallelism
- R1+R2 (纯 offline) 不受此限制

## 三个框架对比

| 特性 | auto-round | llm-compressor | vLLM |
|------|------------|----------------|------|
| R1 | ✅ online + offline | ✅ offline only | ✅ (透明) |
| R2 | ✅ offline | ✅ offline | ✅ (透明) |
| R3 | ✅ online (hook) | ❌ 未实现 | ❌ 未实现 |
| R4 | ✅ online (hook) + offline (weight) | ✅ online + offline | ✅ Linear transform |
| 导出格式 | AutoRound format | compressed_tensors | N/A (推理端) |
| TP 支持 | N/A | N/A | ❌ (online transform) |

## 实际可用的推理链路

```
┌──────────────────────────────────────────────────────────────┐
│ llm-compressor 导出                                          │
│   SpinQuantModifier(rotations=["R1", "R2", "R4"])            │
│   → 保存 compressed_tensors 格式模型                          │
│   → config.json 包含 transform_config                        │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│ vLLM 加载推理                                                │
│   1. 加载权重（R1/R2/R4 offline 部分已融合）                   │
│   2. 解析 transform_config                                   │
│   3. R4 online: HadamardTransform 挂载到 down_proj 输入       │
│   4. 正常推理                                                │
│                                                              │
│   ⚠️ 限制: TP=1 only (online transform)                      │
└──────────────────────────────────────────────────────────────┘
```

## 关于 auto-round 导出模型的 vLLM 推理

auto-round 目前使用自己的格式（AutoRound format），不是 compressed_tensors 格式。
如果要让 vLLM 推理 auto-round 导出的旋转模型，需要：

1. 将旋转完全融合到权重中（offline 模式），然后以标准 HuggingFace 格式保存
2. 或实现 compressed_tensors 兼容的导出格式

对于 R1+R2 offline 的场景，auto-round 可以直接保存标准格式模型，
vLLM 可以正常加载（因为旋转已融合，模型与普通模型无异）。

## 关键代码引用

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `vllm/.../transform/module.py` | L29 | "attention transforms not implemented yet" |
| `vllm/.../transform/module.py` | L49-51 | TP not supported with online transforms |
| `vllm/.../transform/linear.py` | L149-169 | LinearTransformMethod.apply() |
| `vllm/.../transform/linear.py` | L193-224 | get_linear_transform_schemes() |
| `vllm/.../compressed_tensors.py` | L156-198 | get_quant_method() - only LinearBase |
| `llm-compressor/.../spinquant/base.py` | L257-274 | R3 scheme (q_attn + k_cache) |
| `llm-compressor/.../spinquant_example.py` | L1-2 | "R3 will be added in future release" |
