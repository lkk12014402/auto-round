# 面向模型结构的 MXFP 混合精度策略

`mixed_mxfp_policy` 为 Hugging Face 语言模型提供可复用的 MXFP 默认策略，主要基于模块结构关系，而不是依赖 `mlp.experts` 这类特定模型家族的模块名子串。

## 策略类型

- `dense_mxfp4`：选中的 transformer block 内线性层使用 `MXFP4`。
- `moe_conservative`：routed expert 线性层使用 `MXFP4`；其余选中的 transformer block 线性层使用 `MXFP8`；MoE router 保持浮点；attention / linear-attention 投影保持浮点。
- `moe_balanced`：routed expert 线性层使用 `MXFP4`；其余选中的 transformer block 线性层使用 `MXFP8`；MoE router 保持浮点。

如果你保持默认 `scheme="W4A16"`，AutoRound 会自动切换到该策略所需的 MXFP 基础 scheme。若显式传入了不兼容的 scheme，则会直接报错。

## CLI

Dense：

```bash
auto_round /path/to/model \
  --mixed_mxfp_policy dense_mxfp4 \
  --format llm_compressor \
  --output_dir ./dense-mxfp4
```

MoE 保守版：

```bash
auto_round /path/to/model \
  --mixed_mxfp_policy moe_conservative \
  --format llm_compressor \
  --output_dir ./moe-mxfp4-mxfp8-conservative
```

MoE 均衡版：

```bash
auto_round /path/to/model \
  --mixed_mxfp_policy moe_balanced \
  --format llm_compressor \
  --output_dir ./moe-mxfp4-mxfp8-balanced
```

## API

```python
from auto_round import AutoRound

ar = AutoRound(
    "/path/to/model",
    mixed_mxfp_policy="moe_conservative",
)
ar.quantize_and_save("./qmodel", format="llm_compressor")
```

## 优先级

策略默认值会在 `set_layer_config()` 展开最终配置之前先合并进去。

- 显式传入的 `layer_config` 会覆盖策略默认值，包括策略自动生成的 router / attention 浮点跳过规则。
- 显式传入的 `ignore_layers` 同样优先于策略默认值。
- 现有的 lm-head、embedding、shape 对齐跳过、选中 block 过滤、多模态 block 选择逻辑都会继续生效。

## 检测说明

- MoE expert collection 主要通过“重复 expert 子模块 + 关联 router/gate sibling”的结构关系检测，而不是依赖 `mlp.experts`、`self_attn` 这类固定路径。
- router/gate 检测优先使用 expert 关系和张量形状；只有在结构信息不足时才会退回到有限的类名/模块名提示。
- 保守策略中的 attention 保留规则优先依赖 attention 模块结构和常见 attention 元数据，只有在结构元数据缺失时才使用有限的类名/模块名回退。
- dense gated MLP 中的 `gate_proj` 不会被误判为 MoE router，除非它在结构上确实与 expert collection 绑定。

## 限制与回退建议

- 对于未知结构或自定义 `trust_remote_code` 模型，自动检测可能拿不到足够的结构信息。这种情况下，仍建议显式传入 `layer_config` 和/或 `ignore_layers`。
- 策略检测会沿用 AutoRound 现有的 `trust_remote_code` 设置。若模型仓库不可信，请显式传入 `--disable_trust_remote_code` 或 `trust_remote_code=False`。
- 在 model-free 模式下，MXFP 策略会先根据配置构建一个轻量级 meta model 做检测；若该步骤失败，AutoRound 会回退到显式 scheme 与用户自定义覆盖规则。
