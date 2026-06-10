# Selective Per-Linear Hadamard Rotation: Save & vLLM Inference

## 概述

本文档描述了如何将 selective per-linear Hadamard rotation 应用于模型后，保存为 vLLM 可推理的格式。

核心思路：不是所有 Linear 层都需要 Hadamard rotation。通过结构先验（structural）或激活统计（auto）模式，
选择性地仅对部分层做 rotation，跳过对 accuracy 有害的层（如 down_proj、o_proj），
然后保存时只在被 rotate 的层上注入 vLLM 所需的 rotation buffer。

## 原理

### 传统流程（所有层都旋转）

```
quantize time:  所有 Linear → apply Hadamard to weight → 注入 R1 buffer → 保存
vLLM inference: 读取 R1 buffer → 对 activation 做 online Hadamard → GEMM
```

### Selective 流程

```
quantize time:
  1. LayerSelector 决定每层是否旋转（structural/auto mode）
  2. 仅对 enabled=True 的层 apply Hadamard to weight
  3. 保存 _rotated_layers 集合
  4. serialize 时仅在 rotated 的层注入 spinquant_r1_type/size buffer
  5. config.json 中记录 selective_rotation=True + rotated_layers 列表

vLLM inference:
  - 有 buffer 的层：读取 spinquant_r1_type/size → 做 online rotation
  - 没有 buffer 的层（rot_size=0）：跳过，不做 rotation
  → 自然兼容，无需 vLLM 插件额外修改
```

## 修改的文件

### 新增文件

| 文件 | 说明 |
|------|------|
| `auto_round/algorithms/transforms/rotation/selective.py` | 核心选择逻辑：`LayerSelector`, `LayerDecision`, `compute_layer_score`, `structural_decision`, `_profile_activations` |

### 修改文件

| 文件 | 修改 |
|------|------|
| `auto_round/algorithms/transforms/rotation/config.py` | `RotationConfig` 新增字段：`layer_selection`, `include_layers`, `exclude_layers`, `kurtosis_threshold`, `ecr_threshold`, `score_threshold` |
| `auto_round/algorithms/transforms/rotation/apply.py` | `HadamardRotation.apply_to_model()` 新增 selective 逻辑 + `_setup_serialize_bridge()` 函数桥接到 SpinQuant serializer |
| `auto_round/algorithms/transforms/rotation/dispatcher.py` | `apply_hadamard_rotation()` 新增 `calibration_dataloader` 参数并转发 |
| `auto_round/algorithms/transforms/spinquant/algorithm.py` | `inject_buffers_on_layer()` 检查 `model._rotated_layers`，跳过未旋转层 |
| `auto_round/algorithms/transforms/spinquant/serialize.py` | `inject_spinquant_buffers()` 检查 `_rotated_layers`；`save_spinquant_config()` 保存 `selective_rotation`/`rotated_layers` 元数据 |
| `auto_round/vllm_plugin/spinquant_mxfp4.py` | `from_config()` 解析并显示 `[SELECTIVE: N layers rotated]` 日志 |

### 同步到 auto_round_v1/

以上修改同步复制到了 `auto_round_v1/` 目录下对应文件。

## 配置方式

### 方式一：通过 AutoRound API

```python
from auto_round import AutoRound

ar = AutoRound(
    model, tokenizer,
    bits=4, group_size=32,
    data_type="mx_fp",
    rotation_config={
        "hadamard_type": "hadamard",       # "hadamard" 或 "random_hadamard"
        "block_size": 32,                   # MXFP4 group size
        "layer_selection": "structural",    # "all" | "structural" | "auto"
        # 以下为可选参数（仅 "auto" 模式使用）：
        # "kurtosis_threshold": 5.0,
        # "ecr_threshold": 0.05,
        # "score_threshold": 1.5,
        # "include_layers": ["*gate_proj*"],   # fnmatch 强制包含
        # "exclude_layers": ["*layers.0.*"],   # fnmatch 强制排除
    },
    iters=0,  # 0=RTN, >0=SGD tuning
)
ar.quantize()
ar.save_quantized("/path/to/output", format="auto_round")
```

### 方式二：通过测试脚本

```bash
python /data/lkk/quarot/scripts/test_selective_rotation_save_vllm.py \
    --model Qwen/Qwen3-0.6B \
    --output /tmp/qwen3-0.6b-mxfp4-selective \
    --layer-selection structural \
    --hadamard-type hadamard \
    --iters 0
```

### layer_selection 模式说明

| 模式 | 说明 | 性能 |
|------|------|------|
| `"all"` | 所有 Linear 层都旋转（传统行为） | 最快，无额外开销 |
| `"structural"` | 基于层名模式决定（hardcode 规则） | 快，无需数据 |
| `"auto"` | 结构先验 + 激活统计（需要 calibration data） | 需要额外 ~32 samples forward |

### structural 模式的默认规则

| 层类型 | 决定 | 原因 |
|--------|------|------|
| `*q_proj*`, `*k_proj*` | ✅ 旋转 | 输入来自 hidden_state，rotation 有效减少 outlier |
| `*gate_proj*`, `*up_proj*` | ✅ 旋转 | Expand 层 + 后接非线性，rotation 安全 |
| `*down_proj*`, `*o_proj*` | ❌ 跳过 | 直接输出到 residual stream，rotation 可能引入误差 |
| `*v_proj*` | ❌ 跳过(保守) | 条件性，无统计数据时保守跳过 |
| `*lm_head*` | ❌ 跳过 | 最终输出层 |
| `*mlp.gate*` (MoE router) | ❌ 跳过 | 路由层，不应旋转 |

## 保存后的 config.json 格式

```json
{
  "quantization_config": {
    "bits": 4,
    "group_size": 32,
    "data_type": "mx_fp",
    "spinquant_config": {
      "algorithm": "spinquant",
      "r1": true,
      "online_r1_rotation": true,
      "random_r1": false,
      "hidden_size": 896,
      "intermediate_size": 4864,
      "selective_rotation": true,
      "num_rotated_layers": 120,
      "rotated_layers": [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.q_proj",
        "..."
      ]
    }
  }
}
```

## vLLM 推理

### 安装

```bash
pip install --no-build-isolation -e /data/lkk/quarot/latest/new_commit/auto-round
```

### 使用

```python
from vllm import LLM, SamplingParams

llm = LLM("/path/to/saved/model")
# 日志会显示：
# [AutoRound] Model quantization: MXFP4 (group_size=32) |
#   Online rotations: [R1(online, hadamard, size=896, ...) [SELECTIVE: 120 layers rotated]]

output = llm.generate("Hello, world!", SamplingParams(max_tokens=50))
print(output[0].outputs[0].text)
```

### vLLM 推理时的行为

- 有 `spinquant_r1_type` buffer 的层：`_process_rotation()` 读取 type/size，
  重建 Hadamard butterfly 矩阵，在 forward 时对 activation 做 online rotation
- 没有 buffer（或 `rot_size=0`）的层：`_process_rotation()` 直接 return，
  不做任何 rotation → 正确匹配 quantize 时未旋转权重的层

## 内部实现细节

### _setup_serialize_bridge()

`apply.py` 中的桥接函数。Hadamard rotation 完成后，创建一个合成的 `SpinQuantConfig`
设置到 `model._rotation_config`，使得 export 路径能通过 SpinQuant serializer 注入 buffer：

```python
def _setup_serialize_bridge(model, cfg):
    spinquant_cfg = SpinQuantConfig(
        r1=True, online_r1_rotation=True,
        random_r1=(cfg.hadamard_type == "random_hadamard"),
        rotation_size=cfg.block_size,
    )
    spinquant_cfg._selective_mode = cfg.layer_selection != "all"
    model._rotation_config = spinquant_cfg
```

### _rotated_layers 传递链

```
apply_to_model()
  → model._rotated_layers = {"model.layers.0.mlp.gate_proj", ...}  # set of names
  → model._rotation_config = SpinQuantConfig(...)

export path (pack_layer / save_quantized):
  → inject_rotation_buffers_on_layer(name, qlayer, model)
    → checks: name in model._rotated_layers?
      → Yes: inject spinquant_r1_type/size buffer
      → No:  skip (layer stays with default zero buffers)

save_spinquant_config():
  → writes rotated_layers list to config.json
```

### vLLM plugin 加载流

```
SpinQuantMXFP4Config.from_config(config)
  → reads spinquant_config from config.json
  → logs "[SELECTIVE: N layers rotated]"

SpinQuantMXFP4LinearMethod.create_weights()
  → pre-registers spinquant_r1_type/size/matrix (all zeros by default)

state_dict loading:
  → only rotated layers have non-zero values in checkpoint
  → non-rotated layers keep rot_size=0

process_weights_after_loading():
  → _process_rotation(layer, "r1")
    → rot_size = layer.spinquant_r1_size.item()
    → if rot_size == 0: return  ← non-rotated layers stop here
    → else: setup runtime rotation (butterfly/matrix)
```

## 验证结果

测试通过的场景：
- ✅ structural 模式：q/k/gate/up_proj 旋转，down/o_proj 跳过
- ✅ buffer 仅注入到 rotated 层（8/14 layers）
- ✅ config.json 包含 `selective_rotation=True` 和 `rotated_layers`
- ✅ vLLM plugin 正确解析并显示 selective 信息
- ✅ 非 rotated 层 rot_size=0 → vLLM 不做 rotation

## 注意事项

1. **selective 模式产出的模型只能用对应版本的 vLLM plugin 加载**
   旧版 plugin 不识别 selective metadata 但仍能工作（因为 buffer 机制天然兼容）

2. **random_hadamard + selective**
   random_hadamard 类型会在每个 rotated 层存储完整旋转矩阵（int8 ±1），
   模型文件会稍大。deterministic hadamard 只存 type+size（矩阵在 runtime 重建）。

3. **auto 模式需要 calibration data**
   在 AutoRound pipeline 中，calibration dataloader 会自动构建。
   独立使用时需手动传入 `calibration_dataloader` 参数。

4. **多卡模型（device_map="auto"）的 profiling**
   目前对于 accelerate dispatch 的多卡模型，auto 模式 profiling 会跳过，
   退回到 structural-only 决策（因为 MoE patch 有 device mismatch 问题）。
