# Multi-GPU Rotation + Quantization Guide

本文档说明如何在多张 GPU 上运行 Rotation（QuaRot/SpinQuant）+ Quantization + Evaluation。

适用于 32B / 70B / 122B 等单卡放不下的大模型。

---

## 目录

- [快速开始](#快速开始)
- [Device 格式说明](#device-格式说明)
- [使用 run_multi_gpu.sh](#使用-run_multi_gpush)
- [直接使用 Python 脚本](#直接使用-python-脚本)
- [显存需求参考](#显存需求参考)
- [工作原理](#工作原理)
- [三个阶段的多卡处理](#三个阶段的多卡处理)
- [Layer-wise Rotation + 多卡](#layer-wise-rotation--多卡)
- [常见问题](#常见问题)

---

## 快速开始

```bash
# 32B 模型，2 张 A100-80G
GPUS_32B="0,1" bash run_multi_gpu.sh --model 32b --mode quick

# 70B 模型，4 张 A100-80G
GPUS_70B="0,1,2,3" bash run_multi_gpu.sh --model 70b --mode full

# 122B 模型，8 张 A100-80G
GPUS_122B="0,1,2,3,4,5,6,7" bash run_multi_gpu.sh --model 122b --mode layerwise
```

---

## Device 格式说明

`--device` 参数支持三种格式：

| 格式 | 说明 | 示例 |
|------|------|------|
| `cuda:X` | 单卡 | `--device cuda:0` |
| `X,Y,Z` | 多卡（显式指定 GPU ID）| `--device "0,1,2,3"` |
| `auto` | 多卡（自动使用所有可见 GPU）| `--device auto` |

多卡格式会触发以下行为：
- 模型加载到 CPU（不立即放 GPU）
- AutoRound 内部使用 `accelerate.dispatch_model` 分布模型到多张卡
- 评估阶段模型已在多卡上，lm_eval 直接使用分布式模型

---

## 使用 run_multi_gpu.sh

### 基本用法

```bash
# 只跑 32B
bash run_multi_gpu.sh --model 32b

# 只跑 70B，full 模式
bash run_multi_gpu.sh --model 70b --mode full

# 跑所有三个模型（顺序执行）
bash run_multi_gpu.sh

# 多模型并行（需要 GPU 不重叠）
bash run_multi_gpu.sh --parallel
```

### 可选模式

| Mode | 说明 |
|------|------|
| `quick` | limit=100，3 schemes，2 tasks（~30min） |
| `full` | 无 limit，3 schemes，5 tasks，det vs random（~4-8h） |
| `layerwise` | Block-wise rotation（节省 CPU RAM） |
| `tuning` | iters=200 auto-round tuning |
| `layerwise-tuning` | Block-wise + tuning |
| `weight-only` | W2/W3/W4/W8 A16 |

### 环境变量

```bash
# GPU 分配
GPUS_32B="4,5"           # 32B 模型用 GPU 4,5
GPUS_70B="0,1,2,3"      # 70B 模型用 GPU 0-3
GPUS_122B="0,1,2,3,4,5,6,7"  # 122B 模型用全部 8 卡

# 模型指定
MODEL_32B="Qwen/Qwen2.5-32B"
MODEL_70B="meta-llama/Llama-3.1-70B"
MODEL_122B="mistralai/Mistral-Large-Instruct-2411"

# 自定义模型
CUSTOM_GPUS="0,1,2,3" bash run_multi_gpu.sh --model /path/to/my-model
```

### 额外参数

```bash
# 限制评估样本数（快速验证）
bash run_multi_gpu.sh --model 70b --limit 50

# 指定评估任务
bash run_multi_gpu.sh --model 70b --tasks "hellaswag,piqa"
```

---

## 直接使用 Python 脚本

不使用 shell wrapper，直接调用 Python：

```bash
# 70B，4 卡，R1+R2+R3+R4，W4A16 量化
python test_rotation_scheme_matrix_v2.py \
    --model meta-llama/Llama-3.1-70B \
    --device "0,1,2,3" \
    --rotations "R1,R1+R2+R3+R4" \
    --schemes "W4A16,MXFP4" \
    --tasks "hellaswag,piqa" \
    --limit 100

# 32B，auto device，layerwise rotation + tuning
python test_rotation_scheme_matrix_v2.py \
    --model Qwen/Qwen2.5-32B \
    --device auto \
    --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
    --schemes "W4A16" \
    --layerwise \
    --quant-iters 200 \
    --tasks "hellaswag,piqa,winogrande"

# 122B，8 卡，RTN（iters=0），save/load roundtrip
python test_rotation_scheme_matrix_v2.py \
    --model mistralai/Mistral-Large-Instruct-2411 \
    --device "0,1,2,3,4,5,6,7" \
    --rotations "R1,R1+R2+R3+R4" \
    --schemes "W4A16" \
    --save-load \
    --tasks "hellaswag,piqa"
```

### 控制可见 GPU（推荐）

用 `CUDA_VISIBLE_DEVICES` 限定可见 GPU，再用 `--device auto`：

```bash
# 只用物理 GPU 4,5,6,7（对进程来说是 0,1,2,3）
CUDA_VISIBLE_DEVICES=4,5,6,7 python test_rotation_scheme_matrix_v2.py \
    --model meta-llama/Llama-3.1-70B \
    --device auto \
    --rotations "R1+R2+R3+R4" \
    --schemes "W4A16" \
    --tasks "hellaswag,piqa"
```

---

## 显存需求参考

模型参数量与 FP16 显存需求：

| 模型 | 参数量 | FP16 权重 | 推荐配置 |
|------|--------|-----------|----------|
| Qwen3-0.6B | 0.6B | ~1.2 GB | 1× 任意 GPU |
| Llama-3.1-8B | 8B | ~16 GB | 1× A100-40G |
| Qwen2.5-32B | 32B | ~64 GB | 2× A100-80G 或 4× A100-40G |
| Llama-3.1-70B | 70B | ~140 GB | 2× A100-80G 或 4× A100-40G |
| Mistral-Large (122B) | 122B | ~244 GB | 4× A100-80G 或 8× A100-40G |

> **注意**：实际运行时还需要额外显存用于：
> - Calibration 的 activation cache（~2-5 GB/block）
> - Quantization 临时 tensor
> - lm_eval 推理时的 KV cache

---

## 工作原理

### 数据流

```
--device "0,1,2,3"
       │
       ▼
┌──────────────────────────────────┐
│  load_model() → 加载到 CPU       │  ← 不立即放 GPU，避免 OOM
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  AutoRound(device_map="0,1,2,3") │
│    ├─ is_auto_device_mapping()   │  ← 检测到多卡
│    ├─ infer_auto_device_map()    │  ← accelerate 推断分层
│    └─ dispatch_model()           │  ← 模型分布到 4 张卡
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Calibration (forward pass)      │
│    Block 在多卡间流水线执行        │
│    per-block tuning 在主卡进行    │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  evaluate_model()                │
│    HFLM(pretrained=model)        │  ← 直接用已分布的模型
│    model.device = 第一张卡        │
└──────────────────────────────────┘
```

### 关键函数

| 函数 | 单卡行为 | 多卡行为 |
|------|----------|----------|
| `load_model()` | `.to("cuda:X")` | 留在 CPU |
| `load_model(for_eval_only=True)` | `.to("cuda:X")` | `device_map="auto"` |
| `AutoRound(device_map=...)` | 整模型搬到单卡 | `dispatch_model` 分层分布 |
| `evaluate_model()` | `HFLM(device="cuda:X")` | `HFLM(pretrained=model)` 无 device |
| `evaluate_model_from_path()` | `HFLM(device="cuda:X")` | `HFLM(parallelize=True)` |

---

## 三个阶段的多卡处理

### 阶段 1：Rotation

- **Full-model rotation**：在 CPU 上执行（矩阵乘法 `W @ R`），不占 GPU 显存
- **Layer-wise rotation**：同上，rotation 矩阵和权重都在 CPU
- 多卡不影响 rotation 阶段性能（rotation 本身是 CPU bound）

### 阶段 2：Quantization（Calibration）

AutoRound 内部处理：
1. `infer_auto_device_map(model)` — 根据每张卡显存自动分层
2. `dispatch_model(model, device_map)` — 模型层分布到多卡
3. Forward pass — 数据流经多卡（pipeline parallelism）
4. Per-block tuning — 当前 block 在其所在的卡上做梯度优化

### 阶段 3：Evaluation（lm_eval）

- **In-memory model**：模型已分布，直接传给 `HFLM(pretrained=model)`
- **From-disk model**：`HFLM(pretrained=path, parallelize=True)` 让 lm_eval 自行分布

---

## Layer-wise Rotation + 多卡

Layer-wise rotation 适合超大模型（70B+）：

```bash
python test_rotation_scheme_matrix_v2.py \
    --model meta-llama/Llama-3.1-70B \
    --device "0,1,2,3" \
    --layerwise \
    --rotations "R1,R1+R2+R3+R4" \
    --schemes "W4A16"
```

工作方式：
1. 模型加载到 CPU
2. AutoRound calibration 时将 block 逐个搬到 GPU
3. 每个 block 进入 GPU 前，`_on_block_ready` hook 在 CPU 做 rotation
4. Rotation 后的 block 上 GPU → calibration → quantize → 下一个 block

**优势**：CPU RAM 占用更小（不需同时持有全模型 rotation 后的权重）

---

## 常见问题

### Q: 多卡和单卡结果会不同吗？

不会。多卡只是模型分布（tensor parallelism / pipeline parallelism），数学计算完全一致。
Rotation 矩阵、量化参数等都不受设备分布影响。

### Q: 可以混用不同型号的 GPU 吗？

可以，但 accelerate 会根据每张卡的可用显存分配层数。
显存小的卡分配较少层。建议用同型号 GPU 以获得最佳负载均衡。

### Q: OOM 怎么办？

1. 增加 GPU 数量
2. 使用 `--layerwise`（layer-wise rotation 节省 CPU RAM）
3. 减小 `--batch-size`（默认 8，可改为 4 或 2）
4. 减小 `--nsamples`（默认 128，可改为 64）
5. 减小 `--seqlen`（默认 512，可改为 256）

### Q: 如何只用部分 GPU？

```bash
# 方法 1：直接指定 GPU ID
--device "4,5,6,7"

# 方法 2：用 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=4,5,6,7 python ... --device auto
```

### Q: run.sh 和 run_multi_gpu.sh 的区别？

| | run.sh | run_multi_gpu.sh |
|--|--------|-----------------|
| 目标 | 多个小模型并行 | 单个大模型多卡 |
| GPU 分配 | 每个 job 占 1 张卡 | 每个 job 占多张卡 |
| 适用模型 | ≤ 8B | ≥ 32B |
| 并行方式 | 多 job 并发 | 模型层间并行 |

### Q: 评估（lm_eval）支持多卡吗？

支持：
- **In-memory model**（quantize 后直接评估）：模型已在多卡上，lm_eval 直接用
- **From-disk model**（save/load 后评估）：用 `parallelize=True` 让 lm_eval 自动分布

### Q: 单卡的脚本还能正常跑吗？

完全兼容。`--device cuda:0` 走原来的单卡路径，行为不变。
