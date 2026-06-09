# test_save_load_roundtrip.py — 使用文档

## 概述

`test_save_load_roundtrip.py` 是一个端到端验证脚本，用于确认 **SpinQuant rotation + quantization** 后的模型可以正确保存到磁盘并重新加载，且推理精度不变。

### 核心验证逻辑

对于每一种 `rotation × scheme × iters × rotation_size × rotation_mode` 组合，脚本执行5步验证：

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 从 HuggingFace 加载原始模型                         │
│  Step 2: AutoRound 量化 + 保存到磁盘                         │
│  Step 3: 检查保存文件中的 spinquant artifacts                 │
│  Step 4: 从磁盘加载模型 → lm_eval 评估 (Path B: from-disk)   │
│  Step 5: 内存中重新量化 → lm_eval 评估 (Path A: in-memory)   │
│  比较: Path A accuracy == Path B accuracy?                    │
└─────────────────────────────────────────────────────────────┘
```

如果 Path A == Path B，说明 save/load roundtrip 正确（rotation buffers 完整保存并正确恢复）。

---

## 测试维度

| 维度 | 参数 | 可选值 | 说明 |
|------|------|--------|------|
| **Rotation Level** | `--rotations` | `none, R1, R1+R2, R1+R2+R3, R1+R2+R3+R4` | 逐步叠加的旋转级别 |
| **Rotation Mode** | `--rotation-modes` | `hadamard, random` | 确定性 vs 随机正交矩阵 |
| **Quant Scheme** | `--schemes` | `W4A16, W3A16, W8A16, MXFP4, NVFP4, FP8_STATIC` | 覆盖 INT/FP4/FP8 三条路径 |
| **Iters** | `--quant-iters-list` | `"0,200"` 等 | 0=RTN, >0=auto-round tuning |
| **Rotation Size** | `--rotation-sizes` | `"64,128,auto"` 等 | block rotation size |

### Rotation Mode 详解

| Mode | 含义 | 保存行为 |
|------|------|----------|
| `hadamard` | 确定性 Hadamard 矩阵，可从 size 重建 | 仅保存 `rot_type` + `rot_size` (轻量) |
| `random` | H × diag(±1) 随机正交矩阵 | 保存完整矩阵到 safetensors (较大) |

> **注意**: `trainable` (训练优化的旋转矩阵) 不在本脚本测试范围内。

### Scheme 与 Export 路径映射

| Scheme | Export 路径 | 文件 |
|--------|-------------|------|
| W4A16, W3A16, W8A16 | INT pack_layer | `export.py` |
| MXFP4, NVFP4 | MXFP/NVFP pack_layer | `export_to_nvfp_mx.py` |
| FP8_STATIC | FP8 pack_layer | `export_to_fp8.py` |

---

## CLI 参数完整列表

```
python examples/test_save_load_roundtrip.py [OPTIONS]
```

### 模型与设备

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-0.6B` | HuggingFace 模型路径 |
| `--device` | `cuda:0` | 推理设备 |

### 测试组合

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rotations` | `R1,R1+R2,R1+R2+R3,R1+R2+R3+R4` | 逗号分隔的 rotation level 列表 |
| `--schemes` | `W4A16` | 逗号分隔的量化 scheme 列表 |
| `--rotation-modes` | `hadamard` | 逗号分隔: `hadamard,random` |
| `--quant-iters` | `0` | 单个 iters 值 (被 `--quant-iters-list` 覆盖) |
| `--quant-iters-list` | `None` | 逗号分隔多个 iters: `"0,200"` |
| `--rotation-size` | `None` (auto) | 单个 rotation block size |
| `--rotation-sizes` | `None` | 逗号分隔多个 size: `"64,128,auto"` |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tasks` | `hellaswag,piqa` | lm_eval 任务 |
| `--limit` | `None` (full) | 每个 task 评估样本数限制 |
| `--batch-size` | `8` | lm_eval batch size |

### AutoRound 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nsamples` | `128` | 校准数据样本数 |
| `--seqlen` | `512` | 校准序列长度 |

### Rotation 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--online-r1` | `True` | R1 在推理时在线应用 (默认) |
| `--offline-r1` | — | 设置 R1 离线模式 (fuse 到权重) |
| `--random-hadamard` | `False` | (旧接口) 启用 random mode，推荐用 `--rotation-modes` |

### 输出与行为

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | auto-generated | 保存目录 (格式: `roundtrip_{model}_{timestamp}`) |
| `--skip-inmemory` | `False` | 跳过 Step 5 内存对比 (只测试 save+load from disk) |
| `--cleanup` | `False` | 验证完成后删除保存的模型目录 |

---

## 使用示例

### 1. 快速冒烟测试 (单组合)

```bash
python examples/test_save_load_roundtrip.py \
    --device cuda:0 --limit 100 \
    --rotations "R1+R2" --schemes "W4A16" \
    --skip-inmemory
```

约耗时: 2-3 分钟。只验证保存 + 从磁盘加载评估。

### 2. Hadamard vs Random 对比

```bash
python examples/test_save_load_roundtrip.py \
    --device cuda:0 --limit 100 \
    --rotations "R1+R2" --schemes "W4A16" \
    --rotation-modes "hadamard,random"
```

验证两种旋转模式的 save/load 都正确。Random 模式会保存完整矩阵。

### 3. 多 Scheme 覆盖 (INT + FP4 + FP8)

```bash
python examples/test_save_load_roundtrip.py \
    --device cuda:0 --limit 100 \
    --rotations "R1+R2" \
    --schemes "W4A16,MXFP4,NVFP4,FP8_STATIC"
```

覆盖三条不同的 export 路径，验证每条路径的 spinquant buffer 注入。

### 4. RTN vs Tuning

```bash
python examples/test_save_load_roundtrip.py \
    --device cuda:0 --limit 100 \
    --rotations "R1+R2" --schemes "W4A16" \
    --quant-iters-list "0,200"
```

`iters=0` 是 RTN（直接量化），`iters=200` 是 auto-round 优化。两者走同一个 pack/save 路径。

### 5. Rotation Size Sweep

```bash
python examples/test_save_load_roundtrip.py \
    --device cuda:0 --limit 100 \
    --rotations "R1+R2+R3+R4" --schemes "W4A16" \
    --rotation-sizes "64,128,auto"
```

测试不同 block rotation size 的保存/恢复。`auto` = 使用 hidden_size/intermediate_size。

### 6. 全覆盖矩阵 (推荐 CI 测试)

```bash
python examples/test_save_load_roundtrip.py \
    --device cuda:0 --limit 100 \
    --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
    --schemes "W4A16,MXFP4,NVFP4,FP8_STATIC" \
    --rotation-modes "hadamard,random" \
    --quant-iters-list "0,200" \
    --skip-inmemory --cleanup
```

组合数: 4 rotations × 2 modes × 4 schemes × 2 iters = **64 combos**。
使用 `--skip-inmemory` 减半评估时间，`--cleanup` 节省磁盘。

### 7. 使用不同模型

```bash
python examples/test_save_load_roundtrip.py \
    --model "meta-llama/Llama-3.2-1B" \
    --device cuda:0 --limit 100 \
    --rotations "R1+R2" --schemes "W4A16" \
    --rotation-modes "hadamard,random"
```

---

## 输出说明

### 目录结构

```
roundtrip_Qwen3-0.6B_20260511_130800/
├── results.json                        # 所有组合的详细结果
├── Qwen3-0.6B-R1_R2-W4A16/           # rotation + scheme 组合的保存目录
│   └── Qwen3-0.6B-w4g128/            # AutoRound 生成的子目录
│       ├── config.json                 # 包含 spinquant_config
│       ├── model.safetensors           # 包含 spinquant buffer keys
│       └── ...
├── Qwen3-0.6B-R1_R2-W4A16-rand/      # random mode (注意 -rand 后缀)
│   └── ...
├── Qwen3-0.6B-R1_R2-MXFP4/           # MXFP4 scheme
│   └── ...
└── ...
```

### results.json 结构

```json
[
  {
    "rotation": "R1+R2",
    "scheme": "W4A16",
    "rotation_mode": "hadamard",
    "label": "R1+R2 + W4A16",
    "quant_iters": 0,
    "rotation_size": null,
    "save_dir": "roundtrip_.../Qwen3-0.6B-R1_R2-W4A16/Qwen3-0.6B-w4g128",
    "save_checks": {
      "config_has_spinquant": true,
      "config_r1": true,
      "config_r2": true,
      "n_spinquant_keys": 280,
      "has_expected_buffers": true
    },
    "metrics_from_disk": {
      "hellaswag": 0.3920,
      "piqa": 0.6850
    },
    "metrics_in_memory": {
      "hellaswag": 0.3920,
      "piqa": 0.6850
    },
    "roundtrip_match": true,
    "status": "success",
    "total_time_s": 145.2
  }
]
```

### 终端输出摘要

脚本结束时输出两张汇总表:

1. **Roundtrip Comparison** — 每个组合的 disk vs memory accuracy 对比
2. **Save Artifact Checks** — 每个组合的文件检查 (config 是否包含 spinquant、buffer 数量等)

---

## 验证项与判断标准

| 检查项 | 含义 | PASS 条件 |
|--------|------|-----------|
| `config_has_spinquant` | config.json 包含 spinquant_config | `true` |
| `n_spinquant_keys` | safetensors 中 spinquant buffer 数 | `> 0` (rotation 非 none 时) |
| `has_expected_buffers` | buffer 数符合预期 | `true` |
| `roundtrip_match` | disk eval == memory eval | accuracy 差 < 0.0001 |

### 常见失败原因

| 症状 | 可能原因 |
|------|----------|
| `n_spinquant_keys = 0` | ShardWriter 跳过了 meta tensor (injection timing 问题) |
| `config_has_spinquant = False` | `_save_spinquant_config_to_dir()` 未被调用 |
| `roundtrip_match = False` | buffer 保存不完整或加载后未正确 patch forward |
| Random mode disk ≠ memory | 随机矩阵未被持久化 (每次重新生成不同矩阵) |

---

## 与其他测试脚本的关系

| 脚本 | 验证目标 | 是否保存模型 |
|------|----------|--------------|
| `test_rotation_levels.py` | rotation 正确性 (logit 对比) | ❌ |
| `test_rotation_scheme_matrix_v2.py` | rotation × scheme accuracy (内存中) | ❌ |
| **`test_save_load_roundtrip.py`** | **save/load 完整性验证** | ✅ |

---

## 注意事项

1. **磁盘空间**: 每个组合保存约 1-2 GB (取决于模型大小)。使用 `--cleanup` 自动清理。
2. **时间估算**: 每个组合约 2-5 分钟 (取决于 iters 和 limit)。64 combos ≈ 3-5 小时。
3. **Memory**: 脚本在每个组合之间释放 GPU 内存，但建议有 24GB+ GPU 内存。
4. **Random 可重复性**: random mode 使用 `random_hadamard_matrix()`，保存后加载可重现；但两次独立量化产生不同矩阵（预期行为）。
5. **`--skip-inmemory` 模式**: 不做 Path A vs Path B 对比，仅验证模型能从磁盘正确加载并产生合理 accuracy。适合初步验证或 CI 节省时间。
