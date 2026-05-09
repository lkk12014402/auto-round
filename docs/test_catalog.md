# Auto-Round SpinQuant / QuaRot Test & Example Catalog

> 完整的测试和示例脚本索引，包含功能说明、运行命令和依赖关系。

---

## 目录

1. [单元测试 (Unit Tests)](#1-单元测试-unit-tests)
2. [集成测试 / 精度验证 (Integration Tests)](#2-集成测试--精度验证-integration-tests)
3. [批量精度矩阵 (Benchmark Matrix)](#3-批量精度矩阵-benchmark-matrix)
4. [框架对比测试 (Framework Comparison)](#4-框架对比测试-framework-comparison)
5. [示例代码 (Examples / Tutorials)](#5-示例代码-examples--tutorials)
6. [Shell 运行脚本 (Shell Runners)](#6-shell-运行脚本-shell-runners)
7. [测试层级关系图](#7-测试层级关系图)
8. [快速运行指南](#8-快速运行指南)

---

## 1. 单元测试 (Unit Tests)

这些测试使用 mock 模型，不需要 GPU 或真实模型权重，几秒内完成。

### 1.1 `test_reference_equivalence.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | 验证 `rotate_in_channels_()` / `rotate_out_channels_()` 与参考实现 `fuse_layer_rotation()` 的数学等价性 |
| **验证点** | 自定义的 rotation 工具函数是否正确实现了 `W_new = R⁻¹ @ W @ R` |
| **使用模型** | Mock linear layers |
| **依赖** | auto_round.spinquant.rotation_utils |
| **运行命令** | `python test_reference_equivalence.py` |

### 1.2 `test_quark_comparison.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | 对比 Quark 的 rotation fusion 与 auto-round 的实现，使用相同的 mock 模型、随机种子和旋转矩阵 |
| **验证点** | 两个框架对同一模型施加相同旋转后，输出是否一致（cosine similarity > 0.999） |
| **使用模型** | Mock models (RMSNorm, Attention, MLP, DecoderLayer) |
| **依赖** | auto_round, **Quark** |
| **运行命令** | `python test_quark_comparison.py` |

### 1.3 `test_rotation_levels.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | 分步验证 R1 → R1+R2 → R1+R2+R3 → R1+R2+R3+R4 各级旋转的正确性 |
| **验证点** | 每级旋转后模型输出与原始 FP 输出的 cosine similarity (应 > 0.999) |
| **使用模型** | Mock multi-head attention model (power-of-2 dimensions) |
| **依赖** | auto_round.spinquant |
| **运行命令** | `python test_rotation_levels.py` |

### 1.4 `test_spinquant.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | SpinQuant 迁移的综合单元测试：QuaRot 模式 (fixed Hadamard)、SpinQuant trainable 模式 (SGDG optimizer)、边界条件、架构灵活性 |
| **验证点** | 配置解析、旋转矩阵初始化、训练循环基本功能、不同架构兼容性 |
| **使用模型** | Mock models (flat 和 nested 架构) |
| **依赖** | auto_round.spinquant |
| **运行命令** | `python test_spinquant.py` |

### 1.5 `test_wrapper_save_load.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | `InputRotationWrapperHadamard` 的 forward 等价性（旋转+逆旋转=恒等）和 save/load 往返一致性 |
| **验证点** | wrapper 保存后加载，输出不变 |
| **使用模型** | Mock linear layers |
| **依赖** | auto_round.spinquant.rotation_utils |
| **运行命令** | `python test_wrapper_save_load.py` |

---

## 2. 集成测试 / 精度验证 (Integration Tests)

这些测试使用真实模型 (Qwen3-0.6B)，通过 lm_eval 验证精度。需要 GPU。

### 2.1 `test_qwen3_rotation_eval.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | 在 Qwen3-0.6B 上验证各级 QuaRot 旋转后的精度保持（rotation-only, 不做量化） |
| **验证点** | 纯旋转（正交变换）在 FP 精度下不应降低 accuracy（数学等价） |
| **使用模型** | Qwen/Qwen3-0.6B |
| **依赖** | lm_eval, transformers |
| **运行命令** | |

```bash
# 快速测试
python test_qwen3_rotation_eval.py --device cuda:7 --limit 100 --tasks piqa,hellaswag

# 完整测试
python test_qwen3_rotation_eval.py --device cuda:7 --tasks hellaswag,piqa,winogrande,lambada_openai
```

### 2.2 `test_rotation_quantization.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | 对比三种场景的精度：FP16 baseline → RTN W4A16 (无旋转) → Rotation(R1+R2+R3+R4) + RTN W4A16 |
| **验证点** | 旋转使权重分布更均匀 → 量化后精度更高 (outlier redistribution) |
| **使用模型** | Qwen/Qwen3-0.6B |
| **依赖** | lm_eval, auto_round |
| **运行命令** | |

```bash
# 快速测试
python test_rotation_quantization.py --device cuda:7 --limit 200 --tasks piqa,hellaswag

# 选择特定场景
python test_rotation_quantization.py --device cuda:7 --levels baseline_fp16,rotation_rtn
```

### 2.3 `test_rotation_schemes.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | 测试不同旋转级别 (R1→R1+R2+R3+R4) 与不同量化方案 (W4A16, NVFP4, MXFP4) 的组合精度 |
| **验证点** | 各 rotation × scheme 组合的 accuracy 对比 |
| **使用模型** | Qwen/Qwen3-0.6B |
| **依赖** | lm_eval, auto_round |
| **运行命令** | |

```bash
# 快速测试
python test_rotation_schemes.py --device cuda:7 --limit 100

# 指定方案和旋转
python test_rotation_schemes.py --device cuda:7 --schemes W4A16,NVFP4 --rotations R1,R1+R2+R3+R4

# 包含 baseline 对比
python test_rotation_schemes.py --device cuda:7 --include-baselines
```

### 2.4 `test_autoround_rotation_mxfp4.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | Auto-Round 框架下的 rotation + MXFP4 RTN 流水线，与 Quark 版本做 apple-to-apple 对比 |
| **验证点** | FP16 baseline → MXFP4-only → Rotation+MXFP4 的精度变化 |
| **使用模型** | Qwen/Qwen3-0.6B |
| **依赖** | lm_eval, auto_round |
| **运行命令** | |

```bash
python test_autoround_rotation_mxfp4.py --device cuda:7 --limit 200 --tasks piqa,hellaswag
```

### 2.5 `test_quark_rotation_mxfp4.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | Quark 框架下的 rotation (R1+R2) + MXFP4 RTN 流水线（配套 2.4 的 Quark 侧） |
| **验证点** | 同上，但使用 Quark 的实现 |
| **使用模型** | Qwen/Qwen3-0.6B |
| **依赖** | lm_eval, **Quark** |
| **运行命令** | |

```bash
python test_quark_rotation_mxfp4.py --device cuda:7 --limit 200 --tasks piqa,hellaswag
```

---

## 3. 批量精度矩阵 (Benchmark Matrix)

### 3.1 `test_rotation_scheme_matrix.py` ⭐ 主力测试脚本

| 项目 | 说明 |
|------|------|
| **测试内容** | **所有** rotation × quantization scheme 组合的精度矩阵，支持 RTN 和 GPTQ-style 优化 |
| **验证点** | 全面的 rotation + quantization 精度数据，生成 matrix table |
| **使用模型** | Qwen/Qwen3-0.6B (可自定义) |
| **依赖** | lm_eval, auto_round |
| **输出** | JSON、CSV、formatted table |

**测试维度:**

| 维度 | 选项 |
|------|------|
| Rotation | `none`, `R1`, `R1+R2`, `R1+R2+R3`, `R1+R2+R3+R4` |
| Scheme | `W4A16`, `W3A16`, `W2A16`, `W8A16`, `MXFP4`, `NVFP4`, `MXFP8`, `INT8`, `INT4`, `FP8_STATIC`, `FP8_BLOCK` |
| Quant mode | RTN (`iters=0`), GPTQ-style (`iters=200`) |
| Hadamard | Deterministic (default), Random (`--random-hadamard`) |

**7 种预设模式 (通过 `run_rotation_scheme_matrix.sh`):**

| Mode | 内容 | Rotation | Scheme | 耗时 |
|------|------|----------|--------|------|
| `quick` | 快速验证 | none/R1/R1+R2/R1+R2+R3+R4 | W4A16/MXFP4/NVFP4 | ~15min |
| `full` | 完整精度 | 同上 | 同上 | ~2h |
| `full-matrix` | 穷举组合 | 全部5种 | 全部11种 | ~8h+ |
| `weight-only` | 纯权重量化 | 自定义 | W2/W3/W4/W8 | ~3h |
| `weight-act` | 权重+激活量化 | 自定义 | MXFP4/NVFP4/INT8/MXFP8 | ~3h |
| `random` | 随机Hadamard | none/R1/R1+R2/R1+R2+R3+R4 | W4A16/MXFP4/NVFP4 | ~2h |
| `gptq` | GPTQ优化量化 | none/R1/R1+R2 | W4A16 | ~2h |

**运行命令:**

```bash
# 快速测试（limit=100, ~15min）
bash run_rotation_scheme_matrix.sh quick

# 完整精度（no limit, ~2h）
bash run_rotation_scheme_matrix.sh full

# 穷举所有组合
bash run_rotation_scheme_matrix.sh full-matrix

# 只测 weight-only schemes
bash run_rotation_scheme_matrix.sh weight-only

# 只测 weight+activation schemes
bash run_rotation_scheme_matrix.sh weight-act

# Random Hadamard vs Deterministic
bash run_rotation_scheme_matrix.sh random

# GPTQ-style 优化量化
bash run_rotation_scheme_matrix.sh gptq

# 自定义 GPU 和 rotation_size
DEVICE=cuda:4 ROTATION_SIZE=128 bash run_rotation_scheme_matrix.sh full

# 直接调用 Python（更灵活）
python test_rotation_scheme_matrix.py \
    --device cuda:7 \
    --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
    --schemes "W4A16,MXFP4,NVFP4" \
    --quant-iters 0 \
    --tasks "hellaswag,piqa,winogrande,lambada_openai" \
    --rotation-size 128

# RTN vs Tuning 并行测试（两个 GPU）
# GPU 4: RTN
python test_rotation_scheme_matrix.py --device cuda:4 --quant-iters 0 \
    --output-dir results_rtn &

# GPU 5: Tuning
python test_rotation_scheme_matrix.py --device cuda:5 --quant-iters 200 \
    --output-dir results_tuning &
```

---

## 4. 框架对比测试 (Framework Comparison)

### 4.1 `test_three_way_comparison.py` ⭐ 三框架对比

| 项目 | 说明 |
|------|------|
| **测试内容** | Auto-Round vs Quark vs llm-compressor 三框架对比：相同 rotation + MXFP4 量化 |
| **验证点** | 三个框架在相同配置下的 accuracy 差异 |
| **使用模型** | Qwen/Qwen3-0.6B |
| **依赖** | lm_eval, auto_round, **Quark**, **llm-compressor** |

**运行命令:**

```bash
# 快速测试
python test_three_way_comparison.py --device cuda:4 --limit 100

# 完整测试
python test_three_way_comparison.py --device cuda:4

# 指定旋转级别
python test_three_way_comparison.py --device cuda:4 --rotations R1,R1+R2+R3+R4

# 指定框架
python test_three_way_comparison.py --device cuda:4 --frameworks autoround,quark

# 通过 shell 脚本
bash run_three_way_comparison.sh
```

### 4.2 `test_autoround_rotation_mxfp4.py` + `test_quark_rotation_mxfp4.py`

| 项目 | 说明 |
|------|------|
| **测试内容** | Auto-Round 和 Quark 各自的 rotation + MXFP4 pipeline 对比（分别运行，手动比较结果） |
| **验证点** | 两个框架的 pipeline 是否产生相近的精度 |
| **运行命令** | |

```bash
# 分别运行两个框架
python test_autoround_rotation_mxfp4.py --device cuda:6 --limit 200
python test_quark_rotation_mxfp4.py --device cuda:7 --limit 200
```

---

## 5. 示例代码 (Examples / Tutorials)

### 5.1 `spinquant_autoround_example.py`

| 项目 | 说明 |
|------|------|
| **内容** | SpinQuant 集成 AutoRound 的完整示例：手动集成方法 + RotationTrainer 使用 |
| **使用模型** | meta-llama/Llama-3.2-1B-Instruct |
| **注意** | 路径硬编码，需按环境修改 |

### 5.2 `usage_examples.py`

| 项目 | 说明 |
|------|------|
| **内容** | 中文注释的使用示例：RotationTrainer 基本用法、pipeline 模式、与 AutoRound 集成 |
| **使用模型** | Qwen/Qwen3-0.6B |

---

## 6. Shell 运行脚本 (Shell Runners)

| 脚本 | 调用的 Python | 用途 |
|------|-------------|------|
| `run_rotation_scheme_matrix.sh` | `test_rotation_scheme_matrix.py` | 7种模式的批量 rotation × scheme 精度矩阵 |
| `run_three_way_comparison.sh` | `test_three_way_comparison.py` | 三框架对比 |
| `run_comparison_tests.sh` | `test_autoround_rotation_mxfp4.py` / `test_quark_rotation_mxfp4.py` | Quark vs AutoRound 多场景对比 |
| `run_full_comparison.sh` | 同上 | 完整 rotation + MXFP4 对比 |
| `run_rotation_scheme_tests.sh` | `test_rotation_schemes.py` | W4A16/NVFP4 rotation 快速测试 |

---

## 7. 测试层级关系图

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: Unit Tests (mock models, no GPU, seconds)              │
│                                                                 │
│  test_reference_equivalence.py    rotation工具函数正确性        │
│  test_quark_comparison.py         auto-round vs Quark 等价性    │
│  test_rotation_levels.py          R1→R4 分级验证               │
│  test_spinquant.py                SpinQuant 配置和训练基础      │
│  test_wrapper_save_load.py        rotation wrapper 保存/加载   │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: Integration Tests (real model, single config, ~min)    │
│                                                                 │
│  test_qwen3_rotation_eval.py      rotation-only 精度保持       │
│  test_rotation_quantization.py    rotation + W4A16 效果验证    │
│  test_autoround_rotation_mxfp4.py auto-round rotation+MXFP4   │
│  test_quark_rotation_mxfp4.py     Quark rotation+MXFP4        │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: Scheme Matrix (real model, many combos, ~hours)        │
│                                                                 │
│  test_rotation_schemes.py         rotation × scheme (简版)     │
│  test_rotation_scheme_matrix.py   rotation × scheme (完整版) ⭐ │
├─────────────────────────────────────────────────────────────────┤
│ Level 4: Framework Comparison (3 frameworks, ~hours)            │
│                                                                 │
│  test_three_way_comparison.py     AR vs Quark vs llm-comp ⭐    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 快速运行指南

### 8.1 从零开始验证（推荐顺序）

```bash
cd /data/lkk/quarot/auto-round/examples

# Step 1: 单元测试（确认基础功能正常，~30秒）
python test_reference_equivalence.py
python test_rotation_levels.py
python test_spinquant.py
python test_wrapper_save_load.py

# Step 2: Rotation-only 精度验证（确认旋转不损失精度，~5分钟）
python test_qwen3_rotation_eval.py --device cuda:7 --limit 100 --tasks hellaswag

# Step 3: Rotation + Quantization 快速验证（~15分钟）
bash run_rotation_scheme_matrix.sh quick

# Step 4: 完整精度矩阵（~2小时）
bash run_rotation_scheme_matrix.sh full

# Step 5: RTN vs Tuning 对比（并行，~4-6小时）
python test_rotation_scheme_matrix.py --device cuda:4 --quant-iters 0 \
    --output-dir results_rtn &
python test_rotation_scheme_matrix.py --device cuda:5 --quant-iters 200 \
    --output-dir results_tuning &
```

### 8.2 只测特定组合

```bash
# 只测 R1+R2 + W4A16 (最常用配置)
python test_rotation_scheme_matrix.py --device cuda:7 \
    --rotations "R1+R2" --schemes "W4A16" --limit 100

# 测试 random vs deterministic Hadamard
python test_rotation_scheme_matrix.py --device cuda:7 \
    --rotations "R1,R1+R2" --schemes "W4A16" --random-hadamard

# 测试 block rotation (for non-power-of-2 models)
python test_rotation_scheme_matrix.py --device cuda:7 \
    --rotation-size 128 --rotations "R1,R1+R2+R3+R4" --schemes "MXFP4"
```

### 8.3 三框架对比

```bash
# 需要同时安装 auto-round, Quark, llm-compressor
python test_three_way_comparison.py --device cuda:7 --limit 100
```

---

## 附录: 依赖关系

| 依赖 | 安装路径 | 需要的测试 |
|------|---------|-----------|
| auto-round | `/data/lkk/quarot/auto-round` (editable) | 所有 |
| Quark | `/data/lkk/quarot/Quark` | test_quark_comparison, test_quark_rotation_mxfp4, test_three_way_comparison |
| llm-compressor | `/data/lkk/quarot/llm-compressor` (editable) | test_three_way_comparison |
| lm_eval | pip install | 所有 Level 2+ 测试 |
| transformers | pip install | 所有 Level 2+ 测试 |

## 附录: 功能覆盖状态

| 功能 | 覆盖测试 | 状态 |
|------|---------|------|
| Rotation 工具函数 | test_reference_equivalence | ✅ 通过 |
| R1 旋转正确性 | test_rotation_levels, test_qwen3_rotation_eval | ✅ 通过 |
| R2 旋转正确性 | test_rotation_levels | ✅ 通过 |
| R3 旋转正确性 | test_rotation_levels | ✅ 通过 |
| R4 旋转正确性 | test_rotation_levels | ✅ 通过 |
| Quark 对齐 | test_quark_comparison | ✅ 通过 |
| Online R1 | test_qwen3_rotation_eval | ✅ 通过 |
| Offline R1 | test_rotation_levels | ✅ 通过 |
| Deterministic Hadamard | test_rotation_levels | ✅ 通过 |
| Random Hadamard | test_rotation_levels | ✅ 通过 |
| Block rotation (rotation_size) | test_rotation_levels | ✅ 通过 |
| Rotation + W4A16 | test_rotation_scheme_matrix | ✅ 通过 |
| Rotation + MXFP4 | test_rotation_scheme_matrix | ✅ 通过 |
| Rotation + NVFP4 | test_rotation_scheme_matrix | ✅ 通过 |
| SpinQuant 训练 | test_spinquant (mock) | ⚠️ 仅 mock，未在真实模型验证 |
| 模型 save/load | test_wrapper_save_load | ⚠️ 仅 wrapper，完整 save/load 未实现 |
| 三框架对比 | test_three_way_comparison | ✅ 通过 |
