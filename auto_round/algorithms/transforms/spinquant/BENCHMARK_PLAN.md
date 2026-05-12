# Benchmark 运行规划

本文档描述完整的 rotation × quantization benchmark 测试规划，
包含运行命令、优先级、预期耗时和结果分析方法。

脚本位置：`examples/run_benchmark.sh`、`examples/test_save_load_roundtrip.py`

---

## 1. 测试矩阵总览

### 1.1 测试维度

| 维度 | 取值 |
|------|------|
| **模型** | Qwen/Qwen3-0.6B, Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct |
| **Rotation** | none, R1, R1+R2, R1+R2+R3, R1+R2+R3+R4 |
| **量化方案** | W4A16 (INT4), MXFP4 (MX-FP4), NVFP4 (NV-FP4), FP16 (no quant) |
| **量化模式** | RTN (iters=0), Tuning (iters=200) |
| **Rotation size** | 16, 32, 128, auto |
| **Rotation mode** | hadamard (确定性), random (随机), per-rotation random |
| **评测任务** | hellaswag, piqa, winogrande, lambada_openai, wikitext (ppl) |
| **Random seed** | 42 (固定, 保证可复现) |

### 1.2 Parts 分布

| Part | 内容 | 约 combo/模型 | 评测指标 |
|------|------|--------------|---------|
| 1 | 确定性 Hadamard × RTN | 15 | accuracy + roundtrip |
| 2 | 确定性 Hadamard × tuning(200) | 15 | accuracy + roundtrip |
| 3 | Rotation size sweep × RTN | 48 | accuracy + roundtrip |
| 4 | Rotation size sweep × tuning | 48 | accuracy + roundtrip |
| 5 | Random(all) × RTN | 12 | accuracy + roundtrip |
| 6 | Random(all) × tuning | 12 | accuracy + roundtrip |
| 7 | Per-rotation random × RTN | 8 | accuracy + roundtrip |
| 8 | Per-rotation random × tuning | 4 | accuracy + roundtrip |
| 9 | FP16 baseline (无量化) | 5 | accuracy (天花板) |
| 10 | Size × random × RTN | 12 | accuracy + roundtrip |
| 11 | FP16 + rotation (无损验证) | 9 | accuracy (rotation 无损?) |
| 12 | Wikitext perplexity | 10 | perplexity |

---

## 2. 运行规划（4 张 GPU）

### 2.1 Wave 0 (P0)：建立对照基线

**目标**：获得 FP16 精度天花板、验证 rotation 无损性、wikitext perplexity 基线

**预计耗时**：~30 分钟

```bash
cd /data/lkk/quarot/latest/commit/auto-round/examples

bash run_benchmark.sh --gpus 4 --wait \
  --models "Qwen/Qwen3-0.6B" \
  --parts 9,11,12
```

**涉及 Parts**：
- **Part 9** — FP16 baseline：5 种 rotation level 的纯 FP16 精度
- **Part 11** — FP16 + rotation：验证 rotation 对 FP16 模型是否真正无损
  - 11a: hadamard rotation（应与 FP16 baseline 相同）
  - 11b: random rotation（应与 FP16 baseline 相同，因为数学等价）
- **Part 12** — Wikitext perplexity：FP16 / W4A16 / MXFP4 / NVFP4 的 perplexity 基线

**得到的分析**：

| 对比 | 回答的问题 |
|------|-----------|
| Part 9: none vs R1 vs R1+R2 ... (FP16) | rotation 在无量化下是否严格无损？ |
| Part 11a vs Part 9 | hadamard rotation 无损验证 |
| Part 11b vs Part 9 | random rotation 无损验证 |
| Part 12: FP16 ppl vs W4A16 ppl | 量化引入多少 perplexity 劣化？ |

**预期结果**：Part 9 / 11a / 11b 的 accuracy 应完全相同（rotation 数学等价）。若不同，说明实现有 bug。

---

### 2.2 Wave 1 (P1)：核心量化矩阵

**目标**：量化后精度的全面对比——rotation 水平、量化方案、RTN vs tuning

**预计耗时**：~2-3 小时

```bash
bash run_benchmark.sh --gpus 4 --wait \
  --models "Qwen/Qwen3-0.6B" \
  --parts 1,2
```

**涉及 Parts**：
- **Part 1** — 确定性 Hadamard × RTN (15 combos)
- **Part 2** — 确定性 Hadamard × tuning (15 combos)

**得到的分析**：

| 对比 | 回答的问题 |
|------|-----------|
| Part 9 (FP16) vs Part 1 (RTN) | 量化掉了多少精度？ |
| Part 1: none vs R1 vs R1+R2 ... | rotation 恢复了多少精度？递增关系？ |
| Part 1 vs Part 2 (RTN vs tune) | tuning 的额外增益有多大？ |
| Part 1: W4A16 vs MXFP4 vs NVFP4 | 不同量化方案在同一 rotation 下的表现 |
| 所有 combo 的 roundtrip match | save/load 序列化正确性 |

**预期结果**：
- rotation 增加应单调提升或持平（R1+R2+R3+R4 ≥ R1+R2 ≥ R1 ≥ none）
- tuning 应优于 RTN
- 所有 roundtrip 应 match

---

### 2.3 Wave 2 (P2)：随机旋转

**目标**：验证 random rotation 的精度影响和 save/load 正确性

**预计耗时**：~2 小时

```bash
bash run_benchmark.sh --gpus 4 --wait \
  --models "Qwen/Qwen3-0.6B" \
  --parts 5,7
```

**涉及 Parts**：
- **Part 5** — random(all) × RTN (12 combos)
- **Part 7** — per-rotation random × RTN (8 combos)

**得到的分析**：

| 对比 | 回答的问题 |
|------|-----------|
| Part 1 (hadamard) vs Part 5 (random) | random 比 hadamard 有额外增益吗？ |
| Part 7a: 只 R2 random vs Part 7d: 全 random | 哪些位置适合 random？ |
| Part 7b: R2+R4 random vs Part 7c: R1+R3 random | 不同 random 组合对比 |
| 所有 random combo 的 roundtrip match | random 矩阵 save/load 正确性 |

**预期结果**：random 精度应与 hadamard 接近（因为正交矩阵本质等价），但 random 可能在特定量化方案下有轻微差异。

---

### 2.4 Wave 3 (P3)：Rotation size 影响

**目标**：不同 block rotation 尺寸对精度的影响

**预计耗时**：~3-4 小时

```bash
bash run_benchmark.sh --gpus 4 --wait \
  --models "Qwen/Qwen3-0.6B" \
  --parts 3,10
```

**涉及 Parts**：
- **Part 3** — size sweep × RTN (48 combos)
- **Part 10** — size × random × RTN (12 combos)

**得到的分析**：

| 对比 | 回答的问题 |
|------|-----------|
| size=16 vs 32 vs 128 vs auto | 更小的 block 牺牲多少精度？ |
| size=16(快) vs auto(慢) | 速度 vs 精度权衡 |
| size sweep × random | 小 block + random 组合效果 |

---

### 2.5 Wave 4 (P4)：大模型泛化验证

**目标**：验证结论是否在 8B 参数模型上泛化

**预计耗时**：~8-12 小时

```bash
bash run_benchmark.sh --gpus 4 --wait \
  --models "Qwen/Qwen3-8B,meta-llama/Llama-3.1-8B-Instruct" \
  --parts 1,9,12
```

**涉及 Parts**：
- **Part 1** — 确定性 × RTN（核心矩阵）
- **Part 9** — FP16 baseline
- **Part 12** — Wikitext perplexity

**得到的分析**：

| 对比 | 回答的问题 |
|------|-----------|
| 0.6B vs 8B 趋势对比 | rotation 效果是否随模型变大增强/减弱？ |
| Qwen3 vs Llama-3.1 | 不同架构（GQA 配置不同）的影响 |
| 8B wikitext ppl | 大模型的 perplexity 绝对值 |

---

## 3. 快速运行（只验证核心）

如果时间有限，只需跑 P0 + P1 即可得到 80% 的分析价值：

```bash
# 一条命令，约 3 小时
bash run_benchmark.sh --gpus 4 --wait \
  --models "Qwen/Qwen3-0.6B" \
  --parts 1,2,9,11,12
```

---

## 4. 结果分析指南

### 4.1 结果文件

运行完成后，结果位于 `results_benchmark_<timestamp>/`：

```
results_benchmark_<timestamp>/
├── summary.json          # 所有实验结果合并
├── summary.csv           # CSV 格式（可导入 Excel/Sheets）
├── logs/                 # 每个 job 的完整日志
│   ├── Qwen3-0.6B_part1_det_rtn.log
│   ├── Qwen3-0.6B_part9_fp16_baseline.log
│   └── ...
└── Qwen3-0.6B_part1_det_rtn/
    └── results.json      # 单 job 结果
```

### 4.2 核心分析表格

运行后自动打印汇总表。手动分析建议做以下对比表：

**表 1: Rotation 增益分析**（对齐 Part 9 baseline）

| Rotation | FP16 (P9) | W4A16 RTN (P1) | 量化损失 | W4A16 tune (P2) | tune 增益 |
|----------|-----------|----------------|---------|-----------------|----------|
| none     | 0.xxxx    | 0.xxxx         | -0.xxxx | 0.xxxx          | +0.xxxx  |
| R1       | 0.xxxx    | 0.xxxx         | -0.xxxx | 0.xxxx          | +0.xxxx  |
| R1+R2    | 0.xxxx    | 0.xxxx         | -0.xxxx | 0.xxxx          | +0.xxxx  |
| ...      | ...       | ...            | ...     | ...             | ...      |

**表 2: Random vs Hadamard**（P1 vs P5）

| Rotation | Hadamard (P1) | Random (P5) | 差异 |
|----------|-------------|-------------|------|
| R1       | 0.xxxx      | 0.xxxx      | ±0.x |
| R1+R2    | 0.xxxx      | 0.xxxx      | ±0.x |
| ...      | ...         | ...         | ...  |

**表 3: Perplexity (P12)**

| Rotation | FP16 ppl | W4A16 ppl | MXFP4 ppl | NVFP4 ppl |
|----------|----------|-----------|-----------|-----------|
| none     | xx.xx    | xx.xx     | xx.xx     | xx.xx     |
| R1+R2    | xx.xx    | xx.xx     | xx.xx     | xx.xx     |
| ...      | ...      | ...       | ...       | ...       |

### 4.3 关键指标解读

- **Accuracy**: hellaswag/piqa/winogrande/lambada_openai 的 acc 或 acc_norm
- **Perplexity**: wikitext 的 word_perplexity（越低越好）
- **Roundtrip Match**: disk eval ≈ in-memory eval（差 < 1e-4 为 PASS）
- **Rotation 无损性**: FP16 + rotation ≈ FP16（Part 11 vs Part 9）

### 4.4 异常情况处理

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| FP16+rotation ≠ FP16 baseline | rotation 实现 bug | 查 Part 11 vs Part 9 差异 |
| roundtrip mismatch | 序列化 bug | 查 log 中 spinquant_keys 和 warning |
| random 精度远低于 hadamard | random 矩阵 save/load bug | 固定 seed 复现, 查 Part 5/7 日志 |
| R1+R2+R3+R4 < R1+R2 | R3/R4 hook 实现错误 | 单独跑 R1+R2+R3 隔离 |
| tuning 无提升或下降 | tuning 参数问题 | 增加 iters 或检查 lr |

---

## 5. 新增功能说明

### 5.1 FP16 Baseline (scheme="FP16")

`test_save_load_roundtrip.py` 新增 `FP16` scheme：
- 不做量化，只应用 rotation（如有）
- 直接在 FP16 模型上跑 lm_eval
- 用于建立精度天花板和验证 rotation 无损性

```bash
python3 test_save_load_roundtrip.py \
  --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
  --schemes "FP16" --skip-inmemory
```

### 5.2 Wikitext Perplexity

评测函数现在自动提取 perplexity 指标（`word_perplexity` 或 `byte_perplexity`）。
使用 `--tasks "wikitext"` 即可：

```bash
python3 test_save_load_roundtrip.py \
  --rotations "none,R1+R2" --schemes "FP16,W4A16" \
  --tasks "wikitext" --skip-inmemory
```

结果中 perplexity 以 `<task>_ppl` key 保存，如 `"wikitext_ppl": 23.4567`。

### 5.3 Random Seed

`--seed 42` 设置 `torch.manual_seed` / `numpy.random.seed` / `random.seed`，
保证 random rotation 矩阵可复现。`run_benchmark.sh` 默认 seed=42。

```bash
python3 test_save_load_roundtrip.py \
  --rotation-modes "random" --seed 42
```
