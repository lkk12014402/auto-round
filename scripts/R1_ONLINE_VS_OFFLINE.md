# R1 Online vs Offline Rotation 对比文档

本文档说明 auto-round 中 R1 在线（online）与离线（offline）两种模式的区别，以及与 llm-compressor / Quark 的对比。

---

## 目录

- [概念：Online vs Offline](#概念online-vs-offline)
- [各框架 R1 模式](#各框架-r1-模式)
- [Auto-Round Offline R1 支持的组合](#auto-round-offline-r1-支持的组合)
- [代码路径分析](#代码路径分析)
- [Layer-wise Rotation 限制](#layer-wise-rotation-限制)
- [三方对比测试用法](#三方对比测试用法)
- [预期结果差异](#预期结果差异)

---

## 概念：Online vs Offline

### R1 Online（在线）

R1 rotation 作为 **runtime hook** 在推理时执行。模型权重只做部分旋转，hook 在前向传播中补全变换。

```
Forward pass: x → [R1 hook: x @ R1] → layer(rotated_x) → ...
```

- 权重：q/k/v_proj 旋转了 in-channel，o_proj/down_proj 旋转了 out-channel
- `embed_tokens`、`RMSNorm`：**不修改**
- 推理时需要 hook 存在才能正确运行

### R1 Offline（离线）

R1 rotation **永久融合到权重中**。模型是一个完全独立的变换后模型，不需要任何 hook。

```
Forward pass: x → layer(x)    # 权重已经包含了 R1 变换
```

- `embed_tokens`：`W_embed @ R1`
- `RMSNorm`：gamma 融合到 linear 权重中
- `q/k/v_proj`：`W @ R1_inv`（in-channel 旋转）
- `o_proj / down_proj`：`R1 @ W`（out-channel 旋转）
- `lm_head`：`W @ R1_inv`（in-channel 旋转）
- 推理时不需要 hook，模型是 standalone 的

---

## 各框架 R1 模式

| 框架 | R1 模式 | 说明 |
|------|---------|------|
| **auto-round**（默认） | Online | `online_r1_rotation=True`，runtime hook |
| **auto-round**（offline） | Offline | `online_r1_rotation=False`，融合到权重 |
| **Quark**（默认） | Online | `online_r1_rotation=True`，与 auto-round 一致 |
| **llm-compressor** | Offline（仅） | R1/R2 始终 offline，文档明确说明 |

> llm-compressor 源码注释："R1 and R2 are 'offline' rotations, meaning that they can be fused into existing weights and therefore do not induce runtime cost."

---

## Auto-Round Offline R1 支持的组合

**所有 R1/R2/R3/R4 组合在 offline R1 模式下都支持。**

| 组合 | R1 | R2 | R3 | R4 | 说明 |
|------|----|----|----|----|------|
| R1 | offline 融合 | — | — | — | embed + norm + linear 全部融合 |
| R1+R2 | offline 融合 | offline 融合 | — | — | R2 per-head 融合到 v_proj/o_proj |
| R1+R2+R3 | offline 融合 | offline 融合 | online hook | — | R3 是 Q/K post-RoPE，始终 online |
| R1+R2+R4 | offline 融合 | offline 融合 | — | offline 融合 + online hook | R4 融合到 down_proj + hook |
| R1+R2+R3+R4 | offline 融合 | offline 融合 | online hook | offline 融合 + online hook | 完整四级旋转 |

**注意**：R3 和 R4 始终包含 online 部分（runtime hooks），与 R1 的 online/offline 选择无关：
- R3：post-RoPE 的 Q/K rotation，必须在 attention 计算时在线执行
- R4：activation rotation（`x @ R4` 在 down_proj 之前），需要 online hook

---

## 代码路径分析

### 入口：`preprocessor.py` Step 6

```python
# Step 6: apply R1 rotation
if self.config.r1 and self.config.online_r1_rotation:
    # ── Online R1 路径 ──
    self._apply_online_r1()       # 部分权重旋转 + 注册 R1 hook
    self._fuse_r2_rotation()      # R2 照常 offline 融合
    self._fuse_r4_rotation()      # R4 照常 offline 融合
else:
    # ── Offline R1 路径 ──
    self._fuse_offline_rotations()  # R1 + R2 + R4 全部 offline 融合

# Step 7: R3/R4 online hooks（两种路径都会执行）
if self.config.r3 or self.config.r4:
    register_spinquant_hooks(...)   # R3/R4 runtime hooks
```

### `_fuse_offline_rotations()` 详细流程

```
1. embed_tokens.weight = W_embed @ R1                    # 嵌入层旋转
2. 对每个 transformer layer:
   a. q/k/v_proj:  rotate_in_channels_(W, R1_inv)       # W_new = R1_inv @ W
   b. o_proj:      rotate_out_channels_(W, R1)           # W_new = W @ R1
   c. gate/up_proj: rotate_in_channels_(W, R1_inv)
   d. down_proj:   rotate_out_channels_(W, R1)
3. lm_head.weight: rotate_in_channels_(W, R1_inv)        # 输出层旋转
4. _fuse_r2_rotation()                                    # per-head v/o 旋转
5. _fuse_r4_rotation()                                    # down_proj input 旋转
```

### `_apply_online_r1()` 详细流程

```
1. 对每个 transformer layer:
   a. q/k/v_proj: rotate_in_channels_(W, R1_inv)         # 权重部分旋转
   b. o_proj:     rotate_out_channels_(W, R1)
   c. gate/up_proj: rotate_in_channels_(W, R1_inv)
   d. down_proj:  rotate_out_channels_(W, R1)
2. 注册 R1OnlineRotationWrapper 到第一层                    # hook 处理 embed 输出
3. embed_tokens 和 RMSNorm 不修改
```

### 核心区别

| 操作 | Online R1 | Offline R1 |
|------|-----------|------------|
| `embed_tokens` | 不修改 | `W @ R1` |
| `RMSNorm` | 不修改 | gamma 融合到 linear |
| `lm_head` | 不修改 | `rotate_in_channels_` |
| q/k/v/o_proj | 旋转 ✅ | 旋转 ✅ |
| gate/up/down_proj | 旋转 ✅ | 旋转 ✅ |
| Runtime hook | 需要 R1 hook | 不需要 |
| 模型 standalone | ❌ 依赖 hook | ✅ 独立可用 |

---

## Layer-wise Rotation 限制

**Layer-wise rotation（`--layerwise`）不支持 offline R1。**

原因：offline R1 修改 `embed_tokens`，改变了层间隐状态空间。layer-wise 模式先缓存 block 输入再逐块处理，而 offline R1 改变了 embed 的输出维度旋转，导致缓存的输入与后续旋转后的权重不兼容。

```python
# preprocessor.py: prepare() 方法中的校验
if self.config.r1 and not self.config.online_r1_rotation:
    raise ValueError(
        "Layer-wise rotation requires online_r1_rotation=True. "
        "Offline R1 changes inter-layer hidden state space..."
    )
```

**解决方案**：
- 使用 `--layerwise` 时，自动使用 online R1（默认行为）
- 需要 offline R1 时，使用 full-model rotation（不加 `--layerwise`）

---

## 三方对比测试用法

### 对比 auto-round offline 与 llm-compressor（推荐的 apples-to-apples 对比）

```bash
# 只比 R1
python test_three_way_comparison.py --device cuda:4 \
    --frameworks autoround-offline,llmcompressor \
    --rotations R1

# 所有 rotation 级别
python test_three_way_comparison.py --device cuda:4 \
    --frameworks autoround-offline,llmcompressor \
    --rotations R1,R1+R2,R1+R2+R3+R4
```

### auto-round 自身 online vs offline 对比

```bash
python test_three_way_comparison.py --device cuda:4 \
    --frameworks autoround,autoround-offline \
    --rotations R1,R1+R2,R1+R2+R3+R4
```

### 四个框架全量对比

```bash
python test_three_way_comparison.py --device cuda:4 \
    --frameworks autoround,autoround-offline,quark,llmcompressor \
    --rotations R1,R1+R2,R1+R2+R4,R1+R2+R3+R4
```

### 快速验证（limit=100）

```bash
python test_three_way_comparison.py --device cuda:4 --limit 100 \
    --frameworks autoround,autoround-offline,llmcompressor
```

### 包含 MXFP4-only baseline

```bash
python test_three_way_comparison.py --device cuda:4 \
    --frameworks autoround-offline,llmcompressor \
    --include-baselines
```

---

## 预期结果差异

### Online vs Offline R1 精度

理论上 **完全一致**：online 和 offline 只是实现方式不同，数学等价。

```
Online:  x → x @ R1 → (R1_inv @ W) → ...
Offline: x → (embed @ R1) → (R1_inv @ W_fused) → ...
```

实际中可能有 **< 1bp 的微小差异**，原因：
- 浮点运算顺序不同（`(x @ R1) @ W` vs `x @ (R1 @ W)`）
- RMSNorm gamma 融合引入额外精度损失

### auto-round offline vs llm-compressor

预期 **高度接近但不完全相同**，原因：
- 两者都做 R1 offline 融合，数学等价
- MXFP4 量化实现可能有细微差异（scale 计算、rounding）
- 量化 ignore 列表可能不同（auto-round vs llm-compressor 的 `lm_head` 处理）

### 预期对比表（示意）

```
── piqa ───────────────────────────────────────────────
  Rotation Level       | autoround       | autoround-offli | llmcompressor
  ─────────────────────────────────────────────────────
  R1                   | 0.7612          | 0.7612          | 0.7609
  R1+R2                | 0.7634          | 0.7634          | 0.7631
  R1+R2+R3+R4          | 0.7645          | 0.7645          | 0.7642
```

- `autoround` vs `autoround-offline`：精度差异 < 1bp（数学等价）
- `autoround-offline` vs `llmcompressor`：精度差异 < 5bp（实现差异）
