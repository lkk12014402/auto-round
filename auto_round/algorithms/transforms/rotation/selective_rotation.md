# Selective Hadamard Rotation — 设计文档

## 概述

**Selective Hadamard Rotation** 是对 auto-round 现有 Hadamard 旋转机制的增强。核心思想：
**不是对所有 Linear 层统一应用旋转，而是基于结构先验和/或激活统计量，逐层决策是否旋转。**

传统方式（对所有层施加 Hadamard）在大多数 expand 层和 attention projection 层有效，但在
residual-sensitive 层（如 `down_proj`、`o_proj`）上反而会损害精度。

---

## 原理

### 为什么不能无差别旋转？

Hadamard 旋转通过正交变换将权重/激活的 outlier 分散到所有通道，降低量化误差。但：

| 层类型 | Hadamard 效果 | 原因 |
|--------|-------------|------|
| `up_proj`, `gate_proj` | ✅ 正面 | expand 层 + 后有非线性激活缓冲 |
| `q_proj`, `k_proj` | ✅ 正面 | attention expand，旋转后 outlier 均匀分布 |
| `down_proj` | ❌ 负面 | compress 层，直接写入 residual stream，旋转打乱语义对齐 |
| `o_proj` | ❌ 负面 | 输出直接加到 residual，no nonlinearity buffer |
| `v_proj` | ⚠️ 不确定 | 既不 expand 也不 compress，效果取决于具体分布 |
| `lm_head` | ❌ 永远跳过 | embedding alignment，不能旋转 |

### 量化评分公式（auto 模式）

对每一层计算一个 "Hadamard benefit score"：

```
score = 0.0
if is_expand:              score += 1.0     # out_features > in_features
if is_compress:            score -= 1.0     # out_features < in_features
if is_residual_consumer:   score -= 1.5     # down_proj / o_proj
if has_nonlinearity_after: score += 0.5     # up_proj / gate_proj
if kurtosis > threshold:   score += 1.0     # 激活 heavy-tail (有 outlier)
if kurtosis < 3.0:         score -= 0.5     # 接近高斯分布，旋转没意义
if ecr > threshold:        score += 1.0     # 能量集中度高 (单通道 outlier)
```

**判定**: `score >= score_threshold (default=1.5)` → 旋转，否则跳过。

### 两个统计量

- **Kurtosis（峰度）**: 衡量分布的重尾程度。excess kurtosis > 5 表示有严重 outlier。
- **ECR（Energy Concentration Ratio）**: `max_channel_energy / total_energy`。ECR > 0.05 表示单通道集中了过多能量（outlier 通道）。

---

## 修改了哪些代码

### 1. 新增文件

| 文件 | 作用 |
|------|------|
| `auto_round/algorithms/transforms/rotation/selective.py` | **核心实现**：LayerSelector 类、structural_decision()、compute_layer_score()、_profile_activations() |

### 2. 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `auto_round/algorithms/transforms/rotation/config.py` | 新增 6 个配置字段：`layer_selection`、`include_layers`、`exclude_layers`、`kurtosis_threshold`、`ecr_threshold`、`score_threshold` |
| `auto_round/algorithms/transforms/rotation/apply.py` | 在 `HadamardRotation.apply_to_model()` 中集成 LayerSelector，逐层决策 (lines 142-190) |

### 3. 测试脚本

| 文件 | 作用 |
|------|------|
| `/data/lkk/quarot/scripts/test_selective_rotation.py` | 独立对比测试：none / had_all / had_structural / had_auto / had_no_down / had_no_down_oproj |
| `/data/lkk/quarot/scripts/test_rotation_module_compare.py` | 已有脚本中新增了 `had_structural`、`had_auto`、`had_no_down` variants |

---

## 代码结构详解

### selective.py 核心组件

```python
# 1. 结构先验 pattern 列表
STRUCTURAL_SKIP_PATTERNS = ["*lm_head*", "*down_proj*", "*o_proj*"]
STRUCTURAL_PREFER_PATTERNS = ["*up_proj*", "*gate_proj*", "*q_proj*", "*k_proj*"]
STRUCTURAL_CONDITIONAL_PATTERNS = ["*v_proj*"]

# 2. 决策记录
@dataclass
class LayerDecision:
    name: str           # 层名
    enabled: bool       # 是否旋转
    reason: str         # 决策原因
    score: float        # 评分（auto 模式）
    kurtosis: float     # 峰度（auto 模式）
    ecr: float          # 能量集中度（auto 模式）

# 3. 选择器（主入口）
class LayerSelector:
    mode: str           # "all" / "structural" / "auto"
    include_layers: list[str]   # 强制包含的 fnmatch pattern
    exclude_layers: list[str]   # 强制排除的 fnmatch pattern
    
    def should_rotate(name, module) -> bool: ...
    def profile(model, dataloader) -> dict: ...   # 仅 auto 模式
    def get_decisions() -> dict[str, LayerDecision]: ...
    def summary() -> str: ...
```

### apply.py 集成点

```python
# 在 HadamardRotation.apply_to_model() 中：

# 1. 创建 selector
selector = LayerSelector.from_config(cfg)

# 2. (auto 模式) 做 activation profiling
if cfg.layer_selection == "auto":
    selector.profile(model, calibration_dataloader)

# 3. 逐层判定
for name, module in model.named_modules():
    if "lm_head" in name:
        continue
    if not selector.should_rotate(name, module):
        skipped_count += 1
        continue
    _apply_to_module(model, module, cfg, ...)
    applied_count += 1

# 4. 存储决策到 model（可供后续 debug 用）
model.rotation_decisions = selector.get_decisions()
```

### config.py 新增字段

```python
class RotationConfig(BaseModel):
    # ... 已有字段 ...
    
    # ---- 新增 selective rotation 字段 ----
    layer_selection: str = "all"           # "all" | "structural" | "auto"
    include_layers: list[str] | None       # 强制 include fnmatch patterns
    exclude_layers: list[str] | None       # 强制 exclude fnmatch patterns
    kurtosis_threshold: float = 5.0        # auto 模式的峰度阈值
    ecr_threshold: float = 0.05            # auto 模式的 ECR 阈值
    score_threshold: float = 1.5           # auto 模式的总分阈值
```

---

## 使用方法

### 方式 1：通过 AutoRound API（推荐）

```python
from auto_round import AutoRound

# Structural 模式 — 零开销，基于层名规则
autoround = AutoRound(
    model,
    tokenizer=tokenizer,
    rotation_config={
        "hadamard_type": "hadamard",
        "backend": "transform",
        "layer_selection": "structural",    # ← 启用 structural 选择
    },
    scheme="MXFP4",
)
autoround.quantize()
```

```python
# Auto 模式 — 需要 calibration data
autoround = AutoRound(
    model,
    tokenizer=tokenizer,
    rotation_config={
        "hadamard_type": "hadamard",
        "backend": "transform",
        "layer_selection": "auto",          # ← 启用 auto 选择
        "kurtosis_threshold": 5.0,
        "ecr_threshold": 0.05,
        "score_threshold": 1.5,
    },
    scheme="MXFP4",
)
autoround.quantize()
```

```python
# 自定义 include/exclude — 配合任何模式
autoround = AutoRound(
    model,
    tokenizer=tokenizer,
    rotation_config={
        "hadamard_type": "hadamard",
        "backend": "transform",
        "layer_selection": "all",
        "exclude_layers": ["*down_proj*", "*o_proj*"],  # 手动排除
    },
    scheme="MXFP4",
)
autoround.quantize()
```

### 方式 2：直接调用底层 API

```python
from auto_round.algorithms.transforms.rotation.config import RotationConfig
from auto_round.algorithms.transforms.rotation.apply import HadamardRotation

cfg = RotationConfig(
    layer_selection="structural",
    block_size=32,
)
rotation = HadamardRotation(cfg)
rotation.apply_to_model(model, data_type="mx_fp")

# 查看决策
if hasattr(model, "rotation_decisions"):
    for name, d in model.rotation_decisions.items():
        print(f"{name}: {'✅' if d.enabled else '❌'} {d.reason}")
```

### 方式 3：使用 LayerSelector 单独做分析

```python
from auto_round.algorithms.transforms.rotation.selective import LayerSelector

selector = LayerSelector(mode="structural")

# 查询单层
should = selector.should_rotate("model.layers.0.mlp.down_proj")
print(should)  # False

should = selector.should_rotate("model.layers.0.mlp.up_proj")
print(should)  # True

# 打印全部决策
print(selector.summary())
```

---

## 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `layer_selection` | str | `"all"` | 选择模式：`"all"` = 所有层（向后兼容）；`"structural"` = 结构先验；`"auto"` = 统计+结构 |
| `include_layers` | list[str]\|None | None | fnmatch pattern 列表，匹配的层**强制旋转**（优先级最高） |
| `exclude_layers` | list[str]\|None | None | fnmatch pattern 列表，匹配的层**强制跳过**（优先级最高） |
| `kurtosis_threshold` | float | 5.0 | (auto) 峰度超过此值表示 heavy-tail，加分 |
| `ecr_threshold` | float | 0.05 | (auto) ECR 超过此值表示 outlier 集中，加分 |
| `score_threshold` | float | 1.5 | (auto) 总分超过此值才旋转 |

### 优先级

```
exclude_layers (最高) > include_layers > mode-specific logic (最低)
```

即使 `layer_selection="all"`，`exclude_layers` 中匹配的层也会被跳过。

---

## 三种模式对比

| 模式 | 开销 | 需要 calibration data? | 决策质量 | 适用场景 |
|------|------|----------------------|---------|---------|
| `"all"` | 零 | 否 | — | 向后兼容，与现有行为一致 |
| `"structural"` | 零 | 否 | 中等 | 生产推荐默认；无 calibration 但效果明显 |
| `"auto"` | 一次 forward pass (32 samples) | 是 | 最佳 | 追求最优精度-速度平衡 |

### Structural 模式的默认决策

| 层名 Pattern | 决策 | 原因 |
|-------------|------|------|
| `*up_proj*` | ✅ 旋转 | expand + nonlinearity buffer |
| `*gate_proj*` | ✅ 旋转 | expand + nonlinearity buffer |
| `*q_proj*` | ✅ 旋转 | expand in attention |
| `*k_proj*` | ✅ 旋转 | expand in attention |
| `*down_proj*` | ❌ 跳过 | residual-sensitive, compress |
| `*o_proj*` | ❌ 跳过 | residual-sensitive |
| `*v_proj*` | ❌ 跳过 | 保守策略（无明确信号） |
| `*lm_head*` | ❌ 跳过 | embedding 对齐，不可旋转 |
| `*mlp.gate` | ❌ 跳过 | MoE router，参数极少，精度敏感 |
| `*block_sparse_moe.gate` | ❌ 跳过 | MoE router（Mixtral style） |
| `*shared_expert_gate*` | ❌ 跳过 | 共享 expert gate，不量化 |
| `*embed_tokens*` | ❌ 跳过 | embedding，不量化 |

---

## MoE 模型支持

### 核心结论

**Selective per-linear Hadamard 与模型结构无关，MoE 模型可以直接使用。**

原理：我们遍历模型的所有 `nn.Linear`，逐个判定是否旋转。MoE 模型的 expert 内部结构
（`gate_proj` / `up_proj` / `down_proj`）与 dense 模型完全一致，structural 规则直接适用。

### MoE 特殊层的处理

MoE 模型有些层 **不做量化**（保持 fp16/bf16），对它们做旋转没有意义——旋转只在后续有
量化时才有收益。这些层应当同时被 rotation 跳过：

| MoE 层 | 量化中 ignore | 旋转中 skip | 原因 |
|--------|-------------|------------|------|
| `mlp.gate` | ✅ | ✅ | router，参数极少，精度敏感 |
| `shared_expert_gate` | ✅ | ✅ | 门控标量 |
| `embed_tokens` | ✅ | ✅ | embedding |
| `lm_head` | ✅ | ✅ | output head |
| `experts.*.gate_proj` | ❌ 量化 | ✅ 旋转 | expert 内部 expand 层，旋转有益 |
| `experts.*.up_proj` | ❌ 量化 | ✅ 旋转 | expert 内部 expand 层，旋转有益 |
| `experts.*.down_proj` | ❌ 量化 | ❌ 跳过 | residual consumer |

**注意**：`structural` 模式已内置 MoE router/gate 的 skip pattern，无需额外配置。

### MoE 模型使用示例

```python
from auto_round import AutoRound

# Qwen3-30B-A3B MoE 模型
ar = AutoRound(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    scheme="MXFP4",
    iters=0,
    device_map="auto",
    low_gpu_mem_usage=True,
    # 量化中跳过的层
    ignore_layers="lm_head,mlp.gate,shared_expert_gate,embed_tokens",
    # selective rotation 自动处理 MoE 结构
    rotation_config={
        "hadamard_type": "hadamard",
        "backend": "transform",
        "layer_selection": "structural",
    },
)
ar.quantize_and_save(output_dir="./output", format="llm_compressor")
```

### MoE 测试脚本

```bash
cd /data/lkk/quarot/scripts

# MoE 快速测试（--moe 自动设置 ignore_layers 和 variant 集）
python test_selective_rotation.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --moe \
    --device auto \
    --low-gpu-mem \
    --limit 200

# 手动指定 ignore_layers（匹配你的量化脚本）
python test_selective_rotation.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --device auto \
    --ignore-layers "lm_head,mlp.gate,shared_expert_gate,embed_tokens,self_attn" \
    --variants none,had_all,had_structural,had_moe_match_quant \
    --limit 200

# 只用 0.6B 验证 MoE structural pattern（快速）
python test_selective_rotation.py \
    --device cuda:0 \
    --variants none,had_all,had_structural_moe \
    --limit 200
```

---

## 结果分析

### 查看决策报告

量化完成后，模型上会附带 `rotation_decisions` 属性：

```python
model = autoround.model

# 统计
decisions = model.rotation_decisions
enabled = sum(1 for d in decisions.values() if d.enabled)
total = len(decisions)
print(f"Rotated: {enabled}/{total} layers")

# 详细决策
for name, d in sorted(decisions.items()):
    status = "✅" if d.enabled else "❌"
    score_str = f" score={d.score:.2f}" if d.score else ""
    print(f"  {status} {name:<50} {d.reason}{score_str}")
```

### 示例输出 (Qwen3-0.6B, structural 模式)

```
Selective rotation (structural): applied=112, skipped=85 / 197 total

  ✅ model.layers.0.mlp.gate_proj        structural_prefer(*gate_proj*)
  ✅ model.layers.0.mlp.up_proj          structural_prefer(*up_proj*)
  ✅ model.layers.0.self_attn.q_proj     structural_prefer(*q_proj*)
  ✅ model.layers.0.self_attn.k_proj     structural_prefer(*k_proj*)
  ❌ model.layers.0.mlp.down_proj        structural_skip(*down_proj*)
  ❌ model.layers.0.self_attn.o_proj     structural_skip(*o_proj*)
  ❌ model.layers.0.self_attn.v_proj     structural_conditional(*v_proj*) → conservative_skip
```

### 精度验证

使用 lm_eval 对比：

```bash
# 运行对比测试脚本
python scripts/test_selective_rotation.py \
    --model Qwen/Qwen3-0.6B \
    --device cuda:0 \
    --scheme MXFP4 \
    --variants none,had_all,had_structural,had_no_down \
    --tasks piqa,hellaswag \
    --limit 500
```

**预期结果**：
- `had_structural` ≥ `had_all` 精度（跳过有害层）
- `had_no_down` 介于两者之间（只跳过 down_proj）
- `none` 最低（无旋转时量化损失最大）

### Auto 模式的 profiling 结果

```python
# 查看每层的统计量
from auto_round.algorithms.transforms.rotation.selective import LayerSelector

selector = LayerSelector(mode="auto", score_threshold=1.5)
stats = selector.profile(model, dataloader)

for name, s in sorted(stats.items())[:10]:
    print(f"  {name:<50} kurtosis={s['kurtosis']:.2f}  ecr={s['ecr']:.4f}")
```

输出示例：
```
  model.layers.0.mlp.up_proj                kurtosis=8.42  ecr=0.072   → 旋转
  model.layers.0.mlp.down_proj              kurtosis=3.21  ecr=0.012   → 跳过
  model.layers.0.self_attn.q_proj           kurtosis=6.15  ecr=0.053   → 旋转
```

---

## 与 SpinQuant R1 的关系

SpinQuant 的 R1 rotation 本身也是 selective 的（通过 `scaling_layers` 配置只选择部分 module），
但它是全局旋转（乘 Hadamard 矩阵到特定层的 weight），粒度较粗。

Selective Hadamard Rotation 是在 per-Linear transform 级别做选择，粒度更细：

```
SpinQuant R1: 全局 → 选择哪些 module 参与（粗粒度）
Selective Hadamard: 逐 Linear 层判定是否旋转（细粒度）
```

两者可以组合使用：先用 SpinQuant R1 做全局旋转，再用 Selective Hadamard 做 per-Linear 细化。

---

## 扩展性

### 自定义结构先验

修改 `selective.py` 中的 pattern 列表即可：

```python
# 例：对 MoE 模型的 gate 层也跳过
STRUCTURAL_SKIP_PATTERNS.append("*block_sparse_moe.gate*")
```

### 自定义评分函数

继承 `LayerSelector` 并覆盖 `_auto_decision()`：

```python
class MySelector(LayerSelector):
    def _auto_decision(self, name, module):
        # 自定义逻辑
        ...
```

### 添加新的统计量

在 `_profile_activations()` 的 hook 中添加新的计算（如 L2 范数、channel 方差比等）。

---

## 快速开始

```python
# 最简使用：一行配置启用 structural 选择
from auto_round import AutoRound

autoround = AutoRound(
    model, tokenizer=tokenizer,
    rotation_config={"layer_selection": "structural"},
    scheme="MXFP4",
)
autoround.quantize()
# → 自动跳过 down_proj/o_proj/v_proj，比 "all" 模式精度更高
```

---

## FAQ

**Q: structural 模式对性能有影响吗？**
A: 没有。仅在初始化时做 fnmatch 字符串匹配，零开销。

**Q: auto 模式的 profiling 需要多久？**
A: 32 个 calibration sample 的 forward pass，对 0.6B 模型约 2-3 秒。

**Q: 哪些 backend 支持 selective？**
A: 目前只有 `backend="transform"` 支持。`backend="inplace"` 是 QuaRot-style 全局旋转，粒度不在 per-Linear 级别。

**Q: exclude_layers 的 pattern 语法是什么？**
A: Python `fnmatch` 语法，支持 `*`（匹配任意字符）和 `?`（匹配单字符）。

**Q: 如果我想只跳过特定层号（如 layer 0-3）怎么配？**
A: 使用 `exclude_layers=["*layers.0.*", "*layers.1.*", "*layers.2.*", "*layers.3.*"]`。

---

## 日志系统

Selective rotation 内置了两级日志输出，帮助直观理解选择的有效性。

### 日志级别

| 级别 | 显示内容 | 适用场景 |
|------|---------|---------|
| `INFO`（默认） | 汇总统计 + per-type 分类表 + outlier 压缩分析 | 正常使用，快速确认选择合理性 |
| `DEBUG` | INFO 内容 + 每一层的详细决策（含 kurtosis/ECR/score breakdown） | 调试、调参、论文数据收集 |

### 环境变量控制

```bash
# 启用 DEBUG 级别（看每一层的详细决策）
export SELECTIVE_ROTATION_LOG_LEVEL=DEBUG

# 默认 INFO（只看汇总）
export SELECTIVE_ROTATION_LOG_LEVEL=INFO

# 只看 WARNING（静默模式）
export SELECTIVE_ROTATION_LOG_LEVEL=WARNING
```

### INFO 级别示例输出

```
┌─── Selective Rotation: Per-Layer Decisions (structural mode) ───────────────┐
└─── Selective rotation result: applied=112, skipped=85 / 197 total ───┘
  Rotation coverage: 56.9% of Linear layers rotated (43.1% skipped)

╔══════════════════════════════════════════════════════════════════════════════╗
║  Selective Hadamard Rotation — Decision Summary                             ║
║  mode=structural   score_threshold=1.5   kurtosis_threshold=5.0            ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─── Summary ──────────────────────────────────────────────────────┐
  │  Total layers: 196
  │  ✅ Rotated:   112 (57.1%)
  │  ❌ Skipped:   84 (42.9%)
  └─────────────────────────────────────────────────────────────────┘

  ┌─── Per-Layer-Type Breakdown ─────────────────────────────────────┐
  │  Type            Applied    Skipped    Rate     Assessment
  │  ─────────────────────────────────────────────────────────────────
  │  down_proj       0          28         0%       ← all skipped (residual-sensitive)
  │  gate_proj       28         0          100%     ← all rotated (outlier-prone)
  │  k_proj          28         0          100%     ← all rotated (outlier-prone)
  │  o_proj          0          28         0%       ← all skipped (residual-sensitive)
  │  q_proj          28         0          100%     ← all rotated (outlier-prone)
  │  up_proj         28         0          100%     ← all rotated (outlier-prone)
  │  v_proj          0          28         0%       ← all skipped (residual-sensitive)
  └─────────────────────────────────────────────────────────────────┘
```

### DEBUG 级别额外内容

```
# 每一层的逐个决策：
  ✅ APPLY model.layers.0.self_attn.q_proj              | reason: structural_prefer(*q_proj*)
  ✅ APPLY model.layers.0.self_attn.k_proj              | reason: structural_prefer(*k_proj*)
  ❌ SKIP  model.layers.0.self_attn.v_proj              | reason: structural_conditional(*v_proj*) → conservative_skip
  ❌ SKIP  model.layers.0.self_attn.o_proj              | reason: structural_skip(*o_proj*)
  ✅ APPLY model.layers.0.mlp.gate_proj                 | reason: structural_prefer(*gate_proj*)
  ✅ APPLY model.layers.0.mlp.up_proj                   | reason: structural_prefer(*up_proj*)
  ❌ SKIP  model.layers.0.mlp.down_proj                 | reason: structural_skip(*down_proj*)
```

### Auto 模式的 Outlier 压缩分析日志

在 auto 模式下，INFO 级别会额外输出 outlier 统计和压缩分析：

```
╔══════════════════════════════════════════════════════════════════════╗
║  Selective Rotation: Activation Profiling                           ║
║  mode=auto | num_samples=32 | device=cuda:0                        ║
╚══════════════════════════════════════════════════════════════════════╝

  Profiling: registered hooks on 196 layers, running 32 calibration samples...
  Profiling complete: processed 32 samples across 196 layers.

┌─────────────────────────────────────────────────────────────────────┐
│  Activation Profiling Results: 196 layers analyzed
│  ⚠️  Heavy-tail (kurtosis > 5.0): 87 layers
│  ⚠️  Concentrated (ECR > 0.050): 42 layers
│  ✓  Normal distribution: 109 layers
└─────────────────────────────────────────────────────────────────────┘

  Top-5 layers with HIGHEST kurtosis (most outlier-prone → rotation helps):
    🔥 STRONG model.layers.15.mlp.up_proj              | kurtosis= 12.34  ecr=0.0823
    🔥 STRONG model.layers.22.mlp.gate_proj            | kurtosis= 11.87  ecr=0.0756
    ⚠️  moderate model.layers.3.self_attn.q_proj       | kurtosis=  7.21  ecr=0.0512
    ⚠️  moderate model.layers.8.self_attn.k_proj       | kurtosis=  6.89  ecr=0.0487
    ⚠️  moderate model.layers.1.mlp.up_proj            | kurtosis=  6.45  ecr=0.0621

  Bottom-5 layers with LOWEST kurtosis (near-Gaussian → rotation may harm):
    ○ model.layers.27.mlp.down_proj                    | kurtosis=  1.23  ecr=0.0034
    ○ model.layers.26.self_attn.o_proj                 | kurtosis=  1.45  ecr=0.0041
    ○ model.layers.25.mlp.down_proj                    | kurtosis=  1.67  ecr=0.0052
    ○ model.layers.24.self_attn.o_proj                 | kurtosis=  1.89  ecr=0.0063
    ○ model.layers.23.self_attn.v_proj                 | kurtosis=  2.12  ecr=0.0078

  Per-layer-type outlier statistics (averaged across all blocks):
    layer_type       avg_kurt  max_kurt   avg_ecr  verdict
    ───────────────────────────────────────────────────────────────────────────
    up_proj            8.42      12.34    0.0672  ⚠️  outlier-prone → rotation BENEFICIAL
    gate_proj          7.89      11.87    0.0598  ⚠️  outlier-prone → rotation BENEFICIAL
    q_proj             6.21       7.21    0.0456  ⚠️  outlier-prone → rotation BENEFICIAL
    k_proj             5.87       6.89    0.0412  ⚠️  outlier-prone → rotation BENEFICIAL
    v_proj             3.45       4.12    0.0234  ○  moderate → depends on structure
    o_proj             2.12       2.89    0.0078  ✓  near-Gaussian → rotation RISKY
    down_proj          1.78       2.34    0.0045  ✓  near-Gaussian → rotation RISKY
```

### Outlier 压缩分析摘要（auto 模式最终 summary）

```
  ┌─── Outlier Compression Analysis ─────────────────────────────────┐
  │  Why selective rotation works:
  │  • Layers WITH outliers (high kurtosis) → rotation DISPERSES them
  │    across channels, reducing quantization error
  │  • Layers WITHOUT outliers (low kurtosis) → rotation ADDS noise
  │    to an already-uniform distribution (harmful)
  │
  │  Rotated layers avg kurtosis:  7.56 (outlier-prone → rotation beneficial)
  │  Skipped layers avg kurtosis:  2.34 (near-normal → rotation would harm)
  │  Kurtosis separation:          +5.22 (GOOD selective targeting)
  │  Rotated layers avg ECR:       0.0534 (energy concentrated → needs dispersion)
  │  Skipped layers avg ECR:       0.0087 (energy uniform → already balanced)
  │
  │  Interpretation:
  │  • Large kurtosis separation → selective is well-targeted
  │  • If rotated layers have LOW kurtosis → consider relaxing skip
  └─────────────────────────────────────────────────────────────────┘
```

### 如何解读日志

| 指标 | 含义 | 好的信号 |
|------|------|---------|
| Kurtosis separation | 被旋转层 vs 被跳过层的峰度差 | > 2.0 表示选择精准 |
| ECR separation | 被旋转层 vs 被跳过层的能量集中度差 | > 0.02 表示有效 |
| Per-type verdict | 每种层类型的 outlier 程度判定 | up/gate/q/k "BENEFICIAL"，down/o "RISKY" |
| Coverage rate | 旋转覆盖率 | 40-70% 通常最佳 |

### 程序中获取日志数据

```python
# 量化后获取 selector 的决策和统计
decisions = model.rotation_decisions

# 统计分析
for name, d in decisions.items():
    if d.kurtosis is not None:
        print(f"{name}: kurtosis={d.kurtosis:.2f} ecr={d.ecr:.4f} "
              f"score={d.score:.2f} → {'rotated' if d.enabled else 'skipped'}")
```

```python
# 或直接用 summary() 方法
from auto_round.algorithms.transforms.rotation.selective import LayerSelector

selector = LayerSelector(mode="auto")
selector.profile(model, dataloader)
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        selector.should_rotate(name, mod)

# 紧凑版（不含逐层表）
print(selector.summary(verbose=False))
# 完整版（含逐层表 + 统计）
print(selector.summary(verbose=True))
```

---

## 测试脚本

提供了两个测试脚本，位于 `/data/lkk/quarot/scripts/`：

| 脚本 | 用途 |
|------|------|
| `test_selective_rotation.py` | **Selective 专项对比**：none / had_all / had_structural / had_auto / had_no_down 等 |
| `test_rotation_module_compare.py` | **全 rotation 系统对比**：spinquant R1/R4 + hadamard + inplace 等 |

### 运行方式

```bash
cd /data/lkk/quarot/scripts
```

#### 快速验证（~5 分钟）

```bash
# Qwen3-0.6B, piqa, RTN, 限制 200 条加速
python test_selective_rotation.py --device cuda:0 --limit 200
```

#### 完整测试（多 task）

```bash
python test_selective_rotation.py --device cuda:0 \
    --tasks piqa,hellaswag,arc_easy
```

#### 只对比 structural vs all

```bash
python test_selective_rotation.py --device cuda:0 \
    --variants none,had_all,had_structural
```

#### 含 SpinQuant 对比

```bash
python test_selective_rotation.py --device cuda:0 \
    --variants none,had_all,had_structural,spinquant_r1
```

#### 大模型测试

```bash
python test_selective_rotation.py --device cuda:0 \
    --model Qwen/Qwen3-8B --limit 100
```

#### 全 rotation 系统对比

```bash
python test_rotation_module_compare.py --device cuda:0 \
    --variants none,had_default,had_structural,had_auto,spinquant_r1
```

#### 启用 DEBUG 日志看逐层决策

```bash
SELECTIVE_ROTATION_LOG_LEVEL=DEBUG python test_selective_rotation.py \
    --device cuda:0 --variants none,had_structural --limit 100
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | GPU 设备 | `cuda:0` |
| `--model` | 模型名 | `Qwen/Qwen3-0.6B` |
| `--scheme` | 量化方案 | `MXFP4` |
| `--variants` | 逗号分隔的 variant 名 | 全部默认 |
| `--tasks` | lm_eval 评测任务 | `piqa` |
| `--limit` | 每个 task 取多少条（加速） | None（全部） |
| `--quant-iters` | 0=RTN，>0=迭代量化 | `0` |
| `--rotation-size` | 旋转矩阵大小 | `128` |
| `--nsamples` | calibration 样本数 | `128` |
| `--seqlen` | 序列长度 | `512` |
| `--batch-size` | lm_eval batch size | `16` |

### 可选 Variant 列表

`test_selective_rotation.py` 支持的 variants：

| Variant 名 | 系统 | 说明 |
|------------|------|------|
| `none` | — | 无旋转基线 |
| `had_all` | rotation | Hadamard 对所有层（现有默认） |
| `had_structural` | rotation | Hadamard + structural 选择 |
| `had_auto` | rotation | Hadamard + auto 选择（需要 profiling） |
| `had_no_down` | rotation | 手动排除 down_proj |
| `had_no_down_oproj` | rotation | 手动排除 down_proj + o_proj |
| `inplace_all` | rotation | QuaRot-style inplace 全局旋转 |
| `spinquant_r1` | spinquant | SpinQuant R1 |
| `spinquant_r1r4` | spinquant | SpinQuant R1+R4 |

### 预期输出

脚本最后输出精度对比表：

```
════════════════════════════════════════════════════════════════════════════════════════════════════════
  Selective Rotation Comparison — Qwen/Qwen3-0.6B | scheme=MXFP4
════════════════════════════════════════════════════════════════════════════════════════════════════════
  Variant                                  System          piqa     time(s)
  ──────────────────────────────────────────────────────────────────────────────────────────────────
  no rotation (baseline)                   -             0.6842        12
  Hadamard ALL layers                      rotation      0.7012        15
  Hadamard STRUCTURAL selective            rotation      0.7089        14
  Hadamard AUTO selective                  rotation      0.7102        18
  Hadamard skip down_proj only             rotation      0.7045        14
  Hadamard skip down+o_proj                rotation      0.7078        14
════════════════════════════════════════════════════════════════════════════════════════════════════════
```

### 预期结果解读

| 对比 | 预期 | 说明 |
|------|------|------|
| `had_structural` vs `had_all` | structural ≥ all | 跳过有害层提升精度 |
| `had_auto` vs `had_structural` | auto ≥ structural | 统计量更精准定位 |
| `had_no_down` vs `had_all` | no_down ≥ all | 验证 down_proj 确实有害 |
| `had_structural` vs `none` | structural >> none | 旋转本身仍然有大收益 |
| `had_auto` vs `had_structural` | 差距小 | structural 已经很好，auto 微调 |

**核心验证点**：
1. Selective 不比 all 差 → 跳过的层确实不需要旋转
2. Selective 比 all 好 → 跳过有害层避免了精度损失
3. Coverage ~57% → 不是"少做事"，而是"精准做事"


