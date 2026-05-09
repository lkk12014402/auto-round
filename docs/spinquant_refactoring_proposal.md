# SpinQuant / QuaRot Refactoring Proposal

**Date**: 2026-05-09  
**Status**: ✅ Implemented (Phase 1–3 complete)  
**Author**: Copilot  

---

## 1. Executive Summary

我们在 `auto_round/algorithms/transforms/spinquant/` 中实现的 SpinQuant / QuaRot 功能在算法层面已经验证通过（R1–R4 rotation accuracy 正确），但在**代码架构**上与 auto-round 原生的 rotation 实现（`transforms/rotation/`）存在**两套独立的设计模式**，需要统一。

本文档分析两套架构的差异，提出分阶段重构方案，使 spinquant 无缝融入 auto-round 的注册/分发/配置体系。

---

## 2. 现状：两套架构对比

### 2.1 auto-round 原生 rotation（`transforms/rotation/`）

```
设计模式: Registry + Factory + ABC

BaseRotationConfig (dataclass base, algorithm field)
  └─ RotationConfig (Pydantic BaseModel + BaseRotationConfig)
       - algorithm = "hadamard"
       - backend: "auto" | "inplace" | "transform"
       - hadamard_type, block_size, fuse_online_to_weight, ...

BaseRotation (ABC + _REGISTRY dict)
  └─ @register("hadamard") HadamardRotation
       - apply_to_model(model, data_type, **kwargs)
       - from_config(config) → HadamardRotation

统一入口:
  apply_rotation(model, config, data_type)
  └─ normalize_rotation_config(config) → BaseRotationConfig
  └─ BaseRotation.from_config(config) → algorithm instance
  └─ algorithm.apply_to_model(model)

Pipeline 集成:
  BaseCompressor.__init__() → 收集 rotation_configs (BaseRotationConfig 实例)
  BaseCompressor._apply_rotations() → for cfg in rotation_configs: apply_rotation(model, cfg)

AutoRound 入口:
  AutoRound._CONFIG_ALIASES = {"hadamard": RotationConfig, "rtn": RTNConfig, ...}
  AutoRound.__new__(alg_configs=["sign_round", "hadamard"], ...)
```

**优点**: 统一接口、可扩展注册机制、配置验证（Pydantic）、与量化 pipeline 深度集成。

### 2.2 我们的 spinquant（`transforms/spinquant/`）

```
设计模式: 独立 Preprocessor + 独立 Config

SpinQuantConfig (@dataclass, 不继承 BaseRotationConfig)
  - r1, r2, r3, r4 (bool)
  - trainable_rotation, trainable_smooth
  - iters, lr, loss_type, ...
  - random_r1, random_r2
  - rotation_size, online_r1_rotation
  - 无 algorithm field

SpinQuantPreprocessor
  - __init__(model, config)
  - preprocess(dataloader=None) → model (8 步 pipeline)

RotationTrainer (独立 Trainer 类, 有 callbacks/checkpointing)
  - train(dataloader) → metrics
  - fuse() → model

集成方式:
  SpinQuantMixin (compressors_new/spinquant_mixin.py)
  - preprocess_with_spinquant(model, dataloader)
  - 不走 _apply_rotations()
```

**问题**:
1. `SpinQuantConfig` 不是 `BaseRotationConfig` 子类 → `BaseCompressor` 无法识别
2. 没有注册到 `BaseRotation._REGISTRY` → `from_config()` 找不到 "spinquant"
3. `SpinQuantMixin` 是独立旁路 → 与原生 `_apply_rotations()` 互不兼容
4. `AutoRound._CONFIG_ALIASES` 没有 "spinquant" / "quarot" → 用户无法通过字符串别名使用
5. `preprocessor._train_rotations()` 和 `trainer.RotationTrainer` 训练逻辑重复

---

## 3. 目标

1. **统一注册**: SpinQuant/QuaRot 通过 `BaseRotation._REGISTRY` 可发现
2. **统一入口**: `apply_rotation(model, spinquant_config)` 正常工作
3. **Pipeline 集成**: `BaseCompressor._apply_rotations()` 自动处理 spinquant config
4. **用户友好**: `AutoRound(["sign_round", "quarot"], model, ...)` 即可使用
5. **消除重复**: 训练逻辑只在一处实现
6. **向后兼容**: 现有 `SpinQuantPreprocessor` / `RotationTrainer` API 不破坏

---

## 4. 重构方案

### Phase 1: 注册集成（高优先级）

#### 4.1 SpinQuantConfig 继承 BaseRotationConfig

```python
# 现状
@dataclass
class SpinQuantConfig:
    r1: bool = True
    ...

# 重构后
@dataclass
class SpinQuantConfig(BaseRotationConfig):
    algorithm: str = "spinquant"   # 注册标识 (frozen)
    r1: bool = True
    r2: bool = True
    r3: bool = True
    r4: bool = True
    ...
```

**影响范围**: `preprocessor.py` 中 `SpinQuantConfig` 定义  
**风险**: 低。`BaseRotationConfig` 只有一个 `algorithm: str = "base"` 字段，纯增量。  
**向后兼容**: `SpinQuantConfig()` 原有用法全部保留，额外获得 `algorithm` 属性。

#### 4.2 创建 SpinQuantRotation 并注册

新建 `spinquant/algorithm.py`:

```python
from auto_round.algorithms.transforms.base import BaseRotation

@BaseRotation.register("spinquant")
class SpinQuantRotation(BaseRotation):
    """QuaRot/SpinQuant rotation — registered as 'spinquant' in BaseRotation registry."""

    def __init__(self, config: SpinQuantConfig):
        super().__init__(config)

    def apply_to_model(self, model, data_type="mx_fp", **kwargs):
        from auto_round.algorithms.transforms.spinquant.preprocessor import (
            SpinQuantPreprocessor,
        )
        dataloader = kwargs.get("dataloader")
        preprocessor = SpinQuantPreprocessor(model, self.config)
        return preprocessor.preprocess(dataloader)
```

**影响范围**: 新文件，不修改现有代码  
**文件**: `spinquant/algorithm.py` (约 30 行)

#### 4.3 在 _ensure_registry_populated() 中添加 spinquant 懒加载

```python
# transforms/base.py, line 161
def _ensure_registry_populated() -> None:
    global _registry_populated
    if _registry_populated:
        return
    import importlib
    for sub in ("hadamard", "spinquant"):   # ← 添加 spinquant
        try:
            # spinquant 注册入口在 spinquant/algorithm.py
            importlib.import_module(f"auto_round.algorithms.transforms.{sub}")
        except ImportError:
            pass
    _registry_populated = True
```

但这里需要注意：`spinquant/` 子包的 `__init__.py` 导入很多东西但不导入 `algorithm.py`。有两个方案：

- **方案 A**: 在 `spinquant/__init__.py` 中 `import .algorithm` → 加载 `__init__.py` 时自动注册
- **方案 B**: 在 `_ensure_registry_populated()` 中直接 `importlib.import_module("...spinquant.algorithm")`

推荐 **方案 A**，因为用户 `from auto_round.algorithms.transforms.spinquant import SpinQuantRotation` 更自然。

**影响范围**: `transforms/base.py` 加一行，`spinquant/__init__.py` 加一行 import  

#### 4.4 在 transforms/__init__.py 中添加 normalize 支持

```python
# transforms/__init__.py, normalize_rotation_config()
def normalize_rotation_config(config):
    ...
    if isinstance(config, dict):
        alg = config.get("algorithm", "hadamard")
        if alg == "hadamard":
            return RotationConfig.model_validate(config)
        if alg == "spinquant":
            from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
            return SpinQuantConfig(**{k: v for k, v in config.items() if k != "algorithm"})
        raise ValueError(...)
    ...
```

**影响范围**: `transforms/__init__.py` 加一个 elif 分支

#### 4.5 在 AutoRound._CONFIG_ALIASES 中添加别名

```python
# compressors_new/entry.py
class AutoRound:
    _CONFIG_ALIASES = {
        "sign_round": SignRoundConfig,
        "signround": SignRoundConfig,
        "rtn": RTNConfig,
        "hadamard": _NewArchRotationConfig,
        "spinquant": SpinQuantConfig,      # ← 新增
        "quarot": SpinQuantConfig,         # ← 新增 (QuaRot 默认 trainable=False)
    }
```

其中 "quarot" 别名可以用一个默认 `trainable_rotation=False` 的工厂方法：

```python
# 用户可以：
# AutoRound(["sign_round", "quarot"], model, ...)
# AutoRound(["rtn", SpinQuantConfig(r1=True, r2=True, r3=False, r4=False)], model, ...)
```

**注意**: `SpinQuantConfig` 是 `@dataclass`，而 `RotationConfig` 是 Pydantic `BaseModel`。`_resolve_config()` 调用 `cls()` 创建默认实例，两者都支持无参构造，无兼容性问题。但 `BaseCompressor.__init__` 中 `isinstance(_cfg, BaseRotationConfig)` 检查需要通过——Phase 1 Step 4.1 已解决。

---

### Phase 2: 训练逻辑去重（中优先级）

#### 4.6 现状分析

| 功能 | preprocessor._train_rotations() | RotationTrainer.train() |
|------|--------------------------------|------------------------|
| 训练循环 | ✅ 简单 for-loop | ✅ 带 callbacks 的 for-loop |
| KL loss | ✅ kl_top / kl_full / mse | ✅ 相同 (via compute_loss_fn) |
| Optimizer | ✅ AdamAndSGDG | ✅ AdamAndSGDG |
| 原始模型 clone | ✅ copy.deepcopy | ✅ copy.deepcopy |
| Hook 清理 | ✅ remove_spinquant_hooks_from_model | ✅ 同 |
| Callbacks | ❌ | ✅ on_step_begin/end, on_evaluate |
| Checkpointing | ❌ | ✅ save/load_checkpoint |
| Evaluation | ❌ | ✅ mid-training eval |
| Orthogonality 检查 | ✅ _check_orthogonality | ✅ OrthogonalityMonitor callback |
| 日志 | ✅ logger.info 每 50 步 | ✅ LossLogger callback |

**结论**: `preprocessor._train_rotations()` 是 `RotationTrainer.train()` 的功能子集。

#### 4.7 重构方案

**方案**: `preprocessor._train_rotations()` 委托给 `RotationTrainer`

```python
# preprocessor.py
def _train_rotations(self, dataloader):
    """Delegate training to RotationTrainer (single implementation)."""
    from auto_round.algorithms.transforms.spinquant.trainer import (
        RotationTrainer,
        RotationTrainerConfig,
    )
    trainer_config = RotationTrainerConfig(
        r1=self.config.r1, r2=self.config.r2,
        r3=self.config.r3, r4=self.config.r4,
        trainable_rotation=self.config.trainable_rotation,
        trainable_smooth=self.config.trainable_smooth,
        iters=self.config.iters, lr=self.config.lr,
        smooth_lr=self.config.smooth_lr,
        loss_type=self.config.loss_type,
        kl_top_k=self.config.kl_top_k,
        # preprocessor 不需要 checkpoint/eval
        log_interval=50,
        eval_interval=0,
        save_interval=0,
    )
    trainer = RotationTrainer(
        self.model, config=trainer_config,
        callbacks=[LossLogger(50), OrthogonalityMonitor()],
    )
    # Skip trainer's own init (preprocessor already did steps 1-4)
    trainer._setup_training_optimizer_only(dataloader)
    trainer.train(dataloader)
```

**但这里有个细节**: `RotationTrainer._setup_training()` 会重复做 untie/fuse/init，而 preprocessor 已经做过了。

**更好的方案**: 提取共享的训练核心逻辑:

```python
# training_core.py (新文件)
def run_training_loop(
    model, original_model, optimizer,
    dataloader, config, callbacks=None,
) -> dict:
    """共享的训练循环核心，供 preprocessor 和 trainer 复用。"""
    ...
```

然后:
- `preprocessor._train_rotations()` 调用 `run_training_loop()` + 简单日志
- `RotationTrainer.train()` 调用 `run_training_loop()` + callbacks + checkpointing

**影响范围**: 新建 `training_core.py`，修改 `preprocessor.py` 和 `trainer.py` 的训练实现  
**向后兼容**: 两者的公开 API 不变

---

### Phase 3: spinquant_mixin.py 简化（低优先级）

#### 4.8 现状

`compressors_new/spinquant_mixin.py` 提供:
- `SpinQuantMixin.__init__()` — 重新构建 SpinQuantConfig
- `preprocess_with_spinquant()` — 独立调用 SpinQuantPreprocessor
- `patch_compressor_for_spinquant()` — monkeypatch compress method

这是一条**独立于 `_apply_rotations()` 的旁路**。

#### 4.9 重构方案

Phase 1 完成后，`SpinQuantConfig` 已经是 `BaseRotationConfig` 子类，`BaseCompressor._apply_rotations()` 可以自动处理它。此时 `SpinQuantMixin` 可以:

**方案 A**: 完全移除 — 用户直接传 `SpinQuantConfig` 作为 `alg_configs` 之一:
```python
AutoRound(
    alg_configs=[SignRoundConfig(iters=200), SpinQuantConfig(r1=True, r2=True)],
    model=model,
    scheme="W4A16",
)
```

**方案 B**: 保留为简化入口 — 将内部实现改为构造 SpinQuantConfig 并塞进 `rotation_configs`

推荐 **方案 A + 保留 Mixin 文件但标记 deprecated**，给用户迁移时间。

---

### Phase 4: Config 风格对齐（可选，低优先级）

#### 4.10 分析

| 特性 | RotationConfig | SpinQuantConfig |
|------|---------------|-----------------|
| 基类 | Pydantic BaseModel | @dataclass |
| 验证 | @field_validator | __post_init__ |
| 序列化 | .model_dump() | asdict() 或手动 |
| JSON schema | 自动生成 | 需手动 |

**建议**: 暂不对齐。理由:
1. SpinQuantConfig 字段多（20+），用 Pydantic 收益不大
2. `@dataclass` 更轻量，不引入额外依赖（Pydantic 在某些环境可能版本冲突）
3. `BaseRotationConfig` 本身就是 `@dataclass`，继承自然
4. 如果后续需要 JSON schema 或 rich validation，再迁移

---

## 5. 实现计划

### Phase 1: 注册集成（建议先做）

| 步骤 | 文件 | 改动 | 复杂度 |
|------|------|------|--------|
| 1.1 | `spinquant/preprocessor.py` | SpinQuantConfig 加 `BaseRotationConfig` 父类 + `algorithm="spinquant"` | 低 |
| 1.2 | `spinquant/algorithm.py` (新) | SpinQuantRotation class + @register | 低 |
| 1.3 | `spinquant/__init__.py` | import algorithm module + export SpinQuantRotation | 低 |
| 1.4 | `transforms/base.py` | _ensure_registry_populated 加 "spinquant" | 极低 |
| 1.5 | `transforms/__init__.py` | normalize_rotation_config 加 spinquant 分支 | 低 |
| 1.6 | `compressors_new/entry.py` | _CONFIG_ALIASES 加 "spinquant"/"quarot" | 极低 |
| 1.7 | 测试 | 验证 `apply_rotation(model, SpinQuantConfig(...))` 正常工作 | 中 |

**预估工作量**: 约 100 行新代码 + 20 行修改

### Phase 2: 训练去重

| 步骤 | 文件 | 改动 | 复杂度 |
|------|------|------|--------|
| 2.1 | `spinquant/training_core.py` (新) | 提取共享训练循环 | 中 |
| 2.2 | `spinquant/preprocessor.py` | _train_rotations 委托 training_core | 中 |
| 2.3 | `spinquant/trainer.py` | _training_step 路径改用 training_core | 中 |
| 2.4 | 测试 | 验证两个入口结果一致 | 中 |

**预估工作量**: 约 80 行新代码 + 100 行重构

### Phase 3: Mixin 简化

| 步骤 | 文件 | 改动 | 复杂度 |
|------|------|------|--------|
| 3.1 | `compressors_new/spinquant_mixin.py` | 标记 deprecated，改用 rotation_configs 路径 | 低 |
| 3.2 | 文档 | 更新 API guide 迁移指南 | 低 |

### Phase 4: Config 风格对齐（可选）

暂不建议实施，除非有明确的 JSON schema 或 config 序列化需求。

---

## 6. _apply_rotations() 中 spinquant 的特殊需求

auto-round 原生的 `_apply_rotations()` 只传 `data_type`：

```python
# base.py line 871-876
for rotation_cfg in self.rotation_configs:
    self.model_context.model = apply_rotation(
        self.model_context.model,
        rotation_cfg,
        data_type=self.quantize_config.data_type,
    )
```

但 SpinQuant 的 `apply_to_model()` 可能需要 `dataloader`（trainable 模式）。两个解决方案：

**方案 A**: trainable 模式需要 dataloader → 在 `_apply_rotations()` 中传 kwargs:

```python
def _apply_rotations(self):
    for rotation_cfg in self.rotation_configs:
        kwargs = {}
        if isinstance(rotation_cfg, SpinQuantConfig) and rotation_cfg.trainable_rotation:
            kwargs["dataloader"] = self.calibration  # or self.dataloader
        self.model_context.model = apply_rotation(
            self.model_context.model, rotation_cfg,
            data_type=self.quantize_config.data_type,
            **kwargs,
        )
```

**方案 B**: `SpinQuantRotation.apply_to_model()` 自己获取 calibration data → 不推荐（耦合）

**方案 C**: 暂时不支持 trainable 模式通过 pipeline（标注 ⚠️）。用户如需 trainable，仍用 `SpinQuantPreprocessor` + `RotationTrainer` 独立 API。QuaRot（trainable=False）通过 pipeline 完全支持。

**推荐**: 方案 C（短期）+ 方案 A（中期）。理由：trainable 模式本身标为 ⚠️ experimental，不急于集成到 pipeline。

---

## 7. 用户体验目标

重构后，用户可以：

### 7.1 最简用法（QuaRot via pipeline）

```python
from auto_round.compressors_new.entry import AutoRound

# 字符串别名
result = AutoRound(["sign_round", "quarot"], model, scheme="W4A16")

# 或者显式配置
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
config = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False, trainable_rotation=False)
result = AutoRound(["sign_round", config], model, scheme="W4A16")
```

### 7.2 独立 preprocessor（不依赖 pipeline）

```python
# 现有 API 保持不变
from auto_round.algorithms.transforms.spinquant import SpinQuantPreprocessor, SpinQuantConfig

config = SpinQuantConfig(r1=True, r2=True, trainable_rotation=False)
SpinQuantPreprocessor(model, config).preprocess()
```

### 7.3 通用 apply_rotation()

```python
from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

# 与 Hadamard 相同的调用模式
model = apply_rotation(model, SpinQuantConfig(r1=True, r2=True))
```

---

## 8. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Phase 1 破坏现有 SpinQuantConfig 用法 | 低 | 中 | 只加父类，不删改现有字段 |
| Phase 2 训练行为不一致 | 中 | 中 | 对比测试两条路径的 loss 曲线 |
| _apply_rotations 执行顺序问题 | 低 | 中 | spinquant 应在 hadamard 之前 |
| Pydantic/dataclass 混用序列化问题 | 低 | 低 | BaseRotationConfig 本身是 dataclass |

---

## 9. 不在本次重构范围内

1. **SpinQuant 训练验证**: trainable 模式仍标为 ⚠️ experimental，不在此次改
2. **模型 save/load**: rotation hooks 序列化是独立问题
3. **Config 迁移到 Pydantic**: 暂不必要
4. **rotation + quantization 深度融合**: 保持解耦设计

---

## 10. 附录：文件影响矩阵

| 文件 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| `spinquant/preprocessor.py` | 修改 (加父类) | 修改 (委托训练) | — |
| `spinquant/algorithm.py` | **新建** | — | — |
| `spinquant/__init__.py` | 修改 (加 import) | — | — |
| `spinquant/trainer.py` | — | 修改 (用 core) | — |
| `spinquant/training_core.py` | — | **新建** | — |
| `spinquant/training.py` | — | 可能简化 | — |
| `transforms/base.py` | 修改 (1行) | — | — |
| `transforms/__init__.py` | 修改 (elif) | — | — |
| `compressors_new/entry.py` | 修改 (2行) | — | — |
| `compressors_new/base.py` | — | — | 可能修改 |
| `compressors_new/spinquant_mixin.py` | — | — | deprecated |
| `docs/spinquant_quarot_api_guide.md` | 更新 | 更新 | 更新 |

---

## 11. Implementation Status (2026-05-09)

All three phases have been implemented and verified:

### Phase 1: Registry Integration ✅

| Task | Status | Notes |
|------|--------|-------|
| SpinQuantConfig → BaseRotationConfig | ✅ Done | Added `algorithm = "spinquant"` field |
| SpinQuantRotation + @register | ✅ Done | New `algorithm.py` delegates to SpinQuantPreprocessor |
| Registry wiring | ✅ Done | `_ensure_registry_populated()` includes "spinquant" |
| normalize_rotation_config | ✅ Done | Handles dicts and "spinquant"/"quarot" strings |
| AutoRound._CONFIG_ALIASES | ✅ Done | "quarot" → lambda factory with non-trainable defaults |
| Integration tests | ✅ Done | 4 tests pass: config, dict, string, isinstance check |

### Phase 2: Training Logic Deduplication ✅

| Task | Status | Notes |
|------|--------|-------|
| training_core.py created | ✅ Done | ~250 lines, 7 shared primitives |
| preprocessor.py refactored | ✅ Done | `_train_rotations()` → `run_training_loop()` |
| trainer.py refactored | ✅ Done | Uses shared helpers, unused imports cleaned |
| Regression tests pass | ✅ Done | test_reference_equivalence, test_quark_comparison |

### Phase 3: Mixin Simplification ✅

| Task | Status | Notes |
|------|--------|-------|
| Deprecation warnings added | ✅ Done | DeprecationWarning on `__init__` and `preprocess_with_spinquant` |
| Mixin delegates to apply_rotation | ✅ Done | `preprocess_with_spinquant()` → `apply_rotation()` |
| patch_compressor_for_spinquant deprecated | ✅ Done | With deprecation warning |
| Migration examples in docstring | ✅ Done | Shows new `rotation_configs=` API |

### Phase 4: Config Style Alignment

| Task | Status | Notes |
|------|--------|-------|
| Pydantic migration | ⏸️ Deferred | SpinQuantConfig stays as @dataclass (intentional: lighter weight, matches BaseRotationConfig) |

### Files Changed

| File | Action | Lines Changed |
|------|--------|---------------|
| `spinquant/algorithm.py` | **Created** | ~90 lines |
| `spinquant/training_core.py` | **Created** | ~250 lines |
| `spinquant/preprocessor.py` | Modified | Config inheritance, training delegation, import cleanup |
| `spinquant/trainer.py` | Modified | Shared helpers, import cleanup |
| `spinquant/__init__.py` | Modified | Added SpinQuantRotation export |
| `transforms/base.py` | Modified | +1 line in `_ensure_registry_populated()` |
| `transforms/__init__.py` | Modified | +30 lines in `normalize_rotation_config()` |
| `compressors_new/entry.py` | Modified | +5 lines for config aliases |
| `compressors_new/spinquant_mixin.py` | Modified | Deprecated, delegates to apply_rotation |

### Unified API (Post-Refactoring)

```python
# QuaRot via string alias
from auto_round.compressors_new.entry import AutoRound
autoround = AutoRound(model=model, scheme="W4A16", rotation_configs=["quarot"])

# SpinQuant via explicit config
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False,
                      trainable_rotation=False, random_rotation=True)
autoround = AutoRound(model=model, scheme="W4A16", rotation_configs=[cfg])

# Standalone rotation (no quantization)
from auto_round.algorithms.transforms import apply_rotation
model = apply_rotation(model, "quarot")
model = apply_rotation(model, SpinQuantConfig(r1=True, r2=True))
```
