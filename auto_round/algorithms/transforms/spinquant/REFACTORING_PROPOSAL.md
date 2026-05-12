# Rotation Serialization Extensibility: Audit & Refactoring Proposal

> 当前 SpinQuant 序列化实现的问题审查，以及支持多种 rotation 方法的重构方案。

---

## 目录

1. [审查背景](#1-审查背景)
2. [问题清单](#2-问题清单)
   - P1: pack_layer 侵入性
   - P2: 三文件重复注入模式
   - P3: 硬编码 "spinquant" 属性名
   - P4: 硬编码 config 持久化 key
   - P5: 硬编码 buffer 前缀
   - P6: 硬编码投影层名
   - P7: 函数名品牌绑定
   - P8: 无 rotation 时的空跑开销
   - P9: load 侧 `_is_quantlinear()` 原始 bug
3. [已有的抽象基础](#3-已有的抽象基础)
4. [重构方案](#4-重构方案)
   - 方案 A: 轻量统一 — 保持现有结构，参数化命名
   - 方案 B: 中度重构 — 引入 RotationSerializer 接口
   - 方案 C: 深度重构 — export hook 注册机制
5. [方案对比](#5-方案对比)
6. [推荐路径](#6-推荐路径)

---

## 1. 审查背景

当前 SpinQuant/QuaRot 的 rotation 矩阵序列化已可以正常工作（W4A16/MXFP4/NVFP4/FP8）。
但用户计划后续引入新的 rotation 方法，需要评估现有代码的可扩展性。

**审查范围**：

| 文件 | 角色 |
|------|------|
| `export.py` | INT scheme export + 共享注入函数定义 |
| `export_to_nvfp_mx.py` | MXFP/NVFP scheme export |
| `export_to_fp8.py` | FP8 scheme export |
| `serialize.py` | 核心序列化：buffer 注入/预注册/重建 |
| `convert_model.py` | load 侧：buffer 预注册 + forward patch |
| `preprocessor.py` | rotation 预处理（in-memory 路径） |
| `transforms/__init__.py` | 统一入口 + registry |
| `transforms/base.py` | BaseRotation / BaseRotationConfig 抽象基类 |

---

## 2. 问题清单

### P1: pack_layer 侵入性 ⚠️ 高优先

**现状**：三个 export 文件的 `pack_layer()` 末尾都插入了 spinquant 注入代码：

```python
# export.py L219
_inject_spinquant_buffers_on_layer(layer_name, qlayer, model)

# export_to_nvfp_mx.py L122-123
from auto_round.export.export_to_autoround.export import _inject_spinquant_buffers_on_layer
_inject_spinquant_buffers_on_layer(name, qlayer, model)

# export_to_fp8.py L187-188
from auto_round.export.export_to_autoround.export import _inject_spinquant_buffers_on_layer
_inject_spinquant_buffers_on_layer(layer_name, my_linear, model)
```

**问题**：
- `pack_layer()` 本职是量化打包，不应感知 rotation 逻辑
- 每新增一种 rotation 方法，都要改三个 `pack_layer()`
- 即使不做 rotation，也执行了 import + `getattr(model, "_spinquant_config", None)` 检查

**影响面**：3 个文件 × 每个文件 1 处 = 3 处修改点

---

### P2: 三文件重复注入模式 ⚠️ 高优先

**现状**：三个 `save_quantized_as_*()` 函数中，有近乎相同的 spinquant 代码块：

```python
# export.py save_quantized_as_autoround() L403-438
_inject_spinquant_rotation_buffers(model, quantization_config)   # bulk 注入
_save_spinquant_config_to_dir(model, output_dir)                 # config 持久化

# export_to_nvfp_mx.py save_quantized_as_fp() L250-277
_inject_spinquant_rotation_buffers(model, quantization_config)
_save_spinquant_config_to_dir(model, output_dir)

# export_to_fp8.py save_quantized_as_autoround() L260-292
_inject_spinquant_rotation_buffers(model, quantization_config)
_save_spinquant_config_to_dir(model, output_dir)
```

**问题**：
- 三份相同的调用 + 相同的 import → DRY 违反
- 新增 rotation 方法时，三个文件都要加相同的调用
- 容易出现遗漏（如之前 FP8 路径遗漏的 bug）

---

### P3: 硬编码 "spinquant" 属性名 ⚠️ 中优先

**现状**：rotation config 通过固定属性名传递：

```python
# preprocessor.py L308 — 写入
self.model._spinquant_config = self.config

# export.py L227, L405, L428 — 读取
spinquant_config = getattr(model, "_spinquant_config", None)

# convert_model.py L884 — 读取
spinquant_config = getattr(quantization_config, "spinquant_config", None)
```

**问题**：
- 新 rotation 方法需要新属性名 `model._quarot_config`、`model._my_rotation_config`
- 多种 rotation 方法共存时，需要逐一检查每种属性名
- export 代码需要知道所有可能的属性名

**影响面**：6 处硬编码位置

---

### P4: 硬编码 config 持久化 key ⚠️ 中优先

**现状**：config 保存到 JSON 时使用固定 key：

```python
# export.py L419
quantization_config["spinquant_config"] = _config_to_serializable(...)

# serialize.py L160
model_config["quantization_config"]["spinquant_config"] = spinquant_dict

# convert_model.py L886
spinquant_config = quantization_config.get("spinquant_config")
```

**问题**：
- key 名 `"spinquant_config"` 硬编码在 save + load 两侧
- 新 rotation 方法需要新 key（如 `"quarot_config"`），且 load 侧要同时检查多种 key
- 或者，统一为通用 key（如 `"rotation_config"`），但需要同时处理向后兼容

---

### P5: 硬编码 buffer 前缀 ⚠️ 中优先

**现状**：

```python
# serialize.py L52-53
_R1_PREFIX = "spinquant_r1"
_R4_PREFIX = "spinquant_r4"

# convert_model.py L766-767 — 检测
hasattr(module, "spinquant_r1_type") or hasattr(module, "spinquant_r4_type")
```

**问题**：
- 前缀包含 "spinquant" 品牌名
- 新 rotation 方法的 buffer 需要不同前缀
- load 侧检测逻辑需要知道所有可能的前缀

**说明**：buffer 前缀含方法名实际上是**合理的**（避免不同 rotation 方法的 buffer 冲突），
但 load 侧的检测逻辑需要抽象化。

---

### P6: 硬编码投影层名 🔵 低优先

**现状**：

```python
# export.py L233-234
r1_proj_names = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
r4_proj_names = ("down_proj",)
```

**问题**：
- 这些名字假定 LLaMA/Qwen 架构
- 新模型架构（如 Falcon、GPT-NeoX）可能使用不同命名
- 但这在 preprocessor.py 中也有同样的假设，且由 `model_type_resolver` 处理

**评估**：这是架构适配问题，不是 rotation 方法可扩展性问题。暂不纳入重构范围。

---

### P7: 函数名品牌绑定 🔵 低优先

**现状**：所有公共函数名包含 "spinquant"：

| 函数名 | 位置 |
|--------|------|
| `_inject_spinquant_buffers_on_layer()` | export.py L225 |
| `_inject_spinquant_rotation_buffers()` | export.py L403 |
| `_save_spinquant_config_to_dir()` | export.py L426 |
| `inject_spinquant_buffers()` | serialize.py |
| `preregister_spinquant_buffers()` | serialize.py |
| `rebuild_spinquant_online()` | serialize.py |
| `save_spinquant_config()` | serialize.py |
| `_rebuild_spinquant_if_needed()` | convert_model.py L762 |

**问题**：
- 如果引入通用分发层，这些函数名变成内部实现细节
- 公共 API 层应该是 rotation-method-agnostic 的

**评估**：如果采用方案 B/C，这些自然被封装。单独改名收益不大。

---

### P8: 无 rotation 时的空跑开销 🔵 低优先

**现状**：即使没有 rotation，每个 `pack_layer()` 调用都会：

```python
# 1. 导入 serialize 模块 (在函数内 lazy import，影响有限)
# 2. getattr(model, "_spinquant_config", None) — 一次属性查找
# 3. 检查 None → return
```

**评估**：开销极低（一次 `getattr` + `None` 比较），不构成实际性能问题。
但从**代码职责**角度看，pack_layer 不应包含这段逻辑。

---

### P9: load 侧 `_is_quantlinear()` 原始 bug ✅ 已修复

**原问题**：只匹配 `QuantLinear` 类名 + `hasattr(bits)`，FP 系列推理模块不匹配。
**已修复**：扩展为子串匹配 + MRO 检查。详见 SERIALIZATION_ARCHITECTURE.md Section 12.4。

---

## 3. 已有的抽象基础

当前代码已有部分可扩展基础设施：

### 3.1 Registry 机制 (transforms/base.py)

```python
class BaseRotation(ABC):
    _REGISTRY: dict[str, type["BaseRotation"]] = {}

    @classmethod
    def register(cls, algorithm_name: str):
        """Class decorator to register a BaseRotation subclass."""

    @classmethod
    def from_config(cls, config: BaseRotationConfig) -> "BaseRotation":
        """Create rotation instance from config, dispatch by algorithm field."""
```

**覆盖范围**：仅覆盖 preprocessing 阶段（`apply_to_model()`），**不覆盖序列化**。

### 3.2 统一入口 (transforms/\_\_init\_\_.py)

```python
def apply_rotation(model, config, data_type="mx_fp", **kwargs):
    """Dispatch to correct rotation subclass by config.algorithm."""
```

**覆盖范围**：仅覆盖 preprocessing 阶段。

### 3.3 Guard 模式

所有注入函数都有 `if spinquant_config is None: return` 守卫，确保无 rotation 时快速退出。

### 3.4 结论

**差距**：Registry 和统一入口仅覆盖 preprocessing。序列化（save/load）没有对应的抽象层，
所有 spinquant 引用都是直接调用，没有经过 registry dispatch。

---


## 4. 重构方案（选定：方案 B — RotationSerializer Mixin）

### 4.1 设计理念

将序列化能力作为独立 mixin 接口，与现有 `BaseRotation`（preprocessing）组合。
每种 rotation 方法实现自己的 serializer，export 文件通过 registry 统一调用，
**永远不需要感知具体 rotation 方法名**。

```
                  ┌──────────────────┐
                  │ BaseRotation     │  ← 已有：preprocessing（apply_to_model）
                  └────────┬─────────┘
                           │
                  ┌────────┴─────────┐
                  │RotationSerializer│  ← 新增 mixin：serialization 接口
                  └────────┬─────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   SpinQuantRotation  FutureRotation   AnotherRotation
   (BaseRotation +    (BaseRotation +   ...
    RotationSerializer) RotationSerializer)
```

### 4.2 新增接口：RotationSerializer

**文件**: `transforms/base.py`

```python
class RotationSerializer(ABC):
    """Mixin interface for rotation method serialization.

    Any BaseRotation subclass that needs save/load support should also
    inherit from this mixin and implement all abstract methods.

    Separating from BaseRotation keeps preprocessing and serialization
    concerns decoupled — a rotation method CAN exist without save/load
    (e.g. in-memory-only evaluation).
    """

    # ── Save side ──

    @abstractmethod
    def inject_buffers_on_layer(
        self, layer_name: str, qlayer: nn.Module, model: nn.Module,
    ) -> None:
        """Inject rotation buffers onto a single QuantLinear after packing.

        Called inside pack_layer() for ShardWriter path — buffers must be
        on the module BEFORE shard_writer.write() + offload_to_meta().

        Args:
            layer_name: Full qualified name, e.g.
                        "model.layers.0.self_attn.q_proj".
            qlayer: The packed QuantLinear module.
            model: The full model (for reading rotation matrices / config).
        """

    @abstractmethod
    def inject_buffers_bulk(
        self, model: nn.Module, quantization_config: dict,
    ) -> None:
        """Inject rotation buffers on ALL QuantLinear modules + embed config.

        Called in save_quantized_as_*() for non-ShardWriter path.
        Should also write rotation config into quantization_config dict
        under the key "rotation_config".

        Args:
            model: Full quantized model with QuantLinear modules.
            quantization_config: Mutable dict that will be persisted to JSON.
        """

    @abstractmethod
    def save_config(self, model: nn.Module, save_dir: str) -> None:
        """Persist rotation config to config.json after model.save_pretrained().

        Args:
            model: Full model (for architecture info).
            save_dir: Directory where config.json lives.
        """

    # ── Load side ──

    @abstractmethod
    def preregister_buffers(
        self, model: nn.Module, config_dict: dict,
    ) -> int:
        """Pre-register empty rotation buffers on QuantLinear modules.

        Called AFTER convert_hf_model() replaces Linear → QuantLinear,
        BEFORE HuggingFace loads state_dict from safetensors.

        Args:
            model: Model with QuantLinear modules.
            config_dict: Rotation config dict from quantization_config JSON.

        Returns:
            Number of modules that received pre-registered buffers.
        """

    @abstractmethod
    def rebuild_online(self, model: nn.Module) -> nn.Module:
        """Rebuild online rotation hooks after state_dict is loaded.

        Handles: forward patching (R1/R4 buffers), hook re-registration
        (R3), etc.

        Args:
            model: Loaded quantized model with populated rotation buffers.

        Returns:
            The model with online rotations active.
        """

    @abstractmethod
    def has_rotation_buffers(self, module: nn.Module) -> bool:
        """Check if a single module has this rotation method's buffers.

        Used for quick detection on load side to decide whether to call
        rebuild_online().
        """

    @classmethod
    @abstractmethod
    def config_key(cls) -> str:
        """JSON key used in quantization_config for this rotation method.

        e.g. "spinquant_config". Used for legacy compat detection.
        """
```

### 4.3 统一 model 属性名

```python
# ── preprocessor 写入 ──
# Before:
self.model._spinquant_config = self.config
# After:
self.model._rotation_config = self.config

# ── export / convert_model 读取 ──
# Before:
spinquant_config = getattr(model, "_spinquant_config", None)
# After:
rotation_config = getattr(model, "_rotation_config", None)
```

**向后兼容**：load 侧增加 fallback：
```python
config = getattr(model, "_rotation_config", None)
if config is None:
    config = getattr(model, "_spinquant_config", None)  # legacy
```

### 4.4 统一 JSON 持久化 key

使用统一入口 key `"rotation_config"` 包装，内含 `algorithm` 字段标识方法：

```json
{
  "quantization_config": {
    "quant_method": "auto-round",
    "rotation_config": {
      "algorithm": "spinquant",
      "r1": true,
      "r2": true,
      "r3": false,
      "r4": false,
      "online_r1_rotation": true,
      "rotation_size": null,
      "random_r1": false,
      "random_r2": false,
      "head_dim": 128,
      "hidden_size": 1024,
      "intermediate_size": 3072
    }
  }
}
```

**向后兼容策略**：

```
save 侧（过渡期，双写）:
  quantization_config["rotation_config"] = {..., "algorithm": "spinquant"}
  quantization_config["spinquant_config"] = {...}  # legacy compat

load 侧（优先读新 key，fallback 旧 key）:
  rotation_cfg = quantization_config.get("rotation_config")
  if rotation_cfg is None:
      rotation_cfg = quantization_config.get("spinquant_config")
      if rotation_cfg is not None:
          rotation_cfg = dict(rotation_cfg)
          rotation_cfg.setdefault("algorithm", "spinquant")

过渡期结束后（下一个大版本），删除 "spinquant_config" 双写。
```

### 4.5 通用分发层

**文件**: `transforms/__init__.py`（新增函数）或独立 `transforms/serialize_dispatch.py`

这些函数是 export 和 convert_model 唯一调用的入口，内部通过 registry 路由。

```python
# ─── 内部 helper ───

def _get_serializer(model: nn.Module) -> Optional[RotationSerializer]:
    """Get the RotationSerializer for the model's rotation config."""
    config = getattr(model, "_rotation_config", None)
    if config is None:
        config = getattr(model, "_spinquant_config", None)  # legacy
    if config is None:
        return None
    rotation = BaseRotation.from_config(config)
    if not isinstance(rotation, RotationSerializer):
        return None
    return rotation


def _get_serializer_for_config(config_dict: dict) -> Optional[RotationSerializer]:
    """Get serializer from a JSON config dict (load side)."""
    algorithm = config_dict.get("algorithm", "spinquant")  # default for legacy
    try:
        rotation_cls = BaseRotation._REGISTRY[algorithm]
    except KeyError:
        return None
    config = BaseRotationConfig(algorithm=algorithm)
    rotation = rotation_cls(config)
    if not isinstance(rotation, RotationSerializer):
        return None
    return rotation


# ─── Save side (called by export files) ───

def inject_rotation_buffers_on_layer(
    layer_name: str, qlayer: nn.Module, model: nn.Module,
) -> None:
    """Per-layer rotation buffer injection (ShardWriter path).

    Single call replaces the 3 copies of _inject_spinquant_buffers_on_layer()
    in export.py / export_to_nvfp_mx.py / export_to_fp8.py.

    No-op if model has no rotation config.
    """
    serializer = _get_serializer(model)
    if serializer is not None:
        serializer.inject_buffers_on_layer(layer_name, qlayer, model)


def inject_rotation_buffers_bulk(
    model: nn.Module, quantization_config: dict,
) -> None:
    """Bulk rotation buffer injection (non-ShardWriter path).

    Single call replaces the 3 copies of _inject_spinquant_rotation_buffers().

    No-op if model has no rotation config.
    """
    serializer = _get_serializer(model)
    if serializer is not None:
        serializer.inject_buffers_bulk(model, quantization_config)


def save_rotation_config(model: nn.Module, save_dir: str) -> None:
    """Persist rotation config to config.json.

    Single call replaces the 3 copies of _save_spinquant_config_to_dir().

    No-op if model has no rotation config.
    """
    serializer = _get_serializer(model)
    if serializer is not None:
        serializer.save_config(model, save_dir)


# ─── Load side (called by convert_model.py) ───

def preregister_rotation_buffers(
    model: nn.Module, quantization_config: dict,
) -> int:
    """Pre-register empty rotation buffers before state_dict loading.

    Replaces the direct call to preregister_spinquant_buffers().
    """
    # Extract rotation config from quantization_config
    rotation_cfg = None
    if isinstance(quantization_config, dict):
        rotation_cfg = quantization_config.get("rotation_config")
        if rotation_cfg is None:
            rotation_cfg = quantization_config.get("spinquant_config")
            if rotation_cfg is not None:
                rotation_cfg = dict(rotation_cfg)
                rotation_cfg.setdefault("algorithm", "spinquant")
    else:
        rotation_cfg = getattr(quantization_config, "rotation_config", None)
        if rotation_cfg is None:
            rotation_cfg = getattr(quantization_config, "spinquant_config", None)
            if rotation_cfg is not None:
                rotation_cfg = dict(rotation_cfg)
                rotation_cfg.setdefault("algorithm", "spinquant")

    if not rotation_cfg:
        return 0

    serializer = _get_serializer_for_config(rotation_cfg)
    if serializer is None:
        return 0
    return serializer.preregister_buffers(model, rotation_cfg)


def rebuild_rotation_if_needed(model: nn.Module) -> None:
    """Rebuild online rotation hooks after weights are loaded.

    Replaces _rebuild_spinquant_if_needed(). Scans all registered
    serializers to find one whose buffers are present.
    """
    _ensure_registry_populated()

    for name, rotation_cls in BaseRotation._REGISTRY.items():
        try:
            temp = rotation_cls(BaseRotationConfig(algorithm=name))
        except Exception:
            continue
        if not isinstance(temp, RotationSerializer):
            continue

        # Quick scan: does any module have this method's buffers?
        found = False
        for _, module in model.named_modules():
            if temp.has_rotation_buffers(module):
                found = True
                break
        if found:
            temp.rebuild_online(model)
            return  # Only one rotation method expected per model
```

### 4.6 SpinQuantRotation 实现 RotationSerializer

**文件**: `spinquant/algorithm.py`

当前 `SpinQuantRotation` 只有 `apply_to_model()`。扩展为同时实现
`RotationSerializer`，委托给已有 `serialize.py` 函数（不改内部逻辑）。

```python
from auto_round.algorithms.transforms.base import (
    BaseRotation, RotationSerializer,
)


@BaseRotation.register("spinquant")
class SpinQuantRotation(BaseRotation, RotationSerializer):
    """SpinQuant / QuaRot — preprocessing + serialization."""

    # ── Preprocessing (existing, unchanged) ──

    def apply_to_model(self, model, data_type="mx_fp", **kwargs):
        from .preprocessor import SpinQuantPreprocessor
        preprocessor = SpinQuantPreprocessor(model, self.config)
        return preprocessor.preprocess(kwargs.get("dataloader"))

    # ── RotationSerializer: Save side ──

    def inject_buffers_on_layer(self, layer_name, qlayer, model):
        """Per-layer injection. Logic from old export.py
        _inject_spinquant_buffers_on_layer()."""
        config = self._get_model_config(model)
        if config is None:
            return

        from .serialize import (
            _inject_rotation_buffers,
            _get_stored_rotation,
            _get_hidden_size,
            _get_intermediate_size,
            _R1_PREFIX, _R4_PREFIX,
        )

        short_name = layer_name.split(".")[-1]
        r1_proj_names = ("q_proj", "k_proj", "v_proj",
                         "gate_proj", "up_proj")
        r4_proj_names = ("down_proj",)

        try:
            if (config.r1 and config.online_r1_rotation
                    and short_name in r1_proj_names):
                hidden_size = _get_hidden_size(model)
                r1_size = config.rotation_size or hidden_size
                _inject_rotation_buffers(
                    qlayer, _R1_PREFIX, r1_size,
                    random=config.random_r1, is_trained=False,
                    rotation_matrix=_get_stored_rotation(
                        model, "spinquant_R1"),
                )

            if config.r4 and short_name in r4_proj_names:
                intermediate_size = _get_intermediate_size(model)
                r4_size = config.rotation_size or intermediate_size
                _inject_rotation_buffers(
                    qlayer, _R4_PREFIX, r4_size,
                    random=False, is_trained=False,
                    rotation_matrix=None,
                )
        except Exception as e:
            import logging
            logging.getLogger("autoround").warning(
                f"Failed to inject SpinQuant buffers on "
                f"{layer_name}: {e}"
            )

    def inject_buffers_bulk(self, model, quantization_config):
        """Bulk injection. Logic from old export.py
        _inject_spinquant_rotation_buffers()."""
        config = self._get_model_config(model)
        if config is None:
            return

        from .serialize import (
            inject_spinquant_buffers,
            _config_to_serializable,
        )
        try:
            n = inject_spinquant_buffers(model, config)
            if n > 0:
                cfg_dict = _config_to_serializable(config, model)
                cfg_dict["algorithm"] = "spinquant"
                # New key
                quantization_config["rotation_config"] = cfg_dict
                # Legacy compat key (remove in next major version)
                quantization_config["spinquant_config"] = cfg_dict
        except Exception as e:
            import logging
            logging.getLogger("autoround").warning(
                f"Failed to inject SpinQuant buffers: {e}"
            )

    def save_config(self, model, save_dir):
        """Config persistence. Logic from old export.py
        _save_spinquant_config_to_dir()."""
        config = self._get_model_config(model)
        if config is None:
            return
        from .serialize import save_spinquant_config
        try:
            save_spinquant_config(model, save_dir, config)
        except Exception as e:
            import logging
            logging.getLogger("autoround").warning(
                f"Failed to save SpinQuant config: {e}"
            )

    # ── RotationSerializer: Load side ──

    def preregister_buffers(self, model, config_dict):
        from .serialize import preregister_spinquant_buffers
        return preregister_spinquant_buffers(model, config_dict)

    def rebuild_online(self, model):
        from .serialize import rebuild_spinquant_online
        return rebuild_spinquant_online(model)

    def has_rotation_buffers(self, module):
        return (
            hasattr(module, "spinquant_r1_type")
            or hasattr(module, "spinquant_r4_type")
        )

    @classmethod
    def config_key(cls):
        return "spinquant_config"

    # ── Private helper ──

    @staticmethod
    def _get_model_config(model):
        """Get rotation config from model, with legacy fallback."""
        config = getattr(model, "_rotation_config", None)
        if config is None:
            config = getattr(model, "_spinquant_config", None)
        return config
```

### 4.7 export 文件改动（Before → After）

#### 4.7.1 export.py pack_layer()

```python
# ── BEFORE (L217-219) ──
_inject_spinquant_buffers_on_layer(layer_name, qlayer, model)

# ── AFTER ──
from auto_round.algorithms.transforms import inject_rotation_buffers_on_layer
inject_rotation_buffers_on_layer(layer_name, qlayer, model)
```

#### 4.7.2 export.py save_quantized_as_autoround()

```python
# ── BEFORE (L365, L398) ──
_inject_spinquant_rotation_buffers(model, quantization_config)
# ... save_model(...) ...
_save_spinquant_config_to_dir(model, model_output_dir)

# ── AFTER ──
from auto_round.algorithms.transforms import (
    inject_rotation_buffers_bulk,
    save_rotation_config,
)
inject_rotation_buffers_bulk(model, quantization_config)
# ... save_model(...) ...
save_rotation_config(model, model_output_dir)
```

#### 4.7.3 export_to_nvfp_mx.py 和 export_to_fp8.py

**完全相同的替换模式**。三个文件统一调用同一组通用函数。

#### 4.7.4 删除的代码

`export.py` 中以下函数整体删除（逻辑已移入 `SpinQuantRotation`）：

| 函数 | 行号 | 说明 |
|------|------|------|
| `_inject_spinquant_buffers_on_layer()` | L225-264 | → `SpinQuantRotation.inject_buffers_on_layer()` |
| `_inject_spinquant_rotation_buffers()` | L403-423 | → `SpinQuantRotation.inject_buffers_bulk()` |
| `_save_spinquant_config_to_dir()` | L426-438 | → `SpinQuantRotation.save_config()` |

#### 4.7.5 convert_model.py

```python
# ── BEFORE (L758) ──
_rebuild_spinquant_if_needed(model)

# ── AFTER ──
from auto_round.algorithms.transforms import rebuild_rotation_if_needed
rebuild_rotation_if_needed(model)


# ── BEFORE (L881-894) ──
spinquant_config = getattr(quantization_config, "spinquant_config", None)
if spinquant_config is None and isinstance(quantization_config, dict):
    spinquant_config = quantization_config.get("spinquant_config")
if spinquant_config is not None and spinquant_config:
    preregister_spinquant_buffers(model, spinquant_config)

# ── AFTER ──
from auto_round.algorithms.transforms import preregister_rotation_buffers
preregister_rotation_buffers(model, quantization_config)
```

**删除**: `_rebuild_spinquant_if_needed()` 函数（L761-779）

### 4.8 buffer 前缀保持不变

Buffer 前缀 `spinquant_r1_*` / `spinquant_r4_*` **不改**。理由：

1. 前缀含方法名可以**避免不同 rotation 方法的 buffer 冲突**
2. 修改前缀会**破坏已保存模型的兼容性**
3. 新 rotation 方法使用自己的前缀（如 `myrot_r1_*`），自然隔离
4. 检测逻辑已移入 `has_rotation_buffers()` 方法，不再由 export/convert_model 直接判断

### 4.9 无 rotation 时的执行路径

```python
# export.py pack_layer() — 无 rotation 时：
inject_rotation_buffers_on_layer(layer_name, qlayer, model)
  → _get_serializer(model)
    → getattr(model, "_rotation_config", None)  # None
    → getattr(model, "_spinquant_config", None)  # None
    → return None
  → return  # no-op, 2 次 getattr + None 比较
```

开销极低，且无 import 开销（分发函数在 transforms/__init__.py，已加载）。

### 4.10 新增 rotation 方法的完整步骤

假设要添加 "my_rotation" 方法：

```
1. 创建 transforms/my_rotation/ 目录
   ├── __init__.py
   ├── config.py           # MyRotationConfig(BaseRotationConfig)
   ├── preprocessor.py     # 预处理逻辑
   └── serialize.py        # buffer 注入/预注册/重建的具体实现

2. 实现并注册：
   @BaseRotation.register("my_rotation")
   class MyRotation(BaseRotation, RotationSerializer):
       def apply_to_model(self, ...): ...

       # RotationSerializer
       def inject_buffers_on_layer(self, ...): ...
       def inject_buffers_bulk(self, ...): ...
       def save_config(self, ...): ...
       def preregister_buffers(self, ...): ...
       def rebuild_online(self, ...): ...
       def has_rotation_buffers(self, ...): ...

       @classmethod
       def config_key(cls): return "my_rotation_config"

3. 在 transforms/base.py _ensure_registry_populated() 中加入:
   for sub in ("rotation", "spinquant", "my_rotation"):

4. export.py、export_to_nvfp_mx.py、export_to_fp8.py、
   convert_model.py —— 无需任何修改 ✅
```

### 4.11 类图总览

```
transforms/base.py
├── BaseRotationConfig          (dataclass)
├── BaseRotation(ABC)           (existing)
│   ├── _REGISTRY: dict
│   ├── apply_to_model()        (abstract)
│   ├── register()              (classmethod decorator)
│   └── from_config()           (classmethod factory)
└── RotationSerializer(ABC)     (NEW mixin)
    ├── inject_buffers_on_layer()   (abstract)
    ├── inject_buffers_bulk()       (abstract)
    ├── save_config()               (abstract)
    ├── preregister_buffers()       (abstract)
    ├── rebuild_online()            (abstract)
    ├── has_rotation_buffers()      (abstract)
    └── config_key()                (abstract classmethod)

transforms/__init__.py          (NEW dispatch functions)
├── inject_rotation_buffers_on_layer()   → serializer.inject_buffers_on_layer()
├── inject_rotation_buffers_bulk()       → serializer.inject_buffers_bulk()
├── save_rotation_config()               → serializer.save_config()
├── preregister_rotation_buffers()       → serializer.preregister_buffers()
└── rebuild_rotation_if_needed()         → serializer.rebuild_online()

spinquant/algorithm.py
└── SpinQuantRotation(BaseRotation, RotationSerializer)
    ├── apply_to_model()                 (existing, unchanged)
    ├── inject_buffers_on_layer()        (delegates to serialize.py)
    ├── inject_buffers_bulk()            (delegates to serialize.py)
    ├── save_config()                    (delegates to serialize.py)
    ├── preregister_buffers()            (delegates to serialize.py)
    ├── rebuild_online()                 (delegates to serialize.py)
    ├── has_rotation_buffers()           (checks spinquant_r1/r4 prefixes)
    └── config_key() → "spinquant_config"

export.py / export_to_nvfp_mx.py / export_to_fp8.py
└── pack_layer()
    └── inject_rotation_buffers_on_layer()   # 1-line generic call
└── save_quantized_as_*()
    ├── inject_rotation_buffers_bulk()       # 1-line generic call
    └── save_rotation_config()               # 1-line generic call

convert_model.py
├── convert_hf_model()
│   └── preregister_rotation_buffers()       # 1-line generic call
└── post_init()
    └── rebuild_rotation_if_needed()         # 1-line generic call
```

---

## 5. 实施计划

### Phase 1: 基础设施（不改行为，纯新增）

| Step | 文件 | 改动 |
|------|------|------|
| 1.1 | `transforms/base.py` | 新增 `RotationSerializer` ABC |
| 1.2 | `transforms/__init__.py` | 新增 5 个通用分发函数 + `__all__` 更新 |
| 1.3 | `spinquant/algorithm.py` | `SpinQuantRotation` 实现 `RotationSerializer` |

### Phase 2: 替换调用点（改行为，一一对应）

| Step | 文件 | 改动 |
|------|------|------|
| 2.1 | `spinquant/preprocessor.py` | `_spinquant_config` → `_rotation_config` (保留 alias) |
| 2.2 | `export.py` | 替换 pack_layer + save 中的 3 处调用 |
| 2.3 | `export_to_nvfp_mx.py` | 替换 pack_layer + save 中的 2 处调用 |
| 2.4 | `export_to_fp8.py` | 替换 pack_layer + save 中的 2 处调用 |
| 2.5 | `convert_model.py` | 替换 preregister + rebuild 的 2 处调用 |
| 2.6 | `export.py` | 删除 3 个旧 spinquant 包装函数 |
| 2.7 | `convert_model.py` | 删除 `_rebuild_spinquant_if_needed()` |

### Phase 3: JSON key 迁移

| Step | 文件 | 改动 |
|------|------|------|
| 3.1 | `serialize.py` `save_spinquant_config()` | 同时写入 `"rotation_config"` + `"spinquant_config"` |
| 3.2 | `preregister_rotation_buffers()` | 优先读 `"rotation_config"`，fallback `"spinquant_config"` |

### Phase 4: 验证

| Step | 测试 | 验证内容 |
|------|------|---------|
| 4.1 | `test_save_load_roundtrip.py` 全组合 | 新代码 save/load 正确 |
| 4.2 | 用旧模型（只有 `"spinquant_config"`）加载 | 向后兼容 |
| 4.3 | `test_rotation_scheme_matrices.py` | in-memory 路径不受影响 |
| 4.4 | 无 rotation 的正常量化流程 | 无回归 |

---

## 6. 方案对比（A/B/C 参考）

| 维度 | 方案 A (轻量分发) | **方案 B (Serializer 接口)** ✅ | 方案 C (Hook 机制) |
|------|-------------------|-------------------------------|---------------------|
| **export 文件改动** | 小 | 中 | 大 |
| **新 rotation 方法成本** | 加 elif 分支 | **实现接口 + 注册** | 写 hook + 注册 |
| **pack_layer 侵入** | 1行通用调用 | **1行通用调用** | 1行 fire_hook |
| **DRY** | 好 | **好** | 最好 |
| **向后兼容** | 最好 | **好** | 需 migration |
| **OCP (开闭原则)** | ❌ 需改分发函数 | **✅ 只需注册** | ✅ 只需注册 |
| **调试友好** | 最好 | **好** | 一般 |
| **实现时间** | 短 | **中** | 长 |

---

## 附录: 当前 "spinquant" 硬编码位置完整清单

| 文件 | 行号 | 硬编码内容 | 重构后替换为 |
|------|------|-----------|-------------|
| `preprocessor.py` | L308 | `model._spinquant_config` | `model._rotation_config` + alias |
| `export.py` | L219 | `_inject_spinquant_buffers_on_layer()` | `inject_rotation_buffers_on_layer()` |
| `export.py` | L225-264 | 函数定义 | 删除（移入 SpinQuantRotation） |
| `export.py` | L365 | `_inject_spinquant_rotation_buffers()` | `inject_rotation_buffers_bulk()` |
| `export.py` | L398 | `_save_spinquant_config_to_dir()` | `save_rotation_config()` |
| `export.py` | L403-438 | 两个函数定义 | 删除（移入 SpinQuantRotation） |
| `export_to_nvfp_mx.py` | L122 | `_inject_spinquant_buffers_on_layer()` | `inject_rotation_buffers_on_layer()` |
| `export_to_nvfp_mx.py` | L253 | `_inject_spinquant_rotation_buffers()` | `inject_rotation_buffers_bulk()` |
| `export_to_nvfp_mx.py` | L277 | `_save_spinquant_config_to_dir()` | `save_rotation_config()` |
| `export_to_fp8.py` | L188 | `_inject_spinquant_buffers_on_layer()` | `inject_rotation_buffers_on_layer()` |
| `export_to_fp8.py` | L264 | `_inject_spinquant_rotation_buffers()` | `inject_rotation_buffers_bulk()` |
| `export_to_fp8.py` | L292 | `_save_spinquant_config_to_dir()` | `save_rotation_config()` |
| `convert_model.py` | L758 | `_rebuild_spinquant_if_needed()` | `rebuild_rotation_if_needed()` |
| `convert_model.py` | L761-779 | 函数定义 | 删除 |
| `convert_model.py` | L884-894 | `preregister_spinquant_buffers()` | `preregister_rotation_buffers()` |
| `serialize.py` | L52-53 | `_R1_PREFIX = "spinquant_r1"` | 保持不变（方法内部命名） |
| `serialize.py` | L160 | `["spinquant_config"]` | 双写 `rotation_config` + `spinquant_config` |
