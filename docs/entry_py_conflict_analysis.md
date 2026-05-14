# entry.py 冲突分析文档

## 文件信息

- **Main branch (refactor)**:  `refactor/auto-round/auto_round/compressors/entry.py` (585 行)
- **Local branch (new_commit)**: `new_commit/auto-round/auto_round/compressors_new/entry.py` (537 行)

注意：目录名也不同 — main 用 `compressors/`，local 用 `compressors_new/`。

---

## 冲突 1：import 路径（`compressors` vs `compressors_new`）

**性质**：目录重命名导致的 import 路径差异

```python
# === Main (refactor) ===
from auto_round.compressors.data_driven import CalibratedRTNCompressor, DataDrivenCompressor
from auto_round.compressors.utils import check_need_act_calibration
from auto_round.compressors.zero_shot import ZeroShotCompressor
from auto_round.compressors.mllm_mixin import MLLMMixin
from auto_round.compressors.diffusion_mixin import DiffusionMixin

# === Local (new_commit) ===
from auto_round.compressors_new.calib import CalibCompressor, CalibratedRTNCompressor
from auto_round.compressors_new.utils import check_need_act_calibration
from auto_round.compressors_new.zero_shot import ZeroShotCompressor
from auto_round.compressors_new.mllm_mixin import MLLMMixin
from auto_round.compressors_new.diffusion_mixin import DiffusionMixin
```

另外 local 还有类名变化：`DataDrivenCompressor` → `CalibCompressor`。

**建议**：根据最终目录结构决定。类名 `CalibCompressor` vs `DataDrivenCompressor` 需确认哪个是最终命名。

---

## 冲突 2：Local 新增 SpinQuant/QuaRot import

**性质**：Local 新增，Main 没有

```python
# === Local 新增的 import ===
from auto_round.algorithms.transforms import normalize_rotation_config as _normalize_any_rotation_config
from auto_round.algorithms.transforms.base import BaseRotationConfig as _BaseRotationConfig
```

**建议**：保留 local 的（SpinQuant/QuaRot 功能需要）。

---

## 冲突 3：`is_gguf_k_target()` 函数

**性质**：Main 有完整实现，Local 删除了该函数

```python
# === Main 有，Local 删除了 ===
def is_gguf_k_target(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized.startswith("gguf:") and "_k" in normalized
    if isinstance(value, AutoScheme):
        opts = value.options
        if isinstance(opts, str):
            opts = [opts]
        if isinstance(opts, (list, tuple)):
            return any(isinstance(opt, str) and is_gguf_k_target(opt) for opt in opts)
    return False
```

Local 在使用处用简单字符串检查替代：

```python
# === Main 调用处 ===
has_gguf_k = is_gguf_k_target(format) or is_gguf_k_target(scheme)

# === Local 调用处（替代逻辑） ===
has_gguf_k = "gguf" in format.lower() and "_k" in format.lower() if format else False
```

**建议**：保留 main 的 `is_gguf_k_target()` 函数及其调用，它更完善（支持 `AutoScheme` 对象）。

---

## 冲突 4：`SKIP_ARGS` 和 `local_args` 构建方式

**性质**：两种不同的参数传递策略

```python
# === Main：显式构建参数字典 ===
class AutoRound(object):
    # 没有 SKIP_ARGS

    def __new__(...):
        local_args = dict(
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            scheme=scheme,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            layer_config=layer_config,
            nsamples=nsamples,
            seqlen=seqlen,
        )

# === Local：用 locals() 过滤 ===
class AutoRound(object):
    SKIP_ARGS = ("local_args", "kwargs", "cls", "alg_configs", "quant_config", "quant_configs")

    def __new__(...):
        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}
```

**建议**：Main 的显式写法更安全，不会意外泄露 locals() 中的临时变量。但 local 的更灵活（新增参数不需要手动加）。需权衡。

---

## 冲突 5：Mixin 参数清理

**性质**：Main 新增，Local 没有

```python
# === Main 有，Local 没有 ===
# Pop kwargs that are only consumed by specific Mixins so they don't
# leak through to BaseCompressor as unrecognized keys.
if model_type != "diffusion":
    for _k in ("guidance_scale", "num_inference_steps", "generator_seed"):
        kwargs.pop(_k, None)
if model_type != "mllm":
    for _k in ("processor", "image_processor", "template", "extra_data_dir", "quant_nontext_module"):
        kwargs.pop(_k, None)
kwargs.pop("disable_opt_rtn", None)  # consumed by RTN routing above, not a compressor param
```

**建议**：合入 main 的。这段防止不相关的 kwargs 泄漏到 BaseCompressor，避免意外错误。

---

## 冲突 6：Model-free 快速路径

**性质**：Main 新增完整功能，Local 没有

```python
# === Main 有，Local 没有 ===
from auto_round.utils.model import is_model_free_route

# ---- Model-free fast-path detection --------------------------------
if is_model_free_route(model, scheme, iters, kwargs.get("disable_opt_rtn"), kwargs):
    from auto_round.compressors.model_free import ModelFreeCompressor

    if not isinstance(model, str):
        raise ValueError("model_free=True requires `model` to be a HuggingFace ID or local path string.")
    if not bool(kwargs.get("model_free", False)):
        logger.info(
            "Auto-routing to model-free quantization "
            "(iters=0, disable_opt_rtn=True, supported scheme). "
            "Pass disable_model_free=True to use the regular flow."
        )
    return ModelFreeCompressor(
        model_name_or_path=model,
        scheme=scheme,
        layer_config=layer_config,
        tokenizer=tokenizer,
        device_map=device_map,
        **kwargs,
    )
# --------------------------------------------------------------------
```

**建议**：合入 main 的。`ModelFreeCompressor` 是 main 的新功能（不加载模型直接量化）。

---

## 冲突 7：`_resolve_config()` 中 SpinQuant/QuaRot 支持

**性质**：Local 新增，Main 没有

```python
# === Local 新增 ===
@classmethod
def _resolve_config(cls, config):
    if isinstance(config, str):
        key = config.strip().lower()
        # Handle spinquant/quarot via unified normalizer
        if key in ("spinquant", "quarot"):
            return _normalize_any_rotation_config(key)
        ...
```

**建议**：保留 local 的（SpinQuant/QuaRot 功能）。

---

## 冲突 8：`batch_size` 传递位置

**性质**：参数放置位置不同

```python
# === Main：batch_size 传给 AutoRound 构造函数 ===
compressor = AutoRound(
    ...
    seqlen=seqlen,
    batch_size=batch_size,  # ← 在这里
    # MLLM parameters
    processor=processor,
    ...
)

# === Local：batch_size 传给 RTNConfig / SignRoundConfig ===
config = RTNConfig(
    ...
    disable_opt_rtn=disable_opt_rtn,
    batch_size=batch_size,  # ← 在这里
    **common_config_kwargs,
)

config = SignRoundConfig(
    iters=iters,
    batch_size=batch_size,  # ← 在这里
    ...
)

# 而 Local 的 AutoRound 构造函数中没有 batch_size
compressor = AutoRound(
    ...
    seqlen=seqlen,
    # 没有 batch_size
    processor=processor,
    ...
)
```

**建议**：需确认 `batch_size` 应该在 config 里还是 AutoRound 构造函数里。两种方式不能同时存在（会导致重复或不一致）。

---

## 冲突 9：`rotation_config` 处理逻辑

**性质**：Local 扩展了 rotation_config 的类型支持

```python
# === Main ===
_rotation_config_raw = kwargs.pop("rotation_config", None)
if _rotation_config_raw is not None:
    if isinstance(_rotation_config_raw, _NewArchRotationConfig):   # 只检查 RotationConfig
        _rc = _rotation_config_raw
    elif isinstance(_rotation_config_raw, dict):
        _rc = _NewArchRotationConfig.model_validate(_rotation_config_raw)  # 只能解析为 RotationConfig
    else:
        _rc = _NewArchRotationConfig()  # fallback: 默认 RotationConfig
    config = [config, _rc]              # 直接追加

# === Local ===
_rotation_config_raw = kwargs.pop("rotation_config", None)
if _rotation_config_raw is not None:
    if isinstance(_rotation_config_raw, _BaseRotationConfig):      # 检查基类（含 SpinQuantConfig）
        _rc = _rotation_config_raw
    elif isinstance(_rotation_config_raw, dict):
        _rc = _normalize_any_rotation_config(_rotation_config_raw) # 统一 normalizer，支持多种 config
    elif isinstance(_rotation_config_raw, str):                    # 新增：字符串快捷方式
        _rc = _normalize_any_rotation_config(_rotation_config_raw) # "quarot", "spinquant" 等
    else:
        _rc = _NewArchRotationConfig()
    if _rc is not None:                                            # 新增：None 检查
        config = [config, _rc]
```

**建议**：保留 local 的。它是 main 的超集，向后兼容且支持 SpinQuant/QuaRot。

---

## 冲突 10：日志级别

**性质**：minor

```python
# === Main ===
logger.warning_once("Using MLLM mode for multimodal model (new architecture).")
logger.warning_once("Using Diffusion mode for diffusion model (new architecture).")
logger.warning_once("Using LLM mode (new architecture).")

# === Local ===
logger.info("Using MLLM mode for multimodal model (new architecture).")
logger.info("Using Diffusion mode for diffusion model (new architecture).")
logger.info("Using LLM mode (new architecture).")
```

**建议**：Main 用 `warning_once` 是为了避免 LLM-Compressor 多次实例化时重复打印。可选择 main 的。

---

## 冲突 11：`_resolved` 变量

**性质**：Main 有一个未使用的变量

```python
# === Main ===
elif isinstance(quant_config, RTNConfig):
    enable_imatrix = False
    _resolved = {}              # ← Main 有，但似乎未使用
    disable_opt_rtn = ...

# === Local ===
elif isinstance(quant_config, RTNConfig):
    enable_imatrix = False
                                # ← Local 删除了 _resolved
    disable_opt_rtn = ...
```

**建议**：删除（local 的处理正确，如果 main 没有用到就不需要）。

---

## 合并策略总结

| # | 冲突 | 建议 | 理由 |
|---|---|---|---|
| 1 | import 路径 `compressors` vs `compressors_new` | 根据最终目录 | 机械替换 |
| 2 | SpinQuant/QuaRot import | **保留 local** | 新功能 |
| 3 | `is_gguf_k_target()` | **用 main** | 更完善，支持 AutoScheme |
| 4 | `SKIP_ARGS` / `local_args` | **用 main** | 显式更安全 |
| 5 | Mixin 参数清理 | **合入 main** | 防止 kwargs 泄漏 |
| 6 | Model-free 快速路径 | **合入 main** | 新功能 |
| 7 | `_resolve_config` spinquant/quarot | **保留 local** | 新功能 |
| 8 | `batch_size` 位置 | **需确认** | config vs 构造函数 |
| 9 | `rotation_config` 处理 | **保留 local** | 超集，向后兼容 |
| 10 | 日志级别 | **用 main** | `warning_once` 防重复打印 |
| 11 | `_resolved` 变量 | **删除** | 未使用 |

### 合并优先级

- **必须保留 local 的**（3项）：冲突 2、7、9 — SpinQuant/QuaRot 核心功能
- **必须合入 main 的**（4项）：冲突 3、4、5、6 — main 的新功能和改进
- **需要确认**（1项）：冲突 8 — `batch_size` 放哪里
- **Minor**（3项）：冲突 1、10、11

---

## 完整 diff

```diff
--- refactor/auto-round/auto_round/compressors/entry.py     (main branch)
+++ new_commit/auto-round/auto_round/compressors_new/entry.py (local branch)

@@ imports @@
+from auto_round.algorithms.transforms import normalize_rotation_config as _normalize_any_rotation_config
+from auto_round.algorithms.transforms.base import BaseRotationConfig as _BaseRotationConfig
 from auto_round.algorithms.transforms.rotation.config import RotationConfig as _NewArchRotationConfig
-from auto_round.compressors.data_driven import CalibratedRTNCompressor, DataDrivenCompressor
-from auto_round.compressors.utils import check_need_act_calibration
-from auto_round.compressors.zero_shot import ZeroShotCompressor
+from auto_round.compressors_new.calib import CalibCompressor, CalibratedRTNCompressor
+from auto_round.compressors_new.utils import check_need_act_calibration
+from auto_round.compressors_new.zero_shot import ZeroShotCompressor

@@ _get_compressor_class @@
-        from auto_round.compressors.mllm_mixin import MLLMMixin
+        from auto_round.compressors_new.mllm_mixin import MLLMMixin
-        from auto_round.compressors.diffusion_mixin import DiffusionMixin
+        from auto_round.compressors_new.diffusion_mixin import DiffusionMixin

@@ is_gguf_k_target (main only) @@
-def is_gguf_k_target(value) -> bool:
-    ... (11 lines, deleted in local)

@@ AutoRound class @@
+    SKIP_ARGS = ("local_args", "kwargs", "cls", "alg_configs", "quant_config", "quant_configs")

@@ _resolve_config @@
+            if key in ("spinquant", "quarot"):
+                return _normalize_any_rotation_config(key)

@@ __new__ local_args @@
-        local_args = dict(model=model, tokenizer=tokenizer, ..., seqlen=seqlen)
+        local_args = {k: v for k, v in locals().items() if k not in cls.SKIP_ARGS}

@@ __new__ mixin cleanup (main only) @@
-        if model_type != "diffusion":
-            for _k in (...): kwargs.pop(_k, None)
-        if model_type != "mllm":
-            for _k in (...): kwargs.pop(_k, None)
-        kwargs.pop("disable_opt_rtn", None)

@@ SignRoundConfig routing @@
-            return _get_compressor_class(model_type, DataDrivenCompressor)(...)
+            return _get_compressor_class(model_type, CalibCompressor)(...)

@@ RTN routing @@
-            _resolved = {}
-            has_gguf_k = is_gguf_k_target(format) or is_gguf_k_target(scheme)
+            has_gguf_k = "gguf" in format.lower() and "_k" in format.lower() if format else False

@@ AutoRoundCompatible._to_new_api model-free (main only) @@
-        from auto_round.utils.model import is_model_free_route
-        if is_model_free_route(...):
-            ... return ModelFreeCompressor(...)

@@ RTNConfig batch_size (local only) @@
+                batch_size=batch_size,

@@ SignRoundConfig batch_size (local only) @@
+                batch_size=batch_size,

@@ rotation_config handling @@
-            if isinstance(_rotation_config_raw, _NewArchRotationConfig):
+            if isinstance(_rotation_config_raw, _BaseRotationConfig):
-            elif isinstance(_rotation_config_raw, dict):
-                _rc = _NewArchRotationConfig.model_validate(...)
+                _rc = _normalize_any_rotation_config(...)
+            elif isinstance(_rotation_config_raw, str):
+                _rc = _normalize_any_rotation_config(...)
-            config = [config, _rc]
+            if _rc is not None:
+                config = [config, _rc]

@@ logging @@
-            logger.warning_once(...)
+            logger.info(...)

@@ AutoRound constructor call @@
-            batch_size=batch_size,
```
