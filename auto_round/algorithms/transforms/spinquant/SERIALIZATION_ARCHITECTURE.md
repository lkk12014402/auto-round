# SpinQuant Rotation Serialization Architecture

> How SpinQuant rotation matrices are saved alongside quantized weights in auto-round,
> and why the integration point matters.

---

## 1. Problem Statement

SpinQuant/QuaRot applies **rotation matrices** (R1–R4) to model weights before
quantization.  At inference time, some rotations (R1 online, R4) must be applied
on-the-fly, so the rotation matrices need to be **persisted** in the saved
safetensors alongside quantized weights.

The challenge: auto-round's **NEW architecture** uses a `ShardWriter` that saves
tensors **per-layer** during quantization, then offloads modules to `meta` device
to free RAM.  Any buffer registered *after* the shard save is invisible to the
serialization pipeline.

---

## 2. auto-round NEW Architecture: Save Pipeline

```
quantize_and_save()                         # compressors_new/base.py
 ├─ _ensure_shard_writer()                  # creates singleton ShardWriter
 │
 ├─ quantize()                              # calib.py – the main calibration loop
 │   └─ for each transformer block (layer):
 │       ├─ calibrate / optimize weights
 │       │
 │       ├─ immediate_pack()                # compressors_new/utils.py L1226
 │       │   └─ format.immediate_pack()     # formats.py (base OutputFormat)
 │       │       └─ format.pack_layer()     # formats.py → AutoRoundFormat L1241
 │       │           └─ export.pack_layer() # export.py L141  ◄── OUR HOOK POINT
 │       │               ├─ create QuantLinear, call qlayer.pack()
 │       │               └─ _inject_spinquant_buffers_on_layer()  ← buffers registered HERE
 │       │
 │       ├─ shard_writer.write(m)           # shard_writer.py L327
 │       │   └─ save_module(m)              # iterates m.state_dict(), calls _add_tensor()
 │       │       └─ _add_tensor(name, tensor)
 │       │           ├─ SKIP if tensor.device == "meta"
 │       │           └─ accumulate into current shard, flush when shard full
 │       │
 │       └─ _offload_to_meta()              # shard_writer.py L246
 │           └─ module.to("meta")           # frees GPU/CPU RAM
 │
 ├─ save_quantized()                        # compressors_new/base.py L1115
 │   └─ format.save_quantized()
 │       └─ save_quantized_as_autoround()   # export.py L267
 │           ├─ pack_layer loop             # SKIPPED – modules already on meta
 │           ├─ _inject_spinquant_rotation_buffers()  # fallback for non-shard path
 │           ├─ save_model()                # writes config, etc.
 │           └─ _save_spinquant_config_to_dir()  # persists spinquant_config to config.json
 │
 └─ shard_writer.finalize()                 # renames shards, writes index JSON
     └─ model.state_dict() sweep            # captures remaining non-meta tensors
```

### Key Timing Constraint

```
 pack_layer()          ← buffers MUST be registered here (or earlier)
       ↓
 shard_writer.write()  ← saves module.state_dict() to safetensors shard
       ↓
 _offload_to_meta()    ← module moved to meta device, tensors unreachable
```

**Buffers registered after `shard_writer.write()` will never be saved** because:
1. `_add_tensor()` explicitly skips `meta` tensors (L167–169)
2. `_offload_to_meta()` moves the entire module to meta after saving (L258)
3. `finalize()` also skips meta tensors in its sweep (L272–273)

---

## 3. The Root Cause of the Original Bug

Our initial implementation injected spinquant buffers in
`save_quantized_as_autoround()` (export.py L361–365), which runs **after** the
entire quantization + shard-writing loop has completed.  By that point all
`QuantLinear` modules are on `meta` device — the buffers were registered but
their tensors were on meta and got skipped by every save path.

Additionally, even if we forced `device=cpu` for the buffer tensors, the
`_offload_to_meta()` check in ShardWriter looks at the module's **full**
`state_dict()`: if it sees new keys not in `_all_saved`, it won't offload, but
the buffers still won't be in any shard file because `write()` was already called.

---

## 4. The Fix: Per-Layer Buffer Injection in `pack_layer()`

**File: `auto_round/export/export_to_autoround/export.py`**

We inject spinquant buffers at the end of `pack_layer()` (L217–219), immediately
after `qlayer.pack()` and before the function returns:

```python
def pack_layer(layer_name, model, backend, device=None):
    ...
    qlayer.pack(layer, scale, zp, None, device=device)
    qlayer.to(orig_device)

    # Inject SpinQuant rotation buffers right after packing so that
    # ShardWriter.save_module() captures them before offloading to meta.
    _inject_spinquant_buffers_on_layer(layer_name, qlayer, model)
    ...
```

This works because the call chain is:

```
calib.py: immediate_pack()
  → formats.py: AutoRoundFormat.pack_layer()   # L1241
      → export.py: pack_layer()                 # L141 – OUR HOOK (L219)
          └─ return to formats.py
      └─ return to calib.py
  → shard_writer.write(m)                       # buffers are NOW in state_dict ✓
  → _offload_to_meta()                          # safe to offload
```

The crucial verification: `formats.py:AutoRoundFormat.pack_layer()` at L1269
delegates to `export.py:pack_layer()` for standard int-weight quantization:

```python
# formats.py L1268-1272
else:
    from auto_round.export.export_to_autoround.export import pack_layer
    pack_func = pack_layer
return pack_func(layer_name, model, backend, device)
```

### What `_inject_spinquant_buffers_on_layer()` Does

```python
def _inject_spinquant_buffers_on_layer(layer_name, qlayer, model):
    """Inject SpinQuant rotation buffers onto a single QuantLinear after packing."""
    spinquant_config = getattr(model, "_spinquant_config", None)
    if spinquant_config is None:
        return

    short_name = layer_name.split(".")[-1]  # e.g. "q_proj", "down_proj"

    # R1: applied to q/k/v/gate/up projections (online rotation)
    if spinquant_config.r1 and spinquant_config.online_r1_rotation:
        if short_name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            _inject_rotation_buffers(qlayer, "spinquant_r1", ...)

    # R4: applied to down_proj
    if spinquant_config.r4:
        if short_name in ("down_proj",):
            _inject_rotation_buffers(qlayer, "spinquant_r4", ...)
```

Each call to `_inject_rotation_buffers()` registers buffers on the `QuantLinear`:
- `spinquant_r1_type` (int8 scalar) — rotation type enum
- `spinquant_r1_size` (int64 scalar) — matrix dimension
- `spinquant_r1_matrix` (float16 tensor, optional) — the rotation matrix itself

---

## 5. Dual-Path Strategy: Why Two Injection Functions

### 5.1 Two Save Paths in auto-round

auto-round 有两条完全不同的保存路径，由 `_adjust_immediate_packing_and_saving()` 动态决定：

```python
# compressors_new/base.py L1035-1036
if len(formats) == 1 and not formats[0].is_fake() and self.inplace:
    is_immediate_packing = True    # → ShardWriter 路径
else:
    is_immediate_packing = False   # → 非 ShardWriter 路径
```

额外的禁用条件（即使满足上述条件也可能回退到非 ShardWriter）：
- `has_qlayer_outside_block and need_calib` → 强制 `False`
- 非 CausalLM 模型 + tied weights > 1 → 禁用 immediate_saving
- non-inplace 模式 → 不满足条件

### 5.2 Two Functions, Two Purposes

| 函数 | 调用路径 | 生效条件 |
|------|----------|----------|
| `_inject_spinquant_buffers_on_layer()` | `pack_layer()` 内部末尾 | ShardWriter 路径 (`immediate_packing=True`) |
| `_inject_spinquant_rotation_buffers()` | `save_quantized_as_autoround()` 末尾 | 非 ShardWriter 路径 (`immediate_packing=False`) |

### 5.3 为什么不能只用一个函数

**只用函数1（逐层注入）不行：**
- 非 ShardWriter 路径下，`save_quantized_as_autoround()` 中 `unsupported_meta_device(model)` 
  可能为 True，此时**跳过整个 `pack_layer` 循环**（L356），函数1 永远不会执行
- 即使 pack_layer 循环执行了，非 ShardWriter 路径下模型之后还要整体 `model.save_pretrained()`，
  函数1 逐层注入的时机虽然不错但不是必须

**只用函数2（批量注入）不行：**
- ShardWriter 路径下，`shard_writer.write(module)` 保存完 state_dict 后立即调用 
  `_offload_to_meta(module)`，模块被移到 meta device
- 函数2 在**所有层都处理完之后**才运行，此时所有模块已在 meta device
- 注入到 meta 模块上的 buffer 也是 meta tensor，而 `_add_tensor()` 跳过 meta：
  ```python
  # shard_writer.py L167
  if tensor.device.type == "meta":
      return  # ← 直接跳过！
  ```
- 且 shard 文件早已关闭写入

### 5.4 时序对比图

```
═══ ShardWriter 路径 (immediate_packing=True, 大模型/low_cpu_mem) ═══

for each layer:
    calib/rtn → quantize weights
    immediate_pack()
      → format.pack_layer()
          → export.py:pack_layer()
              qlayer.pack(layer, scale, zp)
              ┌─────────────────────────────────────────────────────┐
              │ _inject_spinquant_buffers_on_layer(name, qlayer, m) │ ← 函数1
              └─────────────────────────────────────────────────────┘
              return qlayer
    shard_writer.write(module)          ← state_dict 包含 buffers ✓
    _offload_to_meta(module)            ← 模块移到 meta，不可再写

# 之后...
_inject_spinquant_rotation_buffers(model)  ← 函数2 执行但注入 meta tensor (no-op)
save_model()                               ← 检测到 meta，只保存 configs

═══ 非 ShardWriter 路径 (immediate_packing=False, 小模型/特殊情况) ═══

for each layer:
    pack_layer(name)
        ┌─────────────────────────────────────────────────────┐
        │ _inject_spinquant_buffers_on_layer(name, qlayer, m) │ ← 函数1 也执行
        └─────────────────────────────────────────────────────┘
    # 模块仍在真实设备上 (不 offload)

# 之后...
┌─────────────────────────────────────────────────────────────────────────┐
│ _inject_spinquant_rotation_buffers(model, quantization_config)          │ ← 函数2
│   - 遍历所有 QuantLinear，批量注入 (覆盖或补充函数1 未处理的情况)        │
│   - 同时将 spinquant_config 写入 quantization_config dict               │
└─────────────────────────────────────────────────────────────────────────┘
model.save_pretrained(save_dir)          ← 整体保存，包含所有 buffers ✓
```

### 5.5 互不冲突的双层防御

两个函数在任一路径下**同时运行不会冲突**：

- **ShardWriter 路径**: 函数1 注入真实 tensor → shard 正确保存；函数2 之后运行注入 meta 
  tensor → 等效 no-op（meta tensor 不会被保存，且 shard 已写完）
- **非 ShardWriter 路径**: 函数1 注入真实 tensor → OK；函数2 之后批量注入 → 对已有 buffer 
  是幂等操作（`register_buffer` 对同名 buffer 会覆盖但值相同），对遗漏的 buffer 是补充

### 5.6 何时会走到非 ShardWriter 路径

| 场景 | 原因 |
|------|------|
| `inplace=False` | 条件不满足 |
| 多个 output formats | `len(formats) != 1` |
| 非 CausalLM + tied weights | 主动禁用 immediate_saving |
| `quantize()` 后手动 `save_pretrained()` | 不经过 shard writer |
| 测试/调试场景 | 直接调用 `save_quantized_as_autoround()` |

---

## 6. Multi-Scheme Coverage

### The Problem

`formats.py:AutoRoundFormat.pack_layer()` dispatches to **different** pack
functions depending on the quantization scheme:

```python
# formats.py L1247-1272
if output_format in [NV_FP, MX_FP, MX_FP_RCEIL, NV_FP4_WITH_STATIC_GS]:
    from export_to_nvfp_mx import pack_layer       # MXFP4, NVFP4, MX_INT
elif output_format in [FP8, FP8_STATIC]:
    from export_to_fp8 import pack_layer           # FP8
else:
    from export import pack_layer                  # W4A16, W3A16, W8A16 (INT)
```

Each `pack_layer` must inject spinquant buffers independently.

### Coverage Matrix

| Scheme Family | `pack_layer` File | Per-Layer Hook | Bulk Fallback | Config Save |
|---------------|-------------------|----------------|---------------|-------------|
| **INT** (W4A16, W3A16, W8A16) | `export.py` | ✅ L219 | ✅ L365 | ✅ L398 |
| **MXFP/NVFP** (MXFP4, NVFP4, MX_INT) | `export_to_nvfp_mx.py` | ✅ L119 | ✅ L244 | ✅ L277 |
| **FP8** (FP8, FP8_STATIC) | `export_to_fp8.py` | ✅ L185 | ✅ L260 | ✅ L288 |

### Implementation Pattern

All three files use the same pattern — importing the shared implementation
from `export.py`:

```python
# Per-layer (in pack_layer, after qlayer.pack + qlayer.to):
from auto_round.export.export_to_autoround.export import _inject_spinquant_buffers_on_layer
_inject_spinquant_buffers_on_layer(layer_name, qlayer, model)

# Bulk fallback (in save_quantized_as_*, after filter_quantization_config):
from auto_round.export.export_to_autoround.export import (
    _inject_spinquant_rotation_buffers,
    _save_spinquant_config_to_dir,
)
_inject_spinquant_rotation_buffers(model, quantization_config)

# Config persistence (at end of save, after save_model):
_save_spinquant_config_to_dir(model, output_dir)
```

### `iters` Parameter (AutoRound Tuning)

The `iters` parameter controls optimization iterations (0=RTN, >0=auto-round
tuning). It does **not** affect the pack/save path:

- `iters=0`: quantize weights via RTN → same pack_layer → same shard_writer.write
- `iters>0`: calibrate + optimize weights → same pack_layer → same shard_writer.write

The injection hook is in `pack_layer`, which runs regardless of how the weights
were quantized. Both RTN and tuned paths converge at `immediate_pack → pack_layer`.

### Two-Layer Defense

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Per-layer injection (inside pack_layer)            │
│  ─────────────────────────────────────────────────────────── │
│  Runs BEFORE shard_writer.write() in the NEW arch.           │
│  Ensures buffers exist in module.state_dict() when saved.    │
│  This is the PRIMARY mechanism for shard-based saving.       │
└─────────────────────────────────────────────────────────────┘
                           ↓ (if shard path skipped)
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Bulk injection (in save_quantized_as_*)            │
│  ─────────────────────────────────────────────────────────── │
│  Runs in the legacy non-shard path where modules are still   │
│  on real devices. Iterates all QuantLinear modules at once.  │
│  Also embeds spinquant_config in quantization_config dict.   │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Load Pipeline (Inference)

```
AutoModelForCausalLM.from_pretrained(model_path)
  │
  ├─ _process_model_before_weight_loading()
  │   └─ convert_hf_model()                    # convert_model.py L799
  │       ├─ replace Linear → QuantLinear
  │       └─ preregister_spinquant_buffers()    # L881-893
  │           ├─ reads spinquant_config from quantization_config
  │           └─ registers EMPTY buffers on QuantLinear modules
  │               (so state_dict loader won't drop unknown keys)
  │
  ├─ HuggingFace loads state_dict from safetensors
  │   └─ populates spinquant_r1_matrix, spinquant_r1_type, etc.
  │
  └─ _process_model_after_weight_loading()
      └─ post_init()                            # convert_model.py L755
          └─ _rebuild_spinquant_if_needed()      # L761
              └─ rebuild_spinquant_online()       # serialize.py
                  ├─ reads buffers from QuantLinear
                  ├─ reconstructs rotation matrices (Hadamard or trained)
                  └─ patches QuantLinear.forward() to apply rotations
```

### Why Pre-Registration Matters

HuggingFace's `load_state_dict(strict=False)` will **drop** any keys in the
safetensors that don't match an existing parameter or buffer in the model.  By
pre-registering empty buffers with the correct names, we ensure the spinquant
tensors are loaded from disk.

---

## 8. Config Persistence

SpinQuant configuration is saved to `config.json` in two ways:

1. **In `quantization_config`** — the `spinquant_config` dict is embedded inside
   the `quantization_config` field, read by `convert_hf_model()` at load time.

2. **Standalone `_save_spinquant_config_to_dir()`** — writes spinquant_config
   into the model's `config.json` for robustness.

The config contains: rotation levels (r1/r2/r3/r4), whether R1 is online,
rotation size, random flags, etc.

---

## 9. Comparison with Existing Rotation Serialization

auto-round's **existing rotation** (Hadamard for NVFP/MXFP) uses a different
approach in `auto_round/algorithms/transforms/rotation/patch.py`:

```python
def patch_quantlinear(w_transform):
    """Monkey-patches QuantLinear.pack() to register hadamard_matrix buffer."""
    def _pack_patched(self, linear, scales, ...):
        # ... normal pack logic ...
        self.register_buffer("hadamard_matrix", w_transform.weight)
    QuantLinear.pack = _pack_patched
```

This patches the `pack()` method itself, so the buffer is registered **inside**
the pack call, which is even earlier than our hook.  However, this approach:
- Requires global monkey-patching of `QuantLinear.pack`
- Is tied to a single rotation matrix (not per-module R1/R4 distinction)
- Doesn't scale to multiple rotation types per module

Our approach (post-pack injection in `pack_layer()`) is cleaner for the SpinQuant
use case where different modules get different rotation types.

---

## 10. Buffer Naming Convention

All spinquant buffers follow the naming pattern:

```
{module_path}.spinquant_{rotation}_{field}
```

Examples in safetensors:
```
model.layers.0.self_attn.q_proj.spinquant_r1_type      # int8
model.layers.0.self_attn.q_proj.spinquant_r1_size      # int64
model.layers.0.self_attn.q_proj.spinquant_r1_matrix    # float16 [N, N]
model.layers.0.mlp.down_proj.spinquant_r4_type         # int8
model.layers.0.mlp.down_proj.spinquant_r4_size         # int64
```

For Hadamard rotations, `_matrix` is omitted (reconstructed from type + size).
For trained/random rotations, `_matrix` stores the actual rotation matrix.

---

## 11. Summary of Files

| File | Role |
|------|------|
| `spinquant/serialize.py` | Core serialization: inject, preregister, rebuild functions |
| `export.py` → `pack_layer()` | INT scheme shard-path injection hook |
| `export.py` → `_inject_spinquant_buffers_on_layer()` | Shared per-layer injection logic (imported by all pack paths) |
| `export.py` → `save_quantized_as_autoround()` | INT non-shard fallback + config save |
| `export_to_nvfp_mx.py` → `pack_layer()` | MXFP/NVFP scheme shard-path injection hook |
| `export_to_nvfp_mx.py` → `save_quantized_as_fp()` | MXFP/NVFP non-shard fallback + config save |
| `export_to_fp8.py` → `pack_layer()` | FP8 scheme shard-path injection hook |
| `export_to_fp8.py` → `save_quantized_as_autoround()` | FP8 non-shard fallback + config save |
| `shard_writer.py` | ShardWriter singleton, saves tensors per-layer |
| `calib.py` L1034–1045 | Orchestration: pack → write → offload |
| `formats.py` L1241–1272 | Dispatches to correct `pack_layer()` by scheme |
| `convert_model.py` L881–893 | Load-time buffer pre-registration |
| `convert_model.py` L755–779 | Load-time forward patching |
| `rotation/patch.py` L126 | Reference: existing rotation serialization |


## 12. Serialization Path vs In-Memory Path: Independence Analysis

SpinQuant rotations有两条完全独立的执行路径。理解它们的边界对于调试和测试至关重要。

### 12.1 两条路径总览

| | 序列化路径 (Save / Load) | 内存路径 (In-Memory Evaluation) |
|--|--|--|
| **典型入口** | `serialize.py` | `preprocessor.py` |
| **操作对象** | `QuantLinear` / FP推理模块 (`NVFP4QuantLinear` 等) | 原始 `nn.Linear` |
| **调用 `_is_quantlinear()`** | ✅ (4处) | ❌ (0处) |
| **代表脚本** | `test_save_load_roundtrip.py` | `test_rotation_scheme_matrices.py` |
| **rotation 生效方式** | buffer → forward patch / hook rebuild | 直接修改权重 + 注册 hook |

### 12.2 In-Memory 路径详解 (preprocessor.py)

`preprocessor.py` 直接操作 `nn.Linear` 模块的权重和注意力模块的 hook，**完全不依赖 `serialize.py`**：

- **R1 (offline)**: 直接修改 `nn.Linear.weight`，rotation 矩阵乘进权重 → 量化时被 pack 进 `QuantLinear` → ✅
- **R2 (offline)**: 同 R1，fuse 进 `nn.Linear.weight` → ✅
- **R3 (hook)**: hook 注册在 `self_attn` 模块上（不是 `q_proj`/`k_proj` 等 linear 子模块）→ 量化替换子模块不影响父模块的 hook → ✅
- **R4 (hook)**: 同 R3，hook 注册在上层模块 → ✅

```
preprocessor.py flow:
  nn.Linear weights ──[R1/R2 fuse]──> modified nn.Linear
  self_attn module ──[R3/R4 hook]──> hook on parent module
                                           │
                                     quantization replaces
                                     nn.Linear → QuantLinear
                                           │
                                     hooks on self_attn survive
                                           ▼
                                     lm_eval evaluation ✅
```

### 12.3 Serialization 路径详解 (serialize.py)

序列化路径仅在 save/load 时使用，通过 `_is_quantlinear()` 识别量化模块：

**Save 侧**（在 `pack_layer()` 后）:
- `_inject_rotation_buffers()` → 遍历 `QuantLinear` 模块，将 rotation 数据写入 buffer
- `_is_quantlinear()` 用于识别目标模块

**Load 侧**（在 `convert_hf_model()` 中）:
- `preregister_spinquant_buffers()` → 预注册空 buffer，确保 state_dict 加载不丢弃 rotation keys
- `rebuild_spinquant_online()` → 重建 R3 hook + patch forward
- `_patch_quantlinear_forward_spinquant()` → patch R1/R4 online rotation
- 以上三个函数都依赖 `_is_quantlinear()` 识别模块

### 12.4 `_is_quantlinear()` 的匹配范围

`_is_quantlinear()` 需要识别所有量化线性层变体：

| 模块类型 | 出现场景 | 匹配方式 |
|----------|----------|----------|
| `QuantLinear` | INT W4A16/W3A16/W8A16 (export + inference) | 精确类名匹配 |
| `NVFP4QuantLinear` | NVFP4 inference (load 侧) | 子串 `"QuantLinear" in cls_name` |
| `MXFP4QuantLinear` | MXFP4 inference (load 侧) | 子串匹配 |
| `MXFP8QuantLinear` | MXFP8 inference (load 侧) | 子串匹配 |
| `MXINT4QuantLinear` | MXINT4 inference (load 侧) | 子串匹配 |
| `WeightFP8ActFP8StaticQuantLinear` | FP8 static inference (load 侧) | 子串匹配 |
| `QModuleBase` 子类 | 以上 FP 模块的公共基类 | MRO 检查 |

> **历史 Bug**: 旧版 `_is_quantlinear()` 只检查 `cls_name == "QuantLinear" and hasattr(module, "bits")`，
> 导致 FP 系列推理模块（无 `bits` 属性、类名不同）全部不匹配，load 侧丢失所有 spinquant buffer。
> 已修复为子串匹配 + MRO 检查。

### 12.5 关键结论

1. **`serialize.py` 的 bug 不影响 in-memory evaluation**：`test_rotation_scheme_matrices.py` 从内存直接评估，不走序列化，结果不受 `_is_quantlinear()` bug 影响。
2. **`preprocessor.py` 不依赖 `serialize.py`**：两者之间没有 import 关系，没有共享函数调用。
3. **hook 存活性**：R3/R4 hook 注册在 attention 父模块上，量化替换 linear 子模块不会破坏 hook。
4. **测试策略**：
   - 验证 rotation 正确性 → 用 in-memory 路径 (`test_rotation_scheme_matrices.py`)
   - 验证序列化正确性 → 用 save/load 路径 (`test_save_load_roundtrip.py`)
   - 两条路径的 lm_eval 结果应一致（同 rotation + 同 scheme → 同 accuracy）

---

## 13. Rotation Save/Load 完整逻辑总览

本节给出 R1–R4 保存/加载的全流程 end-to-end 描述，包含 random/trained 矩阵支持（2026-05 更新）。

### 13.1 各 Rotation 的保存策略

| Rotation | 类型 | 保存方式 | 加载方式 |
|----------|------|----------|----------|
| **R1 offline** | 权重 fuse | 已乘进 `nn.Linear.weight`（`W_new = R^T @ W`），量化时被 pack 进 `QuantLinear`，**无需额外保存** | 直接加载权重即可 |
| **R1 online** | 运行时 hook | 在每个目标 QuantLinear（q/k/v/gate/up_proj）上注入 buffer | 加载后 patch `QuantLinear.forward`，自动 `x @ R` |
| **R2** | 权重 fuse | 已 fuse 进 `v_proj`（`R^T @ W`）和 `o_proj`（`W @ R`）权重，**无需额外保存** | 直接加载权重即可 |
| **R3** | 运行时 hook | **不存 buffer，也不存矩阵**，只在 `config.json` 记录 `r3: true` + `head_dim` | 加载后从 config 读取，重新注册 `QKRotationWrapper` monkeypatch |
| **R4** | 运行时 hook + fuse | 在每个 `down_proj` QuantLinear 上注入 buffer；`down_proj.weight` 已 fuse（`W @ R`） | 加载后 patch `QuantLinear.forward`，对输入做 `x @ R` |

> **为什么 R3 不需要存矩阵（包括 random 模式）？**
>
> R3 对 Q 和 K 施加**相同的**正交矩阵 R（在 RoPE 之后）：
> ```
> score = (Q @ R)(K @ R)^T = Q @ R @ R^T @ K^T = Q @ I @ K^T = Q @ K^T
> ```
> R 在注意力分数中完全抵消。因此：
> - **Deterministic R3**：从 `head_dim` 重建 Hadamard 即可
> - **Random R3**：任意正交 R 都能正确抵消，加载时重新生成即可，无需保存
> - 只需要 `config.json` 中的 `r3: true` + `random_r3: true` 标记

### 13.2 Buffer 存储格式（三种类型）

每个需要 buffer 的 QuantLinear 模块上注册以下 tensor：

| Buffer 名称 | dtype | 说明 |
|-------------|-------|------|
| `{prefix}_type` | `int32` 标量 | `0`=Hadamard, `1`=Random, `2`=Trained |
| `{prefix}_size` | `int32` 标量 | rotation 块大小（如 896, 4864） |
| `{prefix}_matrix` | `int8` 或 `float32` | 仅 Random/Trained 类型需要 |

其中 `{prefix}` 为 `spinquant_r1`（R1）或 `spinquant_r4`（R4）。

三种 rotation 类型的存储和推理行为：

| Type | `_type` 值 | `_matrix` | 推理时的计算 | 复杂度 |
|------|-----------|-----------|-------------|--------|
| **Deterministic Hadamard** | `0` | 不存储 | 从 `_size` 重建 Hadamard 矩阵，用 butterfly 算法 `matmul_hadU(x)` | O(n log n) |
| **Random Hadamard** | `1` | `int8` (±1) | `x @ (matrix.float() / √n)` | O(n²) |
| **Trained Orthogonal** | `2` | `float32` | `x @ matrix` | O(n²) |

> **为什么 Random 存 int8？**
> Random Hadamard = `diag(±1) @ H`，其中 `diag(±1)` 只翻转 Hadamard 矩阵行的符号。
> 结果矩阵的每个元素仍然是 ±1（Hadamard 矩阵本身全是 ±1），所以可以用 int8 存储（节省空间 4×），
> 推理时转 float 并除以 √n 还原正交矩阵。

### 13.3 Save 三步流程

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 1: inject_spinquant_buffers(model, config)                │
 │  ─────────────────────────────────────────────────────────────── │
 │  时机：量化 pack 之后、save_pretrained 之前                       │
 │  操作：遍历所有 QuantLinear，按 config 中的 r1/r4 flag 决定       │
 │        是否注入 buffer。register_buffer 使其进入 state_dict。     │
 │                                                                  │
 │  inject 逻辑：                                                   │
 │  • R1 online → q/k/v/gate/up_proj 各注入 _r1_type, _r1_size     │
 │    - deterministic: 只存 type + size                             │
 │    - random: 额外存 _r1_matrix (int8, 从 model.spinquant_R1 读) │
 │    - trained: 额外存 _r1_matrix (float32)                       │
 │  • R4 → down_proj 注入 _r4_type, _r4_size                       │
 │    - random: 额外存 _r4_matrix (int8, 从 model.spinquant_R4_matrix 读) │
 └──────────────────────────────────────────────────────────────────┘
                              ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 2: model.save_pretrained(save_dir)                        │
 │  ─────────────────────────────────────────────────────────────── │
 │  HuggingFace 标准保存，将 state_dict（含 spinquant buffer）     │
 │  写入 safetensors 文件。                                         │
 └──────────────────────────────────────────────────────────────────┘
                              ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 3: save_spinquant_config(model, save_dir, config)         │
 │  ─────────────────────────────────────────────────────────────── │
 │  将 spinquant_config dict 写入 config.json 的                    │
 │  quantization_config.spinquant_config 字段。                     │
 │                                                                  │
 │  保存内容：r1/r2/r3/r4 flags, online_r1_rotation,               │
 │  rotation_size, random_r1/r2/r3/r4, trainable_rotation,         │
 │  head_dim, hidden_size, intermediate_size, algorithm="spinquant" │
 └──────────────────────────────────────────────────────────────────┘
```

实际在 auto-round 的 export 代码中，Step 1 有两层防御（详见第 5 节）：
- **逐层注入**：在 `pack_layer()` 末尾调用，适用于 ShardWriter 路径
- **批量注入**：在 `save_quantized_as_*()` 中调用，适用于非 ShardWriter 路径

### 13.4 Load 三步流程

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 1: preregister_spinquant_buffers(model, spinquant_config)  │
 │  ─────────────────────────────────────────────────────────────── │
 │  时机：convert_hf_model() 中，Linear→QuantLinear 替换之后，      │
 │        HuggingFace load_state_dict 之前。                        │
 │  操作：在 QuantLinear 上预注册 **空** buffer（全零 tensor），     │
 │        确保 load_state_dict(strict=False) 不会丢弃 safetensors   │
 │        中的 spinquant_r1_*, spinquant_r4_* keys。                │
 │                                                                  │
 │  preregister 按 config 判断矩阵类型：                             │
 │  • random/trained → needs_matrix=True → 注册 _matrix 空 tensor  │
 │  • deterministic  → needs_matrix=False → 只注册 _type, _size    │
 └──────────────────────────────────────────────────────────────────┘
                              ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 2: HuggingFace loads state_dict from safetensors          │
 │  ─────────────────────────────────────────────────────────────── │
 │  自动将 safetensors 中的 spinquant_r1_type, spinquant_r1_matrix │
 │  等 tensor 填充到 Step 1 预注册的空 buffer 中。                  │
 └──────────────────────────────────────────────────────────────────┘
                              ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 3: rebuild_spinquant_online(model)                        │
 │  ─────────────────────────────────────────────────────────────── │
 │  时机：post_init()，即模型权重加载完成后。                        │
 │  操作：                                                          │
 │  (a) _patch_quantlinear_forward_spinquant(model)                │
 │      → 找到所有带 spinquant buffer 的 QuantLinear                │
 │      → class-level monkey-patch forward，在原 forward 前插入     │
 │        rotation 逻辑：                                           │
 │        def forward_with_spinquant(self, x):                     │
 │            if has r1 buffer: x = _apply_rotation_from_buffer(r1)│
 │            if has r4 buffer: x = _apply_rotation_from_buffer(r4)│
 │            return original_forward(x)                            │
 │                                                                  │
 │  (b) 如果 config.r3=True:                                       │
 │      → register_spinquant_hooks(model, r3=True)                 │
 │      → 重建 QKRotationWrapper monkeypatch                       │
 └──────────────────────────────────────────────────────────────────┘
```

### 13.5 R1–R4 Random 支持矩阵

| Rotation | Deterministic (Hadamard) | Random Hadamard | Trained Orthogonal |
|----------|-------------------------|-----------------|-------------------|
| **R1 offline** | ✅ fuse 进权重 | ✅ fuse 进权重（`R^T @ W`，R=diag(±1)@H） | ✅ fuse 进权重 |
| **R1 online** | ✅ butterfly O(n log n) | ✅ `x @ R` O(n²)，matrix 存为 int8 buffer | ✅ `x @ R` O(n²)，matrix 存为 float32 buffer |
| **R2** | ✅ fuse（R 对称，`W@R=W@R^T`） | ✅ fuse（einsum 已修复为 `W @ R`） | ✅ fuse |
| **R3** | ✅ hook（butterfly），无需存矩阵 | ✅ hook，**无需存矩阵**（`R@R^T=I` 自动抵消，加载时重新生成即可） | ✅ hook，无需存矩阵 |
| **R4** | ✅ buffer + fuse（butterfly） | ✅ buffer + fuse（`x@R` + `W@R`），matrix 存为 int8 | ✅ buffer + fuse |

**需要存储矩阵的 rotation**：仅 R1 online（random/trained）和 R4（random/trained）。
R2 fuse 进权重不需要存。R3 任意正交矩阵均可抵消（`QR(KR)^T=QK^T`），不需要存。

**config 字段**：`random_r1`, `random_r2`, `random_r3`, `random_r4` 均持久化到 `config.json`。

### 13.6 Safetensors 中的实际 Buffer 示例

以 Qwen3-0.6B + R1(online, random) + R4(deterministic) 为例：

```
# R1 online random → 每个 q/k/v/gate/up_proj 上都有 3 个 buffer
model.layers.0.self_attn.q_proj.spinquant_r1_type      # int32 scalar = 1 (RANDOM)
model.layers.0.self_attn.q_proj.spinquant_r1_size      # int32 scalar = 896
model.layers.0.self_attn.q_proj.spinquant_r1_matrix    # int8 [896, 896] (±1 values)
model.layers.0.self_attn.k_proj.spinquant_r1_type      # 同上
...
model.layers.0.mlp.gate_proj.spinquant_r1_type         # 同上
model.layers.0.mlp.up_proj.spinquant_r1_type           # 同上

# R4 deterministic → 每个 down_proj 上只有 2 个 buffer (无 _matrix)
model.layers.0.mlp.down_proj.spinquant_r4_type         # int32 scalar = 0 (HADAMARD)
model.layers.0.mlp.down_proj.spinquant_r4_size         # int32 scalar = 4864

# R2, R3 → 无 buffer（R2 已 fuse 进权重，R3 从 config.json 重建 hook）
```

### 13.7 config.json 中的 spinquant_config 示例

```json
{
  "quantization_config": {
    "quant_method": "intel/auto-round",
    "bits": 4,
    "group_size": 128,
    "spinquant_config": {
      "algorithm": "spinquant",
      "r1": true,
      "r2": true,
      "r3": true,
      "r4": true,
      "online_r1_rotation": true,
      "rotation_size": null,
      "random_r1": true,
      "random_r2": false,
      "random_r3": false,
      "random_r4": false,
      "trainable_rotation": false,
      "head_dim": 128,
      "hidden_size": 896,
      "intermediate_size": 4864
    }
  }
}
```

### 13.8 RotationSerializer Mixin 调度

export 文件通过 `transforms/__init__.py` 中的 5 个 dispatch 函数调用序列化逻辑，
**不直接 import `serialize.py`**：

```python
# export.py / export_to_nvfp_mx.py / export_to_fp8.py 中的调用模式：

if hasattr(model, "_rotation_config"):
    # pack_layer 内（逐层注入）
    inject_rotation_buffers_on_layer(layer_name, qlayer, model)

    # save_quantized 内（批量注入 fallback）
    inject_rotation_buffers(model)

    # save 结束后（写 config.json）
    save_rotation_config(model, output_dir)
```

这些 dispatch 函数根据 `model._rotation_config` 的类型路由到对应的实现
（目前只有 `SpinQuantRotation`，未来可扩展其他 rotation 方法）。
`_rotation_config` guard 确保无 rotation 时不执行任何序列化代码。

---

## 14. 测试脚本 `test_save_load_roundtrip.py` 使用指南

### 14.1 基本用法

```bash
# 快速验证 R1+R2, W4A16, limit=100
python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
    --rotations "R1+R2" --schemes "W4A16"
```

### 14.2 Random Rotation 控制

有两种方式控制 random rotation：

**方式一：`--rotation-modes`（全局控制）**

所有激活的 rotation 统一使用同一模式（hadamard 或 random）：

```bash
# R1 和 R2 都用 random
--rotations "R1+R2" --rotation-modes "random"

# Sweep: 先跑 hadamard，再跑 random
--rotations "R1+R2" --rotation-modes "hadamard,random"
```

**方式二：`--random-rotations`（按 rotation 级别独立控制，推荐）**

精确指定哪些 rotation 使用 random，其余保持 deterministic：

```bash
# 只有 R2 用 random（R1 保持 deterministic hadamard）
--rotations "R1+R2" --random-rotations "R2"

# R1 和 R3 用 random（R2, R4 保持 deterministic）
--rotations "R1+R2+R3+R4" --random-rotations "R1,R3"

# R1,R2,R3,R4 全部 random（等效于 --rotation-modes "random"）
--rotations "R1+R2+R3+R4" --random-rotations "R1,R2,R3,R4"
```

> **优先级**：`--random-rotations` > `--rotation-modes` > `--random-hadamard`

### 14.3 其他维度

```bash
# 多 scheme 覆盖
--schemes "W4A16,MXFP4,NVFP4,FP8_STATIC"

# RTN vs tuning
--quant-iters-list "0,200"

# Rotation size sweep
--rotation-sizes "64,128,auto"

# 全覆盖矩阵
python test_save_load_roundtrip.py --device cuda:0 --limit 100 \
    --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
    --schemes "W4A16,MXFP4" \
    --random-rotations "R2,R4"
```

### 14.4 输出标签

| 参数 | 标签示例 |
|------|---------|
| `--rotation-modes "random"` | `R1+R2 + W4A16 [random:all]` |
| `--random-rotations "R2"` | `R1+R2 + W4A16 [random:R2]` |
| `--random-rotations "R1,R3"` | `R1+R2+R3+R4 + W4A16 [random:R1+R3]` |
| 无 random 参数 | `R1+R2 + W4A16` |

---

## 15. 已知 Bug 修复记录

### 15.1 R1 Random Online Save/Load 精度崩溃（2026-05-12 修复）

**症状**：`random_r1=True` 时，in-memory eval 正常（hellaswag≈0.44），从磁盘加载后精度崩溃（≈0.28，随机水平）。`random_r1=False, random_r2=True` 正常。

**根因**：`preprocessor.py` 的 `_cleanup()` 方法会删除所有 `spinquant_R*` Parameters：

```python
# 旧代码 — 错误地删除了 R1 矩阵！
for name in list(self.model._parameters.keys()):
    if name.startswith("spinquant_R"):
        delattr(self.model, name)
```

之后 `inject_buffers_on_layer` 调用 `_get_stored_rotation(model, "spinquant_R1")` 返回 `None`，序列化时**静默回退到确定性 Hadamard 矩阵**——与旋转权重时用的 random 矩阵不匹配。

**修复**：
1. `preprocessor.py`：移除 cleanup 中的参数删除循环，rotation 矩阵保留供序列化使用
2. `serialize.py`：`rotation_matrix=None` 时从静默回退改为 `logger.warning` 告警

**验证**：R1+R2 random roundtrip: disk=0.5200, mem=0.5200 ✓ MATCH

### 15.2 R3 加载时 Hook 重建失败（2026-05-12 修复）

**症状**：加载包含 R3 rotation 的模型时报错：
```
Failed to rebuild spinquant rotations: register_spinquant_hooks() got an unexpected keyword argument 'r3'
```
R3 monkeypatch 未重建，导致精度下降。

**根因**：`serialize.py` 的 `rebuild_spinquant_online()` 中调用 `register_spinquant_hooks(model, r3=True, r4=False, ...)`，但该函数签名为 `register_spinquant_hooks(model, config, ...)`，需要传入一个带 `.r3`/`.r4` 属性的 config 对象。

**修复**：构造 `SimpleNamespace(r3=True, r4=False, random_r3=..., ...)` 作为 config 参数传入。

### 15.3 R4 Random 矩阵未保存到 Safetensors（2026-05-12 修复）

**症状**：`random_r4=True` 时，保存的 safetensors 中 `down_proj` 只有 `spinquant_r4_type` 和 `spinquant_r4_size`，缺少 `spinquant_r4_matrix`。加载后 HF 报 "newly initialized" 警告。

**根因**：`algorithm.py` 的 `inject_buffers_on_layer()` 中 R4 的注入硬编码为 `random=False, rotation_matrix=None`，忽略了 `config.random_r4`。

**修复**：改为 `random=config.random_r4`，并在 `random_r4=True` 时从 `model.spinquant_R4_matrix` 读取矩阵传入。
