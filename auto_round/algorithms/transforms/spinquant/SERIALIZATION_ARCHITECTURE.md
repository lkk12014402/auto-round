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
