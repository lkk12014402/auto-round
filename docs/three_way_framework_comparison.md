# Three-Way Framework Comparison: Rotation + Quantization Implementation Details

## Overview

This document provides a detailed comparison of how **auto-round**, **Quark**, and **llm-compressor** implement QuaRot/SpinQuant rotation combined with MXFP4 quantization. All three frameworks implement the same rotation algorithms (R1–R4) but with significantly different architectures, runtime behaviors, and model persistence mechanisms.

---

## 1. Rotation Matrices

### 1.1 Matrix Type

| Framework | R1 Matrix | R2 Matrix | R3 Matrix | R4 Matrix |
|-----------|-----------|-----------|-----------|-----------|
| auto-round | Deterministic Hadamard (normalized H/√N) | Same | Same | Same |
| Quark | Deterministic Hadamard (unnormalized ±1) | Same | Same | Same |
| llm-compressor | Configurable: `hadamard`, `random-hadamard`, `random-matrix` | Same | Same | Same |

### 1.2 Normalization Convention

```
auto-round:     H = scipy.linalg.hadamard(N) / sqrt(N)   → orthogonal (H @ H.T = I)
Quark:          H = ±1 matrix (unnormalized)              → user must divide by sqrt(N)
llm-compressor: H = deterministic_hadamard_matrix(N)      → ±1, normalized by sqrt(N) during apply
```

**Critical difference:** auto-round's `get_hadamard_K()` returns already-normalized matrices.
Quark's `_get_hadamard_K()` returns raw ±1 matrices. The normalization is applied differently
in each framework's multiplication routines. This caused a catastrophic double-normalization
bug in early auto-round development (hellaswag dropped from 0.42 to 0.26).

### 1.3 Matrix Construction Algorithm

All three frameworks use the **butterfly (Fast Walsh-Hadamard Transform)** algorithm for
efficient Hadamard multiplication:

```python
# O(N log N) butterfly algorithm (shared across all three)
def matmul_hadU(X, hadamard_K, K):
    n = X.shape[-1]
    # Phase 1: butterfly O(N log₂(N/K))
    while n > K:
        X = X.view(*X.shape[:-1], 2, n//2)
        X = torch.cat([X[..., 0:1, :] + X[..., 1:2, :],
                       X[..., 0:1, :] - X[..., 1:2, :]], dim=-2)
        X = X.view(*X.shape[:-2], n)
        n //= 2
    # Phase 2: dense multiply with hadamard_K (K×K)
    X = X @ hadamard_K.T
    return X
```

### 1.4 Rotation Size

| Framework | Config Parameter | Default | Semantics |
|-----------|-----------------|---------|-----------|
| auto-round | `SpinQuantConfig.rotation_size` | `None` (= hidden_size) | R1 uses `rotation_size`, R4 uses `rotation_size` or `intermediate_size` |
| Quark | `RotationConfig.rotation_size` | `None` (= hidden_size) | Same semantics |
| llm-compressor | `SpinQuantModifier.transform_block_size` | `None` (= full dim) | Applied uniformly to all rotations |

When `rotation_size < hidden_size`, **block rotation** is used: input is reshaped into
blocks of `rotation_size`, Hadamard is applied per-block, then reshaped back.

---

## 2. How Rotation + Quantization Works (Code Details)

### 2.1 auto-round Pipeline

```python
# Step 1: Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)

# Step 2: Apply rotation (in-place, before quantization)
config = SpinQuantConfig(r1=True, r2=True, r3=True, r4=True,
                         rotation_size=128, online_r1_rotation=True,
                         trainable_rotation=False, trainable_smooth=False)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()
```

**`preprocess()` internal steps (8 steps):**

```
Step 1: Detect model architecture (hidden_size, head_dim, intermediate_size)
Step 2: Untie word embeddings (if tie_word_embeddings=True, clone lm_head from embed_tokens)
Step 3: Skip (placeholder for smooth value init)
Step 4: Initialize rotation matrices (get_hadamard_K for each size)
Step 5: Skip (training disabled)
Step 6: Apply R1 rotation
    _apply_online_r1():
      For each target module (q/k/v_proj, gate/up_proj):
        - W_new = matmul_hadU(W)  # rotate input channels
        - register_forward_pre_hook(hadamard_hook)  # rotate activations at runtime
    _fuse_r2_rotation():
      For v_proj: rotate output channels (weight columns)
      For o_proj: rotate input channels (inverse)
    _fuse_r4_rotation():
      For down_proj: rotate input channels (inverse) — offline weight part only
Step 7: Register online hooks (R3/R4)
    R3: monkeypatch apply_rotary_pos_emb → QKRotationWrapper
    R4: forward_pre_hook on down_proj (block Hadamard on activation)
Step 8: Cleanup
```

```python
# Step 3: Quantize (RTN mode)
ar = AutoRound(model, tokenizer=tokenizer, scheme="MXFP4_RCEIL", iters=0,
               nsamples=128, seqlen=512, device_map="cuda:0")
ar.quantize()
model = ar.model
```

**Quantization internal flow:**
```
1. Collect calibration data (128 samples × 512 tokens)
2. For each transformer layer:
   a. Wrap target nn.Linear with WrapperLinear (or WrapperWALayer for W+A schemes)
   b. WrapperLinear.forward() runs orig_layer._forward_pre_hooks (R1/R4 hooks execute here!)
   c. RTN quantization: compute scales from weight statistics
   d. Pack weights to compressed format
3. Model ready for inference
```

**Key insight:** Hooks are preserved because `WrapperLinear.forward()` explicitly calls
`self.orig_layer._forward_pre_hooks` before the linear computation.

### 2.2 Quark Pipeline

```python
# Step 1: Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16)

# Step 2: Build config (rotation embedded in quant config)
template = LLMTemplate.get(model.config.model_type)
rotation_config = RotationConfig(r1=True, r2=True, r3=False, r4=True, ...)
quant_config = template.get_config(
    scheme="mxfp4",
    algorithm=["rotation"],
    algo_configs={"rotation": rotation_config},
)

# Step 3: Quantize (rotation is applied internally)
quantizer = ModelQuantizer(quant_config)
model = quantizer.quantize_model(model, calib_dataloader)
model = quantizer.freeze(model)
```

**`quantize_model()` internal flow:**
```
1. RotationProcessor.apply():
   - fuse_normalization(): merge RMSNorm gamma into downstream linears
   - r1(): offline rotation of ALL weights + InputRotationWrapper on targets
   - r2(): head-wise rotation of v_proj/o_proj
   - r4(): InputRotationWrapper on down_proj + weight inverse
2. torch.export.export_for_training() → FX graph trace
3. For each aten.linear.default node:
   - Apply quantization observers
   - Compute scales/zero_points
   - Pack to quantized format
4. freeze(): finalize quantization parameters
```

**Key difference from auto-round:** Quark uses `InputRotationWrapper` (nn.Module subclass)
instead of hooks. This works because Quark uses FX graph tracing, which sees through
wrapper modules to the underlying `aten.linear` operation.

### 2.3 llm-compressor Pipeline

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Declarative recipe — rotation and quantization are separate modifiers
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        transform_type="hadamard",
        transform_block_size=128,
    ),
    QuantizationModifier(targets="Linear", scheme="MXFP4", ignore=["lm_head"]),
]

# Single call applies both
oneshot(model=model, recipe=recipe, pipeline="datafree")
```

**`oneshot()` internal flow:**
```
Pipeline: DataFreePipeline (no calibration data needed for RTN)

Phase 1: SpinQuantModifier lifecycle
  on_initialize():
    - Infer layer mappings from model architecture (regex patterns)
    - Create TransformConfig with R1/R2/R4 schemes
  on_start():
    - center_embeddings(): weight -= weight.mean(dim=-1)
    - fuse_norm_linears(): linear.weight *= norm.weight; norm.weight = 1
    - apply_transform_config(model, transform_config):
      For each scheme (R1, R2, R4):
        factory = TransformFactory.from_scheme(scheme)  # HadamardFactory
        factory.apply_to_model(model):
          For each matching module (via regex targets):
            if location is WEIGHT_INPUT/WEIGHT_OUTPUT:
              → Fuse rotation into weight matrix (offline)
            if location is INPUT:
              → Register forward_pre_hook (online)

Phase 2: QuantizationModifier lifecycle
  on_initialize():
    - Attach quantization schemes to all Linear modules
    - Create weight observers
  on_start():
    - Calibrate: observe weight distributions
    - Quantize: apply scales and pack weights
  on_end():
    - Freeze quantization parameters
```

---

## 3. R1 Online vs Offline Comparison

| Aspect | auto-round (Online R1) | Quark (Online R1) | llm-compressor (Offline R1) |
|--------|----------------------|-------------------|---------------------------|
| embed_tokens | **Unchanged** | **Unchanged** | Rotated (WEIGHT_OUTPUT) |
| RMSNorm layers | **Not fused** | **Not fused** (but fused before R1) | **Fused** into downstream |
| q/k/v_proj | W@H + hook | W@H + Wrapper | W@H⁻¹ fused (WEIGHT_INPUT) |
| gate/up_proj | W@H + hook | W@H + Wrapper | W@H⁻¹ fused (WEIGHT_INPUT) |
| o_proj | **Unchanged** | **Unchanged** | R@W fused (WEIGHT_OUTPUT) |
| down_proj | **Unchanged** | **Unchanged** | R@W fused (WEIGHT_OUTPUT) |
| lm_head | **Unchanged** | **Unchanged** | W@H⁻¹ fused (WEIGHT_INPUT) |
| Inference cost | Hook per target (5/layer) | Wrapper per target | **Zero** (all fused) |
| Weight distribution | Only targets changed | Only targets changed | **ALL weights changed** |
| Quantization impact | Minimal | Minimal | Higher (all distributions shift) |

**Why auto-round uses online R1:** Offline R1 changes ALL weight distributions, which hurts
MXFP4 quantization quality. Online R1 only modifies target modules, preserving the original
distribution of prev_modules (o_proj, down_proj, embed_tokens).

**Why llm-compressor uses offline R1:** Zero inference overhead. The accuracy impact from
distribution shift is acceptable for their use cases. They center embeddings first to
mitigate the effect.

---

## 4. Inference After Rotation + Quantization

### 4.1 auto-round Inference

After quantization, the model has:
- `WrapperLinear` modules wrapping original nn.Linear layers
- R1 `forward_pre_hook`s on q/k/v/gate/up_proj (inside WrapperLinear.orig_layer)
- R3 monkeypatch on `apply_rotary_pos_emb`
- R4 `forward_pre_hook` on down_proj

```python
# Inference path for a target module (e.g., q_proj):
def WrapperLinear.forward(x):
    # 1. Run pre-hooks (R1 Hadamard rotation on input)
    for hook in self.orig_layer._forward_pre_hooks.values():
        x = hook(self.orig_layer, (x,))[0]  # x_rotated = H @ x
    # 2. Dequantize weight
    weight = self.dequant(self.packed_weight, self.scale, ...)
    # 3. Linear computation
    return F.linear(x, weight, bias)
```

### 4.2 Quark Inference

After export, target modules become `QParamsLinearWithRotation`:

```python
class QParamsLinearWithRotation(QParamsLinear):
    def forward(self, inp):
        # 1. Apply rotation transform (Hadamard via butterfly)
        inp = self.transform(inp)  # HadamardTransform or OrthogonalTransform
        # 2. Parent class handles dequant + linear
        return super().forward(inp)

class HadamardTransform:
    def __call__(self, x):
        return matmul_hadU(x, hadamard_K=self.hadamard_K, K=self.K) / sqrt(self.rotation_size)
```

### 4.3 llm-compressor Inference

R1 is fully offline — no runtime cost. R4 has a forward_pre_hook:

```python
# R1: No hook needed. Weight already contains R @ W or W @ R⁻¹
# Inference is just standard quantized linear:
y = dequant(packed_weight) @ x

# R4: forward_pre_hook registered as a TransformBase submodule
class HadamardTransform(TransformBase):
    def forward(self, value):
        # Apply Hadamard rotation to activation
        return apply_transform_weight(self.weight, value, self.args.location, ...)
```

For vLLM inference, llm-compressor supports **Hadacore kernels** — fused CUDA kernels
that combine Hadamard rotation + quantized linear into a single operation, eliminating
the overhead of online R4 entirely.

---

## 5. Model Save and Load

### 5.1 auto-round: Currently Hooks are LOST on Save

```python
# Save (currently broken for rotation):
model.save_pretrained(output_dir)
# → Hooks are Python closures, NOT serialized
# → rotation_configs actively REMOVED by export/utils.py line 374

# Load: rotation hooks are NOT re-applied
model = AutoModelForCausalLM.from_pretrained(output_dir)
# → Model runs WITHOUT rotation → incorrect inference results
```

**Planned fix:** Save `rotation_config.json` alongside the model, then re-register hooks
on load (design documented in `docs/rotation_save_load_solution.md`).

### 5.2 Quark: Full Save/Load with `QParamsLinearWithRotation`

```python
# Save:
from quark.torch.export.api import export_hf_model
export_hf_model(model, quant_config, save_dir, custom_mode="quark")
# → Replaces nn.Linear with QParamsLinearWithRotation
# → Saves rotation matrix as buffer: self.register_buffer("input_rotation", ...)
#   - int8 for deterministic Hadamard (only ±1 values, saves space)
#   - float64 for trained rotation (full precision)
# → Saves quantization_config.json with rotation_config

# Load:
model = AutoModelForCausalLM.from_pretrained(save_dir)
# → QParamsLinearWithRotation loaded with input_rotation buffer
# → post_process_after_loading() reconstructs HadamardTransform or OrthogonalTransform
# → Inference works correctly with rotation
```

**Save format:**
```
model_dir/
├── config.json              # includes quantization_config with rotation info
├── model.safetensors        # includes input_rotation buffers (int8 or float64)
└── quantization_config.json # detailed quant + rotation config
```

### 5.3 llm-compressor: First-Class Transform Serialization

```python
# Save:
model.save_pretrained(save_dir, save_compressed=True)
# → ModelCompressor.from_pretrained_model(model) extracts:
#   - quantization_config from attached schemes
#   - transform_config from model.transform_config attribute
# → config.json includes full transform_config
# → Online transform weights (R4 hooks) saved as named parameters
# → Shared weights deduplicated via _tied_weights_keys

# Load:
model = AutoModelForCausalLM.from_pretrained(save_dir)
# → CompressedTensorsHfQuantizer reads config.json
# → Finds quantization_config + transform_config
# → _process_model_before_weight_loading():
#     - apply_quantization_config(model, q_config)
#     - Creates quantized module structure
# → Load weights from safetensors
# → _process_model_after_weight_loading():
#     - Optional decompression
# → Online transforms (R4 hooks) are loaded as nn.Module submodules
#     (TransformBase instances registered via factory.apply_to_model)
```

**Save format:**
```
model_dir/
├── config.json
│   └── quantization_config:
│       ├── quantization_method: "compressed-tensors"
│       ├── transform_config:    # ← FULL transform config here
│       │   └── config_groups:
│       │       ├── R1: {type: "hadamard", apply: [...]}
│       │       ├── R2: {type: "hadamard", apply: [...], head_dim: 128}
│       │       └── R4: {type: "hadamard", apply: [...]}
│       └── config_groups: {...}  # quantization config
├── model.safetensors           # quantized weights + online transform weights
└── tokenizer files
```

**On reload, online transforms (R4) are re-registered automatically** because:
1. `TransformBase` modules are registered as named submodules during save
2. Their `weight` parameters are included in `state_dict`
3. After loading weights, the module's `forward()` method (which applies the transform)
   is already connected via PyTorch hooks

---

## 6. Comparison Table: Save/Load Capability

| Feature | auto-round | Quark | llm-compressor |
|---------|-----------|-------|----------------|
| R1 offline saved? | N/A (online only) | N/A (online only) | ✅ Fused into weights |
| R1 online saved? | ❌ Hooks lost | ✅ `input_rotation` buffer | ❌ (R1 is offline) |
| R2 saved? | ✅ Fused into weights | ✅ Fused into weights | ✅ Fused into weights |
| R3 saved? | ❌ Monkeypatch lost | ❌ Not supported in export | ❌ Partially implemented |
| R4 saved? | ❌ Hooks lost | ✅ `input_rotation` buffer | ✅ TransformBase submodule |
| Rotation matrix format | Not saved | int8 (Hadamard) / float64 (trained) | safetensors parameter |
| Config persistence | ❌ Actively removed | ✅ quantization_config.json | ✅ config.json |
| Load & infer correctly | ❌ | ✅ | ✅ |

---

## 7. Test Script Execution Details

### 7.1 auto-round Test Flow

```python
# In test_three_way_comparison.py → run_autoround():
model = load_model(model_name, device)                        # Fresh FP16 model
model = apply_rotation(model, rotation_flags, rotation_size)  # Apply R1-R4 hooks
ar = AutoRound(model, tokenizer, scheme="MXFP4_RCEIL", iters=0, ...)
ar.quantize()                                                 # RTN quantize
model = ar.model.eval().to(device)                            # Quantized model with hooks
# → lm_eval evaluates using model.forward() which triggers hooks
```

### 7.2 Quark Test Flow

```python
# In test_three_way_comparison.py → run_quark():
model = load_model(model_name, device)
template = LLMTemplate.get(model.config.model_type)           # Architecture template
quant_config = template.get_config(scheme="mxfp4", algorithm=["rotation"], ...)
calib_dataloader = get_calib_dataloader(...)                   # Required by Quark
quantizer = ModelQuantizer(quant_config)
model = quantizer.quantize_model(model, calib_dataloader)     # Rotation + quantization
model = quantizer.freeze(model)                               # Finalize
# → lm_eval evaluates; InputRotationWrapper.forward() applies rotation before linear
```

### 7.3 llm-compressor Test Flow

```python
# In test_three_way_comparison.py → run_llmcompressor():
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
recipe = [
    SpinQuantModifier(rotations=[...], transform_type="hadamard", transform_block_size=128),
    QuantizationModifier(targets="Linear", scheme="MXFP4", ignore=["lm_head"]),
]
oneshot(model=model, recipe=recipe, pipeline="datafree")      # Rotation + quantization
# → lm_eval evaluates; R1 is fused (no overhead), R4 hook applies rotation on down_proj
```

---

## 8. Key Architectural Differences Summary

### 8.1 Module Discovery

| Framework | Method | Implication |
|-----------|--------|-------------|
| auto-round | `type(m) in SUPPORTED_LAYER_TYPES` | Only recognizes `nn.Linear`, `Conv1D` |
| Quark | `torch.export.export_for_training()` → FX graph | Sees through wrappers to `aten.linear` |
| llm-compressor | Regex patterns on module names | Flexible, works with any naming convention |

### 8.2 Quantization + Rotation Ordering

```
auto-round:       Rotation (hooks) → Quantization (wraps nn.Linear) → Inference (hook inside wrapper)
Quark:            Rotation (wrappers) → FX trace → Quantization (on aten ops) → Export (QParamsLinear)
llm-compressor:   Rotation (modifier 1, fuse/hook) → Quantization (modifier 2, observe + pack)
```

### 8.3 Online Rotation Implementation

| Framework | R1 online | R4 online | Mechanism |
|-----------|-----------|-----------|-----------|
| auto-round | `forward_pre_hook` closure | `forward_pre_hook` closure | PyTorch hook system |
| Quark | `InputRotationWrapper(nn.Module)` | `InputRotationWrapper(nn.Module)` | Module wrapping |
| llm-compressor | N/A (offline) | `TransformBase` submodule + hook | Module + hook |

---

## 9. Performance Characteristics

| Framework | R1 Inference Cost | R4 Inference Cost | Save/Load |
|-----------|-------------------|-------------------|-----------|
| auto-round | O(N log N) per target × 5/layer | O(N log N) per down_proj | ❌ |
| Quark | O(N log N) per target × 5/layer | O(N log N) per down_proj | ✅ |
| llm-compressor | **Zero** (fused offline) | O(N log N) per down_proj (or Hadacore kernel) | ✅ |

For a 28-layer model with hidden_size=1024:
- Online R1: 28 × 5 = 140 Hadamard multiplications per forward pass
- Online R4: 28 Hadamard multiplications per forward pass
- llm-compressor's offline R1: 0 extra cost at inference

---

## 10. Which Framework to Choose?

| Use Case | Recommendation | Reason |
|----------|---------------|--------|
| Best quantization quality | auto-round (online R1) | Preserves original weight distributions |
| Production deployment | llm-compressor | Full save/load, vLLM integration, Hadacore kernels |
| Research / flexibility | Quark | Most features, trainable rotation, comprehensive export |
| No save/load needed | auto-round | Simplest API, hook-based, scheme-agnostic |
| Need R3 support | auto-round or Quark | llm-compressor R3 is incomplete |
