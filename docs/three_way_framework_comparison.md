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

### 3.1 Per-Module Comparison

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

### 3.2 Online R1 vs Offline R1: Accuracy Impact on Quantization

Online R1 and offline R1 are **mathematically equivalent in FP** (produce identical outputs
before quantization), but they produce **significantly different results after quantization**.
This is the key reason auto-round and Quark default to online R1.

**Why offline R1 degrades quantization accuracy:**

```
Offline R1 global change-of-basis:
  1. embed_tokens: W_embed → W_embed @ H         (distribution changed)
  2. RMSNorm: gamma fused into downstream weights (distribution changed)
  3. q/k/v_proj: W → H.T @ diag(gamma) @ W       (distribution changed)
  4. o_proj: W → W @ H                            (distribution changed)
  5. gate/up_proj: W → H.T @ diag(gamma) @ W      (distribution changed)
  6. down_proj: W → W @ H                          (distribution changed)
  7. lm_head: W → H.T @ W                          (distribution changed)
  → ALL 7 module types have shifted weight distributions
  → Quantization (MXFP4/INT4) applied to shifted distributions → more error
```

```
Online R1 local rotation:
  1. q/k/v_proj: W → W @ H    (distribution changed, but only right-multiply)
  2. gate/up_proj: W → W @ H  (distribution changed, but only right-multiply)
  3. o_proj, down_proj, embed_tokens, lm_head: UNCHANGED
  4. RMSNorm: UNCHANGED (not fused)
  → Only 5 module types changed, and only by right-multiplication
  → Other modules keep ORIGINAL weight distributions → less quantization error
```

**Observed accuracy difference (Qwen3-0.6B, hellaswag, MXFP4 RTN):**

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| FP16 baseline (no rotation, no quant) | ~0.4250 | Reference |
| Online R1 + MXFP4 | ~0.4550 | Rotation **helps** quantization |
| Offline R1 + MXFP4 | ~0.2628 | Catastrophic degradation |
| MXFP4 only (no rotation) | ~0.4200 | Slightly below FP16 |

The offline R1 result (0.2628) is **worse than random** for hellaswag (4-choice → 0.25),
demonstrating that the global weight distribution shift is devastating for MXFP4.

**Why the difference is so large with MXFP4:**
MXFP4 uses **microscaling** — per-block scaling factors for groups of values. When offline
R1 fuses RMSNorm gamma into weights, it creates large dynamic range variations that MXFP4's
limited 4-bit precision cannot represent. Online R1 avoids this by keeping RMSNorm separate.

### 3.3 auto-round R1: Both Online and Offline Supported

auto-round is the only framework in our comparison that **supports both modes via a config flag**:

```python
@dataclass
class SpinQuantConfig:
    online_r1_rotation: bool = True   # ← Toggle: True=online, False=offline
```

**Online R1 path** (`preprocessor.py → _apply_online_r1()`):
```python
def _apply_online_r1(self):
    # Only rotates weight RIGHT side: W_new = W @ H
    for name, module in model.named_modules():
        if name in target_modules:
            rotate_in_channels_(module, R_in=R1_mat)  # W → W @ H
            # Register hook: x → H @ x (activation rotation)
            hook = module.register_forward_pre_hook(self._make_online_r1_hook(...))
            self._r1_hook_handles.append(hook)
    # RMSNorm NOT fused, embed_tokens/o_proj/down_proj NOT touched
```

**Offline R1 path** (`preprocessor.py → _fuse_offline_rotations()`):
```python
def _fuse_offline_rotations(self):
    # Global change-of-basis — modifies EVERY layer
    rotate_embeddings(model, R1_mat)          # embed_tokens @ H
    fuse_layer_norms(model)                   # gamma → downstream weights
    rotate_head(model, R1_mat)                # lm_head @ H.T
    for layer in layers:
        rotate_in_channels_(q_proj, R1_mat)   # H.T @ W_q
        rotate_in_channels_(k_proj, R1_mat)
        # ... all projections rotated
        rotate_out_channels_(o_proj, R1_mat)  # W_o @ H
        rotate_out_channels_(down_proj, R1_mat)
    # No hooks needed — everything is fused
```

**Recommendation:** Always use `online_r1_rotation=True` (the default) when quantizing.
Offline R1 is only useful for pure rotation experiments without quantization.

### 3.4 Quark R1: Both Online and Offline Supported

Quark also supports both via `online_r1_rotation` config:

```python
# rotation.py
if self.rotation_config.online_r1_rotation:
    self.apply_online_r1()    # InputRotationWrapperHadamard on targets
else:
    self._r1_offline()        # Global weight fusion
```

### 3.5 llm-compressor R1: Offline Only

llm-compressor only supports offline R1. It uses `TransformLocation.WEIGHT_INPUT` and
`WEIGHT_OUTPUT` which directly modify weight data:

```python
# _create_r1_scheme() → all weight_input/weight_output locations
# TransformFactory._apply_to_module():
if args.location in (WEIGHT_INPUT, WEIGHT_OUTPUT):
    # Eager weight fusion — no hook registered
    module.weight.data = apply_transform_weight(transform.weight, module.weight.data, ...)
```

There is **no `INPUT` location** in the R1 scheme, so no hook → always offline.

To mitigate the accuracy impact, llm-compressor applies **embedding centering** before R1:
```python
# base.py → on_start():
center_embeddings(self.model)   # Subtract mean from embeddings first
fuse_norm_linears(self.model)   # Fuse RMSNorm into weights
apply_transform_config(...)     # Then apply R1 (offline)
```

---

## 3A. Complete Online/Offline Summary: All Rotations × All Frameworks

### 3A.1 During Rotation + Quantization Preparation

| Rotation | auto-round | Quark | llm-compressor |
|----------|-----------|-------|----------------|
| **R1** | **Online** (default): `forward_pre_hook` on q/k/v/gate/up + weight right-multiply. **Offline** (opt-in): global weight fusion + RMSNorm fusion | **Online** (default): `InputRotationWrapperHadamard` on targets + weight right-multiply. **Offline** (opt-in): global fusion | **Offline** (only): `WEIGHT_INPUT`/`WEIGHT_OUTPUT` fusion on all layers. No online mode |
| **R2** | **Offline** only: `apply_hadamard_to_linear()` fuses into v_proj output + o_proj input per head | **Offline** only: `rotate_out_channels_(v_proj)` + `rotate_in_channels_(o_proj)` per head | **Offline** only: `WEIGHT_OUTPUT` on v_proj + `WEIGHT_INPUT` (inverse) on o_proj |
| **R3** | **Online** only: monkeypatch `apply_rotary_pos_emb` → `QKRotationWrapper` | **Online** only: monkeypatch `apply_rotary_pos_emb` with closure | **Online** only: `Q_ATTN` hook + `K_CACHE` hook via `QuantizedAttentionImpl` |
| **R4** | **Hybrid**: Offline weight fusion (`rotate_in_channels_` on down_proj) + Online `forward_pre_hook` on down_proj activation | **Hybrid**: Offline weight fusion + Online `InputRotationWrapperHadamard` on down_proj | **Hybrid**: Offline `WEIGHT_INPUT` inverse on down_proj + Online `INPUT` hook (TransformBase submodule) |

### 3A.2 During Inference After Quantization

| Rotation | auto-round | Quark | llm-compressor |
|----------|-----------|-------|----------------|
| **R1 online** | ✅ Hook runs inside `WrapperLinear.forward()` → O(N·log N) per target | ✅ `InputRotationWrapperHadamard.forward()` runs before linear → O(N·log N) per target | N/A (R1 always offline) |
| **R1 offline** | No runtime cost (all fused) | No runtime cost (all fused) | ✅ No runtime cost (all fused) |
| **R2** | No runtime cost (fused into v_proj/o_proj weights) | No runtime cost (fused) | No runtime cost (fused) |
| **R3** | ✅ Monkeypatch intercepts every `apply_rotary_pos_emb` call → O(seq × heads × d·log d) | ✅ Same monkeypatch mechanism | ✅ Hook on `QuantizedAttentionImpl` + `QuantizedKVCache` |
| **R4 activation** | ✅ `forward_pre_hook` on down_proj → O(seq × inter × log K) | ✅ `InputRotationWrapperHadamard.forward()` | ✅ `TransformBase.forward()` hook (or fused via Hadacore kernel) |
| **R4 weight** | No runtime cost (inverse pre-fused) | No runtime cost (inverse pre-fused) | No runtime cost (inverse pre-fused) |

### 3A.3 Why Each Rotation Level is Online or Offline

| Rotation | Online/Offline | Mathematical Reason |
|----------|---------------|---------------------|
| **R1** | Can be either | Online: x' = Hx then y = W·x' = (WH)·x, only need W→WH. Offline: global basis change, fuse H into ALL connected layers + RMSNorm. Both mathematically equivalent in FP. |
| **R2** | Always offline | Per-head rotation on V output channels and O input channels. V and O are directly connected (Attn @ V → O input), so both sides can be fused. No activation hook needed. |
| **R3** | Always online | Must apply AFTER RoPE (position-dependent rotation). Cannot be fused into weights because RoPE uses element-wise ops (⊙ cos, ⊙ sin) that don't commute with matrix multiply. |
| **R4** | Hybrid | Activation side (act → H·act) must be online (depends on runtime activation). Weight side (W_down → W_down·H.T inverse) can be pre-fused. |

### 3A.4 Which Rotations Can auto-round Switch Between Online/Offline?

| Rotation | Switchable? | Config | Notes |
|----------|------------|--------|-------|
| **R1** | ✅ Yes | `online_r1_rotation=True/False` | **Default: online.** Offline hurts quantization accuracy significantly |
| **R2** | ❌ No | Always offline | No reason to make online — both sides (v_proj, o_proj) fully fuseable |
| **R3** | ❌ No | Always online | **Cannot** be offline — RoPE is position-dependent, cannot pre-fuse |
| **R4** | ❌ No | Always hybrid | Activation must be online; weight inverse must be offline |

---

## 4. R2 Head-Wise Attention Rotation (OFFLINE) — Detailed Comparison

R2 applies a **per-head rotation** to the attention value and output projections.
It's offline (fused into weights) in all three frameworks — zero inference cost.

### 4.1 Mathematical Formulation

Standard multi-head attention value path:

```
V = X @ W_v     # [seq, hidden] @ [hidden, hidden] → [seq, hidden]
V_heads = reshape(V, [seq, num_heads, head_dim])
O_heads = Attention(Q_heads, K_heads) @ V_heads   # per-head
O = reshape(O_heads, [seq, hidden]) @ W_o          # [seq, hidden] @ [hidden, hidden]
```

R2 applies head-wise Hadamard H_d (d = head_dim) to each head:

```
V_heads_rotated[h] = V_heads[h] @ H_d    ∀h ∈ [0, num_heads)
O_heads_rotated[h] = Attn @ V_heads_rotated[h]  (rotation absorbed in V)
O_input_rotated = reshape(O_heads_rotated) @ W_o
                → need W_o to absorb H_d⁻¹ per head on input side
```

Fusion into weights:
```
W_v_new:  output channels rotated per head
          W_v_new[h*d:(h+1)*d, :] = H_d @ W_v[h*d:(h+1)*d, :]

W_o_new:  input channels rotated per head (inverse)
          W_o_new[:, h*d:(h+1)*d] = W_o[:, h*d:(h+1)*d] @ H_d.T
```

### 4.2 auto-round R2 Implementation

```python
# preprocessor.py → _fuse_r2_rotation()
def _fuse_r2_rotation(self):
    R2_mat = self.R2   # head_dim × head_dim Hadamard matrix
    for layer in self._get_layers():
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj

        # v_proj: rotate output channels per head
        # Weight shape: [num_heads * head_dim, hidden_size]
        # Reshape to [num_heads, head_dim, hidden_size], apply R2 per head
        apply_hadamard_to_linear(v_proj, R2_mat, mode="output", head_dim=self.head_dim)

        # o_proj: rotate input channels per head (inverse)
        # Weight shape: [hidden_size, num_heads * head_dim]
        # Reshape to [hidden_size, num_heads, head_dim], apply R2.T per head
        apply_hadamard_to_linear(o_proj, R2_mat, mode="input", head_dim=self.head_dim)
```

Key helper (`rotation_utils.py`):
```python
def apply_hadamard_to_linear(linear, R, mode="output", head_dim=None):
    W = linear.weight.data.to(torch.float64)
    if head_dim and mode == "output":
        # Reshape [out, in] → [num_heads, head_dim, in]
        W = W.view(-1, head_dim, W.shape[-1])
        W = torch.einsum("ij,hjk->hik", R.to(torch.float64), W)
        W = W.reshape(-1, W.shape[-1])
    elif head_dim and mode == "input":
        # Reshape [out, in] → [out, num_heads, head_dim]
        W = W.view(W.shape[0], -1, head_dim)
        W = torch.einsum("onh,hd->ond", W, R.T.to(torch.float64))
        W = W.reshape(W.shape[0], -1)
    linear.weight.data = W.to(linear.weight.dtype)
```

### 4.3 Quark R2 Implementation

```python
# rotation.py → r2()
def r2(self):
    for layer in self.layers:
        v_proj = getattr(layer, self.rotation_config.self_attn).v_proj
        o_proj = getattr(layer, self.rotation_config.self_attn).o_proj

        # Rotate v_proj output channels per head
        rotate_out_channels_(v_proj, self.Q_r2, self.head_dim)
        # Rotate o_proj input channels per head (inverse)
        rotate_in_channels_(o_proj, self.Q_r2, self.head_dim)

# rotation_utils.py → rotate_out_channels_()
def rotate_out_channels_(layer, Q, head_dim):
    W = layer.weight.data.to(torch.float64)
    num_heads = W.shape[0] // head_dim
    # Reshape: [num_heads * head_dim, in] → [num_heads, head_dim, in]
    W = W.reshape(num_heads, head_dim, -1)
    # Per-head rotation: Q @ W[h]
    W = torch.einsum("ij,hjk->hik", Q.to(torch.float64), W)
    layer.weight.data = W.reshape(-1, W.shape[-1]).to(layer.weight.dtype)
```

**Quark normalization note:** Quark's Hadamard is unnormalized (±1), so division by
√head_dim happens internally in `rotate_out_channels_` / `rotate_in_channels_`.

### 4.4 llm-compressor R2 Implementation

```python
# base.py → _create_r2_scheme()
def _create_r2_scheme(self, head_dim: int) -> TransformScheme:
    return TransformScheme(
        type=self.transform_type,       # "hadamard"
        randomize=self.randomize,
        requires_grad=self.learnable,
        precision=self.precision,
        head_dim=head_dim,              # ← enables block-diagonal
        apply=[
            TransformArgs(
                targets=[self.mappings.attn_v],     # "re:.*v_proj$"
                location="weight_output",
            ),
            TransformArgs(
                targets=[self.mappings.attn_o],     # "re:.*o_proj$"
                location="weight_input",
                inverse=True,
            ),
        ],
    )
```

The `head_dim` parameter triggers block-diagonal multiplication via `_multihead_matmul`:

```python
# matrix.py → _multihead_matmul()
def _multihead_matmul(A: Tensor, B: Tensor) -> Tensor:
    """Block-diagonal multiplication for multi-headed attention.
    A is head_dim × head_dim, B is [num_heads * head_dim, ...]
    """
    head_dim = A.shape[0]
    if B.shape[0] == head_dim:
        return A @ B  # single head, direct multiply
    # Multi-head: unflatten → batch matmul → flatten
    num_heads = B.shape[0] // head_dim
    B = B.unflatten(0, (num_heads, head_dim))    # [num_heads, head_dim, ...]
    result = torch.einsum("ij,hjk->hik", A, B)   # per-head rotation
    return result.flatten(0, 1)                    # [num_heads * head_dim, ...]
```

### 4.5 R2 Comparison Table

| Aspect | auto-round | Quark | llm-compressor |
|--------|-----------|-------|----------------|
| Modules modified | v_proj (output), o_proj (input) | v_proj (output), o_proj (input) | v_proj (weight_output), o_proj (weight_input, inverse) |
| Math | `einsum("ij,hjk->hik", R, W)` per head | `einsum("ij,hjk->hik", Q, W)` per head | `_multihead_matmul(H, W)` unflatten/flatten |
| Precision | float64 during fusion | float64 during fusion | float64 for offline, scheme.precision for online |
| Head detection | `self.head_dim` from config | `self.head_dim` from config | `TransformScheme.head_dim` parameter |
| Save/Load | ✅ Fused into weights (transparent) | ✅ Fused into weights (transparent) | ✅ Fused into weights (transparent) |
| Inference cost | **Zero** (offline) | **Zero** (offline) | **Zero** (offline) |

---

## 5. R3 Q/K Rotation After RoPE (ONLINE) — Detailed Comparison

R3 applies Hadamard rotation to **query and key states after RoPE**. This is the most
architecturally complex rotation because it must be injected at a precise point in the
attention computation — after positional encoding but before the dot-product.

### 5.1 Mathematical Formulation

Standard attention:
```
Q = RoPE(X @ W_q)    # [seq, num_heads, head_dim]
K = RoPE(X @ W_k)    # [seq, num_kv_heads, head_dim]
Attn = softmax(Q @ K.T / √d)
```

R3 applies head-wise Hadamard H_d to Q and K after RoPE:
```
Q_rotated = Q @ H_d     # per head: [seq, head_dim] @ [head_dim, head_dim]
K_rotated = K @ H_d     # per head

Attn = softmax(Q_rotated @ K_rotated.T / √d)
     = softmax(Q @ H_d @ H_d.T @ K.T / √d)
     = softmax(Q @ I @ K.T / √d)        # H @ H.T = I (orthogonal)
     = softmax(Q @ K.T / √d)            # identical to original!
```

**Why R3 doesn't change accuracy (in FP):** Since H is orthogonal (H @ H.T = I), the
attention scores are mathematically identical. However, R3 changes the **basis** of Q/K,
which reduces outliers → improves quantization quality when Q/K are quantized.

### 5.2 auto-round R3 Implementation

auto-round uses **monkeypatching** — replacing the `apply_rotary_pos_emb` function in the
attention module's forward method scope with a wrapper.

```python
# monkeypatch.py → QKRotationWrapper
class QKRotationWrapper:
    """Wraps apply_rotary_pos_emb to inject Hadamard rotation after RoPE."""

    def __init__(self, original_fn):
        self._original_fn = original_fn
        self._hadamard_K = None
        self._K = None
        self._head_dim = None

    def set_hadamard(self, hadamard_K, head_dim, K=None):
        """Store Hadamard parameters for the rotation."""
        self._head_dim = head_dim
        self._hadamard_K, self._K = get_hadamard_K(head_dim)
        self._hadamard_K = self._hadamard_K.to(hadamard_K.device if hadamard_K.numel() > 0
                                                 else "cpu")

    def __call__(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Step 1: Apply original RoPE
        q_roped, k_roped = self._original_fn(q, k, cos, sin,
                                              position_ids=position_ids,
                                              unsqueeze_dim=unsqueeze_dim)
        # Step 2: Apply Hadamard rotation to q and k (per head)
        had_K = self._hadamard_K.to(q_roped.device, q_roped.dtype)
        q_rotated = matmul_hadU(q_roped, hadamard_K=had_K, K=self._K)
        k_rotated = matmul_hadU(k_roped, hadamard_K=had_K, K=self._K)

        return q_rotated, k_rotated


# monkeypatch.py → add_qk_rotation_after_rope()
def add_qk_rotation_after_rope(attn_module, rope_function_name="apply_rotary_pos_emb"):
    """Replace apply_rotary_pos_emb in the attention module's forward globals."""
    fwd = attn_module.forward
    # Get the forward method's global namespace
    if hasattr(fwd, "__globals__"):
        globals_dict = fwd.__globals__
    elif hasattr(fwd, "__func__"):
        globals_dict = fwd.__func__.__globals__
    else:
        raise RuntimeError("Cannot find forward method globals")

    original_fn = globals_dict[rope_function_name]
    wrapper = QKRotationWrapper(original_fn)
    globals_dict[rope_function_name] = wrapper   # ← Replace in globals!
    return wrapper
```

**Hook registration** (`inplace/apply.py`):
```python
# R3 section in register_spinquant_hooks()
if config.r3 and head_dim > 0:
    if not is_pow2(head_dim):
        logger.warning(f"R3 requires head_dim={head_dim} to be power-of-2. Skipping R3.")
    else:
        for name, module in model.named_modules():
            if name.endswith("self_attn") and hasattr(module, "q_proj"):
                wrapper = add_qk_rotation_after_rope(module, "apply_rotary_pos_emb")
                wrapper.set_hadamard(torch.empty(0), head_dim, compute_device)
                handles.append(("r3_monkeypatch", name, wrapper))
                r3_count += 1
```

**Runtime inference path:**
```
1. attn.forward(hidden_states, ...) called
2. q = self.q_proj(hidden_states)
3. k = self.k_proj(hidden_states)
4. q, k = apply_rotary_pos_emb(q, k, cos, sin)
         ↓ This is now the QKRotationWrapper!
   4a. wrapper calls original apply_rotary_pos_emb → q_roped, k_roped
   4b. wrapper applies matmul_hadU(q_roped) → q_rotated
   4c. wrapper applies matmul_hadU(k_roped) → k_rotated
5. attn_output = scaled_dot_product_attention(q_rotated, k_rotated, v)
```

### 5.3 Quark R3 Implementation

Quark also uses **monkeypatching** with a very similar approach:

```python
# rotation.py → r3()
def r3(self):
    for layer in self.layers:
        attn = getattr(layer, self.rotation_config.self_attn)
        add_qk_rotation_after_rope(attn, self.head_dim, self.rotation_config)

# rotation_utils.py → add_qk_rotation_after_rope()
def add_qk_rotation_after_rope(module, head_dim, config):
    """Replace apply_rotary_pos_emb in attention forward globals."""
    fwd = module.forward
    if hasattr(fwd, "__func__"):
        globals_dict = fwd.__func__.__globals__
    else:
        globals_dict = fwd.__globals__

    original_fn = globals_dict["apply_rotary_pos_emb"]

    def rotated_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_roped, k_roped = original_fn(q, k, cos, sin, position_ids, unsqueeze_dim)
        # Apply Hadamard per-head
        hadamard_K, K = _get_hadamard_K(head_dim)
        hadamard_K = hadamard_K.to(q_roped.device)
        q_rotated = matmul_hadU(q_roped, hadamard_K, K) / math.sqrt(head_dim)
        k_rotated = matmul_hadU(k_roped, hadamard_K, K) / math.sqrt(head_dim)
        return q_rotated, k_rotated

    globals_dict["apply_rotary_pos_emb"] = rotated_apply_rotary_pos_emb
```

**Key difference from auto-round:** Quark normalizes by `/ math.sqrt(head_dim)` because
Quark's `_get_hadamard_K()` returns unnormalized matrices (±1 values). auto-round's
`get_hadamard_K()` returns already-normalized matrices (H/√N), so no extra division.

### 5.4 llm-compressor R3 Implementation

llm-compressor uses a fundamentally different approach: **attention implementation swapping**
with `QuantizedAttentionImpl` and `QuantizedKVCache` modules.

```python
# base.py → _create_r3_scheme()
def _create_r3_scheme(self, head_dim: int) -> TransformScheme:
    return TransformScheme(
        type=self.transform_type,
        head_dim=head_dim,
        apply=[
            TransformArgs(targets=[self.mappings.attn], location="q_attn"),   # query hook
            TransformArgs(targets=[self.mappings.attn], location="k_cache"),  # key hook
        ],
    )
```

**How hooks are registered** (`factory/base.py`):
```python
# In TransformFactory._apply_to_module():
elif args.location == TransformLocation.Q_ATTN:
    def query_hook(_, query_states):
        return transform(query_states)      # HadamardTransform applied to Q
    initialize_hooked_attention(model, module)
    register_query_hook(module, query_hook)

elif args.location == TransformLocation.K_CACHE:
    def key_hook(_, key_states):
        return transform(key_states)        # HadamardTransform applied to K
    initialize_hooked_kv_cache(model, module)
    register_key_hook(module, key_hook)
```

**Attention implementation swapping** (`modeling/attention.py`):
```python
def initialize_hooked_attention(model, module):
    # 1. Register QuantizedAttentionImpl as submodule
    module.register_module("impl", QuantizedAttentionImpl(model.config))

    # 2. Save original attention implementation
    QuantizedAttentionImpl._original_impl = model.config._attn_implementation

    # 3. Register new "ct_hooked_attention" function
    ALL_ATTENTION_FUNCTIONS.register("ct_hooked_attention", _hooked_attention)

    # 4. Switch model to use hooked implementation
    model.set_attn_implementation("ct_hooked_attention")

class QuantizedAttentionImpl(InternalModule):
    def forward(self, module, query, key, value, *args, **kwargs):
        # Hooks on this module intercept query/key AFTER RoPE
        # Then call original attention implementation
        return ALL_ATTENTION_FUNCTIONS[self._original_impl](
            module, query, key, value, *args, **kwargs
        )
```

**KV Cache interception** (`modeling/kvcache.py`):
```python
class QuantizedKVCache(InternalModule):
    def forward(self, key_states, value_states, *args, **kwargs):
        # Hooks intercept key_states here (after RoPE, before cache update)
        # Then call original cache update
        return self.past_key_values().update(key_states, value_states, *args, **kwargs)
```

**Runtime inference path (llm-compressor):**
```
1. attn.forward(hidden_states, ...) called
2. q = self.q_proj(hidden_states)
3. k = self.k_proj(hidden_states)
4. q, k = apply_rotary_pos_emb(q, k, cos, sin)  ← standard RoPE (not monkeypatched)
5. _hooked_attention(module, q, k, v, ...) called (swapped implementation)
   5a. attn.impl.forward(module, q, k, v, ...)
       → forward_pre_hook intercepts args
       → query_hook(_, q) → HadamardTransform(q) → q_rotated
   5b. kv_cache.forward(k, v, ...)
       → forward_pre_hook intercepts args
       → key_hook(_, k) → HadamardTransform(k) → k_rotated
6. Original attention: scaled_dot_product_attention(q_rotated, k_rotated, v)
```

### 5.5 R3 Comparison Table

| Aspect | auto-round | Quark | llm-compressor |
|--------|-----------|-------|----------------|
| **Mechanism** | Monkeypatch `apply_rotary_pos_emb` in forward globals | Monkeypatch `apply_rotary_pos_emb` in forward globals | Attention implementation swapping + hook system |
| **Injection point** | Inside RoPE call (wrapper calls original RoPE, then rotates) | Inside RoPE call | After RoPE, before attention dot-product |
| **Matrix** | H/√N (already normalized) | H (unnormalized, divides by √N in hook) | H (normalized by HadamardTransform._scale) |
| **head_dim restriction** | Must be power of 2 | Must be power of 2 | Must be power of 2 (or use random-hadamard) |
| **Save/Load** | ❌ Monkeypatch lost on save | ❌ Monkeypatch lost on save | ✅ QuantizedAttentionImpl + HadamardTransform saved as submodules |
| **Serialization** | Must manually call `register_spinquant_hooks()` after load | Must manually call `r3()` after load | Automatic — transforms loaded, hooks re-attached |
| **Inference overhead** | O(seq × num_heads × head_dim × log₂(head_dim)) per layer | Same | Same |
| **Compatible with SDPA/Flash?** | ✅ (applied before attention) | ✅ | ✅ (falls back to eager, then calls original impl) |

### 5.6 Why R3 is Applied After RoPE (Not Before)

RoPE applies **position-dependent** rotation to Q and K:
```
Q_roped = Q ⊙ cos(θ) + rotate_half(Q) ⊙ sin(θ)
```

If R3 were applied **before** RoPE:
```
Q' = Q @ H → RoPE(Q') ≠ RoPE(Q) @ H
```
because RoPE uses element-wise operations (⊙), which don't commute with matrix
multiplication. Applying R3 **after** RoPE ensures the rotation reduces outliers in
the post-RoPE representation without interfering with positional encoding.

---

## 6. R4 MLP Activation Rotation (ONLINE + OFFLINE) — Detailed Comparison

R4 rotates the MLP intermediate activation before `down_proj`. It has both an
**online** part (activation hook) and an **offline** part (weight inverse fusion).

### 6.1 Mathematical Formulation

Standard MLP:
```
up   = X @ W_up      # [seq, hidden] → [seq, intermediate]
gate = X @ W_gate     # [seq, hidden] → [seq, intermediate]
act  = up * SiLU(gate)  # element-wise
out  = act @ W_down   # [seq, intermediate] → [seq, hidden]
```

R4 applies Hadamard H_k (k = rotation_size or intermediate_size) to `act` before `down_proj`:
```
act_rotated = act @ H_k        # online (forward_pre_hook on down_proj)
out = act_rotated @ W_down     # need W_down to absorb H_k⁻¹

W_down_new = W_down @ H_k.T   # offline fusion (weight inverse)
→ out = (act @ H_k) @ (W_down @ H_k.T).T = act @ H_k @ H_k @ W_down.T = act @ W_down.T  ✓
```

### 6.2 Block Rotation (When rotation_size < intermediate_size)

For Qwen3-0.6B: intermediate_size=3072, rotation_size=128

```
3072 / 128 = 24 blocks
act: [seq, 3072] → reshape [seq, 24, 128] → Hadamard per block → reshape [seq, 3072]
```

### 6.3 auto-round R4

**Offline weight fusion** (`preprocessor.py → _fuse_r4_rotation()`):
```python
def _fuse_r4_rotation(self):
    if not self.config.r4:
        return
    r4_size = self.r4_rotation_size
    R4_mat, K4 = get_hadamard_K(r4_size)
    for layer in self._get_layers():
        down_proj = layer.mlp.down_proj
        # Rotate input channels of down_proj (inverse)
        rotate_in_channels_(down_proj, R_in=R4_mat)
```

**Online activation hook** (`inplace/apply.py`):
```python
if config.r4 and intermediate_size > 0:
    r4_size = r4_rotation_size or intermediate_size
    hadamard_K, K = get_hadamard_K(r4_size)
    need_block_rotation = (r4_size < intermediate_size)

    for name, module in model.named_modules():
        if name.endswith("down_proj") and isinstance(module, nn.Linear):
            if need_block_rotation:
                # Block rotation: reshape → per-block Hadamard → reshape
                def hook(mod, args, had_K=hadamard_K, K_val=K, blk=r4_size, inter=intermediate_size):
                    x = args[0]
                    orig_shape = x.shape
                    num_blocks = inter // blk
                    x = x.view(*orig_shape[:-1], num_blocks, blk)
                    x = matmul_hadU(x, hadamard_K=had_K.to(x.device, x.dtype), K=K_val)
                    return (x.view(orig_shape),) + args[1:]
            else:
                # Full rotation
                def hook(mod, args, had_K=hadamard_K, K_val=K):
                    x = args[0]
                    return (matmul_hadU(x, hadamard_K=had_K.to(x.device, x.dtype), K=K_val),) + args[1:]

            module.register_forward_pre_hook(hook)
```

### 6.4 Quark R4

Quark uses `InputRotationWrapper` for R4 (same as R1 online):
```python
# rotation.py → r4()
def r4(self):
    for layer in self.layers:
        mlp = getattr(layer, self.rotation_config.mlp)
        down_proj = mlp.down_proj
        # Wrap down_proj with InputRotationWrapper
        wrapper = InputRotationWrapper(down_proj, rotation_size=self.r4_size)
        setattr(mlp, "down_proj", wrapper)
        # Weight inverse already fused into down_proj
```

### 6.5 llm-compressor R4

```python
# base.py → _create_r4_scheme()
def _create_r4_scheme(self) -> TransformScheme:
    return TransformScheme(
        type=self.transform_type,
        apply=[
            TransformArgs(
                targets=self.mappings.mlp_out,    # "re:.*down_proj$"
                location="input",                  # online hook
            ),
            TransformArgs(
                targets=self.mappings.mlp_out,
                location="weight_input",           # offline fusion
                inverse=True,
            ),
        ],
    )
```

### 6.6 R4 Comparison Table

| Aspect | auto-round | Quark | llm-compressor |
|--------|-----------|-------|----------------|
| Online part | `forward_pre_hook` closure | `InputRotationWrapper` module | `HadamardTransform` submodule + hook |
| Offline part | `rotate_in_channels_()` on down_proj | Weight inverse in wrapper | `weight_input` inverse via factory |
| Block rotation | ✅ reshape → per-block Hadamard | ✅ | ✅ (via transform_block_size) |
| Save/Load | ❌ Hook lost on save | ✅ Wrapper saved as `input_rotation` buffer | ✅ HadamardTransform saved as submodule |
| Inference cost | O(seq × inter × log₂K) per layer | Same | Same (or zero with Hadacore kernel) |

---

## 7. Inference After Rotation + Quantization

### 7.1 auto-round Inference

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

### 7.2 Quark Inference

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

### 7.3 llm-compressor Inference

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

## 8. Model Save and Load

### 8.1 auto-round: Currently Hooks are LOST on Save

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

### 8.2 Quark: Full Save/Load with `QParamsLinearWithRotation`

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

### 8.3 llm-compressor: First-Class Transform Serialization

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

## 9. Comparison Table: Save/Load Capability

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

## 10. Test Script Execution Details

### 10.1 auto-round Test Flow

```python
# In test_three_way_comparison.py → run_autoround():
model = load_model(model_name, device)                        # Fresh FP16 model
model = apply_rotation(model, rotation_flags, rotation_size)  # Apply R1-R4 hooks
ar = AutoRound(model, tokenizer, scheme="MXFP4_RCEIL", iters=0, ...)
ar.quantize()                                                 # RTN quantize
model = ar.model.eval().to(device)                            # Quantized model with hooks
# → lm_eval evaluates using model.forward() which triggers hooks
```

### 10.2 Quark Test Flow

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

### 10.3 llm-compressor Test Flow

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

## 11. Key Architectural Differences Summary

### 11.1 Module Discovery

| Framework | Method | Implication |
|-----------|--------|-------------|
| auto-round | `type(m) in SUPPORTED_LAYER_TYPES` | Only recognizes `nn.Linear`, `Conv1D` |
| Quark | `torch.export.export_for_training()` → FX graph | Sees through wrappers to `aten.linear` |
| llm-compressor | Regex patterns on module names | Flexible, works with any naming convention |

### 11.2 Quantization + Rotation Ordering

```
auto-round:       Rotation (hooks) → Quantization (wraps nn.Linear) → Inference (hook inside wrapper)
Quark:            Rotation (wrappers) → FX trace → Quantization (on aten ops) → Export (QParamsLinear)
llm-compressor:   Rotation (modifier 1, fuse/hook) → Quantization (modifier 2, observe + pack)
```

### 11.3 Online Rotation Implementation

| Framework | R1 online | R4 online | Mechanism |
|-----------|-----------|-----------|-----------|
| auto-round | `forward_pre_hook` closure | `forward_pre_hook` closure | PyTorch hook system |
| Quark | `InputRotationWrapper(nn.Module)` | `InputRotationWrapper(nn.Module)` | Module wrapping |
| llm-compressor | N/A (offline) | `TransformBase` submodule + hook | Module + hook |

---

## 12. Performance Characteristics

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

## 13. Which Framework to Choose?

| Use Case | Recommendation | Reason |
|----------|---------------|--------|
| Best quantization quality | auto-round (online R1) | Preserves original weight distributions |
| Production deployment | llm-compressor | Full save/load, vLLM integration, Hadacore kernels |
| Research / flexibility | Quark | Most features, trainable rotation, comprehensive export |
| No save/load needed | auto-round | Simplest API, hook-based, scheme-agnostic |
| Need R3 support | auto-round or Quark | llm-compressor R3 is incomplete |
