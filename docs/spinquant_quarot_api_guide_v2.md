# Auto-Round SpinQuant / QuaRot API & Usage Guide (v2)

> **Rotation-based weight transforms for improved quantization accuracy.**
> This module implements [SpinQuant](https://arxiv.org/abs/2405.16406) and [QuaRot](https://arxiv.org/abs/2404.00456)
> rotation techniques as a preprocessing step before Auto-Round quantization.

> **v2 changes:** Unified `apply_rotation()` API, known Hadamard matrices for non-power-of-2 dimensions,
> `rotation_size` rarely needed, R1+R2+R3 rotation level supported.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
3. [Quick Start](#3-quick-start)
4. [API Reference](#4-api-reference)
5. [Configuration Guide](#5-configuration-guide)
6. [Usage Patterns](#6-usage-patterns)
7. [Rotation Levels Explained](#7-rotation-levels-explained)
8. [Quantization Scheme Compatibility](#8-quantization-scheme-compatibility)
9. [Online vs Offline Rotation](#9-online-vs-offline-rotation)
10. [Advanced Topics](#10-advanced-topics)
11. [Dimension Support & Known Hadamard Matrices](#11-dimension-support)
12. [Troubleshooting](#12-troubleshooting)
13. [Examples](#13-examples)

---

## 1. Overview

### What Problem Does Rotation Solve?

Large Language Models (LLMs) have **outlier channels** — a small number of hidden dimensions
with values 10–100× larger than average. These outliers make quantization difficult: the
large dynamic range forces the quantizer to use coarse step sizes, degrading accuracy for
the majority of channels.

**Rotation** applies an orthogonal transform (Hadamard matrix) to **redistribute** these
outliers across all channels. Since orthogonal transforms preserve the L2 norm, the model's
output is mathematically identical — but the weight distribution becomes much more uniform,
leading to better quantization.

```
Before rotation:    [0.1, 0.2, 50.0, 0.3, 0.1, ...]   ← outlier at channel 3
After rotation:     [2.1, 1.8, 2.3, 1.9, 2.0, ...]     ← uniform distribution
```

### QuaRot vs SpinQuant

| Aspect | QuaRot | SpinQuant |
|--------|--------|-----------|
| **Rotation matrix** | Fixed Hadamard (deterministic or random) | Learned orthogonal (trained via Cayley SGD) |
| **Requires training** | No | Yes (calibration data needed) |
| **Speed** | Instant (seconds) | Slow (minutes to hours) |
| **Accuracy** | Good | Best (optimized for specific model) |
| **Reproducibility** | Deterministic: always identical; Random: must save matrix | Must save trained matrix |
| **When to use** | Quick experiments, large models | When accuracy matters most |

### What's Supported

| Feature | Status | Notes |
|---------|--------|-------|
| QuaRot (fixed Hadamard, R1-R4) | ✅ Fully supported | Production-ready, no training needed |
| SpinQuant (trainable rotation) | ⚠️ Experimental | Training loop exists but NOT validated end-to-end on real models |
| Unified API (`apply_rotation`) | ✅ New in v2 | Single entry point, registry-based dispatch |
| String shorthand (`"quarot"`) | ✅ New in v2 | `apply_rotation(model, "quarot")` |
| Non-power-of-2 dimensions | ✅ New in v2 | Known Hadamard matrices (12,20,28,...,172) via Quark port |
| Deterministic Hadamard | ✅ Default | Same matrix every run, no need to persist |
| Random Hadamard (H×D) | ✅ Via `random_r1=True`/`random_r2=True` | Must persist the matrix for reproducibility |
| Online R1 (hook-based) | ✅ Default, recommended | Small runtime overhead per layer |
| Offline R1 (weight fusion) | ✅ Via `online_r1_rotation=False` | ⚠️ May degrade accuracy when combined with quantization |
| R2 (per-head V/O rotation) | ✅ Offline fused | Zero runtime cost |
| R3 (Q/K after RoPE) | ✅ Online monkeypatch | Works across all HF architectures |
| R4 (MLP activation) | ✅ Online hook | Hybrid: activation online + weight offline |
| Block rotation (`rotation_size`) | ✅ Rarely needed in v2 | Known Hadamard matrices handle most sizes |
| Model save/load after rotation | ❌ Not yet implemented | Online hooks not serialized by `save_pretrained()` |
| Pre-trained rotation matrices | ❌ Not shipped | No pre-trained SpinQuant matrices available |
| All HuggingFace architectures | ✅ Llama, Qwen, Mistral, Phi, Gemma, etc. | Generic monkeypatch approach |

---

## 2. Core Concepts

### 2.1 Four Rotation Levels

The rotation scheme has four independent levels (R1–R4), each targeting different parts
of the transformer:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Transformer Block                           │
│                                                                 │
│  Input (residual) ─── [R1] ──→ LayerNorm → Q/K/V projection    │
│                                                                 │
│  Attention:                                                     │
│    Q, K ─── RoPE ─── [R3] ──→ Attention scores                 │
│    V ─── [R2] ──→ V_rotated → Attention output → O_proj        │
│                                     ↓                           │
│    O_proj output ─── [R2⁻¹] ──→ residual connection            │
│                                                                 │
│  MLP:                                                           │
│    gate/up_proj → activation ─── [R4] ──→ down_proj             │
│                                                                 │
│  Output (residual) ──→ next block                               │
└─────────────────────────────────────────────────────────────────┘
```

| Level | Target | Dimension | Online/Offline | Effect |
|-------|--------|-----------|----------------|--------|
| **R1** | Residual stream | `hidden_size` | Both (configurable) | Smooths all linear layers in the block |
| **R2** | Attention V/O | `head_dim` | Offline only | Smooths per-head attention values |
| **R3** | Query/Key | `head_dim` | Online only | Applied after RoPE, improves K/Q quantization |
| **R4** | MLP activation | `intermediate_size` | Online only | Smooths gate/up activations before down_proj |

### 2.2 Offline vs Online

- **Offline rotation**: Fused into weights during model preparation. Zero runtime cost.
  `W_new = R⁻¹ @ W @ R` — mathematically absorbed into the weight matrix.
- **Online rotation**: Applied at every inference step via hooks or monkeypatches.
  Small runtime cost but necessary when rotation cannot be fused (e.g., after RoPE).

### 2.3 Rotation Matrix Types

| Type | Formula | Reproducible? | Config |
|------|---------|---------------|--------|
| **Deterministic Hadamard** | `H = Sylvester(N) / √N` | Yes (same every run) | `random_r1=False` (default) |
| **Random Hadamard** | `R = H × diag(±1)` | No (must save matrix) | `random_r1=True` |
| **Trainable** (SpinQuant) | `R = Cayley(A)` optimized | No (must save) | `trainable_rotation=True` |

---

## 3. Quick Start

### 3.1 QuaRot via Unified API (Recommended — New in v2)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

# Load model
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Step 1: Apply rotation via unified API
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,  # R1+R2
    trainable_rotation=False,                # QuaRot (fixed Hadamard)
    trainable_smooth=False,
    online_r1_rotation=True,                 # Online R1 (recommended)
)
model = apply_rotation(model, config)

# Step 2: Quantize with Auto-Round
autoround = AutoRound(model, tokenizer, scheme="W4A16", iters=0)
autoround.quantize()
```

### 3.2 QuaRot with String Shorthand (Simplest — New in v2)

```python
from auto_round.algorithms.transforms import apply_rotation

# "quarot" shorthand = SpinQuantConfig(trainable_rotation=False, trainable_smooth=False)
# Uses R1+R2+R3+R4 by default
model = apply_rotation(model, "quarot")
```

### 3.3 Full Rotation (R1+R2+R3+R4)

```python
# No rotation_size needed for Qwen3-0.6B (intermediate_size=3072)
# v2 handles non-power-of-2 via known Hadamard matrices automatically
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False,
    trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)

# Quantize
autoround = AutoRound(model, tokenizer, scheme="MXFP4", iters=0)
autoround.quantize()
```

### 3.4 Direct SpinQuantPreprocessor (Legacy, Still Supported)

```python
from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig, SpinQuantPreprocessor
)

config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()
```

### 3.5 Evaluation After Rotation + Quantization

```python
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8, device="cuda:0")
results = simple_evaluate(
    model=lm,
    tasks=["hellaswag", "piqa", "winogrande", "lambada_openai", "mmlu"],
    batch_size=8,
    limit=None,   # None for full eval, integer for quick test
)

for task, data in results["results"].items():
    acc = data.get("acc_norm,none") or data.get("acc,none")
    print(f"{task}: {acc:.4f}")
```

---

## 4. API Reference

### 4.1 Unified Entry Point (New in v2)

```python
from auto_round.algorithms.transforms import apply_rotation
```

**Signature:**
```python
def apply_rotation(
    model: torch.nn.Module,
    config: Any,            # SpinQuantConfig, dict, str, or None
    data_type: str = "mx_fp",
    **kwargs,
) -> torch.nn.Module
```

**Config forms:**

| Form | Example | Description |
|------|---------|-------------|
| `SpinQuantConfig` | `SpinQuantConfig(r1=True, r2=True, ...)` | Full control |
| `str "quarot"` | `apply_rotation(model, "quarot")` | QuaRot defaults (all R1-R4, no training) |
| `str "spinquant"` | `apply_rotation(model, "spinquant")` | SpinQuant defaults (with training) |
| `dict` | `{"algorithm": "spinquant", "r1": True, "r2": True}` | Dict-based config |
| `None` | `apply_rotation(model, None)` | No-op, returns model unchanged |

### 4.2 `SpinQuantConfig`

Configuration dataclass for rotation preprocessing.

```python
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r1` | `bool` | `True` | Enable R1 (hidden_size rotation) |
| `r2` | `bool` | `True` | Enable R2 (per-head V/O rotation) |
| `r3` | `bool` | `True` | Enable R3 (Q/K online rotation after RoPE) |
| `r4` | `bool` | `True` | Enable R4 (MLP activation online rotation) |
| `rotation_size` | `Optional[int]` | `None` | Custom rotation dimension. `None` = use model's native dimensions. **v2: rarely needed** — known Hadamard matrices handle non-pow2 sizes (768, 3072, 5120, 11008, etc.). Only set if the preprocessor warns about unsupported dimensions. |
| `random_r1` | `bool` | `False` | Use random Hadamard (H×D) for R1 instead of deterministic. |
| `random_r2` | `bool` | `False` | Use random Hadamard for R2. |
| `trainable_rotation` | `bool` | `True` | Train rotation via Cayley SGD (SpinQuant). `False` = fixed Hadamard (QuaRot). |
| `trainable_smooth` | `bool` | `True` | Train SmoothQuant scaling factors via Adam. |
| `online_r1_rotation` | `bool` | `True` | Online R1: rotate weights per-module + hook. `False` = offline global fusion. |
| `iters` | `int` | `200` | Training iterations (SpinQuant mode only). |
| `lr` | `float` | `1e-4` | SGDG learning rate for rotation matrices. |
| `smooth_lr` | `float` | `1e-3` | Adam learning rate for smooth values. |
| `batch_size` | `int` | `1` | Training batch size. |
| `loss_type` | `str` | `"kl_top"` | Loss: `"kl_top"`, `"kl_full"`, or `"mse"`. |
| `kl_top_k` | `int` | `1000` | Top-k tokens for KL divergence loss. |
| `fuse_rmsnorm` | `bool` | `True` | Fuse RMSNorm gamma into linear weights. |
| `untie_embeddings` | `bool` | `True` | Untie input/output embeddings. |
| `dtype` | `torch.dtype` | `torch.float32` | Computation dtype for rotation. |
| `device` | `Optional[str]` | `None` | Auto-detects "cuda" or "cpu". |

### 4.3 `SpinQuantPreprocessor`

Main class that orchestrates the 8-step rotation pipeline.

```python
from auto_round.algorithms.transforms.spinquant import SpinQuantPreprocessor

preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess(dataloader=None)
```

**8-Step Pipeline:**
1. Untie embeddings (if offline R1)
2. Fuse RMSNorm gamma into weights (if offline R1)
3. Add trainable smooth values (if `trainable_smooth=True`)
4. Initialize rotation matrices (Hadamard or identity)
5. Train rotations (if `trainable_rotation=True` — requires dataloader)
6. Apply R1 (online or offline), fuse R2/R4 offline
7. Register R3/R4 online hooks
8. Cleanup training artifacts

### 4.4 `BaseRotation` Registry (New in v2)

```python
from auto_round.algorithms.transforms.base import BaseRotation

# SpinQuantRotation is registered as "spinquant"
rotation = BaseRotation.from_config(config)
model = rotation.apply_to_model(model)
```

### 4.5 `RotationTrainer` (Alternative Interface)

> ⚠️ **Experimental**: The `RotationTrainer` has basic infrastructure (Cayley SGD,
> KL-divergence loss, callbacks, checkpointing) but the SpinQuant training path
> has **NOT been validated end-to-end** on real models. For production use, prefer
> QuaRot mode (`trainable_rotation=False`).

```python
from auto_round.algorithms.transforms.spinquant import (
    RotationTrainer, RotationTrainerConfig
)

trainer = RotationTrainer(model, config=RotationTrainerConfig(iters=200, lr=1e-4))
metrics = trainer.train(dataloader)
model = trainer.fuse()
```

### 4.6 Hook Management

```python
from auto_round.algorithms.transforms.spinquant import (
    register_spinquant_hooks, remove_spinquant_hooks
)

handles = register_spinquant_hooks(
    model, config,
    head_dim=128,
    intermediate_size=3584,
    r4_rotation_size=128,
)

# Later, to remove hooks:
remove_spinquant_hooks(handles)
```

### 4.7 Rotation Utilities

```python
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    deterministic_hadamard_matrix,  # H / √N, always the same
    random_hadamard_matrix,         # H × diag(±1) / √N (supports non-pow2 via known Hadamard)
    matmul_hadU,                    # O(N log N) butterfly Hadamard multiply (supports non-pow2)
    is_pow2,                        # Check power-of-2
    get_hadamard_K,                 # Get (hadamard_matrix, K) for size N (uses known Hadamard)
    rotate_in_channels_,            # W_new = W @ R (in-place)
    rotate_out_channels_,           # W_new = R @ W (in-place)
    fuse_rmsnorm_in_model,          # Fuse all RMSNorm gamma
    untie_word_embeddings_if_needed,# Untie embeddings
    get_model_arch_info,            # Extract model dimensions
)
```

---

## 5. Configuration Guide

### 5.1 Common Configuration Presets

**QuaRot R1 only (simplest, good baseline):**
```python
config = SpinQuantConfig(
    r1=True, r2=False, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)
```

**QuaRot R1+R2 (most common, best cost/benefit):**
```python
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)
```

**QuaRot R1+R2+R3 (adds Q/K rotation):**
```python
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)
```

**QuaRot Full (R1+R2+R3+R4):**
```python
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)
```

**String shorthand (QuaRot R1+R2+R3+R4):**
```python
model = apply_rotation(model, "quarot")
```

**Random Hadamard (better outlier distribution):**
```python
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    random_r1=True, random_r2=True,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)
```

**SpinQuant (trainable — ⚠️ experimental, training under development):**
```python
# ⚠️ Training loop exists but NOT validated end-to-end.
# Use QuaRot mode above for production.
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=True, trainable_smooth=True,
    online_r1_rotation=True,
    iters=200, lr=1e-4, smooth_lr=1e-3,
    loss_type="kl_top", kl_top_k=1000,
)
model = apply_rotation(model, config)  # needs dataloader via kwargs
```

### 5.2 Rotation Level Selection Guide

| Scenario | Recommended Levels | Why |
|----------|-------------------|-----|
| Quick test / baseline | R1 only | Minimal overhead, noticeable improvement |
| Production (weight-only quant) | R1+R2 | Best cost/benefit for W4A16, W8A16 |
| Weight+Activation quant (MXFP4, NVFP4) | R1+R2+R3+R4 | Activation quantization benefits from R3/R4 |
| Accuracy-critical deployment | R1+R2+R3+R4 + SpinQuant training | Best possible accuracy (⚠️ training experimental) |

### 5.3 `rotation_size` vs Known Hadamard Matrices

These are **two independent mechanisms** solving different problems:

| | `rotation_size=None` (default) | `rotation_size=128` (custom) |
|---|---|---|
| **R1** | Full `hidden_size` rotation | hidden_size/128 independent blocks |
| **R4** | Full `intermediate_size` rotation | intermediate_size/128 independent blocks |
| **Hadamard construction** | Needs decomposition for non-pow2 dims | Each 128-block is pow2, always works |

**Before v2:** If `intermediate_size=3072` (not pow2), R4 failed with `rotation_size=None`.
Users had to set `rotation_size=128` as a workaround (block rotation).

**After v2 (known Hadamard):** `rotation_size=None` works directly — 3072 is decomposed
as 12×256 using a pre-computed Hadamard matrix for size 12. No workaround needed.

**v2: In most cases, you don't need `rotation_size`.** The known Hadamard matrix
decomposition handles non-power-of-2 dimensions automatically:

| Dimension | Decomposition | Status |
|-----------|--------------|--------|
| 768 | 12 × 64 | ✅ Supported natively |
| 1024 | pow2 | ✅ Supported natively |
| 1536 | 12 × 128 | ✅ Supported natively |
| 2048 | pow2 | ✅ Supported natively |
| 3072 | 12 × 256 | ✅ Supported natively |
| 4096 | pow2 | ✅ Supported natively |
| 5120 | 20 × 256 | ✅ Supported natively |
| 7168 | 28 × 256 | ✅ Supported natively |
| 8192 | pow2 | ✅ Supported natively |
| 11008 | 172 × 64 | ✅ Supported natively |
| 14336 | 28 × 512 | ✅ Supported natively |

**When to use `rotation_size`:**
- Only if the preprocessor warns about unsupported dimensions
- Or if you want smaller block rotations for memory/speed reasons

**Constraints on `rotation_size`:**
- Must be a **power of 2** (16, 32, 64, 128, 256, ...)
- Must **divide** both `hidden_size` (for R1) and `intermediate_size` (for R4)
- Does **not** affect R2/R3 (they always use `head_dim`)

The preprocessor automatically validates and falls back gracefully:
```
[SpinQuant] WARNING: Cannot find Hadamard decomposition for intermediate_size=XXXX.
                     R4 rotation disabled for this model.
```

### 5.4 Known Limitation: R1 Validation Strictness

> ⚠️ **Current inconsistency:** R4 validation uses `get_hadamard_K()` (supports non-pow2 via
> known Hadamard matrices), but R1/R2/R3 validation still requires strict power-of-2 dimensions.
>
> This means: if a model has `hidden_size=768` (e.g., GPT-2), R1 is auto-disabled even though
> 768 = 12 × 64 could be handled by `matmul_hadU`. R4 with `intermediate_size=3072` works fine.
>
> **Impact:** Models with non-pow2 `hidden_size` cannot use R1/R2/R3 at full dimension.
> Workaround: set `rotation_size=64` or `rotation_size=128` (must divide `hidden_size`).
>
> This will be relaxed in a future update to use `get_hadamard_K()` for R1/R2/R3 validation.

---

## 6. Usage Patterns

### 6.1 Pattern A: Rotation + RTN Quantization (Fastest)

```python
from auto_round.algorithms.transforms import apply_rotation

# Rotation (QuaRot, no training)
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)

# RTN quantization (iters=0)
autoround = AutoRound(model, tokenizer, scheme="W4A16", iters=0)
autoround.quantize()
```

### 6.2 Pattern B: Rotation + Auto-Round Tuning (Better Accuracy)

```python
# Rotation is decoupled from quantization — any rotation combo
# works with any quantization method (RTN or tuning)
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)

# Auto-round tuning (iters=200, typically +3-5% accuracy over RTN)
autoround = AutoRound(
    model, tokenizer,
    scheme="W4A16",
    iters=200,         # Optimization iterations
    nsamples=128,      # Calibration samples
    seqlen=2048,       # Sequence length
)
autoround.quantize()
```

### 6.3 Pattern C: Rotation-Only (No Quantization, for Debugging)

```python
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)

# Evaluate directly — should match FP16 baseline (rotation is lossless in FP)
results = evaluate_model(model, tokenizer, tasks="hellaswag")
```

### 6.4 Pattern D: With lm_eval for Accuracy Benchmarking

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

# After rotation + quantization...
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8, device="cuda:0")
results = lm_eval.evaluator.simple_evaluate(
    model=lm,
    tasks=["hellaswag", "piqa", "winogrande", "lambada_openai", "mmlu"],
    batch_size=8,
)

for task, data in results["results"].items():
    acc = data.get("acc_norm,none") or data.get("acc,none", 0)
    print(f"  {task}: {acc:.4f}")
```

---

## 7. Rotation Levels Explained

### 7.1 R1: Residual Stream Rotation

**What:** Applies a full `hidden_size × hidden_size` Hadamard rotation to the residual stream.

**Math:**
```
For each linear layer W in the block:
  W_new = R1⁻¹ @ W @ R1       (for internal layers)
  W_new = W @ R1               (for first layer)
  W_new = R1⁻¹ @ W             (for last layer)
```

**Online mode** (default, recommended):
- Each target linear layer's weight is rotated: `W_rotated = W @ R1`
- A `forward_pre_hook` applies `x_rotated = x @ R1.T` before the layer
- No change to embeddings, norms, or output head

**Offline mode** (`online_r1_rotation=False`):
- Rotation is fused globally: `embed → R1`, `head → R1⁻¹`, all internal layers `R1⁻¹ @ W @ R1`
- Requires untying embeddings and fusing RMSNorm
- More complex, same mathematical result

### 7.2 R2: Attention Head Rotation

**What:** Per-head Hadamard rotation on V/O projection pairs.

**Math (per head h):**
```
V_proj_new[h] = V_proj[h] @ R2        (output channels of v_proj)
O_proj_new[h] = R2.T @ O_proj[h]      (input channels of o_proj)
```

**Always offline** — V and O are directly connected, fully fuseable.

### 7.3 R3: Query/Key Rotation (After RoPE)

**What:** Hadamard rotation applied to Q and K **after** RoPE position embedding.

**Math:**
```
Q_rotated = matmul_hadU(Q_after_RoPE)
K_rotated = matmul_hadU(K_after_RoPE)
```

**Always online** — must happen after RoPE because position embedding is input-dependent.

**Implementation:** Architecture-generic monkeypatch replaces `apply_rotary_pos_emb` in
the attention forward's globals. Works for Llama, Qwen, Mistral, Phi, Gemma, etc.

### 7.4 R4: MLP Activation Rotation

**What:** Block Hadamard rotation on MLP hidden activations before `down_proj`.

**Math:**
```
x_gate_up = activation(gate(x)) * up(x)      # MLP intermediate
x_rotated = matmul_hadU(x_gate_up)            # Apply Hadamard
output = down_proj(x_rotated)                 # down_proj sees smoother input
```

**Always online** (activation part) — applied via `forward_pre_hook` on `down_proj`.
The inverse rotation `R4⁻¹` is fused into `down_proj` weights (offline).

**v2 improvement:** Now supports non-power-of-2 `intermediate_size` (e.g., 3072 = 12 × 256)
via known Hadamard matrices. Previously required `rotation_size` for these models.

---

## 8. Quantization Scheme Compatibility

Auto-Round supports 30+ quantization schemes. Here are the most commonly used with rotation:

### 8.1 Weight-Only Quantization (WxA16)

| Scheme | Config | Best Rotation | Notes |
|--------|--------|---------------|-------|
| `W4A16` | INT4, group_size=128 | R1+R2 | Most common, best supported |
| `W3A16` | INT3, group_size=128 | R1+R2+R3+R4 | Aggressive, rotation helps more |
| `W2A16` | INT2, group_size=128 | R1+R2+R3+R4 | Very aggressive |
| `W8A16` | INT8, group_size=128 | R1 | Already high precision, minimal rotation needed |

### 8.2 Weight+Activation Quantization (WxAx)

| Scheme | Config | Best Rotation | Notes |
|--------|--------|---------------|-------|
| `MXFP4` | MX FP4, group_size=32, W4A4 | R1+R2+R3+R4 | R3/R4 crucial for activation quant |
| `MXFP4_RCEIL` | MX FP4 with RCEIL rounding | R1+R2+R3+R4 | Slightly better rounding |
| `NVFP4` | NV FP4, group_size=16, W4A4 | R1+R2+R3+R4 | NVIDIA hardware-optimized |
| `INT8` | INT8 W8A8, per-tensor | R1+R2 | Good balance |
| `MXFP8` | MX FP8, group_size=32 | R1+R2 | High precision, less rotation benefit |

### 8.3 Rotation + Quantization is Decoupled

Rotation (preprocessing) and quantization (AutoRound) are fully decoupled:
- **Any rotation combo** works with **any quantization scheme**
- **Any quantization method** works: RTN (iters=0) or auto-round tuning (iters=200)
- Tuning (iters=200) consistently adds +3-5% accuracy over RTN regardless of rotation

```python
# Same rotation setup
model = apply_rotation(model, config)

# Choice 1: RTN (fast, good baseline)
AutoRound(model, tokenizer, scheme="W4A16", iters=0).quantize()

# Choice 2: Auto-round tuning (slower, better accuracy)
AutoRound(model, tokenizer, scheme="W4A16", iters=200).quantize()
```

---

## 9. Online vs Offline Rotation

### 9.1 Summary Table

| Rotation | Online | Offline | Default | Configurable? |
|----------|--------|---------|---------|---------------|
| R1 | ✅ (hook) | ✅ (weight fusion) | Online | `online_r1_rotation=True/False` |
| R2 | ❌ | ✅ (always fused) | Offline | No |
| R3 | ✅ (monkeypatch) | ❌ | Online | No |
| R4 | ✅ (activation hook) + offline (weight inverse) | Hybrid | Hybrid | No |

### 9.2 Online R1 vs Offline R1

**Online R1** (recommended, `online_r1_rotation=True`):
- Rotates each target module's weight: `W_rotated = W @ R1`
- Registers `forward_pre_hook`: `x → x @ R1.T`
- Pros: No need to untie embeddings or fuse RMSNorm, simpler
- Cons: Small per-module runtime cost for the hook

**Offline R1** (`online_r1_rotation=False`):
- Fuses rotation globally into all weights
- Rotates embed_tokens, LM head, and all internal layers
- Requires: untie embeddings + fuse RMSNorm (automatically handled)
- Pros: Zero runtime overhead
- Cons: More complex transformations, must handle embeddings carefully

**⚠️ Accuracy Warning:** In quantized models, offline R1 can degrade accuracy significantly
because the fused RMSNorm changes the effective precision of normalization. Online R1 is
recommended for quantized models.

---

## 10. Advanced Topics

### 10.1 Block Rotation (`rotation_size`)

When `rotation_size` is set, the dimension is divided into blocks and Hadamard is applied
independently per block:

```
Input: [x₁, x₂, ..., x₁₀₂₄]
With rotation_size=128:  8 blocks of 128
  Block 1: H × [x₁, ..., x₁₂₈]
  Block 2: H × [x₁₂₉, ..., x₂₅₆]
  ...
```

**Which levels use `rotation_size`:**
- R1: uses `rotation_size` or `hidden_size` (must divide `hidden_size`)
- R2: always `head_dim` (not affected)
- R3: always `head_dim` (not affected)
- R4: uses `rotation_size` or `intermediate_size` (must divide `intermediate_size`)

**Note:** `rotation_size` must be a power of 2. With v2's known Hadamard matrices,
block rotation is rarely needed — most non-pow2 dimensions are handled natively.
See §5.3 for details on the relationship between `rotation_size` and known Hadamard matrices.

### 10.2 Normalization Convention

Auto-round normalizes Hadamard matrices by `1/√N` after computing `H_K ⊗ H_butterfly`:
```python
# For dimension n = K × 2^m:
# H = kron(H_K, I) applied via butterfly, then divided by √n
result = matmul_hadU(x)  # internally divides by √n
```

This matches Quark's convention and ensures `H @ H.T ≈ I` (orthogonal).

### 10.3 Butterfly Algorithm (matmul_hadU)

The online rotations (R3, R4) use the Fast Walsh-Hadamard Transform for O(N·log N)
complexity instead of O(N²) dense matrix multiplication:

```python
# O(N log N) butterfly — works for non-pow2 via known Hadamard decomposition
matmul_hadU(x)

# vs O(N²) dense
x @ hadamard_matrix
```

For non-power-of-2 dimensions, `matmul_hadU` first applies the known Hadamard
matrix for the K-factor, then uses butterfly for the power-of-2 part:
```
n = K × 2^m
H_n = H_K ⊗ H_{2^m}
```

### 10.4 Supported Model Architectures

The rotation implementation is architecture-generic thanks to:
- `get_model_arch_info()`: Auto-detects hidden_size, head_dim, etc. from model config
- R3 monkeypatch: Works by replacing `apply_rotary_pos_emb` in attention's `forward.__globals__`
- Layer iteration: Uses HuggingFace naming conventions (`model.layers`, `q_proj`, etc.)

Tested architectures: Llama, Qwen2, Qwen3, Mistral, Phi, Gemma.

### 10.5 Fallback Behavior

The preprocessor validates dimensions for each rotation level. If a dimension is unsupported,
the rotation is automatically disabled with a warning:

- **R1**: Checks `r1_rotation_size` is power-of-2 → ⚠️ disables for non-pow2 hidden_size (e.g., 768). See §5.4.
- **R2**: Checks `head_dim` is power-of-2 → disables if not
- **R3**: Checks `head_dim` is power-of-2 → disables if not
- **R4**: Uses `get_hadamard_K()` try/except → **broadest coverage**, supports all known Hadamard sizes + pow2. Only falls back for truly unsupported sizes.

> The R1/R2/R3 strictness vs R4's flexibility is a known inconsistency (see §5.4).
> R4 was updated to use `get_hadamard_K()` when we fixed the non-pow2 Hadamard issue;
> R1/R2/R3 will be relaxed in a future update.

---

## 11. Dimension Support & Known Hadamard Matrices

### 11.1 Known Hadamard Matrices (New in v2)

v2 ports AMD Quark's pre-computed Hadamard matrices for 11 non-power-of-2 sizes:

| Size K | Used for decomposition |
|--------|----------------------|
| 12 | 768=12×64, 1536=12×128, 3072=12×256 |
| 20 | 5120=20×256 |
| 28 | 7168=28×256, 14336=28×512 |
| 36 | |
| 40 | |
| 52 | |
| 60 | |
| 108 | |
| 140 | |
| 156 | |
| 172 | 11008=172×64 |

### 11.2 Decomposition Algorithm

For a dimension `n`, `get_hadamard_K(n)` finds the decomposition:
```
n = K × 2^m
where K ∈ {1, 12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172}
```

The Hadamard matrix is then: `H_n = H_K ⊗ H_{2^m}`

This covers virtually all mainstream LLM dimensions:
- **GPT-2 / small models**: 768 = 12 × 64 ✅
- **Llama / Qwen (1B-7B)**: 2048, 4096 = pow2 ✅
- **Qwen3-0.6B**: hidden=1024 (pow2) ✅, intermediate=3072 (12×256) ✅
- **Llama-7B**: intermediate=11008 (172×64) ✅
- **Llama-13B/70B**: intermediate=13824, 28672 = 108×128, 28×1024 ✅

### 11.3 Unsupported Dimensions

If `get_hadamard_K()` cannot find a decomposition, the rotation level is disabled:
```
[SpinQuant] WARNING: Cannot find Hadamard decomposition for size XXXX.
                     Rotation disabled for this dimension.
```

Workaround: Set `rotation_size` to a supported power-of-2 value (e.g., 64, 128, 256)
that divides the dimension.

---

## 12. Troubleshooting

### Q: Accuracy drops significantly after rotation + quantization

**A:** Check these in order:
1. **Rotation-only test**: Run rotation without quantization and compare with FP16 baseline.
   Accuracy should be identical (within floating-point tolerance). If not, there's a rotation bug.
2. **Online vs Offline R1**: Switch to `online_r1_rotation=True` — offline R1 can degrade
   accuracy in quantized models.
3. **Try auto-round tuning**: Use `iters=200` instead of `iters=0`. Tuning consistently
   adds +3-5% accuracy.

### Q: "Cannot find suitable Hadamard decomposition" error

**A (v2):** This should be rare with v2's known Hadamard matrices. If it occurs:
1. The dimension is truly unsupported (not decomposable as K × 2^m for known K values)
2. Workaround: Set `rotation_size` to a power-of-2 that divides the dimension

### Q: R3 or R4 disabled automatically

**A:** The preprocessor validates dimensions. Check the log messages for details.
With v2, R4 supports many more dimensions than before (via known Hadamard matrices).

### Q: Can I use rotation without quantization?

**A:** Yes. Just skip the `AutoRound` step. The rotated model is mathematically equivalent
to the original (in FP precision). This is useful for verifying rotation correctness.

### Q: Does rotation slow down inference?

**A:** Online rotations (R3, R4, online R1) add small overhead:
- R1 online: ~1% overhead (one matmul per hook per layer)
- R3: ~2% overhead (butterfly matmul on Q,K per layer)
- R4: ~1% overhead (butterfly matmul on MLP activation per layer)
- R2: Zero overhead (offline fused)

### Q: How do I use the old API?

**A:** The old `SpinQuantPreprocessor` API is still fully supported:
```python
# Old API (still works)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()

# New unified API (recommended)
model = apply_rotation(model, config)
```

---

## 13. Examples

### 13.1 Complete: Rotation + W4A16 + Evaluation (v2 API)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

# Load
model_name = "Qwen/Qwen3-0.6B"
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Rotate (QuaRot R1+R2, deterministic)
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
model = apply_rotation(model, config)

# Quantize (W4A16 RTN)
ar = AutoRound(model, tokenizer, scheme="W4A16", iters=0, device_map=device)
ar.quantize()

# Evaluate
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
lm = HFLM(pretrained=ar.model, tokenizer=tokenizer, batch_size=8, device=device)
results = simple_evaluate(model=lm, tasks=["hellaswag", "piqa"], batch_size=8)
for task, data in results["results"].items():
    print(f"{task}: {data.get('acc_norm,none', data.get('acc,none', 'N/A'))}")
```

### 13.2 Batch: Multiple Rotations × Multiple Schemes

See `examples/test_rotation_scheme_matrix_v2.py` for a comprehensive script that tests
all combinations of rotation levels and quantization schemes with lm_eval evaluation.

```bash
# Quick test
bash examples/run_rotation_scheme_matrix_v2.sh quick

# Full eval with all 5 rotation levels
bash examples/run_rotation_scheme_matrix_v2.sh full

# With auto-round tuning
bash examples/run_rotation_scheme_matrix_v2.sh tuning
```

### 13.3 Comparison: Auto-Round vs Quark vs llm-compressor

See `examples/test_three_way_comparison.py` for side-by-side framework comparison
using identical rotation configurations and quantization settings.

> **Note:** `test_three_way_comparison.py` uses the legacy `SpinQuantPreprocessor` API
> with `rotation_size=128` (block rotation). This still works after refactoring — no
> modifications needed. However, with v2's known Hadamard matrices, block rotation is
> no longer required for Qwen3-0.6B (intermediate_size=3072 = 12×256 is handled natively).
> To compare full-dimension rotation across frameworks, change `rotation_size=128` to
> `rotation_size=None` in the script.

---

## Appendix A: Test Scripts Summary

| Script | API | Description |
|--------|-----|-------------|
| `test_rotation_scheme_matrix.py` | Legacy (`SpinQuantPreprocessor`) | v1: Rotation × quantization accuracy matrix |
| `test_rotation_scheme_matrix_v2.py` | Unified (`apply_rotation`) | v2: Same as above, adds R1+R2+R3, mmlu, non-pow2 support |
| `run_rotation_scheme_matrix.sh` | — | v1 shell wrapper (modes: quick/full/full-matrix/...) |
| `run_rotation_scheme_matrix_v2.sh` | — | v2 shell wrapper (adds tuning/tuning-matrix modes) |
| `test_three_way_comparison.py` | Legacy (`SpinQuantPreprocessor`) | Auto-Round vs Quark vs llm-compressor comparison |
| `test_reference_equivalence.py` | Legacy | Numerical equivalence vs Quark reference |
| `test_quark_comparison.py` | Legacy | Side-by-side Quark vs auto-round rotation |
| `test_rotation_levels.py` | Legacy | Per-level rotation correctness (no quantization) |

All legacy-API scripts are **fully compatible** with the refactored code — `SpinQuantConfig`
and `SpinQuantPreprocessor` are unchanged. The unified API (`apply_rotation`) is recommended
for new scripts.

## Appendix B: Module Structure

```
auto_round/algorithms/transforms/
├── __init__.py              # Unified apply_rotation() entry point
├── base.py                  # BaseRotation (ABC + registry), BaseRotationConfig
├── rotation/                # Hadamard rotation (original auto-round)
│   └── ...
└── spinquant/               # SpinQuant / QuaRot implementation
    ├── __init__.py          # Public API exports
    ├── algorithm.py         # SpinQuantRotation (BaseRotation subclass, registered as "spinquant")
    ├── preprocessor.py      # SpinQuantConfig, SpinQuantPreprocessor (8-step pipeline)
    ├── trainer.py           # RotationTrainer, RotationTrainerConfig
    ├── training_core.py     # Shared training primitives (loss, optimizer, etc.)
    ├── training.py          # Training hooks & state management
    ├── rotation_utils.py    # Hadamard matrices, matmul_hadU, weight rotation
    ├── known_hadamard.py    # Pre-computed Hadamard matrices for non-pow2 sizes (ported from Quark)
    ├── cayley_optimizer.py  # SGDG (Stiefel manifold), AdamAndSGDG
    ├── monkeypatch.py       # QKRotationWrapper (R3 after RoPE)
    └── inplace/
        ├── __init__.py
        └── apply.py         # register_spinquant_hooks, apply_spinquant_in_place
```

## Appendix C: Migration from v1 API

```python
# ──── v1 (still works, not deprecated) ────
from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig, SpinQuantPreprocessor
)
config = SpinQuantConfig(r1=True, r2=True, ...)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()

# ──── v2 (recommended) ────
from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

config = SpinQuantConfig(r1=True, r2=True, ...)
model = apply_rotation(model, config)

# ──── v2 shorthand ────
model = apply_rotation(model, "quarot")  # QuaRot defaults (R1+R2+R3+R4)

# ──── v2 dict-based ────
model = apply_rotation(model, {
    "algorithm": "spinquant",
    "r1": True, "r2": True, "r3": False, "r4": False,
    "trainable_rotation": False, "trainable_smooth": False,
})
```
