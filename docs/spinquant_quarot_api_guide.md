# Auto-Round SpinQuant / QuaRot API & Usage Guide

> **Rotation-based weight transforms for improved quantization accuracy.**
> This module implements [SpinQuant](https://arxiv.org/abs/2405.16406) and [QuaRot](https://arxiv.org/abs/2404.00456)
> rotation techniques as a preprocessing step before Auto-Round quantization.

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
11. [Troubleshooting](#11-troubleshooting)
12. [Examples](#12-examples)

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

| Feature | Status |
|---------|--------|
| QuaRot (fixed Hadamard, R1-R4) | ✅ Fully supported |
| SpinQuant (trainable rotation) | ⚠️ Scaffolding exists, training loop under development |
| Deterministic Hadamard | ✅ Default |
| Random Hadamard (H×D) | ✅ Via `random_r1=True`/`random_r2=True` |
| Online R1 (hook-based) | ✅ Default |
| Offline R1 (weight fusion) | ✅ Via `online_r1_rotation=False` |
| R2 (per-head V/O rotation) | ✅ Offline fused |
| R3 (Q/K after RoPE) | ✅ Online monkeypatch |
| R4 (MLP activation) | ✅ Online hook |
| Block rotation (`rotation_size`) | ✅ For non-power-of-2 models |
| All HuggingFace architectures | ✅ Llama, Qwen, Mistral, Phi, Gemma, etc. |

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

### 3.1 QuaRot: Fixed Rotation + Quantization (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig, SpinQuantPreprocessor
)

# Load model
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 1: Apply rotation (QuaRot mode — no training needed)
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,  # R1+R2
    trainable_rotation=False,                # QuaRot (fixed Hadamard)
    trainable_smooth=False,                  # No SmoothQuant training
    online_r1_rotation=True,                 # Online R1 (recommended)
)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()  # No dataloader needed for QuaRot

# Step 2: Quantize with Auto-Round
autoround = AutoRound(
    model, tokenizer,
    scheme="W4A16",   # or "MXFP4", "NVFP4", "INT8", etc.
    iters=0,          # RTN (0) or GPTQ-style (200+)
)
autoround.quantize()
```

### 3.2 Full Rotation (R1+R2+R3+R4) with Block Rotation

```python
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    rotation_size=128,           # Block rotation for non-power-of-2 dimensions
    trainable_rotation=False,
    trainable_smooth=False,
    online_r1_rotation=True,
)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()

# Quantize
autoround = AutoRound(model, tokenizer, scheme="MXFP4", iters=0)
autoround.quantize()
```

### 3.3 Evaluation After Rotation + Quantization

```python
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8, device="cuda:0")
results = simple_evaluate(
    model=lm,
    tasks=["hellaswag", "piqa", "winogrande", "lambada_openai"],
    batch_size=8,
    limit=None,   # None for full eval, integer for quick test
)

for task, data in results["results"].items():
    acc = data.get("acc_norm,none") or data.get("acc,none")
    print(f"{task}: {acc:.4f}")
```

---

## 4. API Reference

### 4.1 `SpinQuantConfig`

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
| `rotation_size` | `Optional[int]` | `None` | Custom rotation dimension (must be power-of-2). `None` = use model's hidden_size/intermediate_size. Use for models where dimensions aren't power-of-2 (e.g., Qwen3-0.6B: hidden_size=1024 → ok, but if 768 → set `rotation_size=128`). |
| `random_r1` | `bool` | `False` | Use random Hadamard (H×D) for R1 instead of deterministic. Only relevant when `trainable_rotation=False`. |
| `random_r2` | `bool` | `False` | Use random Hadamard for R2. Same constraint as `random_r1`. |
| `trainable_rotation` | `bool` | `True` | Train rotation via Cayley SGD (SpinQuant). `False` = fixed Hadamard (QuaRot). |
| `trainable_smooth` | `bool` | `True` | Train SmoothQuant scaling factors via Adam. |
| `online_r1_rotation` | `bool` | `True` | Online R1: rotate weights per-module + hook. `False` = offline global fusion (requires untie embeddings + RMSNorm fusion). |
| `iters` | `int` | `200` | Training iterations (SpinQuant mode). |
| `lr` | `float` | `1e-4` | SGDG learning rate for rotation matrices. |
| `smooth_lr` | `float` | `1e-3` | Adam learning rate for smooth values. |
| `batch_size` | `int` | `1` | Training batch size. |
| `loss_type` | `str` | `"kl_top"` | Loss: `"kl_top"` (top-k KL divergence), `"kl_full"` (full KL), `"mse"`. |
| `kl_top_k` | `int` | `1000` | Top-k tokens for KL divergence loss. |
| `fuse_rmsnorm` | `bool` | `True` | Fuse RMSNorm gamma into linear weights. |
| `untie_embeddings` | `bool` | `True` | Untie input/output embeddings. |
| `dtype` | `torch.dtype` | `torch.float32` | Computation dtype for rotation. |
| `device` | `Optional[str]` | `None` | Auto-detects "cuda" or "cpu". |

### 4.2 `SpinQuantPreprocessor`

Main class that orchestrates the 8-step rotation pipeline.

```python
from auto_round.algorithms.transforms.spinquant import SpinQuantPreprocessor

preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess(dataloader=None)
```

**Constructor:**
- `model: nn.Module` — HuggingFace causal LM model
- `config: Optional[SpinQuantConfig]` — Config (defaults to `SpinQuantConfig()`)

**Methods:**

| Method | Description |
|--------|-------------|
| `preprocess(dataloader=None) → nn.Module` | Run full 8-step pipeline. Returns modified model. Dataloader required only when `trainable_rotation=True` or `trainable_smooth=True`. |

**8-Step Pipeline:**
1. Untie embeddings (if offline R1)
2. Fuse RMSNorm gamma into weights (if offline R1)
3. Add trainable smooth values (if `trainable_smooth=True`)
4. Initialize rotation matrices (Hadamard or identity)
5. Train rotations (if `trainable_rotation=True` — requires dataloader)
6. Apply R1 (online or offline), fuse R2/R4 offline
7. Register R3/R4 online hooks
8. Cleanup training artifacts

### 4.3 `RotationTrainer` (Alternative Interface)

HuggingFace-Trainer-style interface for SpinQuant training.

```python
from auto_round.algorithms.transforms.spinquant import (
    RotationTrainer, RotationTrainerConfig
)

trainer = RotationTrainer(model, config=RotationTrainerConfig(iters=200, lr=1e-4))
metrics = trainer.train(dataloader)
model = trainer.fuse()
```

**RotationTrainerConfig fields:** Same as `SpinQuantConfig`, plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_interval` | `int` | `50` | Print loss every N steps |
| `eval_interval` | `int` | `0` | Run evaluation every N steps (0=never) |
| `save_interval` | `int` | `0` | Save checkpoint every N steps (0=never) |
| `checkpoint_dir` | `Optional[str]` | `None` | Directory for checkpoints |

**RotationTrainer methods:**

| Method | Description |
|--------|-------------|
| `train(dataloader) → dict` | Run training, return metrics dict |
| `fuse() → nn.Module` | Fuse rotations into weights, return model |
| `evaluate(dataloader) → dict` | Run evaluation, return metrics |
| `save_checkpoint(path) → str` | Save training state |
| `load_checkpoint(path)` | Load training state |

**Callbacks:**
```python
from auto_round.algorithms.transforms.spinquant import (
    RotationTrainerCallback, LossLogger, OrthogonalityMonitor
)

trainer = RotationTrainer(
    model,
    config=config,
    callbacks=[LossLogger(), OrthogonalityMonitor()],
)
```

### 4.4 In-Place Application (One-Shot)

```python
from auto_round.algorithms.transforms.spinquant import apply_spinquant_in_place

model = apply_spinquant_in_place(model, config, dataloader=None)
```

### 4.5 Hook Management

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

### 4.6 Rotation Utilities

```python
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    deterministic_hadamard_matrix,  # H / √N, always the same
    random_hadamard_matrix,         # H × diag(±1) / √N
    matmul_hadU,                    # O(N log N) butterfly Hadamard multiply
    is_pow2,                        # Check power-of-2
    get_hadamard_K,                 # Get (hadamard_matrix, K) for size N
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
SpinQuantConfig(
    r1=True, r2=False, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
```

**QuaRot R1+R2 (most common, best cost/benefit):**
```python
SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
```

**QuaRot Full (R1+R2+R3+R4):**
```python
SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
```

**QuaRot Full with block rotation (for non-power-of-2 models):**
```python
SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    rotation_size=128,  # Must divide hidden_size and intermediate_size
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
```

**Random Hadamard (better outlier distribution):**
```python
SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    random_r1=True, random_r2=True,   # Random sign diagonal
    online_r1_rotation=True,
)
```

**SpinQuant (trainable, best accuracy — training under development):**
```python
SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=True, trainable_smooth=True,
    online_r1_rotation=True,
    iters=200, lr=1e-4, smooth_lr=1e-3,
    loss_type="kl_top", kl_top_k=1000,
)
```

### 5.2 Rotation Level Selection Guide

| Scenario | Recommended Levels | Why |
|----------|-------------------|-----|
| Quick test / baseline | R1 only | Minimal overhead, noticeable improvement |
| Production (weight-only quant) | R1+R2 | Best cost/benefit for W4A16, W8A16 |
| Weight+Activation quant (MXFP4, NVFP4) | R1+R2+R3+R4 | Activation quantization benefits from R3/R4 |
| Accuracy-critical deployment | R1+R2+R3+R4 + SpinQuant training | Best possible accuracy |

### 5.3 When to Use `rotation_size`

Use `rotation_size` when:
- Model `hidden_size` is not a power of 2 (e.g., 768, 1536, 3072)
- Model `intermediate_size` is not a power of 2 (e.g., 8960)
- The preprocessor will warn you if dimensions don't match

The preprocessor automatically validates and disables rotations that can't fit:
```
[SpinQuant] R4 disabled: intermediate_size=8960 not divisible by rotation_size=128
```

Common `rotation_size` values: `64`, `128`, `256`, `512`, `1024`.

---

## 6. Usage Patterns

### 6.1 Pattern A: Rotation + RTN Quantization (Fastest)

```python
# No training, no calibration data for rotation
config = SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
SpinQuantPreprocessor(model, config).preprocess()

# RTN quantization (iters=0)
autoround = AutoRound(model, tokenizer, scheme="W4A16", iters=0)
autoround.quantize()
```

### 6.2 Pattern B: Rotation + GPTQ-Style Quantization (Better Accuracy)

```python
# Same rotation setup
config = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)
SpinQuantPreprocessor(model, config).preprocess()

# GPTQ-style quantization (iters=200)
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
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()

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

### 8.3 Floating Point Formats

| Scheme | Config | Rotation | Notes |
|--------|--------|----------|-------|
| `FP8_STATIC` | FP8 static, per-tensor | R1 | Simple, per-tensor |
| `FP8_BLOCK` | FP8, block 128×128 | R1+R2 | Block format |
| `BF16` | No quantization | For rotation-only testing | |

### 8.4 Usage Example

```python
# W4A16 + R1+R2 (weight-only, most common)
config = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False, ...)
SpinQuantPreprocessor(model, config).preprocess()
AutoRound(model, tokenizer, scheme="W4A16", iters=0).quantize()

# MXFP4 + R1+R2+R3+R4 (weight+activation, best with full rotation)
config = SpinQuantConfig(r1=True, r2=True, r3=True, r4=True, ...)
SpinQuantPreprocessor(model, config).preprocess()
AutoRound(model, tokenizer, scheme="MXFP4_RCEIL", iters=0).quantize()
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

### 9.3 Why R3 Must Be Online

R3 applies after RoPE (Rotary Position Embedding). RoPE is position-dependent — it changes
with each token position. Therefore R3 cannot be fused into weights and must run at every
forward pass.

### 9.4 Why R4 Is Hybrid

R4 consists of two parts:
1. **Activation rotation** (online): `x_rotated = matmul_hadU(activation)` — runtime dependent
2. **Weight inverse** (offline): `down_proj_new = down_proj @ R4⁻¹` — fused once

---

## 10. Advanced Topics

### 10.1 Block Rotation (`rotation_size`)

When model dimensions aren't powers of 2, block rotation divides the dimension into
blocks of `rotation_size` and applies Hadamard independently per block:

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

### 10.2 Normalization Convention

Auto-round uses **normalized** Hadamard matrices: `H / √N` where `H` is the Sylvester matrix
and `N` is the dimension. This means `H @ H.T = I` (orthogonal).

This differs from Quark which uses unnormalized ±1 matrices and applies `/ √N` during
multiplication. Auto-round's convention avoids a class of double-normalization bugs.

### 10.3 Butterfly Algorithm (matmul_hadU)

The online rotations (R3, R4) use the Fast Walsh-Hadamard Transform for O(N·log N)
complexity instead of O(N²) dense matrix multiplication:

```python
# O(N log N) butterfly
matmul_hadU(x)

# vs O(N²) dense
x @ hadamard_matrix
```

This is why R3 and R4 always use **deterministic** Hadamard — the butterfly algorithm
requires the recursive `[[H,H],[H,-H]]` structure of Sylvester Hadamard. A random
diagonal D would break this structure.

### 10.4 Supported Model Architectures

The rotation implementation is architecture-generic thanks to:
- `get_model_arch_info()`: Auto-detects hidden_size, head_dim, etc. from model config
- R3 monkeypatch: Works by replacing `apply_rotary_pos_emb` in attention's `forward.__globals__`
- Layer iteration: Uses HuggingFace naming conventions (`model.layers`, `q_proj`, etc.)

Tested architectures: Llama, Qwen2, Qwen3, Mistral, Phi, Gemma.

### 10.5 Dimension Validation

The preprocessor automatically validates dimensions and disables incompatible rotations:

```python
# If hidden_size=768 (not power of 2) and no rotation_size set:
# → R1 disabled with warning
# → R2 still works if head_dim=64 (power of 2)

# If rotation_size=128 but intermediate_size=8960 (not divisible by 128):
# → R4 disabled with warning
```

---

## 11. Troubleshooting

### Q: Accuracy drops significantly after rotation + quantization

**A:** Check these in order:
1. **Rotation-only test**: Run rotation without quantization and compare with FP16 baseline.
   Accuracy should be identical (within floating-point tolerance). If not, there's a rotation bug.
2. **Online vs Offline R1**: Switch to `online_r1_rotation=True` — offline R1 can degrade
   accuracy in quantized models.
3. **Rotation size**: If dimensions aren't power-of-2, ensure `rotation_size` is set and
   divides both `hidden_size` and `intermediate_size`.

### Q: "rotation_size must be a power of 2" error

**A:** `rotation_size` must be 16, 32, 64, 128, 256, 512, 1024, etc.

### Q: R3 or R4 disabled automatically

**A:** The preprocessor validates dimensions. If `head_dim` isn't power-of-2, R3 is disabled.
If `intermediate_size` isn't divisible by `rotation_size`, R4 is disabled. Check the log
messages for details.

### Q: Can I use rotation without quantization?

**A:** Yes. Just skip the `AutoRound` step. The rotated model is mathematically equivalent
to the original (in FP precision). This is useful for verifying rotation correctness.

### Q: Does rotation slow down inference?

**A:** Online rotations (R3, R4, online R1) add small overhead:
- R1 online: ~1% overhead (one matmul per hook per layer)
- R3: ~2% overhead (butterfly matmul on Q,K per layer)
- R4: ~1% overhead (butterfly matmul on MLP activation per layer)
- R2: Zero overhead (offline fused)

---

## 12. Examples

### 12.1 Complete: Rotation + W4A16 + Evaluation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig, SpinQuantPreprocessor

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
SpinQuantPreprocessor(model, config).preprocess()

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

### 12.2 Batch: Multiple Rotations × Multiple Schemes

See `examples/test_rotation_scheme_matrix.py` for a comprehensive script that tests
all combinations of rotation levels and quantization schemes with lm_eval evaluation.

### 12.3 Comparison: Auto-Round vs Quark vs llm-compressor

See `examples/test_three_way_comparison.py` for side-by-side framework comparison
using identical rotation configurations and quantization settings.

---

## Appendix: Module Structure

```
auto_round/algorithms/transforms/spinquant/
├── __init__.py              # Public API exports
├── preprocessor.py          # SpinQuantConfig, SpinQuantPreprocessor
├── trainer.py               # RotationTrainer, RotationTrainerConfig
├── training.py              # Training hooks & state management
├── rotation_utils.py        # Hadamard matrices, matmul_hadU, weight rotation
├── cayley_optimizer.py      # SGDG (Stiefel manifold), AdamAndSGDG
├── monkeypatch.py           # QKRotationWrapper (R3 after RoPE)
└── inplace/
    ├── __init__.py
    └── apply.py             # register_spinquant_hooks, apply_spinquant_in_place
```
