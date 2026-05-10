# Rotation Guide — QuaRot & SpinQuant in Auto-Round

> Rotation-based weight transforms for improved quantization accuracy.
> Implements [QuaRot](https://arxiv.org/abs/2404.00456) and [SpinQuant](https://arxiv.org/abs/2405.16406)
> as a preprocessing step before quantization.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Rotation Levels (R1–R4)](#3-rotation-levels-r1r4)
4. [Online vs Offline Mode](#4-online-vs-offline-mode)
5. [Dimension Requirements & Non-Power-of-2 Support](#5-dimension-requirements--non-power-of-2-support)
6. [Configuration Reference](#6-configuration-reference)
7. [Usage Patterns](#7-usage-patterns)
8. [Rotation Matrix Types](#8-rotation-matrix-types)
9. [Supported Model Architectures](#9-supported-model-architectures)
10. [Troubleshooting](#10-troubleshooting)
11. [Three-Framework Comparison](#11-three-framework-comparison)

---

## 1. Overview

### What Problem Does Rotation Solve?

Quantization accuracy degrades when weight/activation distributions have outlier channels —
a few dimensions with magnitudes 10–100× larger than the rest. Rotation applies an orthogonal
transform (Hadamard matrix) to redistribute these outliers uniformly across all channels, making
the distribution more quantization-friendly.

```
Before rotation:  [ 0.1, 0.2, 50.0, 0.3 ]   ← outlier in channel 3
After rotation:   [ 12.7, 12.5, 12.8, 12.6 ] ← uniform distribution
```

Since orthogonal transforms preserve mathematical equivalence (Q @ Q^T = I), the model's
FP16 output is unchanged — only quantization behavior improves.

### QuaRot vs SpinQuant

| Feature | QuaRot | SpinQuant |
|---------|--------|-----------|
| Rotation matrix | Fixed Hadamard | Learnable (trained) |
| Training required | No | Yes (10–50 steps) |
| Typical accuracy | Good | Slightly better |
| Speed | Fast (seconds) | Slower (minutes) |

In auto-round, both share the same codebase. The difference is a config flag:
- `trainable_rotation=False, trainable_smooth=False` → **QuaRot**
- `trainable_rotation=True, trainable_smooth=True` → **SpinQuant**

---

## 2. Quick Start

### Simplest — String Shorthand

```python
from auto_round import AutoRound

# "quarot" applies R1+R2+R3+R4 with fixed Hadamard, then quantizes
autoround = AutoRound(model, tokenizer, rotation_config="quarot", scheme="W4A16", iters=0)
autoround.quantize()
```

### Full Control — SpinQuantConfig

```python
from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

cfg = SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False,    # QuaRot (fixed Hadamard)
    trainable_smooth=False,
    online_r1_rotation=True,     # Online R1 (recommended)
)

autoround = AutoRound(model, tokenizer, rotation_config=cfg, scheme="W4A16", iters=200)
autoround.quantize()
```

### Standalone Rotation (No Quantization)

```python
from auto_round.algorithms.transforms import apply_rotation

model = apply_rotation(model, "quarot")
# Model is now rotated — evaluate directly or quantize separately
```

### Supported `rotation_config` Values

| Value | Description |
|-------|-------------|
| `"quarot"` | QuaRot defaults: R1+R2+R3+R4, fixed Hadamard |
| `"spinquant"` | SpinQuant defaults: R1+R2+R3+R4, with learnable rotations |
| `"hadamard"` | Simple Hadamard rotation (original auto-round) |
| `SpinQuantConfig(...)` | Full control over all parameters |
| `{"algorithm": "spinquant", "r1": True, ...}` | Dict-based config |

---

## 3. Rotation Levels (R1–R4)

Rotation is applied at four levels, each targeting a different part of the transformer block.
You can enable any combination of R1–R4.

### R1 — Residual Stream Rotation

**What:** Rotates the hidden states flowing through the residual stream.

**Affects:** All linear layers connected to the residual stream — `q_proj`, `k_proj`, `v_proj`,
`gate_proj`, `up_proj` (input side) and `o_proj`, `down_proj` (output side), plus `embed_tokens`
and `lm_head`.

**Modes:** Online (forward hook, recommended) or Offline (weight fusion).

```python
SpinQuantConfig(r1=True, r2=False, r3=False, r4=False)
```

### R2 — Attention Head Rotation

**What:** Rotates each attention head's value and output projection independently.

**Affects:** `v_proj` and `o_proj`, operating on `head_dim` dimensions within each head.

**Mode:** Always offline (weight fusion). Head dimension is typically 64 or 128 (always pow2).

```python
SpinQuantConfig(r1=True, r2=True, r3=False, r4=False)
```

### R3 — Query/Key Rotation (After RoPE)

**What:** Rotates Q and K after positional encoding (RoPE) to improve attention quantization.

**Affects:** Attention computation — implemented by wrapping the attention module's Q/K matmul.

**Mode:** Always online (module monkeypatch on the attention layer). Applied per-head on `head_dim`.

```python
SpinQuantConfig(r1=True, r2=True, r3=True, r4=False)
```

### R4 — MLP Activation Rotation

**What:** Rotates the activation between `up_proj`/`gate_proj` and `down_proj` in the MLP block.

**Affects:** MLP intermediate activations — `up_proj`/`gate_proj` output columns and `down_proj`
input rows, plus a forward hook for online activation rotation.

**Mode:** Hybrid — weight fusion (offline) + forward hook (online) for activation rotation.
Operates on `intermediate_size` dimensions.

```python
SpinQuantConfig(r1=True, r2=True, r3=True, r4=True)
```

### Summary Table

| Level | Target | Dimension | Mode | Impact |
|-------|--------|-----------|------|--------|
| R1 | Residual stream | `hidden_size` | Online or Offline | All layers |
| R2 | Attention V/O | `head_dim` | Offline only | Per-head |
| R3 | Q/K after RoPE | `head_dim` | Online only | Attention |
| R4 | MLP activation | `intermediate_size` | Hybrid (offline + online) | MLP |

---

## 4. Online vs Offline Mode

### Offline Rotation

The rotation matrix is fused directly into the model weights: `W' = R @ W` or `W' = W @ R^T`.
This has **zero inference overhead** but permanently modifies the weights.

**Used by:** R1 (optional), R2 (always), R4 (weight part).

### Online Rotation

A forward hook applies rotation to activations at runtime: `x' = x @ H / √n`.
This adds a small inference cost but doesn't modify weights.

**Used by:** R1 (recommended), R3 (always), R4 (activation part).

### R1 Online vs Offline

| | Online R1 | Offline R1 |
|---|-----------|------------|
| **Mechanism** | Forward pre-hook with butterfly Hadamard | Weight fusion: `W' = W @ R^T` |
| **Inference cost** | O(n log n) per layer (butterfly) | Zero |
| **Quantization accuracy** | Better (rotation applied fresh each forward) | May degrade (fused into quantized weights) |
| **Supports random** | No (butterfly = deterministic only) | Yes (`random_r1=True`) |
| **Recommendation** | ✅ Default, use this | Only if inference speed is critical |

> **Why is online R1 recommended?** When using offline R1 with quantization, the rotation
> is baked into the weights *before* quantization, and the quantization error affects the
> rotated representation. Online R1 applies rotation at inference time on the full-precision
> activations, after the quantized weights are dequantized, avoiding this compounding error.

### R3 and R4 Online Components

R3 and R4 always have online components that cannot be made offline:
- **R3**: Q/K rotation must happen after RoPE encoding, which is computed at runtime
- **R4**: Activation rotation between gate/up and down projections must happen at runtime

---

## 5. Dimension Requirements & Non-Power-of-2 Support

### Power-of-2 Requirement

The standard Hadamard matrix construction (Sylvester method) requires dimensions to be powers of 2
(e.g., 64, 128, 256, 512, 1024, 2048, 4096). This affects which rotation levels work out of the box.

### Known Hadamard Matrices for Non-Power-of-2

Auto-round includes **11 pre-computed Hadamard matrices** for common non-power-of-2 sizes.
When a dimension matches one of these, the matrix is decomposed as `H_n = H_K ⊗ H_{n/K}` where
`K` is the largest power-of-2 factor and `n/K` has a known Hadamard matrix.

**Supported non-pow2 sizes:**

| n/K | Supported sizes (examples) |
|-----|---------------------------|
| 12 | 768, 1536, 3072 |
| 20 | 1280, 2560, 5120 |
| 28 | 1792, 3584, 7168 |
| 36 | 2304, 4608, 9216 |
| 40 | 2560, 5120, 10240 |
| 52 | 3328, 6656, 13312 |
| 60 | 3840, 7680, 15360 |
| 108 | 6912, 13824 |
| 140 | 8960, 17920 |
| 156 | 9984, 19968 |
| 172 | 11008, 22016 |

### Dimension per Rotation Level

| Level | Dimension Used | Typical Values | Pow2? |
|-------|---------------|----------------|-------|
| R1 | `hidden_size` | 768, 1024, 2048, 4096 | Mixed — often non-pow2 |
| R2 | `head_dim` | 64, 128 | ✅ Always pow2 |
| R3 | `head_dim` | 64, 128 | ✅ Always pow2 |
| R4 | `intermediate_size` | 3072, 8192, 11008, 14336 | Mixed — often non-pow2 |

### What Happens When a Dimension is Unsupported?

If a dimension is not a power of 2 **and** has no known Hadamard matrix, the affected rotation
level is **automatically skipped** with a warning:

```
WARNING: R4 intermediate_size=13337 is not supported (no known Hadamard matrix).
R4 will be disabled. Use rotation_size to set a compatible block size.
```

### `rotation_size` — Manual Override

Use `rotation_size` to force a specific (pow2) block rotation dimension. The matrix is applied
as blocks along the diagonal:

```python
# Force 128×128 block rotation for R1 (handles any hidden_size divisible by 128)
SpinQuantConfig(r1=True, rotation_size=128)
```

When `rotation_size < hidden_size`, the rotation matrix is block-diagonal: `R = I ⊗ H_{rotation_size}`.
This is less effective than full-dimensional rotation but works for any size divisible by `rotation_size`.

---

## 6. Configuration Reference

### SpinQuantConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r1` | bool | True | Enable R1 (residual stream rotation) |
| `r2` | bool | True | Enable R2 (attention head rotation) |
| `r3` | bool | True | Enable R3 (Q/K rotation after RoPE) |
| `r4` | bool | True | Enable R4 (MLP activation rotation) |
| `online_r1_rotation` | bool | True | Use online R1 (hook) vs offline (weight fusion) |
| `trainable_rotation` | bool | True | Learn rotation matrices (SpinQuant) vs fixed (QuaRot) |
| `trainable_smooth` | bool | True | Learn smooth parameters (SpinQuant only) |
| `random_r1` | bool | False | Use random Hadamard (H×D) for R1 offline |
| `random_r2` | bool | False | Use random Hadamard for R2 |
| `rotation_size` | int\|None | None | Override rotation dimension (None = auto) |

### Common Presets

```python
# QuaRot (recommended for most users)
SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=True,
)

# QuaRot R1+R2 only (faster, simpler)
SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
)

# QuaRot with random Hadamard for R1/R2 (offline R1 only)
SpinQuantConfig(
    r1=True, r2=True, r3=False, r4=False,
    trainable_rotation=False, trainable_smooth=False,
    online_r1_rotation=False,  # Must be offline for random to take effect
    random_r1=True, random_r2=True,
)

# SpinQuant (learnable rotations)
SpinQuantConfig(
    r1=True, r2=True, r3=True, r4=True,
    trainable_rotation=True, trainable_smooth=True,
)
```

> **Note:** `random_r1=True` has no effect when `online_r1_rotation=True` (online mode always
> uses deterministic Hadamard via butterfly algorithm). A warning is logged if both are set.

---

## 7. Usage Patterns

### Pattern A: Rotation + RTN Quantization (Fastest)

```python
autoround = AutoRound(model, tokenizer, rotation_config="quarot", scheme="W4A16", iters=0)
autoround.quantize()
```

### Pattern B: Rotation + Auto-Round Tuning (Better Accuracy)

```python
autoround = AutoRound(
    model, tokenizer,
    rotation_config="quarot",
    scheme="W4A16",
    iters=200,
    nsamples=128,
    seqlen=2048,
)
autoround.quantize()
```

### Pattern C: Rotation-Only (No Quantization, for Debugging)

```python
from auto_round.algorithms.transforms import apply_rotation

model = apply_rotation(model, "quarot")
# Evaluate — should match FP16 baseline (rotation is lossless in FP)
```

### Pattern D: With lm_eval for Accuracy Benchmarking

```python
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM

lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8, device="cuda:0")
results = simple_evaluate(model=lm, tasks=["hellaswag", "piqa", "winogrande"], batch_size=8)
```

### Pattern E: Legacy API (Direct Preprocessor)

```python
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig, SpinQuantPreprocessor

config = SpinQuantConfig(r1=True, r2=True, ...)
preprocessor = SpinQuantPreprocessor(model, config)
model = preprocessor.preprocess()
```

### Internal Pipeline Flow

When you pass `rotation_config` to `AutoRound`, here's what happens internally:

```
AutoRound(rotation_config="quarot")
  → normalize_rotation_config("quarot") → SpinQuantConfig(...)
  → BaseCompressor.__init__(config=[QuantConfig, SpinQuantConfig])
  → BaseCompressor.quantize()
    → Phase 4.5: _apply_rotations()
      → apply_rotation(model, SpinQuantConfig)
        → SpinQuantPreprocessor(model, config).preprocess()
    → Phase 5+: layer-wise quantization (on rotated model)
```

---

## 8. Rotation Matrix Types

### Matrix Types per Rotation Level

| Rotation | Mode | Matrix | Normalization | Supports Random |
|----------|------|--------|--------------|----------------|
| R1 | Online | `get_hadamard_K()` → butterfly | Output ÷√n | ❌ Deterministic only |
| R1 | Online block | `get_hadamard_K()` → kron | Manual ÷√n | ❌ |
| R1 | Offline | `deterministic_hadamard_matrix()` | Pre-normalized | ✅ `random_r1=True` |
| R2 | Offline | `deterministic/random_hadamard_matrix()` | Pre-normalized | ✅ `random_r2=True` |
| R3 | Online | `deterministic_hadamard_matrix()` as buffer | Via `matmul_hadU` | ❌ |
| R4 | Hybrid | `get_hadamard_K()` → `matmul_hadU` | Output ÷√n | ❌ |

### Normalization Convention

| Function | Returns | Convention |
|----------|---------|-----------|
| `get_hadamard_K(n)` | Unnormalized (±1 entries) | For butterfly algorithm input |
| `matmul_hadU(X, K, H_K)` | `X @ H / √n` | Normalized output |
| `deterministic_hadamard_matrix(n)` | `H / √n` | Orthogonal (H @ H^T = I) |
| `random_hadamard_matrix(n)` | `H × diag(±1) / √n` | Orthogonal with random signs |

---

## 9. Supported Model Architectures

Currently supported model families (Llama-style architectures):
- **Qwen** (Qwen2, Qwen2.5, Qwen3)
- **Llama** (Llama-2, Llama-3, Llama-3.1, Llama-3.2)
- **Mistral** / **Mixtral**
- **Gemma** (Gemma-2)
- **DeepSeek** (DeepSeek-V2)
- **Phi** (Phi-2, Phi-3)

The implementation relies on identifying model structure patterns (attention modules with
`q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP modules with `gate_proj`, `up_proj`, `down_proj`).
Most Llama-family architectures are automatically supported.

---

## 10. Troubleshooting

### Accuracy drops significantly after rotation + quantization

- Try online R1 (`online_r1_rotation=True`) — this is the default and recommended
- Add R3 and R4 if using weight+activation quantization (MXFP4, NVFP4)
- Increase auto-round tuning iterations (`iters=200`)

### "Cannot find suitable Hadamard decomposition" error

The dimension is not a power of 2 and has no known Hadamard matrix. Solutions:
- Set `rotation_size` to a power of 2 that divides the dimension (e.g., 128)
- Disable the problematic rotation level (e.g., `r4=False`)

### R3 or R4 disabled automatically

The model's `head_dim` or `intermediate_size` doesn't support Hadamard decomposition.
Check the log for dimension values and refer to the supported sizes in §5.

### `random_r1=True` has no effect

Online R1 always uses deterministic Hadamard (butterfly algorithm). Set `online_r1_rotation=False`
to use offline R1 if you need random Hadamard.

### How do I verify rotation is lossless?

Apply rotation without quantization and compare logits:
```python
model_rotated = apply_rotation(copy.deepcopy(model), "quarot")
# Compare logits — should be nearly identical (FP16 accumulation error < 1%)
```

---

## 11. Three-Framework Comparison

### Auto-Round vs Quark vs llm-compressor

| Feature | Auto-Round | Quark | llm-compressor |
|---------|-----------|-------|----------------|
| **R1** | ✅ Online (hook) + Offline (fusion) | ✅ Online (wrapper) + Offline (fusion) | ✅ Offline only |
| **R2** | ✅ Offline | ✅ Offline | ✅ Offline |
| **R3** | ✅ Online (monkeypatch) | ✅ Online (monkeypatch) | ✅ Online (monkeypatch) |
| **R4** | ✅ Hybrid (fusion + hook) | ✅ Hybrid (fusion + wrapper) | ✅ Hybrid |
| **Learnable rotation** | ✅ SpinQuant training | ✅ SpinQuant training | ❌ NotImplementedError |
| **Random Hadamard** | ✅ Offline R1/R2 | ✅ Offline R1/R2 | ✅ (called `random-hadamard`) |
| **Non-pow2 support** | ✅ Known Hadamard (11 sizes) | ✅ Known Hadamard | ✅ QR decomposition fallback |
| **Online R1 mechanism** | Forward pre-hook | Module replacement | N/A (offline only) |
| **RMSNorm fusion** | ✅ Absorbed into linear | ✅ Absorbed into linear | ✅ Absorbed into linear |

### Key Differences

- **Auto-Round** uses `forward_pre_hook` for online R1 (lightweight, no module replacement)
- **Quark** uses `InputRotationWrapperHadamard` module replacement for online R1
- **llm-compressor** has no online R1 — always fuses into weights
- **llm-compressor** supports `random-matrix` (QR decomposition) as a fallback for any size,
  but at O(n²) cost. Auto-round and Quark use butterfly O(n log n) for online rotations.

---

## Module Structure

```
auto_round/algorithms/transforms/
├── __init__.py          # Unified entry: apply_rotation(), normalize_rotation_config()
├── base.py              # BaseRotation registry, BaseRotationConfig
├── rotation/            # Simple Hadamard rotation (original auto-round)
│   ├── algorithm.py     # HadamardRotation implementation
│   └── config.py        # RotationConfig (Pydantic)
└── spinquant/           # QuaRot / SpinQuant multi-level rotation
    ├── __init__.py      # Public exports: SpinQuantConfig, SpinQuantPreprocessor, ...
    ├── preprocessor.py  # Core: SpinQuantConfig + SpinQuantPreprocessor pipeline
    ├── rotation_utils.py # Hadamard construction, model inspection, RMSNorm fusion
    ├── known_hadamard.py # 11 pre-computed Hadamard matrices for non-pow2
    ├── monkeypatch.py   # R3 Q/K rotation wrapper
    ├── training.py      # SpinQuantTrainingHook interface
    ├── training_core.py # Training loop, loss functions, orthogonality checks
    ├── trainer.py       # RotationTrainer (HuggingFace-style training)
    ├── cayley_optimizer.py # SGDG + AdamAndSGDG for Stiefel manifold
    └── inplace/         # Inplace rotation application
        └── apply.py     # R3/R4 hook/monkeypatch registration
```
