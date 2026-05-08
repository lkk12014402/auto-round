# llm-compressor Rotation Implementation Analysis

## Overview

llm-compressor (by Neural Magic / vLLM) implements QuaRot/SpinQuant rotation via a
**Modifier + Transform** architecture split across two packages:

| Package | Role |
|---------|------|
| **compressed-tensors** | Transform framework: matrix factories, hook registration, serialization |
| **llm-compressor** | SpinQuant/QuIP modifiers: layer mappings, norm fusion, lifecycle |

## Architecture Comparison

| Feature | auto-round | Quark | llm-compressor |
|---------|-----------|-------|----------------|
| Graph tracing | Module tree + `type()` | `torch.fx` export | Module tree + regex |
| Rotation config | `SpinQuantConfig` dataclass | `RotationConfig` | `SpinQuantModifier` (pydantic) |
| Quantization API | `AutoRound(scheme=...)` | `ModelQuantizer(config)` | `oneshot(recipe=[...])` |
| Pipeline pattern | Imperative Python | Imperative Python | **Recipe/Modifier** declarative |
| Transform serialization | Not yet (hooks lost) | `rotation_config` in save dict | `transform_config` in `config.json` |
| Online rotation | `forward_pre_hook` | `InputRotationWrapper` | `forward_pre_hook` / `forward_hook` |
| Offline rotation | Weight fusion + norm fusion | Weight fusion + norm fusion | Weight fusion + norm fusion |
| R3 implementation | Monkeypatch `apply_rotary_pos_emb` | Monkeypatch attention | `Q_ATTN` / `K_CACHE` hooks |
| Trainable rotation | SpinQuant Cayley SGD (partial) | SpinQuant Cayley SGD | `requires_grad` + parametrize (planned) |

## R1–R4 Implementation Details

### R1 — Full Network Rotation (OFFLINE)

**llm-compressor applies R1 as fully offline rotation** (unlike auto-round's online approach):

```
embed_tokens → WEIGHT_OUTPUT: R @ W    (rotate embedding output)
q/k/v_proj   → WEIGHT_INPUT:  W @ R⁻¹  (rotate linear input weights)
gate/up_proj → WEIGHT_INPUT:  W @ R⁻¹
o_proj       → WEIGHT_OUTPUT: R @ W
down_proj    → WEIGHT_OUTPUT: R @ W
lm_head      → WEIGHT_INPUT:  W @ R⁻¹
```

Before R1, norms are fused: `linear.weight *= norm.weight; norm.weight = 1`

**Key difference from auto-round:** llm-compressor's R1 is always offline (all weights
fused, norms fused, embeddings centered). Auto-round uses online R1 (hooks on target
modules only) to preserve weight distributions for better quantization quality.

### R2 — Head-wise Attention Rotation (OFFLINE)

```
v_proj → WEIGHT_OUTPUT (head_dim block diagonal)
o_proj → WEIGHT_INPUT inverse (head_dim block diagonal)
```

Uses `head_dim` parameter in `TransformScheme` for block-diagonal multiplication.
Same approach across all three frameworks.

### R3 — Q/K Cache Rotation (ONLINE)

```
q_proj → Q_ATTN location (hook on attention query states)
k_proj → K_CACHE location (hook on key cache values)
```

llm-compressor uses dedicated `TransformLocation.Q_ATTN` and `K_CACHE` enum values
with specialized hooks, rather than monkeypatching `apply_rotary_pos_emb`.

**Status:** Partially implemented, not in default rotations list.

### R4 — MLP Activation Rotation (ONLINE)

```
down_proj → INPUT location (forward_pre_hook on activation)
down_proj → WEIGHT_INPUT inverse (offline weight fusion)
```

Split into online (activation hook) + offline (weight inverse fusion).

## Transform Framework (compressed-tensors)

### TransformLocation Enum

| Location | Type | Applied To |
|----------|------|-----------|
| `INPUT` | Online | Activations (forward_pre_hook) |
| `OUTPUT` | Online | Activations (forward_hook) |
| `WEIGHT_INPUT` | Offline | Weight matrix (right multiply) |
| `WEIGHT_OUTPUT` | Offline | Weight matrix (left multiply) |
| `Q_ATTN` | Online | Query states in attention |
| `K_CACHE` | Online | Key values in KV cache |

### Factory Registry

```
TransformFactory (abstract, registry pattern)
├── HadamardFactory ("hadamard")      — deterministic Sylvester construction
├── RandomHadamardFactory ("random-hadamard") — random Hadamard from safetensors library
└── RandomMatrixFactory ("random-matrix")     — dense random with explicit inverse
```

### Serialization

Transform config is saved as `transform_config` in the model's `config.json`:

```json
{
  "transform_config": {
    "config_groups": {
      "R1": {
        "type": "hadamard",
        "apply": [
          {"targets": ["re:.*q_proj$"], "location": "weight_input", "inverse": true},
          {"targets": ["re:.*o_proj$"], "location": "weight_output"}
        ]
      }
    }
  }
}
```

Online transforms (hooks) are re-registered on model load via `apply_transform_config()`.
Shared transform weights are deduplicated via `_tied_weights_keys`.

## Recipe/Modifier Pattern

llm-compressor uses a declarative **recipe** pattern:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        transform_type="hadamard",
        transform_block_size=128,
    ),
    QuantizationModifier(targets="Linear", scheme="MXFP4", ignore=["lm_head"]),
]

oneshot(model=model, recipe=recipe, pipeline="datafree")
```

Modifier lifecycle:
1. `on_initialize()` — infer layer mappings, create `TransformConfig`
2. `on_start()` — center embeddings, fuse norms, apply transforms
3. `on_end()` / `on_finalize()` — cleanup

## Key Differences from auto-round

### 1. R1 is Offline (not Online)

llm-compressor fuses R1 into ALL weights + norms. This means:
- Zero inference overhead for R1
- But changes ALL weight distributions → may impact quantization quality
- auto-round's online R1 only modifies target modules, preserving distributions

### 2. Proper Serialization Built-in

compressed-tensors has first-class transform serialization:
- `TransformConfig` saved to `config.json`
- Online hooks re-registered on load
- Shared weights deduplicated
- auto-round currently loses hooks on save (documented in `rotation_save_load_solution.md`)

### 3. Block-Diagonal via `head_dim` Parameter

R2 uses `TransformScheme.head_dim` for automatic block-diagonal multiplication.
The framework handles unflatten/flatten internally via `_multihead_matmul()`.

### 4. Architecture Mapping

llm-compressor uses regex patterns (`"re:.*q_proj$"`) for layer targeting.
Default mappings exist for `LlamaForCausalLM`; other architectures fall back to
generic defaults (which work for Qwen3 as the module naming is compatible).

## Quantization Scheme Support

llm-compressor supports rotation with any quantization scheme:
- `MXFP4`, `MXFP4A16`, `MXFP8`
- `W4A16`, `W8A8` (INT)
- `FP8` variants

Rotation and quantization are independent modifiers — any combination works.

## Current Limitations

1. `randomize=True` — NotImplementedError (no per-layer unique matrices)
2. `learnable=True` — NotImplementedError (no trainable rotation)
3. R3 — partial, not in default rotation list
4. Architecture mappings — only Llama has explicit mappings (others use generic defaults)
