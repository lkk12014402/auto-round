# MXFP4 Activation QDQ — Architecture & Implementation Notes

## Overview

The MXFP4 activation Quantize-Dequantize (QDQ) module performs runtime activation
quantization for MXFP4 inference in vLLM. It quantizes activations to FP4 E2M1
format with block-wise e8m0 scaling (block_size=32), then immediately dequantizes
back to bf16/fp16 for computation with dequantized weights.

## Backend Architecture

Three QDQ backends are available, automatically selected by priority:

```
┌─────────────────────────────────────────────────────────┐
│                   qdq_mxfp4(x)                          │
├─────────────────────────────────────────────────────────┤
│  Priority 1: local_ext (CUDA JIT kernel)                │
│  Priority 2: triton (vendored Triton kernel)            │
│  Priority 3: fallback (pure PyTorch)                    │
└─────────────────────────────────────────────────────────┘
```

### Backend Details

| Backend | Implementation | Speed (256×4096) | Shape Constraint | Source |
|---------|---------------|-----------------|-----------------|--------|
| `local_ext` | CUDA C++ kernel via `torch.utils.cpp_extension.load()` | ~0.012ms | `numel % 64 == 0` | Vendored from Quark (MIT) |
| `triton` | Triton JIT kernel | ~0.085ms | `numel % 32 == 0` | Vendored from Quark (MIT) |
| `fallback` | Pure PyTorch (no compilation needed) | ~0.186ms | `last_dim % 32 == 0` | Original implementation |

### GEMM Backend

The QDQ is used in the `preunpack_fp8` runtime path:

```
Input (bf16) → qdq_mxfp4() → quantized activation (bf16)
                                        ↓
Weight (packed MXFP4) → dequant_to_fp8 → fp8 + bf16 scale
                                        ↓
                            cuBLAS FP8 GEMM (torch._scaled_mm)
```

The GEMM backend is **cuBLAS** (via PyTorch's `torch._scaled_mm` for FP8, or
`torch.matmul` for the dense dequant path). This is NOT a custom kernel —
it's NVIDIA's optimized library routine.

Note: The `quark_like_dense` backend dequantizes weight back to bf16 before GEMM,
so a 32B MXFP4 model will use ~64GB VRAM (not 16GB) since weights are expanded
to full precision at runtime.

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AUTO_ROUND_MXFP4_QDQ_BACKEND` | `local_ext`, `triton`, `fallback` | (auto) | Force a specific QDQ backend |
| `AUTO_ROUND_DISABLE_LOCAL_MXFP4_QDQ` | `0`, `1` | `0` | Disable CUDA kernel JIT build |

## CUDA Kernel Constraints

- Requires `numel % 64 == 0` (warp-level shuffle with block_size 64/128)
- Only supports `group_size = 32`
- Only supports bf16 and fp16 input
- First invocation triggers JIT compilation (~30s); subsequent uses hit cache

### Why `numel % 64 == 0`?

The CUDA kernel computes the MXFP4 block-max scale using **warp-level shuffle
reduction** (`__shfl_xor`):

```cuda
// Each thread holds 1 element; 32 threads cooperate to find block-max
for (int i = 1; i < 32; i *= 2) {
    block_max = hmax(block_max, habs(__shfl_xor_sync(0xffffffff, block_max, i)));
}
```

This requires the CUDA thread block size to be a multiple of 32 (warp size).
The kernel launch uses `blockDim = 64` or `128` (2 or 4 MXFP4 blocks per thread
block), so `numel` must be evenly divisible by the thread block size.

**This is Quark's original design** — our vendored kernel is unchanged from
`quark/torch/kernel/hw_emulation/csrc/mxfp4/fake.cu`.

### Graceful Fallback

When `numel % 64 != 0` (e.g., Qwen2.5-72B with `intermediate_size=29568`,
TP=4 → `down_proj` input dim = 7392, batch=1 → numel=7392, `7392 % 64 = 32`),
the CUDA kernel cannot be used. The dispatch automatically falls back to:

1. **Triton kernel** (no `numel % 64` constraint — only requires `numel % 32 == 0`)
2. **Pure PyTorch fallback** (works on any shape divisible by group_size)

This only affects small batch sizes during CUDA graph warmup. For batch ≥ 2,
`numel = 2 × 7392 = 14784` which is divisible by 64.

### Triton Kernel — No Such Constraint

The Triton kernel (also vendored from Quark, `quark/torch/kernel/mx/triton.py`)
uses `tl.program_id` with block_size=32, so it only needs `numel % 32 == 0`.
This makes it a natural fallback for edge cases the CUDA kernel cannot handle.

## Known Issue: CUDA vs Triton Tie-Breaking Difference

### Problem

The CUDA kernel and Triton kernel produce slightly different results (~0.6% of
elements differ) for the same input.

### Root Cause

**FP4 E2M1 midpoint tie-breaking policy:**

When an input value falls exactly at the midpoint between two representable FP4
values, the two kernels round differently:

| Kernel | Tie-Breaking Policy | Example: input=1.25 (midpoint of 1.0 and 1.5) |
|--------|-------------------|-----------------------------------------------|
| CUDA | Round toward zero | → 1.0 |
| Triton | Round away from zero | → 1.5 |

### Affected Values

FP4 E2M1 representable positive values: `0, 0.5, 1, 1.5, 2, 3, 4, 6`

Midpoints (before scaling): `0.25, 0.75, 1.25, 2.5, 3.5, 5.0`

After block scaling, any input that lands exactly on a scaled midpoint will
trigger the tie-breaking difference.

### Verification

```python
# Confirmed: Quark itself has the EXACT SAME difference
Quark CUDA vs Quark Triton:    6974 diffs (0.665%)
Our local_ext vs Quark CUDA:   0 diffs  ← identical
Our triton vs Quark Triton:    0 diffs  ← identical
```

### Impact

- **Negligible** — only affects ~0.6% of elements
- Both are valid MXFP4 implementations per the MX OCP spec
- The MX spec does NOT mandate a specific tie-breaking policy
- In practice, model accuracy (measured by lm_eval) is indistinguishable

### Resolution

**Not a bug — by design.** This is inherited from Quark's own implementation.
In production, `local_ext` (CUDA) is always preferred for performance. The Triton
kernel serves as a fallback when CUDA JIT compilation is not available.

If exact bit-for-bit matching is required between backends, the Triton kernel's
rounding logic would need modification. This is not planned as it provides no
practical benefit.

## File Structure

```
auto_round_extension/vllm_ext/
├── sitecustomize.py           # Entry point (VLLM_ENABLE_AR_EXT=1)
├── __init__.py                # apply() function
├── auto_round_ext.py          # AutoRoundExtensionConfig (extends vLLM built-in)
├── spinquant_mxfp4.py         # SpinQuantMXFP4Config — rotation plugin
├── _weight_loading_patch.py   # Patches AutoWeightsLoader for spinquant_R* keys
├── mxfp4_qdq_utils.py         # Main QDQ dispatch + GEMM helper functions
├── _mxfp4_qdq_ext.py          # Lazy JIT loader for CUDA kernel
├── _triton_mxfp4_qdq.py       # Vendored Triton MXFP4 QDQ (from Quark, MIT)
├── envs_ext.py                 # Environment variable registration
├── linear_impl_mxfp4.py       # MXFP4 linear layer (calls qdq_mxfp4)
├── linear_impl_mxfp8.py       # MXFP8 linear layer
├── csrc/mxfp4_qdq/
│   ├── common.h               # CUDA kernel constants
│   ├── fake.h                 # Forward declarations
│   ├── fake.cu                # CUDA kernel implementation
│   └── binding.cpp            # pybind11 bindings
└── tests/
    └── test_mxfp4_qdq.py      # Unit tests for all backends
```

## Logging

When a QDQ backend is selected, it is logged once per process:

```
INFO: [AutoRound] MXFP4 activation QDQ backend: local_ext
```

To see which GEMM backend is used, check the vLLM model loading logs for the
weight processing path (`preunpack_fp8`, `quark_like_dense`, etc.).

## Dependencies

- **No Quark runtime dependency** — both CUDA and Triton QDQ kernels are vendored from
  AMD Quark (MIT license) and compiled/run independently
- PyTorch (required)
- Triton (optional — for triton QDQ backend; serves as fallback when CUDA kernel
  cannot handle the tensor shape)
- CUDA toolkit (optional — for local_ext JIT compilation, highest performance)

## SpinQuant/QuaRot Rotation Plugin

The `spinquant_mxfp4.py` module implements the online rotation plugin for vLLM,
supporting SpinQuant and QuaRot style MXFP4 models with rotation-before-quantization.

### Supported Rotations

| Rotation | Type | Description |
|----------|------|-------------|
| R1 | Online | Pre-linear Hadamard/random/trained rotation on q/k/v/gate/up inputs |
| R2 | Offline | Fused into weights at export time (no runtime cost) |
| R3 | Online | Head-dim rotation on Q/K after RoPE |
| R4 | Online | Activation rotation before down_proj |

### Runtime Backends

| Backend | Weight Format | GEMM Library | VRAM Usage |
|---------|--------------|--------------|------------|
| `packed_fused` (default) | Packed uint8 MXFP4 | Triton fused kernel | Minimal (weight stays packed) |
| `quark_like_dense` | Dense bf16 (load-time dequant) | cuBLAS (torch.matmul) | 2× weight size |
| `preunpack_fp8` | FP8 + bf16 scale | cuBLAS FP8 (torch._scaled_mm) | ~1.5× weight size |

### Configuration (model's config.json)

```json
{
  "quantization_config": {
    "quant_method": "spinquant_mxfp4",
    "bits": 4,
    "group_size": 32,
    "spinquant_config": {
      "online_r1_rotation": true,
      "r2": true,
      "r3": false,
      "r4": true,
      "hidden_size": 4096,
      "intermediate_size": 14336,
      "head_dim": 128
    }
  }
}
```

### Environment Variables (Rotation Plugin)

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND` | `packed_fused`, `quark_like_dense`, `preunpack_fp8` | `packed_fused` | Force runtime GEMM backend |

## Unified Activation

Both `auto_round` (MXFP4/MXFP8 quantization) and `spinquant_mxfp4` (rotation
plugin) are activated by the single environment variable:

```bash
VLLM_ENABLE_AR_EXT=1 python -m vllm.entrypoints.openai.api_server --model ...
```

The `sitecustomize.py` entry point:
1. Replaces vLLM's built-in `AutoRoundConfig` with `AutoRoundExtensionConfig`
2. Registers `SpinQuantMXFP4Config` as a new quantization method
3. Applies the weight-loading patch for `spinquant_R*` keys

At runtime, the model's `quant_method` field determines which path is used:
- `"auto_round"` → `AutoRoundExtensionConfig` (existing MXFP4/MXFP8 path)
- `"spinquant_mxfp4"` → `SpinQuantMXFP4Config` (rotation + MXFP4 path)

## Running Tests

```bash
# Unit tests
cd auto_round_extension/vllm_ext/tests
pytest test_mxfp4_qdq.py -v

# Full verification with benchmark
python verify_mxfp4_qdq.py
```
