<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work spans implementing R1-R4 rotations, fixing accuracy bugs, validating against Quark with lm_eval benchmarks, and implementing a proper `InputRotationWrapperHadamard` (nn.Module) to enable model save/load with online rotations. The current focus is fixing a critical compatibility issue where the new wrapper breaks auto-round's quantization pipeline (WrapperLinear/WrapperWALayer) because auto-round discovers and wraps the inner `nn.Linear` inside the wrapper, causing shape mismatches at inference time.
</overview>

<history>
1. User asked to fix broken tests, resolve training code duplication, and expand rotation level validation (R1-R4)
   - Explored both codebases and reference implementations
   - Fixed device mismatches, R4 hook signature, added `apply_hadamard_to_linear()`
   - Rewrote `_fuse_offline_rotations()` with proper R2/R4 fusion
   - Created comprehensive `test_rotation_levels.py` — all 7 mock tests pass

2. User asked to validate with Qwen3-0.6B using lm_eval
   - Created `test_qwen3_rotation_eval.py` evaluation script
   - Fixed R4 crash (intermediate_size=3072 not power-of-2 → block Hadamard)
   - Fixed R3 (must be AFTER RoPE, not on q_proj output)
   - Verified all levels: cosine_sim >0.99996, 100% argmax agreement

3. User asked for auto-round's existing implementations, Quark-style logging
   - Implemented `matmul_hadU` in rotation_utils.py (butterfly algorithm)
   - Created `monkeypatch.py` with architecture-generic R3
   - Updated preprocessor with Python logging module throughout

4. User reported R1 accuracy degradation (hellaswag -14%)
   - **Root cause:** `untie_word_embeddings_if_needed()` didn't set `model.config.tie_word_embeddings = False`
   - lm_eval's HFLM calls `model.tie_weights()` which re-tied weights, overwriting rotated lm_head
   - **Fix:** Added `model.config.tie_word_embeddings = False` after untying

5. User asked for rotation + quantization test scripts and MXFP4 comparison with Quark
   - Created test scripts for both frameworks, `run_comparison_tests.sh`
   - Added `rotation_size` configurability matching Quark's semantics

6. User reported rotation+MXFP4 accuracy much worse than MXFP4-only in auto-round
   - Root cause: offline R1 rotates ALL weights (including o_proj, down_proj, fuses RMSNorm) → changes weight distributions → hurts MXFP4 quantization
   - Quark's default is online R1: only rotates target modules + activation hooks

7. Implemented online R1 rotation matching Quark's behavior
   - Added `_apply_online_r1()` method with activation hooks
   - Changed `online_r1_rotation` default to `True`
   - Quick test (limit=200) showed online R1 dramatically better than offline R1

8. User ran full eval with rotation_size=128 — catastrophic accuracy (hellaswag=0.2628)
   - **Root cause: DOUBLE NORMALIZATION BUG** — `get_hadamard_K()` returns already-normalized matrix (H/√N), but block rotation code divided by √N again
   - **Fix:** Removed extra `/ math.sqrt(r1_size)` from both `_apply_online_r1()` and `_make_online_r1_hook()`
   - Verified: cosine_sim=1.0 for rotation cancellation, hellaswag recovered to 0.4250

9. User asked to understand offline vs online R1 precision difference and model save implications
   - Explained: offline R1 changes ALL weight distributions → hurts quantization quality
   - Online R1 only modifies target modules, preserving original distributions for other layers
   - Documented: model saving requires nn.Module wrapper (not hooks) for online R1

10. User asked for comprehensive Quark rotation analysis documentation
    - Created `docs/quark_rotation_analysis.md` — complete analysis of R1-R4, save/load flow
    - Key findings: R1(offline)+R2 fully fuseable (no wrapper), Online R1/R4 need wrappers, R3 needs monkeypatch

11. User asked for full accuracy comparison: R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
    - Created `run_full_comparison.sh` — runs all levels on both frameworks in parallel
    - **Results (limit=200):** Auto-round and Quark accuracy aligned within ±0.03 for all rotation levels

12. User asked to implement InputRotationWrapper in auto-round for save/load support
    - Implemented `InputRotationWrapperHadamard` in `rotation_utils.py`
    - Updated `_apply_online_r1()` to use wrappers instead of hooks
    - Updated R4 in `inplace/apply.py` to use wrappers for down_proj
    - Fixed `apply_hadamard_to_linear()` to handle wrapped modules
    - All 7 unit tests pass, full Qwen3-0.6B save/load roundtrip works (cosine_sim=1.0)
    - R1+R2 rotation_fp16 accuracy: hellaswag=0.5200 (matches hook-based approach)

13. User asked whether online R1 can be converted to offline R1 after quantization
    - Analyzed mathematically and concluded: **No, they are NOT equivalent after quantization**
    - Two reasons: (1) RMSNorm gamma is not fused in online R1, can't fuse after quantization; (2) quantization is non-linear: Q(W@H) ≠ Q(W)@H
    - Created `docs/online_vs_offline_r1_after_quantization.md` with detailed analysis

14. User reported all auto-round rotation tests in `run_full_comparison.sh` crash
    - Examined `logs_comparison/` — all rotation levels fail with `RuntimeError: size mismatch`
    - **Root cause identified:** `InputRotationWrapperHadamard` stores `original_module` as a submodule (nn.Module). Auto-round's `wrapper_block()` iterates `named_modules()`, finds the inner `nn.Linear` at path `q_proj.original_module`, and wraps it with `WrapperLinear`. After unwrapping, the inner module becomes `WrapperWALayer` whose weight gets compressed to shape [0] for MXFP4, and `WrapperWALayer.forward()` calls `self.orig_layer.forward(x)` which fails because the weight is empty.
    - **Was actively debugging this when compaction occurred**
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection (QKRotation after RoPE)
- `examples/test_rotation_quantization.py`: Rotation + W4A16 RTN test
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison script
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison script
- `examples/run_full_comparison.sh`: Full R1→R1+R2+R3+R4 comparison script
- `examples/test_wrapper_save_load.py`: 7-test suite for InputRotationWrapperHadamard (forward, block rotation, state_dict, load, proxy, full model save/load)
- `docs/quark_rotation_analysis.md`: Complete Quark rotation analysis
- `docs/online_r1_wrapper_and_rotation_size.md`: Wrapper comparison and rotation_size docs
- `docs/online_vs_offline_r1_after_quantization.md`: Analysis of why online/offline R1 aren't interchangeable after quantization
- `docs/r3_r4_online_rotation.md`: R3/R4 documentation

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`:
  - Added `InputRotationWrapperHadamard` class (~lines 183-361)
  - state_dict() uses `nn.Module.state_dict()` to temp dict, then strips `original_module.` prefix
  - `_load_from_state_dict()` remaps keys: `prefix + "weight"` → `prefix + "original_module.weight"`
  - Fixed `apply_hadamard_to_linear()` to handle wrapped modules (extract `original_module`)
  - Added to `__all__`

- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - Added `InputRotationWrapperHadamard` to imports
  - Removed old `_make_online_r1_hook()` function
  - Rewrote `_apply_online_r1()`: uses `setattr(parent, attr_name, wrapper)` instead of `register_forward_pre_hook`
  - Changed log message from "activation hooks" to "wrapper"

- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`:
  - Added `InputRotationWrapperHadamard` import
  - Replaced R4 hook-based approach with wrapper: finds parent module, creates `InputRotationWrapperHadamard`, does `setattr(parent, attr_name, wrapper)`
  - Updated `remove_spinquant_hooks()` to handle `r4_wrapper` entries (unwrap → original_module)

- `auto_round/algorithms/transforms/spinquant/__init__.py`:
  - Added `InputRotationWrapperHadamard` import and export

Work completed:
- [x] InputRotationWrapperHadamard implementation (forward, state_dict, load)
- [x] Online R1 uses wrapper instead of hooks
- [x] R4 uses wrapper instead of hooks  
- [x] Unit tests pass (7/7)
- [x] Full model save/load roundtrip works (Qwen3-0.6B, cosine_sim=1.0)
- [x] Rotation-only accuracy preserved (hellaswag=0.5200)
- [ ] **BROKEN: Rotation + MXFP4 quantization fails at inference** — critical bug

Current state:
- Wrapper works correctly for rotation-only (no quantization)
- Wrapper + auto-round quantization (MXFP4) crashes at inference
- All auto-round rotation entries in `run_full_comparison.sh` fail
- Quark comparison entries still work fine
</work_done>

<technical_details>
**Normalization Convention (CRITICAL):**
- Quark's `_get_hadamard_K()` returns UNNORMALIZED matrices (±1 values)
- Auto-round's `get_hadamard_K()` returns NORMALIZED matrices (H/√N, orthogonal)
- Must NOT re-normalize — this caused the double-normalization bug (rotation_size=128 → hellaswag=0.2628)

**Online R1 vs Offline R1:**
- Offline R1: global change-of-basis. Modifies ALL layers, fuses RMSNorm. No wrapper needed. Quantization quality suffers because ALL weight distributions change.
- Online R1: local rotation per target module. Only modifies q/k/v/gate/up weights + wraps with InputRotationWrapperHadamard. Residual stream in original basis. Better quantization quality but needs wrapper at inference.
- After quantization: online ↔ offline are NOT interchangeable because (1) RMSNorm gamma not fused in online mode, (2) Q(W@H) ≠ Q(W)@H

**R1-R4 Fusion Summary:**
| Rotation | Offline fuseable? | Needs inference wrapper? |
|----------|------------------|------------------------|
| R1 offline | ✅ fully | ❌ |
| R1 online | weight side fused | ✅ InputRotationWrapper |
| R2 | ✅ fully (V out + O in) | ❌ |
| R3 | ❌ (after RoPE, non-linear) | ✅ monkeypatch |
| R4 | down_proj input fused | ✅ wrapper on down_proj |

**CRITICAL BUG — InputRotationWrapperHadamard + auto-round quantization incompatibility:**

The root cause chain:
1. `InputRotationWrapperHadamard` stores `original_module` (nn.Linear) as a **submodule** (via `self.original_module = original_module` in `__init__`, which PyTorch's `__setattr__` auto-registers)
2. Auto-round's `wrapper_block()` calls `block.named_modules()` which discovers the inner `nn.Linear` at path `q_proj.original_module`
3. `wrapper_block` wraps it with `WrapperLinear` because `type(m) == torch.nn.Linear`
4. After quantization, `unwrapper_block` unwraps to `WrapperWALayer` wrapping the nn.Linear
5. For MXFP4, the inner `nn.Linear` weight gets compressed to shape `[0]` (packed format)
6. At inference: `InputRotationWrapperHadamard.forward()` → `WrapperWALayer.forward()` → `self.orig_layer.forward(x)` → `F.linear(x, weight, bias)` where weight has shape [0] → crash

Additional sub-issue: `WrapperLinear.__init__` line 127 does `type(self.orig_layer) == torch.nn.Linear`, which is True here. But there's also the `WrapperWALayer` which calls `self.orig_layer.forward(x)` on the inner linear — for MXFP4 with packed weights this fails.

**Possible fix approaches (not yet implemented):**
1. **Don't register `original_module` as submodule**: Use `object.__setattr__` to bypass PyTorch's auto-registration. Steal weight/bias as own Parameters. Use `F.linear()` directly in forward. But then auto-round won't quantize the weight (not in SUPPORTED_LAYER_TYPES).
2. **Make wrapper inherit from nn.Linear** or add to SUPPORTED_LAYER_TYPES: Auto-round would wrap the wrapper itself. But WrapperLinear.forward bypasses our rotation logic (calls `F.linear` directly).
3. **Hybrid approach**: Store rotation info in the wrapper, but let auto-round quantize the weight normally. The wrapper's forward does rotation then calls `F.linear(x_rotated, self.weight, self.bias)`. Need to ensure auto-round's quantization applies to our weight.
4. **Override `named_modules()`** to hide inner module. But still need auto-round to quantize the weight.

The fundamental tension: auto-round needs to quantize the weight, but it needs to find `nn.Linear` modules to do so. The wrapper hides/wraps the nn.Linear, breaking the discovery.

**Qwen3-0.6B Architecture:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q, 8 kv)
- intermediate_size=3072 (= 3 × 1024, NOT pow2)
- tie_word_embeddings=True in config

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- auto-round installed editable at /data/lkk/quarot/auto-round
- Quark at /data/lkk/quarot/Quark
- GPUs 0-3 often busy; use GPUs 4-7 for testing

**Key auto-round wrapping infrastructure:**
- `SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)` (in `auto_round/utils/common.py:607`)
- `wrapper_block()` (wrapper.py:746): iterates `named_modules()`, wraps anything with `type(m) in SUPPORTED_LAYER_TYPES`
- `WrapperLinear.__init__` (wrapper.py:127): `self.orig_forward = self.linear_forward if type(self.orig_layer) == torch.nn.Linear else self.conv1d_forward`
- `WrapperLinear.forward` (wrapper.py:489): quantizes weight via `_qdq_weight()`, calls `self.orig_forward(x, weight_q, bias)`
- `WrapperWALayer` (wrapper.py:540): wraps nn.Linear for activation quantization at inference, calls `self.orig_layer.forward(x)` directly
- `unwrapper_block()` (wrapper.py:833): finds modules with `orig_layer` attr, calls `m.unwrapper()`
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Contains `InputRotationWrapperHadamard` class (~lines 183-361) — THE module causing the bug
  - Also: `get_hadamard_K()`, `matmul_hadU()`, `rotate_in_channels_()`, `apply_hadamard_to_linear()`
  - `apply_hadamard_to_linear()` was updated to handle wrapped modules (extracts original_module)

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()` (line ~220)
  - `_apply_online_r1()` (line ~642): uses wrapper instead of hooks — creates InputRotationWrapperHadamard, does setattr
  - `_fuse_offline_rotations()` (line ~735): offline R1 path
  - `SpinQuantConfig` dataclass with `online_r1_rotation=True` default

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - `register_spinquant_hooks()`: R3 monkeypatch + R4 wrapper
  - R4 now uses InputRotationWrapperHadamard instead of hooks
  - `remove_spinquant_hooks()` handles r4_wrapper entries

- `/data/lkk/quarot/auto-round/auto_round/wrapper.py`
  - `WrapperLinear` (line 62): auto-round's quantization wrapper — wraps nn.Linear modules
  - `wrapper_block()` (line 746): discovers nn.Linear via `named_modules()`, wraps with WrapperLinear
  - `WrapperWALayer` (line 540): inference wrapper for activation quantization
  - **Line 127**: `type(self.orig_layer) == torch.nn.Linear` — exact type check, the key compatibility point
  - **Line 769**: `type(m) in SUPPORTED_LAYER_TYPES` — discovers modules to quantize

- `/data/lkk/quarot/auto-round/auto_round/utils/common.py`
  - **Line 607**: `SUPPORTED_LAYER_TYPES = (torch.nn.Linear, ...)` — controls what gets quantized

- `/data/lkk/quarot/auto-round/examples/test_wrapper_save_load.py`
  - 7-test suite: forward equivalence, block rotation, state_dict, load, model integration, proxy
  - All pass for rotation-only (no quantization)

- `/data/lkk/quarot/auto-round/examples/test_autoround_rotation_mxfp4.py`
  - Main test script for rotation + MXFP4 comparison
  - `apply_rotation()` at line 73, `quantize_mxfp4()` at line 88

- `/data/lkk/quarot/auto-round/examples/run_full_comparison.sh`
  - Full comparison script — currently all autoround rotation entries FAIL

- `/data/lkk/quarot/auto-round/examples/logs_comparison/`
  - Contains latest test logs: `20260508_093228_autoround_R1.log` etc.
  - All autoround rotation logs show same error: `size mismatch, got input (...), mat (...), vec (0)`

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation_utils.py`
  - Reference: `InputRotationWrapperHadamard` at line 290
  - Quark's implementation stores original_module as submodule too, but Quark's quantization system knows about wrappers

- `/data/lkk/quarot/auto-round/docs/online_vs_offline_r1_after_quantization.md`
  - Analysis explaining why online R1 can't be converted to offline after quantization
</important_files>

<next_steps>
**IMMEDIATE — Fix InputRotationWrapperHadamard + auto-round quantization incompatibility:**

The wrapper breaks auto-round's quantization because auto-round's `wrapper_block()` finds and wraps the inner `nn.Linear` inside the wrapper.

**Recommended approach (not yet implemented):**

Refactor `InputRotationWrapperHadamard` to NOT store `original_module` as an nn.Module submodule. Instead:

1. **Steal weight and bias as own Parameters:**
   ```python
   class InputRotationWrapperHadamard(nn.Module):
       def __init__(self, original_module, rotation_size, ...):
           super().__init__()
           # Steal parameters — don't store original_module as submodule
           self.weight = original_module.weight  # nn.Parameter, auto-registered
           self.register_parameter('bias', original_module.bias)
           self._in_features = original_module.in_features
           self._out_features = original_module.out_features
           # ... hadamard buffers ...
       
       def forward(self, x):
           x = <rotate x>
           return F.linear(x, self.weight, self.bias)
   ```

2. **Make auto-round recognize it for quantization** — either:
   - Add `InputRotationWrapperHadamard` to `SUPPORTED_LAYER_TYPES` (requires small change to utils/common.py or a registration mechanism)
   - OR make the wrapper subclass `nn.Linear` (skip `nn.Linear.__init__`, just `nn.Module.__init__()`)
   - OR modify `WrapperLinear` to use `isinstance(m, nn.Linear)` instead of `type(m) == nn.Linear` and make wrapper inherit from Linear

3. **Ensure WrapperLinear compatibility:**
   - If auto-round wraps InputRotationWrapperHadamard, `WrapperLinear.__init__` must pick `linear_forward` (not `conv1d_forward`)
   - Need to verify `WrapperLinear.forward` works: it calls `self._qdq_weight()` on the weight, then `self.orig_forward(x, weight_q, bias)`. The rotation must happen BEFORE weight quantization in the forward.
   - This is problematic because `WrapperLinear.forward` calls `self.orig_forward(x, weight_q, bias)` which is `F.linear` — it bypasses the wrapper's rotation

4. **Alternative: Keep wrapper structure but prevent auto-round from descending into it:**
   - Use `object.__setattr__(self, 'original_module', original_module)` to bypass nn.Module registration
   - Override `_apply()` to ensure `.to()`, `.cuda()` etc. still propagate to original_module
   - Auto-round won't find the inner nn.Linear (it's not in `named_modules()`)
   - But then the wrapper's weight also won't be quantized...
   - Could register pre/post hooks or override forward to intercept quantized weight from elsewhere

**The fundamental design question:** How should online rotation interact with auto-round's quantization pipeline? The rotation must happen in the forward path BEFORE the linear computation, but auto-round's WrapperLinear.forward takes control of the entire forward pass (weight quant → act quant → F.linear).

**Most promising path:** Make InputRotationWrapperHadamard own its weight directly (no inner nn.Linear visible to auto-round), AND register it as a supported layer type so auto-round can quantize it. Override forward to do rotation then F.linear. Auto-round's WrapperLinear would wrap it, and since WrapperLinear.forward calls `self.orig_forward(x, weight_q, bias)` which is `F.linear`, the rotation needs to happen BEFORE WrapperLinear's forward... This means we need to register the rotation as a forward_pre_hook on the wrapper's weight, which WrapperLinear.forward explicitly runs (lines 503-518).

Actually the cleanest path: **Go back to hooks + save rotation config separately.** The hook approach actually works with auto-round's pipeline (WrapperLinear explicitly runs forward_pre_hooks). For save/load, save the rotation config (which modules, rotation_size, hadamard_K) as a JSON file alongside the model, and provide a `load_rotated_model()` utility that re-applies wrappers/hooks after loading. This matches how Quark's `prepare_model_for_reloading_fake()` works.

**Other pending tasks:**
- After fixing the wrapper/quantization issue, re-run `run_full_comparison.sh` to verify auto-round accuracy
- Validate save/load roundtrip WITH quantization (not just rotation-only)
</next_steps>