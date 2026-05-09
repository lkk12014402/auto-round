<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work spans implementing R1-R4 rotations, fixing accuracy bugs, validating against Quark with lm_eval benchmarks, and resolving a critical incompatibility between `InputRotationWrapperHadamard` (nn.Module wrapper for online rotation) and auto-round's quantization pipeline. The current phase focused on understanding why the wrapper breaks quantization, fixing it by reverting to hook-based rotation, and writing analysis documents about the architectural constraints.
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
   - Quick test showed online R1 dramatically better than offline R1

8. User ran full eval — catastrophic accuracy (hellaswag=0.2628)
   - **Root cause: DOUBLE NORMALIZATION BUG** — `get_hadamard_K()` returns already-normalized matrix (H/√N), but block rotation code divided by √N again
   - **Fix:** Removed extra `/ math.sqrt(r1_size)` from both `_apply_online_r1()` and hook
   - Verified: cosine_sim=1.0, hellaswag recovered to 0.4250

9. User asked about offline vs online R1 precision difference and model save implications
   - Explained: offline R1 changes ALL weight distributions → hurts quantization
   - Online R1 only modifies target modules, preserving original distributions
   - Documented: model saving requires nn.Module wrapper (not hooks) for online R1

10. User asked for comprehensive Quark rotation analysis documentation
    - Created `docs/quark_rotation_analysis.md` — complete analysis of R1-R4, save/load flow

11. User asked for full accuracy comparison: R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
    - Created `run_full_comparison.sh` — runs all levels on both frameworks
    - Results (limit=200): Auto-round and Quark accuracy aligned within ±0.03

12. User asked to implement InputRotationWrapper for save/load support
    - Implemented `InputRotationWrapperHadamard` in `rotation_utils.py`
    - Updated `_apply_online_r1()` to use wrappers instead of hooks
    - R4 also uses wrappers
    - All 7 unit tests pass, rotation-only save/load works (cosine_sim=1.0)

13. User asked whether online R1 can be converted to offline R1 after quantization
    - Analyzed and concluded: **No** — (1) RMSNorm gamma not fused, (2) Q(W@H) ≠ Q(W)@H
    - Created `docs/online_vs_offline_r1_after_quantization.md`

14. User reported all auto-round rotation tests crash after wrapper implementation
    - **Root cause:** `InputRotationWrapperHadamard` stores `original_module` as submodule → auto-round's `wrapper_block()` finds inner `nn.Linear` via `named_modules()` → wraps it with `WrapperLinear` → after MXFP4 quantization, inner weight compressed to shape [0] → inference crash
    - Identified the full 6-point coupling chain in auto-round's pipeline

15. Fixed the crash by reverting from wrapper to hook-based rotation
    - Rewrote `_apply_online_r1()` to use `forward_pre_hook` instead of wrapper
    - Rewrote R4 in `inplace/apply.py` to use hooks instead of wrappers
    - Kept `InputRotationWrapperHadamard` class in rotation_utils.py (refactored to self-own weight/bias) as a utility
    - **Verified: R1 + MXFP4 works! hellaswag=0.4550** (vs 0.4200 mxfp4-only, 0.5250 baseline)

16. User asked to explain why input wrapper can't be used instead of embed_tokens fusion for inference
    - Wrote detailed mathematical derivation showing two independent fatal issues:
      (a) RMSNorm γ and H don't commute: diag(γ)·H ≠ H·diag(γ)
      (b) Residual stream basis mismatch: rotated embed + unrotated o_proj/down_proj outputs
    - Created `docs/input_wrapper_vs_embed_fusion.md`

17. User asked why InputRotationWrapper can't work with auto-round
    - Explained the 6 hardcoded coupling points in auto-round's pipeline
    - Documented 4 possible fix approaches (A-D), concluded hook (D) is optimal

18. User asked whether compressors_new can decouple the pipeline to support wrappers
    - Analyzed compressors_new architecture in depth (entry.py, base.py, calib.py, quantizer)
    - Created `docs/wrapper_vs_hook_refactoring_analysis.md` — detailed analysis of 4 approaches
    - Created `docs/compressors_new_decoupling_analysis.md` — analysis of 6 coupling points and 3 abstraction layers needed

19. User asked if compressors_new architecture can decouple the complex quantization flow
    - Analyzed: compressors_new refactored the compressor hierarchy (mixins, routing) but NOT the core quantization path (wrapper_block → WrapperLinear → F.linear)
    - Identified 3 minimal abstraction points that could enable wrapper support:
      (1) `BaseQuantizer.is_quantizable()` — module discovery
      (2) `WrapperLinear._resolve_forward_strategy()` — forward path
      (3) `BaseCompressor._get_wrapper_block_fn()` — wrapping strategy
    - Wrote this analysis into `docs/compressors_new_decoupling_analysis.md`
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection (QKRotation after RoPE)
- `examples/test_rotation_quantization.py`: Rotation + W4A16 RTN test
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison script
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison script
- `examples/run_full_comparison.sh`: Full R1→R1+R2+R3+R4 comparison script
- `examples/test_wrapper_save_load.py`: 7-test suite for InputRotationWrapperHadamard
- `docs/quark_rotation_analysis.md`: Complete Quark rotation analysis
- `docs/online_r1_wrapper_and_rotation_size.md`: Wrapper comparison and rotation_size docs
- `docs/online_vs_offline_r1_after_quantization.md`: Why online/offline R1 aren't interchangeable after quantization
- `docs/r3_r4_online_rotation.md`: R3/R4 documentation
- `docs/input_wrapper_vs_embed_fusion.md`: Mathematical proof that input wrapper ≠ embed_tokens fusion
- `docs/wrapper_vs_hook_refactoring_analysis.md`: 4 approaches analysis for wrapper support
- `docs/compressors_new_decoupling_analysis.md`: Pipeline decoupling analysis with 6 coupling points

Files modified (current state):
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`:
  - `InputRotationWrapperHadamard` refactored: now self-owns weight/bias (no `original_module` submodule), uses `F.linear` in forward. Kept as utility class but NOT used in main rotation flow.
  - `apply_hadamard_to_linear()` updated to handle both `nn.Linear` and `InputRotationWrapperHadamard` (accesses `.weight` directly, not via `.original_module`)
  - Added to `__all__`

- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - Removed `InputRotationWrapperHadamard` import (no longer used)
  - Rewrote `_apply_online_r1()`: uses `forward_pre_hook` (not wrapper) — creates hook via `_make_online_r1_hook()`, registers on each target module
  - Added static method `_make_online_r1_hook()` that creates butterfly or block rotation hooks

- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`:
  - Removed `InputRotationWrapperHadamard` import
  - Rewrote R4: uses `forward_pre_hook` on `down_proj` (not wrapper)
  - Simplified `remove_spinquant_hooks()`: removed `r4_wrapper` handling, only handles R3 monkeypatch + standard hook handles

- `auto_round/algorithms/transforms/spinquant/__init__.py`:
  - Updated comment for `InputRotationWrapperHadamard` export (now "utility, used for rotation-only save/load")

Work completed:
- [x] Hook-based online R1 rotation (compatible with auto-round quantization)
- [x] Hook-based R4 rotation (compatible with auto-round quantization)
- [x] Verified R1 + MXFP4 works: hellaswag=0.4550
- [x] InputRotationWrapperHadamard refactored to self-own weight (utility only)
- [x] Mathematical proof: input wrapper ≠ embed_tokens fusion
- [x] Analysis documents for all architectural decisions
- [ ] Full comparison (R1, R1+R2, R1+R2+R3, R1+R2+R3+R4) with hook-based approach — was about to run
- [ ] Rotation config persistence (save rotation_config.json alongside model)
- [ ] Model load with automatic hook re-registration
</work_done>

<technical_details>
**Normalization Convention (CRITICAL):**
- Quark's `_get_hadamard_K()` returns UNNORMALIZED matrices (±1 values)
- Auto-round's `get_hadamard_K()` returns NORMALIZED matrices (H/√N, orthogonal)
- Must NOT re-normalize — this caused the double-normalization bug (rotation_size=128 → hellaswag=0.2628)

**Online R1 vs Offline R1:**
- Offline R1: global change-of-basis. Modifies ALL layers, fuses RMSNorm. No wrapper/hook needed at inference. Quantization quality suffers because ALL weight distributions change.
- Online R1: local rotation per target module. Only modifies q/k/v/gate/up weights + registers forward_pre_hook. Residual stream in original basis. Better quantization quality but needs hook at inference.
- After quantization: online ↔ offline are NOT interchangeable because (1) RMSNorm gamma not fused in online mode, (2) Q(W@H) ≠ Q(W)@H

**Why Input Wrapper ≠ Embed Tokens Fusion (after quantization):**
- Input wrapper: `x_rot = (x/rms(x) ⊙ γ) · H` — **先 γ 后 H**
- Embed fusion: `x'_norm = (x·H/rms(x)) ⊙ γ` — **先 H 后 γ**
- `diag(γ) · H ≠ H · diag(γ)` unless γ is constant
- Additionally, residual stream mixes rotated (embed) and unrotated (o_proj/down_proj) bases

**R1-R4 Fusion Summary:**
| Rotation | Offline fuseable? | Needs inference hook? |
|----------|------------------|----------------------|
| R1 offline | ✅ fully | ❌ |
| R1 online | weight side fused | ✅ forward_pre_hook |
| R2 | ✅ fully (V out + O in) | ❌ |
| R3 | ❌ (after RoPE, non-linear) | ✅ monkeypatch |
| R4 | down_proj input fused | ✅ forward_pre_hook on down_proj |

**Why InputRotationWrapperHadamard breaks auto-round (6 coupling points):**
1. `wrapper_block()` uses `block.named_modules()` which recurses into wrapper → finds inner `nn.Linear`
2. `type(m) in SUPPORTED_LAYER_TYPES` — only recognizes `nn.Linear` and `Conv1D`
3. `WrapperLinear.__init__` selects `linear_forward` or `conv1d_forward` based on `type(orig_layer) == nn.Linear`
4. `WrapperLinear.forward` calls `F.linear(x, weight_q, bias)` — bypasses wrapper's rotation
5. `WrapperWALayer.forward` calls `self.orig_layer.forward(x)` — after MXFP4 packing, weight is shape [0]
6. `pack_layer()` checks `type(layer) not in SUPPORTED_LAYER_TYPES`

**Why hooks work perfectly:**
- `WrapperLinear.forward()` explicitly runs `self.orig_layer._forward_pre_hooks` (lines 503-518) BEFORE quantized linear computation
- `WrapperWALayer.__init__()` steals hooks from `orig_layer` (line 557), runs them in `forward()` (lines 572-575) before activation quantization
- Module tree stays clean (all targets remain `nn.Linear`), auto-round discovers and wraps them normally

**compressors_new architecture:**
- Refactored hierarchy: `entry.py` routes → `BaseCompressor` + mixins (MLLM, Diffusion, SpinQuant) → `CalibCompressor`/`ZeroShotCompressor`
- 7-phase `post_init()`: resolve_scheme → resolve_formats → patch_model → build_layer_config → apply_rotations → hardware_setup
- BUT core quantization path (wrapper_block → WrapperLinear → F.linear) is unchanged
- `_apply_rotations()` runs at Phase 4.5 — after layer_config, before quantization
- `self.wrapper_block = wrapper_block` is directly bound at line 729 — not overridable

**Qwen3-0.6B Architecture:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q, 8 kv)
- intermediate_size=3072 (= 3 × 1024, NOT pow2)
- tie_word_embeddings=True in config

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- auto-round installed editable at /data/lkk/quarot/auto-round
- Quark at /data/lkk/quarot/Quark
- GPUs 0-3 often busy; use GPUs 4-7 for testing
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Core rotation utilities: `get_hadamard_K()`, `matmul_hadU()`, `rotate_in_channels_()`, `apply_hadamard_to_linear()`
  - `InputRotationWrapperHadamard` class (refactored: self-owns weight/bias, no submodule) — kept as utility
  - Fixed `apply_hadamard_to_linear()` to access `.weight` directly on both `nn.Linear` and wrapper

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()` (line ~220)
  - `_apply_online_r1()` (line ~642): **reverted to hook-based** — registers `forward_pre_hook` on target modules
  - `_make_online_r1_hook()`: static method creating butterfly or block rotation hooks
  - `_fuse_offline_rotations()` (line ~735): offline R1 path
  - `SpinQuantConfig` dataclass with `online_r1_rotation=True` default

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - `register_spinquant_hooks()`: R3 monkeypatch + R4 hooks
  - R4 now uses `forward_pre_hook` on `down_proj` (not wrapper)
  - `remove_spinquant_hooks()` handles R3 monkeypatch + standard hook handles

- `/data/lkk/quarot/auto-round/auto_round/wrapper.py`
  - `WrapperLinear` (line 62): auto-round's quantization wrapper
  - **Line 503-518**: `forward()` explicitly runs `orig_layer._forward_pre_hooks` — this is WHY hooks work
  - **Line 554-558**: `WrapperWALayer.__init__()` steals hooks from orig_layer
  - **Line 570-607**: `WrapperWALayer.forward()` runs stolen hooks before activation quant
  - **Line 768-769**: `wrapper_block()` uses `type(m) in SUPPORTED_LAYER_TYPES` — the discovery bottleneck

- `/data/lkk/quarot/auto-round/auto_round/utils/common.py`
  - **Line 607**: `SUPPORTED_LAYER_TYPES = (torch.nn.Linear, ...)` — controls what gets quantized

- `/data/lkk/quarot/auto-round/auto_round/compressors_new/base.py`
  - `BaseCompressor`: 7-phase pipeline, `_apply_rotations()` at Phase 4.5
  - **Line 729**: `self.wrapper_block = wrapper_block` — direct binding, not overridable
  - `supported_types = SUPPORTED_LAYER_TYPES` at class level

- `/data/lkk/quarot/auto-round/auto_round/compressors_new/calib.py`
  - `CalibCompressor` / `CalibratedRTNCompressor`: block-wise quantization
  - Uses `WrapperLinear` directly (line 1252) for RTN quantization of individual layers

- `/data/lkk/quarot/auto-round/auto_round/algorithms/quantization/sign_round/quantizer.py`
  - `self.wrapper_block(block, ...)` at line 186 — calls wrapper_block for each block
  - `unwrapper_block(block, best_params)` at line 330

- `/data/lkk/quarot/auto-round/docs/input_wrapper_vs_embed_fusion.md`
  - Mathematical proof that input wrapper ≠ embed_tokens fusion (γ·H ≠ H·γ + basis mismatch)

- `/data/lkk/quarot/auto-round/docs/wrapper_vs_hook_refactoring_analysis.md`
  - 4 approaches analysis: skip internal (A), add to types (B), self-own weight (C), hook (D)
  - Conclusion: hook (D) is optimal — zero intrusion, already supported by WrapperLinear/WrapperWALayer

- `/data/lkk/quarot/auto-round/docs/compressors_new_decoupling_analysis.md`
  - 6 coupling points in auto-round's pipeline
  - 3 minimal abstraction layers needed for future wrapper support
  - Short/medium/long term strategy recommendations
</important_files>

<next_steps>
Remaining work:
- Run full comparison (R1, R1+R2, R1+R2+R3, R1+R2+R3+R4) with hook-based approach to verify all levels work with MXFP4
- Implement rotation config persistence: save `rotation_config.json` alongside quantized model
- Implement model load with automatic hook re-registration (`load_rotated_model()` utility)
- Update `test_wrapper_save_load.py` to test hook-based save/load flow
- Consider contributing minimal abstraction points to auto-round's compressors_new

Immediate next steps:
- User was about to test R1+R2, R1+R2+R3, R1+R2+R3+R4 with MXFP4 — they said "我来测试就好" (I'll test it myself)
- Discuss with user about the hook vs wrapper architectural decision and next priorities
- The user's latest question was about whether compressors_new can decouple the pipeline — analysis is complete and documented

Open questions from user:
- Whether to pursue minimal abstraction changes to compressors_new for wrapper support
- Priority of rotation config persistence vs other tasks
</next_steps>