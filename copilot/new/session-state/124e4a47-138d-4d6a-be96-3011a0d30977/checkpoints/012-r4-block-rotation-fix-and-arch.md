<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work spans implementing R1-R4 rotations, fixing accuracy bugs, validating against Quark with lm_eval benchmarks, resolving incompatibilities between rotation wrappers and auto-round's quantization pipeline, and producing comprehensive architecture analysis documents. The current phase focused on analyzing Quark vs auto-round architectural differences (FX graph tracing vs module tree type checking), writing documentation, fixing an R4 block rotation bug, and planning multi-scheme test coverage.
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
    - Identified 6-point coupling chain in auto-round's pipeline

15. Fixed the crash by reverting from wrapper to hook-based rotation
    - Rewrote `_apply_online_r1()` to use `forward_pre_hook` instead of wrapper
    - Rewrote R4 in `inplace/apply.py` to use hooks instead of wrappers
    - Kept `InputRotationWrapperHadamard` class as utility
    - **Verified: R1 + MXFP4 works! hellaswag=0.4550**

16. User asked about input wrapper vs embed_tokens fusion mathematical equivalence
    - Wrote detailed mathematical derivation showing two fatal issues: (a) diag(γ)·H ≠ H·diag(γ), (b) residual stream basis mismatch
    - Created `docs/input_wrapper_vs_embed_fusion.md`

17. User asked why InputRotationWrapper can't work with auto-round
    - Explained 6 hardcoded coupling points
    - Documented 4 approaches (A-D), concluded hook (D) is optimal
    - Created `docs/wrapper_vs_hook_refactoring_analysis.md`

18. User asked whether compressors_new can decouple the pipeline
    - Analyzed compressors_new architecture — refactored compressor hierarchy but NOT core quantization path
    - Created `docs/compressors_new_decoupling_analysis.md`

19. User asked why Quark CAN use InputRotationWrapper
    - **Key finding:** Quark uses `torch.fx` graph tracing (traces actual operations), auto-round uses module tree + `type()` checking
    - Quark quantizes FIRST (on clean nn.Linear), THEN wraps with rotation wrapper
    - FX graph automatically sees through wrappers — no type checking needed
    - Created `docs/quark_vs_autoround_wrapper_architecture.md`

20. User asked to update rotation save/load docs for all three matrix types
    - Identified 3 types: Hadamard (deterministic, just save size), QuaRot random (must save matrix), SpinQuant trained (must save matrix)
    - Quark uses buffer dtype to distinguish: int8 → Hadamard, float64 → random/trained
    - Updated `docs/rotation_save_load_solution.md` with complete 3-type persistence scheme
    - Analyzed invasiveness: ~26 lines of auto-round core changes + 1 new file (~100 lines)
    - **Critical finding:** `export/utils.py` line 374 actively CLEANS `rotation_configs` from save dict

21. User asked whether Quark has custom acceleration kernels for rotation
    - **Finding: NO rotation-specific kernels in Quark**
    - Quark has CUDA kernels only for quantization (MXFP4 dequantize, MX scale)
    - Hadamard uses pure PyTorch butterfly algorithm (same as auto-round)
    - No fused rotation+linear kernel exists

22. User asked about rotation + different quantization scheme compatibility
    - Rotation is completely scheme-agnostic in both Quark (17 schemes) and auto-round
    - Created `docs/rotation_scheme_compatibility.md`

23. User reported R1+R2+R3+R4 MXFP4 test crash
    - **Root cause:** R4 hook used `matmul_hadU(x, K=128)` on 3072-dim input. Butterfly halves 3072→96, but 96 < 128 → dimension mismatch
    - **Fix:** Added `need_block_rotation = (r4_size < intermediate_size)` check. When true, reshape input to blocks of r4_size, apply Hadamard per block, reshape back
    - Verified: cosine_sim=1.0, Qwen3-0.6B forward pass works, hellaswag=0.4900

24. User asked to create test scripts for W4A16 and NVFP4 schemes with rotation combinations
    - Was about to create the test script when compaction triggered
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection
- `examples/test_rotation_quantization.py`: Rotation + W4A16 RTN test
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison
- `examples/run_full_comparison.sh`: Full R1→R1+R2+R3+R4 comparison script
- `examples/test_wrapper_save_load.py`: 7-test suite for InputRotationWrapperHadamard
- `docs/quark_rotation_analysis.md`: Complete Quark rotation analysis
- `docs/online_r1_wrapper_and_rotation_size.md`: Wrapper comparison and rotation_size docs
- `docs/online_vs_offline_r1_after_quantization.md`: Why online/offline R1 aren't interchangeable
- `docs/r3_r4_online_rotation.md`: R3/R4 documentation
- `docs/input_wrapper_vs_embed_fusion.md`: Mathematical proof input wrapper ≠ embed fusion
- `docs/wrapper_vs_hook_refactoring_analysis.md`: 4 approaches analysis
- `docs/compressors_new_decoupling_analysis.md`: Pipeline decoupling analysis
- `docs/quark_vs_autoround_wrapper_architecture.md`: FX tracing vs type checking architecture
- `docs/rotation_save_load_solution.md`: Complete 3-type persistence scheme (Hadamard/random/trained)
- `docs/rotation_scheme_compatibility.md`: Rotation + quantization scheme compatibility analysis

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`:
  - `InputRotationWrapperHadamard` refactored (self-owns weight/bias, utility only)
  - `apply_hadamard_to_linear()` handles both nn.Linear and wrapper
  - `matmul_hadU()` with butterfly algorithm

- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - `_apply_online_r1()`: uses forward_pre_hook (not wrapper)
  - `_make_online_r1_hook()`: static method for butterfly/block rotation hooks

- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`:
  - **Latest fix (this session):** R4 hook completely rewritten to handle block rotation when `r4_size < intermediate_size`. Added `need_block_rotation` flag — when true, reshapes input to `[..., num_blocks, r4_size]`, applies Hadamard per block, reshapes back. Fixes crash on Qwen3-0.6B where `r4_rotation_size=128` but `intermediate_size=3072`.
  - R4 uses forward_pre_hook (not wrapper)
  - `remove_spinquant_hooks()` simplified

- `auto_round/algorithms/transforms/spinquant/__init__.py`:
  - Updated exports

Work completed:
- [x] Hook-based online R1 rotation (compatible with auto-round quantization)
- [x] Hook-based R4 rotation with block rotation support
- [x] R4 block rotation fix for r4_size < intermediate_size
- [x] Verified R1 + MXFP4: hellaswag=0.4550
- [x] Verified R1+R2+R3+R4 rotation-only: hellaswag=0.4900
- [x] All architecture analysis documents written
- [x] Rotation scheme compatibility analysis
- [ ] Test scripts for W4A16 + rotation combinations — user requested, not yet created
- [ ] Test scripts for NVFP4 + rotation combinations — user requested, not yet created
- [ ] Rotation config persistence (save/load) — design documented, implementation deferred
</work_done>

<technical_details>
**Normalization Convention (CRITICAL):**
- Quark's `_get_hadamard_K()` returns UNNORMALIZED matrices (±1 values)
- Auto-round's `get_hadamard_K()` returns NORMALIZED matrices (H/√N, orthogonal)
- Must NOT re-normalize — caused double-normalization bug (hellaswag=0.2628)

**Online R1 vs Offline R1:**
- Offline R1: global change-of-basis. Modifies ALL layers, fuses RMSNorm. No hook needed at inference. Quantization quality suffers because ALL weight distributions change.
- Online R1: local rotation per target module. Only modifies q/k/v/gate/up weights + registers forward_pre_hook. Better quantization quality but needs hook at inference.
- After quantization: online ↔ offline NOT interchangeable: (1) RMSNorm gamma not fused, (2) Q(W@H) ≠ Q(W)@H

**Why InputRotationWrapper breaks auto-round (6 coupling points):**
1. `wrapper_block()` uses `type(m) in SUPPORTED_LAYER_TYPES` — wrapper not recognized
2. `named_modules()` recurses into wrapper → finds inner nn.Linear with wrong path
3. `WrapperLinear.__init__` only recognizes nn.Linear/Conv1D
4. `WrapperLinear.forward` bypasses wrapper's rotation with F.linear
5. `WrapperWALayer.forward` — after MXFP4 packing, weight=[0] → crash
6. `pack_layer()` type check fails

**Why Quark CAN use wrappers:**
- Quark uses `torch.export.export_for_training()` → FX graph tracing → sees `ops.aten.linear.default` nodes regardless of wrapper depth
- Quark quantizes FIRST (clean nn.Linear → QuantLinear), THEN wraps with rotation
- Auto-round operates on module tree with type checking — fundamentally incompatible

**Why hooks work in auto-round:**
- `WrapperLinear.forward()` explicitly runs `self.orig_layer._forward_pre_hooks` (line 503-518)
- `WrapperWALayer.__init__()` steals hooks from orig_layer (line 557), runs them in forward (line 572-575)
- Module tree stays clean (all targets remain nn.Linear)

**R4 Block Rotation Bug (fixed this session):**
- When `r4_rotation_size=128` but `intermediate_size=3072`, `matmul_hadU(x, K=128)` called on full 3072 dim
- Butterfly: 3072→1536→768→384→192→96, stops at 96 < 128 → dimension mismatch
- Fix: detect `r4_size < intermediate_size`, reshape to blocks first, apply Hadamard per block

**Three Rotation Matrix Types:**
| Type | Init | Must persist matrix |
|------|------|-------------------|
| Hadamard | `scipy.linalg.hadamard(N)` deterministic | No, just save rotation_size |
| QuaRot random | `random ±1 diagonal × Hadamard` | Yes (float64, N×N) |
| SpinQuant trained | Cayley/Stiefel optimization | Yes (float64, N×N) |

**Rotation Config Persistence (designed, not implemented):**
- `export/utils.py` line 374: `clean_list` actively REMOVES `rotation_configs` — must fix
- `inference/convert_model.py` line 819-835: already has framework to read rotation_config and apply hooks
- Core invasiveness: ~26 lines of auto-round changes + 1 new file (~100 lines)

**Rotation is scheme-agnostic:** Works with any quantization scheme (MXFP4, W4A16 INT4, NVFP4, FP8, etc.) because rotation executes at Phase 4.5 (before quantization) and hooks are preserved by WrapperLinear/WrapperWALayer.

**No custom rotation kernels exist** — neither Quark nor auto-round has CUDA/Triton kernels for Hadamard. Both use pure PyTorch butterfly O(N log N) algorithm.

**R1-R4 Fusion Summary:**
| Rotation | Offline fuseable? | Needs inference hook? |
|----------|------------------|----------------------|
| R1 offline | ✅ fully | ❌ |
| R1 online | weight side fused | ✅ forward_pre_hook |
| R2 | ✅ fully (V out + O in) | ❌ |
| R3 | ❌ (after RoPE) | ✅ monkeypatch |
| R4 | down_proj input fused | ✅ forward_pre_hook on down_proj |

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
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - R3 monkeypatch + R4 hook registration
  - **Latest fix:** R4 block rotation for r4_size < intermediate_size (lines 118-220+)
  - `register_spinquant_hooks()` and `remove_spinquant_hooks()`

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()` (line ~220)
  - `_apply_online_r1()` (line ~642): hook-based online R1
  - `_make_online_r1_hook()`: static method for rotation hooks
  - `_fuse_offline_rotations()` (line ~735): offline R1 path
  - `SpinQuantConfig` dataclass (line 63): r1/r2/r3/r4/rotation_size/online_r1_rotation/trainable_rotation/trainable_smooth
  - `r4_rotation_size = config.rotation_size or intermediate_size` (line 181)

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - `get_hadamard_K()`, `matmul_hadU()`, `rotate_in_channels_()`, `apply_hadamard_to_linear()`
  - `InputRotationWrapperHadamard` (utility only, not used in main flow)
  - `deterministic_hadamard_matrix()` returns NORMALIZED H/√N

- `/data/lkk/quarot/auto-round/auto_round/wrapper.py`
  - `WrapperLinear` (line 62): auto-round's quantization wrapper
  - Line 503-518: forward() runs orig_layer._forward_pre_hooks — WHY hooks work
  - Line 554-558: WrapperWALayer steals hooks from orig_layer
  - Line 768-769: `wrapper_block()` type(m) in SUPPORTED_LAYER_TYPES

- `/data/lkk/quarot/auto-round/auto_round/export/utils.py`
  - Line 374: `clean_list = ("supported_types", "quant_block_list", "rotation_configs")` — REMOVES rotation_configs from save
  - Line 243-245: saves quantization_config.json

- `/data/lkk/quarot/auto-round/auto_round/compressors_new/base.py`
  - Line 102: `SerializedCompressorConfig.rotation_configs` field exists
  - Line 147: `self.rotation_configs` stored
  - Line 845-876: `_apply_rotations()` at Phase 4.5
  - Line 1113-1190: `save_quantized()` serialization

- `/data/lkk/quarot/auto-round/auto_round/inference/convert_model.py`
  - Line 819-835: existing rotation_config loading framework (uses experimental.transform)

- `/data/lkk/quarot/auto-round/examples/test_autoround_rotation_mxfp4.py`
  - Existing MXFP4 test script with rotation + quantization evaluation

- `/data/lkk/quarot/auto-round/docs/` — 9 analysis documents covering architecture, save/load, scheme compatibility

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - `QKRotationWrapper`: R3 injection after RoPE
  - Architecture-generic implementation
</important_files>

<next_steps>
Remaining work:
- Create test script for W4A16 INT4 + rotation combinations (R1, R1+R2, R1+R2+R3, R1+R2+R3+R4) — user explicitly requested
- Create test script for NVFP4 + rotation combinations — user explicitly requested
- MXFP4 tests already exist in `test_autoround_rotation_mxfp4.py`
- Run full comparison (R1, R1+R2, R1+R2+R3, R1+R2+R3+R4) with hook-based approach across all schemes
- Implement rotation config persistence (save rotation_config.json, load and re-register hooks) — design complete, implementation deferred per user
- Contribute minimal abstraction points to auto-round's compressors_new (longer term)

Immediate next steps:
- Create `examples/test_autoround_rotation_schemes.py` that tests W4A16 INT4 and NVFP4 with different rotation level combinations
- Model the script on existing `test_autoround_rotation_mxfp4.py` structure
- For W4A16: use RTN mode (iters=0 or similar), no calibration needed
- For NVFP4: needs calibration for input global scale computation
- User will run their own tests for MXFP4 R1+R2+R3+R4 with the R4 fix

Open questions:
- User's preference on test script structure (single script with scheme arg vs separate scripts)
- Whether to include SignRound (iters>0) testing or just RTN
</next_steps>