<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work spans fixing broken tests, ensuring rotation equivalence on real models (Qwen3-0.6B), debugging critical accuracy degradation bugs, building comprehensive test scripts for rotation + quantization (RTN W4A16 and MXFP4), adding detailed logging, implementing `rotation_size` configurability, and creating comparison test infrastructure between Quark and auto-round. My approach has been to systematically validate each rotation level, compare against Quark's reference implementation, and build test infrastructure for accuracy verification.
</overview>

<history>
1. User asked to fix broken test cases, resolve training code duplication in preprocessor.py/trainer.py, and expand validation for R1-R4
   - Explored both codebases and reference implementations
   - Fixed device mismatches, R4 hook signature, added `apply_hadamard_to_linear()`
   - Rewrote `_fuse_offline_rotations()` with proper R2/R4 fusion
   - Fixed all example files, created comprehensive `test_rotation_levels.py`
   - All 7 mock tests pass

2. User asked to validate with Qwen3-0.6B using lm_eval
   - Created `test_qwen3_rotation_eval.py` evaluation script
   - Fixed R4 crash (intermediate_size=3072 not power-of-2 → block Hadamard)
   - Fixed R3 (must be AFTER RoPE, not on q_proj output)
   - Implemented R3 as architecture-generic monkey-patch of attention forward
   - Verified all levels: cosine_sim >0.99996, 100% argmax agreement

3. User asked to avoid `fast_hadamard_transform`, use auto-round's existing implementations, add Quark-style logging
   - Implemented `matmul_hadU` in rotation_utils.py (butterfly algorithm)
   - Created `monkeypatch.py` with architecture-generic R3
   - Rewrote `inplace/apply.py` with generic R3, proper logging, dimension validation
   - Updated `preprocessor.py` with Python logging module throughout

4. User reported R1 accuracy degradation (hellaswag -14%) — investigated root cause
   - **Found root cause:** `untie_word_embeddings_if_needed()` cloned lm_head.weight but didn't set `model.config.tie_word_embeddings = False`
   - lm_eval's HFLM wrapper calls `model.tie_weights()` which RE-TIES weights, overwriting the rotated lm_head
   - **Fix:** Added `model.config.tie_word_embeddings = False` after untying
   - Verified: all rotation levels (R1 through R1+R2+R3+R4) show 0% accuracy degradation on full piqa+hellaswag

5. User asked about Quark's rotation + quantization (PTQ) flow
   - Analyzed `quantize_quark.py`, `RotationProcessor`, `ModelQuantizer`
   - Documented architecture: rotation is preprocessing before calibration/quantization

6. User asked to create rotation + RTN W4A16 quantization test
   - Created `test_rotation_quantization.py` testing baseline_fp16, rtn_only, rotation_rtn
   - Made R1-R4 individually configurable via CLI flags

7. User asked for documentation on R3/R4 online rotation mechanism
   - Created comprehensive markdown: `docs/r3_r4_online_rotation.md`

8. User asked to add per-layer transformation summary table to logging
   - Added `_print_transformation_summary()` to preprocessor.py
   - Fixed logging visibility (added `logging.basicConfig()` to test scripts)

9. User asked about Hadamard matrix size constraints and Quark's approach
   - Quark has pre-stored non-pow2 Hadamard matrices (12,20,28,36,40,52,60,108,140,156,172)
   - Our approach: only uses pow2, does block Hadamard with largest pow2 factor
   - Recorded as TODO for future improvement

10. User asked to create MXFP4 comparison scripts for Quark vs auto-round
    - Created `test_quark_rotation_mxfp4.py` and `test_autoround_rotation_mxfp4.py`
    - Both default to R1+R2 only (matching Quark's default)

11. User asked about Quark's rotation_size semantics and training compatibility
    - Analyzed: R1 uses rotation_size, R2 always head_dim, R3 raises NotImplementedError, R4 uses rotation_size
    - Created training compatibility plan (3 phases)

12. User asked to implement Phase 1: rotation_size configurable
    - Added `rotation_size: Optional[int] = None` to SpinQuantConfig with validation
    - Resolved effective sizes: `r1_rotation_size`, `r2_rotation_size`, `r4_rotation_size`
    - Updated `rotate_in_channels_` and `rotate_out_channels_` to support block rotation (reshape → per-block rotate → flatten)
    - Updated `_validate_dimensions()`, `_init_rotation_matrices()`, `_fuse_offline_rotations()`, `_fuse_r4_rotation()`
    - Updated `_print_transformation_summary()` to show block rotation info
    - Updated `inplace/apply.py` R4 hook to use `r4_rotation_size`
    - Added `--rotation-size` CLI arg to all test scripts
    - Verified: 7/7 mock tests pass, Qwen3-0.6B with rotation_size=128 gives cosine_sim=0.9999

13. User asked for test commands comparing Quark vs auto-round accuracy
    - Created `run_comparison_tests.sh` with 4 scenarios (MXFP4 r128, MXFP4 full, W4A16 RTN, rotation-only)
    - Supports `--scenario`, `--limit`, `--dry-run` flags

14. User ran the scripts and got `AttributeError: 'dict' object has no attribute 'name'` in Quark script
    - Root cause: passed raw dict instead of `RotationConfig` object to Quark's `algo_configs`
    - Fixed: replaced `get_rotation_config_dict()` with `get_rotation_config()` returning proper `RotationConfig` object
    - Added `target_modules` to scaling_layers (needed for online R1)

15. User asked why Quark logs show "R1 Rotation: 56" and "R2 Rotation: 28"
    - Explained: R1 iterates over `scaling_layers` which has 2 sub-patterns per layer × 28 layers = 56, but `last_layer` is SKIPPED when `online_r1_rotation=True` (line 652 in rotation.py)
    - R2 = 28 = one per decoder layer

16. User reported device mismatch error running `test_autoround_rotation_mxfp4.py`
    - Error: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
    - Traceback shows error in `input_layernorm` during lm_eval inference AFTER rotation+quantization
    - Error at `self.weight * hidden_states.to(input_dtype)` in RMSNorm — the `self.weight` is on CPU while `hidden_states` is on CUDA
    - This happens because after fuse_rmsnorm, the RMSNorm weight is set to `torch.ones_like()` but auto-round's quantization pipeline may move layers between devices
    - **Was actively debugging this when compaction occurred**
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection via globals patching
- `examples/test_rotation_quantization.py`: Tests rotation + W4A16 RTN quantization with configurable R1-R4
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison script
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison script
- `examples/run_comparison_tests.sh`: Shell script running all 4 comparison scenarios
- `docs/r3_r4_online_rotation.md`: Detailed documentation on R3/R4 online rotation mechanism
- Session files: `training_compatibility_plan.md` (Phase 1-3 plan)

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`:
  - Added `is_pow2()`, `get_hadamard_K()`, `matmul_hadU()` (butterfly Hadamard)
  - **CRITICAL FIX:** `untie_word_embeddings_if_needed()` now sets `model.config.tie_word_embeddings = False`
  - Updated `rotate_in_channels_()` and `rotate_out_channels_()` to support block rotation
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`: Complete rewrite — generic R3 via monkeypatch, R4 with validation, added `r4_rotation_size` parameter
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - All `print` → `logger`, added dimension validation, rich logging
  - Added `rotation_size: Optional[int] = None` to SpinQuantConfig
  - Resolved effective rotation sizes: `r1_rotation_size`, `r2_rotation_size`, `r4_rotation_size`
  - Updated `_validate_dimensions()`, `_init_rotation_matrices()`, `_fuse_offline_rotations()`, `_fuse_r4_rotation()`
  - Added `_print_transformation_summary()` — per-layer table of all transforms and hooks
- `examples/test_qwen3_rotation_eval.py`: Added `logging.basicConfig()`, `--rotation-size` arg
- `examples/test_quark_rotation_mxfp4.py`: Fixed to use `RotationConfig` object instead of dict, added `--rotation-size`

Work completed:
- [x] Fix R1 accuracy degradation bug (tie_word_embeddings)
- [x] Verify all rotation levels preserve accuracy (hellaswag, piqa — full eval)
- [x] Analyze Quark's rotation+quantization architecture
- [x] Create rotation+quantization test script (W4A16 RTN)
- [x] Make R1-R4 configurable in test scripts
- [x] Document R3/R4 online rotation mechanism
- [x] Add per-layer transformation summary table logging
- [x] Fix logging visibility (basicConfig)
- [x] Create MXFP4 comparison scripts (Quark + auto-round)
- [x] Implement Phase 1: rotation_size configurable
- [x] Fix Quark test script (RotationConfig object vs dict)
- [x] Create run_comparison_tests.sh
- [ ] Fix device mismatch in auto-round MXFP4 + rotation test (IN PROGRESS)
- [ ] Port non-pow2 Hadamard matrices from Quark (TODO)
- [ ] Phase 2: RotationLinear training wrapper (TODO)

Current state:
- All mock tests pass (7/7 in test_rotation_levels.py)
- Real model rotation-only evaluation works perfectly (all levels, Δ ≤ 0.005)
- Quark MXFP4 script: RotationConfig fix applied but not yet verified end-to-end
- Auto-round MXFP4 script: **DEVICE ERROR** — RMSNorm weight on CPU after quantization pipeline
</work_done>

<technical_details>
**Critical Bug Found & Fixed — tie_word_embeddings:**
- When `model.config.tie_word_embeddings = True` (Qwen3-0.6B), HuggingFace `model.tie_weights()` re-ties lm_head to embed_tokens
- lm_eval's HFLM wrapper calls this during initialization
- After rotation, embed_tokens has `W @ R1` but lm_head has `(W * γ_final) @ R1` — must be different
- Fix: set `model.config.tie_word_embeddings = False` after untying

**RMSNorm Fusion + R1 Math:**
- RMSNorm(x@R1) with non-trivial gamma ≠ RMSNorm(x) @ R1 (gamma doesn't commute with rotation)
- Therefore fuse_rmsnorm (absorb gamma into weights) is REQUIRED for R1 correctness
- After fusion, RMSNorm weight becomes `torch.ones_like()` — this is key to the current device bug

**Block Rotation (rotation_size < hidden_size):**
- `rotate_in_channels_()`: reshape W to `(..., -1, rot_size)`, apply `@ R.T` per block, reshape back
- `rotate_out_channels_()`: transpose → reshape → rotate → reshape → transpose back
- Embedding: same block rotation pattern on last dim
- Verified: exact match (0.00e+00 diff per block)

**Quark's `rotation_size` semantics:**
- R1: `rotation_size` overrides `hidden_size` → enables block rotation
- R2: always uses `head_dim`, ignores rotation_size
- R3: raises `NotImplementedError` when rotation_size is set
- R4: `rotation_size` overrides `intermediate_size`
- When `rotation_size < hidden_size` + `online_r1=True`: Quark wraps each linear with `InputRotationWrapperHadamard`

**Quark's online R1 iteration count (56 for 28-layer model):**
- `scaling_layers` is flattened: first_layer(2) + middle_layers(2×27=54) = 56
- `last_layer` (lm_head) is SKIPPED when `online_r1_rotation=True` (rotation.py line 652)
- R2 = 28 (one per decoder layer)

**Quark RotationConfig vs dict:**
- Quark's `_apply_advanced_quant_algo()` accesses `config.algo_config[i].name`
- Must pass `RotationConfig(...)` object, NOT a raw dict
- `RotationConfig` needs `scaling_layers` with `target_modules` for online R1

**Current Device Bug (ACTIVE):**
- Error: `self.weight * hidden_states` in RMSNorm — weight on CPU, hidden_states on CUDA
- Happens during lm_eval inference AFTER rotation + MXFP4 quantization
- Likely cause: auto-round's quantization pipeline moves model layers between CPU/CUDA during quantization, and the RMSNorm `weight` (set to ones after fusion) doesn't get moved back
- The quantization pipeline uses `device_map` which may not handle the fused RMSNorm properly

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- auto-round installed editable: `pip install -e .` in /data/lkk/quarot/auto-round
- Quark at /data/lkk/quarot/Quark, uses sys.path.insert

**Qwen3-0.6B Architecture:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q heads, 8 kv heads)
- intermediate_size=3072 (NOT power-of-2 → R4 uses block Hadamard K=1024, blocks=3)
- Has q_norm and k_norm (per-head RMSNorm after q/k projection)
- tie_word_embeddings=True in config
- 28 transformer layers, model_type="qwen3"
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()`
  - SpinQuantConfig dataclass with `rotation_size` field (line ~54-100)
  - Resolved sizes: `r1_rotation_size`, `r2_rotation_size`, `r4_rotation_size` in `__init__`
  - `_validate_dimensions()` checks rotation_size divides target dimension
  - `_init_rotation_matrices()` uses resolved sizes
  - `_fuse_offline_rotations()` handles block embedding rotation when r1_rotation_size < hidden_size
  - `_print_transformation_summary()` shows per-layer table with block rotation info

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Core rotation math: `rotate_in_channels_()`, `rotate_out_channels_()` now support block rotation
  - `untie_word_embeddings_if_needed()` — CRITICAL: sets `config.tie_word_embeddings=False` (line ~365)
  - `deterministic_hadamard_matrix`, `random_hadamard_matrix`, `matmul_hadU`, `is_pow2`, `get_hadamard_K`

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - `register_spinquant_hooks()` — accepts `r4_rotation_size` parameter
  - R3: Generic monkeypatch via `add_qk_rotation_after_rope()`
  - R4: forward_pre_hook on down_proj with block Hadamard

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - QKRotationWrapper: wraps apply_rotary_pos_emb to apply Hadamard after RoPE

- `/data/lkk/quarot/auto-round/examples/test_autoround_rotation_mxfp4.py`
  - Auto-round MXFP4 + rotation comparison script
  - **CURRENTLY BROKEN** — device mismatch in RMSNorm after quantization
  - `apply_rotation()` at line ~73 accepts `rotation_size` parameter
  - `quantize_mxfp4()` uses `AutoRound(scheme="MXFP4", iters=0)`

- `/data/lkk/quarot/auto-round/examples/test_quark_rotation_mxfp4.py`
  - Quark MXFP4 + rotation comparison script
  - Fixed: uses `RotationConfig` object (not dict) — `get_rotation_config()` at line ~67
  - `quantize_with_quark_mxfp4()` builds proper RotationConfig with `scaling_layers` including `target_modules`

- `/data/lkk/quarot/auto-round/examples/run_comparison_tests.sh`
  - Shell script running 4 comparison scenarios with `--scenario`, `--limit`, `--dry-run`

- `/data/lkk/quarot/auto-round/examples/test_rotation_quantization.py`
  - Tests rotation + W4A16 RTN with configurable R1-R4 and `--rotation-size`

- `/data/lkk/quarot/auto-round/examples/test_qwen3_rotation_eval.py`
  - Real model lm_eval evaluation of rotation levels with `--rotation-size`

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - Quark's RotationProcessor: `r1()` line 659, `r2()` line 933, `apply_online_r1()` line 879
  - Key: `get_scaling_layers()` line 571 — skips `last_layer` when online_r1=True (line 652)
  - `InputRotationWrapperHadamard` used for online R1 inference

- `/data/lkk/quarot/Quark/quark/torch/quantization/config/config.py`
  - `RotationConfig` class at line 2084 — proper config object with `.name` attribute
  - Required fields: backbone, model_decoder_layers, v_proj, o_proj, self_attn, mlp, scaling_layers
</important_files>

<next_steps>
**Immediate — Fix device mismatch bug:**
- Error: RMSNorm weight on CPU, hidden_states on CUDA during lm_eval after rotation + MXFP4
- Root cause: after `fuse_rmsnorm`, weight becomes `torch.ones_like()`. When auto-round's quantization pipeline offloads/moves layers, this RMSNorm weight may end up on CPU
- Possible fixes:
  1. After quantization completes, ensure all model parameters are on the correct device
  2. Check if auto-round's `AutoRound` with `device_map=device` properly handles all submodules
  3. May need to explicitly `.to(device)` the model after `ar.quantize()`
  4. Check if the issue is specific to passing a pre-rotated model object (vs model name string) to AutoRound

**Remaining work:**
- Verify both Quark and auto-round MXFP4 scripts run end-to-end and compare accuracy
- Run the full comparison test suite (`run_comparison_tests.sh`)
- Port non-pow2 Hadamard matrices from Quark (TODO in SQL: id='non-pow2-hadamard')
- Phase 2: RotationLinear training wrapper (future)
- Phase 3: Integration with auto-round tuning loop (future)
</next_steps>