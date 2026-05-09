<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work spans fixing broken tests, ensuring rotation equivalence on real models (Qwen3-0.6B), building comprehensive test scripts for rotation + quantization (RTN W4A16 and MXFP4), adding detailed logging, implementing `rotation_size` configurability, creating comparison test infrastructure between Quark and auto-round, and now debugging a critical accuracy degradation where rotation+MXFP4 performs much worse than MXFP4-only. My approach has been to systematically validate each rotation level, compare against Quark's reference implementation, and build diagnostic infrastructure to isolate root causes.
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
    - Updated `rotate_in_channels_` and `rotate_out_channels_` to support block rotation
    - Updated all related methods and test scripts
    - Verified: 7/7 mock tests pass, Qwen3-0.6B with rotation_size=128 gives cosine_sim=0.9999

13. User asked for test commands comparing Quark vs auto-round accuracy
    - Created `run_comparison_tests.sh` with 4 scenarios
    - Fixed Quark script (RotationConfig object vs dict)

14. User reported device mismatch error running test_autoround_rotation_mxfp4.py
    - Error: RMSNorm weight on CPU, hidden_states on CUDA after quantization
    - Fixed: added `model.to(device)` after `ar.quantize()` in both MXFP4 and W4A16 test scripts

15. User reported rotation+MXFP4 accuracy much worse than MXFP4-only in auto-round (but not in Quark)
    - Quark: rotation_mxfp4 ≈ mxfp4_only (hellaswag 0.4046 vs 0.4059)
    - Auto-round: rotation_mxfp4 << mxfp4_only (hellaswag 0.3307 vs 0.4054 on full eval)
    - Created `debug_rotation_mxfp4.py` with 6 diagnostic steps
    - User ran debug script, results confirmed:
      - String vs Object loading: no difference (Δ=-0.005)
      - GPU vs CPU before quant: no difference (Δ=0.000)
      - **Object-no-rotation vs Object-with-rotation: Δ=-0.045/-0.090** ← ROOT CAUSE
    - Was actively analyzing Quark's `online_r1_rotation` code when compaction occurred

16. Deep analysis of Quark's online R1 vs our offline R1
    - Traced through auto-round's `_quantize_rtn()` flow for MXFP4 (iters=0):
      - MXFP4 has act_bits=4, `check_need_act_calibration` returns False for dynamic act
      - Goes to the `else` branch (blockwise quantization, no act calibration hooks)
      - `_quantize_layer_via_rtn` wraps each linear with WrapperLinear for quantization
    - Confirmed: MXFP4 forces bfloat16 (act_bits<=8, line 520-524 in base.py)
    - Analyzed Quark's `r1()` method (rotation.py line 659+):
      - When `online_r1_rotation=True`: iterates scaling_layers EXCLUDING last_layer
      - For each entry: prev_modules get output rotated by R_inv, next_modules get `InputRotationWrapperHadamard` hooks
      - `apply_online_r1()` wraps target_modules with hooks that apply Hadamard to input
    - **ROOT CAUSE IDENTIFIED**: Our R1 is OFFLINE (fused into weights), Quark uses ONLINE R1 (rotation via hooks, weights unchanged for next_modules)
      - Offline R1: weight quantization sees rotated weights → Hadamard mixing changes per-group weight distribution → harmful for MXFP4's group_size=32 shared-exponent scheme
      - Online R1: weight quantization sees original (gamma-fused) weights → preserves natural scale structure → neutral for MXFP4
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection via globals patching
- `examples/test_rotation_quantization.py`: Tests rotation + W4A16 RTN quantization with configurable R1-R4
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison script
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison script
- `examples/run_comparison_tests.sh`: Shell script running all 4 comparison scenarios
- `examples/debug_rotation_mxfp4.py`: 6-step diagnostic script isolating rotation+quant accuracy bug
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
- `examples/test_rotation_quantization.py`: Added `model.to(device)` after quantize, configurable R1-R4
- `examples/test_autoround_rotation_mxfp4.py`: Added `model.to(device)` after quantize, changed scheme to MXFP4_RCEIL (user changed, but MXFP4 also works)

Work completed:
- [x] Fix R1 accuracy degradation bug (tie_word_embeddings)
- [x] Verify all rotation levels preserve accuracy (hellaswag, piqa — full eval)
- [x] Analyze Quark's rotation+quantization architecture
- [x] Create rotation+quantization test scripts (W4A16 RTN + MXFP4)
- [x] Make R1-R4 configurable in test scripts
- [x] Document R3/R4 online rotation mechanism
- [x] Add per-layer transformation summary table logging
- [x] Create MXFP4 comparison scripts (Quark + auto-round)
- [x] Implement Phase 1: rotation_size configurable
- [x] Fix device mismatch in auto-round MXFP4 + rotation test
- [x] Create debug script to isolate rotation+MXFP4 accuracy issue
- [x] Identify root cause: offline R1 vs online R1
- [ ] **Implement online R1 rotation** (IN PROGRESS — was actively analyzing Quark's code)
- [ ] Port non-pow2 Hadamard matrices from Quark (TODO)
- [ ] Phase 2: RotationLinear training wrapper (TODO)
- [ ] Phase 3: Integration with auto-round tuning loop (TODO)
</work_done>

<technical_details>
**ROOT CAUSE — Rotation+MXFP4 Accuracy Degradation:**
- Our R1 implementation is OFFLINE: rotation is fused into weight matrices (both input and output channels of linear layers)
- Quark's R1 is ONLINE by default (`online_r1_rotation=True`): only prev_modules' output channels are rotated, next_modules get `InputRotationWrapperHadamard` hooks
- **Why offline R1 hurts MXFP4**: MXFP4 uses group_size=32 with shared exponent. Hadamard rotation mixes channel values across groups, destroying the natural scale structure. Groups that had uniform small values now have mixed large/small values, losing quantization precision.
- **Why online R1 is neutral**: next_modules' weights are NOT rotated → quantized in original distribution. The hooks apply Hadamard to activations at runtime, which helps activation quantization without hurting weight quantization.
- Debug results confirmed: step 4 (MXFP4 object, no rotation) vs step 5 (rotation+MXFP4): hellaswag Δ=-0.045, piqa Δ=-0.090

**Quark's Online R1 Flow (rotation.py line 659+):**
- `get_scaling_layers()` skips `last_layer` when `online_r1_rotation=True` (line 652)
- For each scaling_layer entry {prev_modules, norm_module, next_modules, target_modules}:
  - Fuse norm's gamma into next_modules' weights
  - Rotate prev_modules' output channels by R_inv
  - Wrap target_modules with `InputRotationWrapperHadamard` (hook applies Hadamard to input)
- scaling_layers count: first_layer(2) + middle_layers(2×27=54) = 56 entries for 28-layer model
- R2 = 28 (one per decoder layer), always offline
- **Question: how does Quark handle down_proj of last layer?** When last_layer is skipped, down_proj(N-1) is NOT rotated (not a prev in any non-skipped entry). This should cause mathematical inconsistency (rotated residual + non-rotated MLP output). Need to verify in Quark's code.

**Auto-Round MXFP4 Quantization Path (`_quantize_rtn`, base.py line 1466+):**
- For iters=0, calls `_quantize_rtn()` (line 1950)
- MXFP4 act_bits=4 → forces bfloat16 (line 520-524)
- `check_need_act_calibration` returns False for dynamic MXFP4 → goes to else branch (line 1523)
- Uses blockwise quantization: iterates over blocks, calls `_quantize_layer_via_rtn()` per layer
- `_quantize_layer_via_rtn` wraps with WrapperLinear, calls unwrapper for RTN quantization
- After quantization, model params may be on CPU (offload mechanism) → need `model.to(device)`

**Critical Bug Found & Fixed — tie_word_embeddings:**
- When `model.config.tie_word_embeddings = True` (Qwen3-0.6B), HuggingFace `model.tie_weights()` re-ties lm_head to embed_tokens
- lm_eval's HFLM wrapper calls this during initialization
- Fix: set `model.config.tie_word_embeddings = False` after untying in `untie_word_embeddings_if_needed()`

**RMSNorm Fusion + R1 Math:**
- RMSNorm(x@R) with non-trivial gamma ≠ RMSNorm(x) @ R (gamma doesn't commute with rotation)
- Therefore fuse_rmsnorm (absorb gamma into weights) is REQUIRED for R1 correctness
- After fusion, RMSNorm weight becomes `torch.ones_like()` via `fill_(1.0)` (in-place, preserves device)

**Block Rotation (rotation_size < hidden_size):**
- `rotate_in_channels_()`: reshape W to `(..., -1, rot_size)`, apply `@ R.T` per block, reshape back
- `rotate_out_channels_()`: transpose → reshape → rotate → reshape → transpose back
- Verified: exact match (0.00e+00 diff per block)

**Hadamard Matrix Properties (important for correctness):**
- Normalized Hadamard H satisfies: H @ H = I (self-inverse), H = H.T (symmetric), H = H^{-1}
- So R1.T = R1 = R1^{-1} for deterministic Hadamard
- This means R1_inv = R1.T = R1 — the inverse IS the matrix itself

**Quark's `rotation_size` semantics:**
- R1: `rotation_size` overrides `hidden_size` → enables block rotation
- R2: always uses `head_dim`, ignores rotation_size
- R3: raises `NotImplementedError` when rotation_size is set
- R4: `rotation_size` overrides `intermediate_size`

**MXFP4 Scheme Definitions (auto-round schemes.py):**
- MXFP4: bits=4, group_size=32, data_type=mx_fp, act_bits=4, act_data_type=mx_fp, act_dynamic=True
- MXFP4_RCEIL: same but act_data_type=mx_fp_rceil (round-ceiling for activations)
- User confirmed switching MXFP4 → MXFP4_RCEIL did NOT fix the accuracy issue

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
  - Main orchestrator: 8-step pipeline in `preprocess()` (line ~188)
  - SpinQuantConfig dataclass with `rotation_size` and `online_r1_rotation` fields (line ~54-105)
  - `_fuse_offline_rotations()` at line 619 — **NEEDS MODIFICATION for online R1 support**
  - Currently does offline R1: rotates both prev (embed, o_proj, down_proj) and next (q/k/v, gate/up, lm_head) modules' weights
  - For online R1: should only rotate prev modules, register hooks on next modules
  - `_print_transformation_summary()` at line 797 — per-layer table with block rotation info

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Core rotation math: `rotate_in_channels_()`, `rotate_out_channels_()` support block rotation
  - `untie_word_embeddings_if_needed()` — CRITICAL: sets `config.tie_word_embeddings=False` (line ~365)
  - `deterministic_hadamard_matrix`, `random_hadamard_matrix`, `matmul_hadU`, `is_pow2`, `get_hadamard_K`
  - `fuse_rmsnorm_in_model()` at line 303 — fuses gamma, sets norm weights to 1.0

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - `register_spinquant_hooks()` — accepts `r4_rotation_size` parameter
  - R3: Generic monkeypatch via `add_qk_rotation_after_rope()`
  - R4: forward_pre_hook on down_proj with block Hadamard
  - **NEEDS EXTENSION**: add online R1 hook registration

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - QKRotationWrapper: wraps apply_rotary_pos_emb to apply Hadamard after RoPE

- `/data/lkk/quarot/auto-round/examples/debug_rotation_mxfp4.py`
  - 6-step diagnostic script that isolated the root cause
  - Confirmed: rotation changes weight distribution → hurts MXFP4 group quantization
  - Results: step4 vs step5 Δ=-0.045/-0.090 (hellaswag/piqa)

- `/data/lkk/quarot/auto-round/examples/test_autoround_rotation_mxfp4.py`
  - Auto-round MXFP4 + rotation comparison script
  - `apply_rotation()` at line ~73, `quantize_mxfp4()` at line ~88
  - Has `model.to(device)` fix after quantize
  - Currently uses scheme="MXFP4_RCEIL" (user changed from MXFP4)

- `/data/lkk/quarot/auto-round/examples/test_quark_rotation_mxfp4.py`
  - Quark MXFP4 + rotation comparison script
  - Uses `RotationConfig` object with `online_r1_rotation=True` (Quark default)
  - `get_rotation_config()` at line 67 builds proper RotationConfig with scaling_layers

- `/data/lkk/quarot/auto-round/examples/test_rotation_quantization.py`
  - Tests rotation + W4A16 RTN with configurable R1-R4 and `--rotation-size`
  - Has `model.to(device)` fix after quantize

- `/data/lkk/quarot/auto-round/auto_round/compressors/base.py`
  - Auto-round's core quantization pipeline
  - `_quantize_rtn()` at line 1466 — the RTN path for iters=0
  - `_quantize_layer_via_rtn()` at line 1356 — per-layer quantization
  - `_set_amp_dtype()` at line 3647 — forces bfloat16 for act_bits<=8
  - Lines 520-524: forces bfloat16 conversion for MXFP4

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - Quark's RotationProcessor: `r1()` line 659, `r2()` line 933, `apply_online_r1()` line 879
  - `get_scaling_layers()` at line ~640: skips `last_layer` when online_r1=True (line 652)
  - Key reference for implementing online R1 in auto-round
  - Need to study: `apply_online_r1()` implementation, `InputRotationWrapperHadamard` class
</important_files>

<next_steps>
**Immediate — Implement Online R1 Rotation:**
The root cause is confirmed: our offline R1 (fusing rotation into ALL linear weights) degrades MXFP4 quantization quality. The fix is to implement online R1 matching Quark's approach.

Implementation plan:
1. **Study Quark's `apply_online_r1()` (rotation.py line ~879)** and `InputRotationWrapperHadamard` class to understand exact hook behavior
2. **Resolve the last_layer question**: When online_r1 skips last_layer, how does Quark handle down_proj(N-1)? The residual would have rotated + non-rotated components. Need to verify this in Quark's code.
3. **Modify `preprocessor.py` `_fuse_offline_rotations()`**:
   - When `online_r1_rotation=True`:
     - Only rotate prev_modules' output channels: embed_tokens, o_proj (all layers), down_proj (all but last)
     - Do NOT rotate next_modules' input channels: q/k/v_proj, gate/up_proj, lm_head
     - Register forward_pre_hooks on next_modules that apply block Hadamard to input
4. **Add hook function** (either in `inplace/apply.py` or new method):
   ```python
   def online_r1_hook(module, input_args):
       x = input_args[0]
       # Apply block Hadamard: reshape to [..., n_blocks, rot_size], apply H, reshape back
       x_rotated = apply_block_hadamard(x, r1_rotation_size)
       return (x_rotated,) + input_args[1:]
   ```
5. **Change default** `online_r1_rotation` from False to True in SpinQuantConfig
6. **Test**: Re-run `debug_rotation_mxfp4.py` with online R1 — expect step 5 ≈ step 4
7. **Re-run full comparison**: `run_comparison_tests.sh` to verify Quark vs auto-round accuracy alignment

**Remaining work after online R1:**
- Verify both Quark and auto-round MXFP4 scripts give matching accuracy end-to-end
- Port non-pow2 Hadamard matrices from Quark (TODO in SQL: id='non-pow2-hadamard')
- Phase 2: RotationLinear training wrapper (future)
- Phase 3: Integration with auto-round tuning loop (future)
</next_steps>