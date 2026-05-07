<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work spans fixing broken tests, ensuring rotation equivalence on real models (Qwen3-0.6B), debugging critical accuracy degradation bugs, building comprehensive test scripts for rotation + quantization (RTN W4A16 and MXFP4), adding detailed logging, and now planning training compatibility improvements. My approach has been to systematically validate each rotation level, compare against Quark's reference implementation, and build test infrastructure for accuracy verification.
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
   - Script was running successfully

7. User asked to make R1-R4 individually configurable in test_rotation_quantization.py
   - Added `--r1/--no-r1`, `--r2/--no-r2`, `--r3/--no-r3`, `--r4/--no-r4` CLI flags
   - Updated rotation description and apply_rotation calls to use flags

8. User asked for documentation on how R3/R4 online rotation works during inference and quantization
   - Created comprehensive markdown: `/data/lkk/quarot/auto-round/docs/r3_r4_online_rotation.md`
   - Covers monkeypatch mechanism (R3), forward_pre_hook (R4), half-fusion strategy, quantization interaction

9. User asked to add a per-layer transformation summary table to logging
   - Added `_print_transformation_summary()` to `preprocessor.py`
   - Prints: global transforms, per-layer table (R1-R4 for each layer), registered hooks, totals
   - Fixed generator exhaustion bug (needed `list(self._get_layers())`)
   - Compact format: shows first 2 + last layer + "..." for large models

10. User reported summary table not visible when running test scripts
    - Root cause: Python logging not configured — `logger.info()` goes nowhere without `logging.basicConfig()`
    - Added `logging.basicConfig(level=logging.INFO)` and `logging.getLogger("auto_round.spinquant").setLevel(logging.INFO)` to both `test_qwen3_rotation_eval.py` and `test_rotation_quantization.py`

11. User asked about Hadamard matrix size constraints
    - Explained: must be power-of-2 AND divide the target dimension
    - Compared our approach (block Hadamard with largest pow2 factor) vs Quark's (pre-stored non-pow2 Hadamard matrices: 12,20,28,36,40,52,60,108,140,156,172 + kron product)
    - Our approach is simpler but less effective for non-pow2 dimensions (cross-block outliers not mixed)
    - Recorded as TODO for future improvement

12. User asked to create MXFP4 comparison scripts for Quark vs auto-round
    - Created `test_quark_rotation_mxfp4.py` — uses Quark's `ModelQuantizer` + `LLMTemplate` with mxfp4 scheme
    - Created `test_autoround_rotation_mxfp4.py` — uses auto-round's `AutoRound(scheme="MXFP4", iters=0)`
    - Both default to R1+R2 only (matching Quark's default), with configurable R1-R4 flags
    - Verified Quark imports work, qwen3 template available

13. User asked why Quark defaults to R1+R2 and uses online R1
    - Explained: R3/R4 have runtime overhead; R1+R2 are zero-cost after fusion
    - Quark's online R1 with `rotation_size=128 < hidden_size=1024`: enables block rotation + training compatibility via `InputRotationWrapperHadamard`
    - Updated auto-round MXFP4 test to default R1+R2 (R3/R4 disabled) to match Quark

14. User asked about training compatibility plan and whether rotation_size affects R1 only
    - Analyzed Quark's `rotation_size` usage across R1-R4
    - **R1**: `rotation_size` overrides `hidden_size` — core use case
    - **R2**: always uses `head_dim` from config, NOT `rotation_size`
    - **R3**: explicitly `raise NotImplementedError` when `rotation_size is not None`
    - **R4**: `rotation_size` overrides `intermediate_size` when set
    - Created training compatibility plan document with 3 phases
    - User asked to implement Phase 1 first (rotation_size configurable)
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection via globals patching
- `examples/test_rotation_quantization.py`: Tests rotation + W4A16 RTN quantization with configurable R1-R4
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison script
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison script
- `docs/r3_r4_online_rotation.md`: Detailed documentation on R3/R4 online rotation mechanism
- Session files: `training_compatibility_plan.md` (Phase 1-3 plan)

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`:
  - Added `is_pow2()`, `get_hadamard_K()`, `matmul_hadU()` (butterfly Hadamard)
  - **CRITICAL FIX:** `untie_word_embeddings_if_needed()` now sets `model.config.tie_word_embeddings = False`
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`: Complete rewrite — generic R3 via monkeypatch, R4 with validation
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - All `print` → `logger`, added dimension validation, rich logging
  - Added `_print_transformation_summary()` — per-layer table of all transforms and hooks
- `examples/test_qwen3_rotation_eval.py`: Added `logging.basicConfig()` for visibility

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
- [x] Analyze Quark's rotation_size usage across R1-R4
- [ ] Implement Phase 1: rotation_size configurable in auto-round (PENDING)
- [ ] Port non-pow2 Hadamard matrices from Quark (TODO recorded)

Current state:
- All mock tests pass (7/7 in test_rotation_levels.py)
- All reference tests pass (test_quark_comparison.py, test_reference_equivalence.py)
- Real model evaluation: rotation preserves accuracy perfectly (R1 through R1+R2+R3+R4, Δ ≤ 0.005 on full eval)
- MXFP4 comparison scripts created but not yet run
- Transformation summary table works and displays correctly
</work_done>

<technical_details>
**Critical Bug Found & Fixed — tie_word_embeddings:**
- When `model.config.tie_word_embeddings = True` (Qwen3-0.6B), HuggingFace `model.tie_weights()` re-ties lm_head to embed_tokens
- lm_eval's HFLM wrapper calls this during initialization
- After rotation, embed_tokens has `W @ R1` but lm_head has `(W * γ_final) @ R1` — must be different
- Fix: set `model.config.tie_word_embeddings = False` after untying
- Quark avoids this because they don't use lm_eval's HFLM; they have their own eval path

**RMSNorm Fusion + R1 Math:**
- RMSNorm(x@R1) with non-trivial gamma ≠ RMSNorm(x) @ R1 (gamma doesn't commute with rotation)
- Therefore fuse_rmsnorm (absorb gamma into weights) is REQUIRED for R1 correctness
- Without fusion: cosine_sim drops to 0.12 (completely wrong)

**Quark's `rotation_size` semantics:**
- `rotation_size` is a single config value that affects R1 and R4, but NOT R2 and NOT R3
- R1: `rotation_size` overrides `hidden_size` — enables block rotation (e.g., 128 instead of 1024)
- R2: always uses `head_dim` from model config, ignores `rotation_size`
- R3: raises `NotImplementedError` when `rotation_size is not None`
- R4: `rotation_size` overrides `intermediate_size` when set
- When `rotation_size < hidden_size`, Quark uses online R1 (`InputRotationWrapperHadamard`) because block rotation can't be simply fused into full-dimension weights

**Quark's online R1 vs offline R1:**
- Offline R1: `weight = W @ R1` fused permanently, zero inference cost
- Online R1: `InputRotationWrapperHadamard` wraps each linear, applies Hadamard to activation before matmul
  - Needed when `rotation_size < hidden_size` (block rotation)
  - Enables training compatibility: gradient flows through the wrapper
  - Quantizer can insert QDQ between rotation and linear

**Quark's `RotationLinear`:**
- Training wrapper: holds `rotation_in` and `rotation_out` as `nn.Parameter`
- Forward: online computes `(x @ R_in) @ (W @ R_in)^T` — gradient flows through R
- After training: `post_process_trained_rotation()` fuses R into weights, replaces with `InputRotationWrapperHadamard`
- Supports `smooth_values_in/out` for joint SmoothQuant training (OSTQuant-style)

**Quark MXFP4 scheme:**
- W4A4: weight=static MXFP4, activation=dynamic MXFP4
- `template.get_config(scheme="mxfp4", algorithm=["rotation"], algo_configs=...)` integrates rotation
- Default rotation for Qwen3: R1+R2 only, online_r1=True, rotation_size=128

**Non-pow2 Hadamard — Quark vs auto-round:**
- Quark: pre-stored Hadamard matrices for sizes 12,20,28,36,40,52,60,108,140,156,172
  - Can construct full-dim Hadamard via `kron(H_K, H_{n/K})` for any n that factors
  - E.g., n=3072 = 172 × 16^1 → `kron(H_172, H_16)` = full 3072×3072
- Auto-round: only finds largest pow2 factor K, does block Hadamard
  - E.g., n=3072 → K=1024, 3 independent blocks (no cross-block mixing)
- Improvement TODO recorded in database

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- In newer transformers: `dtype=` is correct (not `torch_dtype=`; the latter still works but deprecated for from_pretrained)
- auto-round installed editable: `pip install -e .` in /data/lkk/quarot/auto-round
- Quark installed at /data/lkk/quarot/Quark, uses sys.path.insert for imports

**Qwen3-0.6B Architecture:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q heads, 8 kv heads)
- intermediate_size=3072 (NOT power-of-2 → R4 uses block Hadamard K=1024, blocks=3)
- Has q_norm and k_norm (per-head RMSNorm after q/k projection)
- tie_word_embeddings=True in config
- 28 transformer layers, model_type="qwen3"
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Core rotation math: rotate_in_channels_, rotate_out_channels_, fuse_rmsnorm_in_model, matmul_hadU
  - CRITICAL FIX at line ~365: untie_word_embeddings_if_needed sets config.tie_word_embeddings=False
  - Contains: deterministic_hadamard_matrix, random_hadamard_matrix, is_pow2, get_hadamard_K, apply_hadamard_to_linear, get_model_arch_info

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()`
  - Key methods: `_fuse_offline_rotations()`, `_validate_dimensions()`, `_init_rotation_matrices()`, `_print_transformation_summary()`
  - SpinQuantConfig dataclass defines all rotation options (r1-r4, trainable, etc.)
  - `_print_transformation_summary()` added: per-layer table showing all transforms and hooks

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - Online hook registration for R3 and R4
  - R3: Generic monkeypatch via `add_qk_rotation_after_rope()`
  - R4: forward_pre_hook on down_proj with block Hadamard

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - Architecture-generic R3 injection: copies forward, patches globals to replace apply_rotary_pos_emb
  - QKRotationWrapper: wraps apply_rotary_pos_emb to apply Hadamard after RoPE

- `/data/lkk/quarot/auto-round/examples/test_rotation_quantization.py`
  - Tests rotation + W4A16 RTN quantization with configurable R1-R4
  - Compares baseline_fp16, rtn_only, rotation_rtn, rotation_fp16

- `/data/lkk/quarot/auto-round/examples/test_quark_rotation_mxfp4.py`
  - Quark-side MXFP4 + rotation comparison script
  - Uses Quark's LLMTemplate, ModelQuantizer, rotation config dict

- `/data/lkk/quarot/auto-round/examples/test_autoround_rotation_mxfp4.py`
  - Auto-round-side MXFP4 + rotation comparison script
  - Uses AutoRound(scheme="MXFP4", iters=0) for RTN mode

- `/data/lkk/quarot/auto-round/examples/test_qwen3_rotation_eval.py`
  - Real model lm_eval evaluation of all rotation levels (R1 through R1+R2+R3+R4)
  - Added logging.basicConfig for visibility

- `/data/lkk/quarot/auto-round/docs/r3_r4_online_rotation.md`
  - Comprehensive documentation on R3/R4 online rotation mechanism

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - Quark's RotationProcessor: r1(), r2(), r3(), r4(), apply(), post_process_trained_rotation()
  - RotationLinear: training-compatible wrapper with rotation_in/rotation_out as nn.Parameter
  - Key reference for rotation_size handling: R1 line 660-663, R3 line 988-989, R4 line 999-1004

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/hadamard.py`
  - Quark's Hadamard implementation with KNOWN_HADAMARD_MATRICES (non-pow2 sizes)
  - `_get_hadamard_K()`, `matmul_hadU()`, `random_hadamard_matrix()`
  - Lines 4269-4281: KNOWN_HADAMARD_MATRICES dict (12,20,28,36,40,52,60,108,140,156,172)

- `/data/lkk/quarot/Quark/examples/torch/language_modeling/rotation/train_rotation.py`
  - Quark's full rotation training + quantization pipeline
  - Shows two-phase: train rotation → post_process → quantize → freeze
</important_files>

<next_steps>
Remaining work:
1. **Implement Phase 1: rotation_size configurable** — User explicitly requested this. Changes needed:
   - Add `rotation_size: int | None = None` to `SpinQuantConfig` in preprocessor.py
   - Modify `_init_rotation_matrices()` to use `rotation_size or hidden_size` for R1
   - Modify `_init_rotation_matrices()` to use `rotation_size or intermediate_size` for R4
   - Handle R3: raise error or skip when rotation_size is set (matching Quark)
   - R2 stays at head_dim (no change)
   - Update `_validate_dimensions()` for new rotation_size
   - Update `_fuse_offline_rotations()` to handle block rotation when rotation_size < hidden_size
   - Update `_print_transformation_summary()` to show rotation_size
   - Update logging messages

2. **Run MXFP4 comparison tests** — Scripts created, user will run them

3. **Phase 2: RotationLinear training wrapper** — Future work, planned
   - Implement RotationLinear (wrap nn.Linear with trainable rotation_in/rotation_out)
   - Training loop with KL loss
   - post_process_trained_rotation() to fuse and convert to inference wrappers

4. **Port non-pow2 Hadamard matrices from Quark** — TODO recorded in SQL database (id='non-pow2-hadamard')

Immediate next action:
- Implement Phase 1 rotation_size support in SpinQuantConfig and preprocessor
- Key insight from analysis: `rotation_size` in Quark affects R1 and R4, NOT R2, and R3 raises NotImplementedError
</next_steps>