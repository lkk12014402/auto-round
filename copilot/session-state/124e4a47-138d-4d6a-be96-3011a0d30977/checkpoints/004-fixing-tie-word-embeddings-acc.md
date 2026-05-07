<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work involves fixing broken tests, ensuring rotation equivalence on real models (Qwen3-0.6B), debugging a critical accuracy degradation bug, and now testing rotation combined with quantization (RTN W4A16). My approach has been to systematically validate each rotation level, identify and fix bugs by comparing against Quark's reference implementation, and build comprehensive test scripts.
</overview>

<history>
1. User asked to fix broken test cases, resolve training code duplication, and expand validation for all rotation levels (R1-R4)
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

4. User reported R1 accuracy degradation (hellaswag -14%) and asked to investigate
   - Initially suspected float16 precision, but testing with float32 showed SAME degradation (16%)
   - Discovered cosine_sim was 0.99997 on single inputs but lm_eval still degraded
   - **Found root cause:** `untie_word_embeddings_if_needed()` cloned lm_head.weight but didn't set `model.config.tie_word_embeddings = False`
   - lm_eval's HFLM wrapper calls `model.tie_weights()` which checks this flag and RE-TIES the weights, overwriting the norm-fused+rotated lm_head with embed_tokens values
   - **Fix:** Added `model.config.tie_word_embeddings = False` after untying
   - Verified: all rotation levels (R1 through R1+R2+R3+R4) now show 0% accuracy degradation

5. User asked about Quark's rotation + quantization (PTQ) flow
   - Analyzed `quantize_quark.py`, `train_rotation.py`, `RotationProcessor`, `ModelQuantizer`
   - Documented Quark's architecture: rotation is a preprocessing step in `apply_advanced_quant_algo()`, executed BEFORE calibration/quantization
   - Identified the flow: untie → fuse_norm → R1 → R2 → R3 → R4 → quantize → calibrate → freeze

6. User asked to create a rotation + quantization test (RTN W4A16) using auto-round, and optionally a Quark comparison
   - Created `test_rotation_quantization.py` that tests: baseline_fp16, rtn_only (W4A16), rotation_rtn (rotation + W4A16)
   - Script was running successfully — baseline_fp16 evaluated (acc_norm=0.5000/0.6900), RTN quantization completed for all 28 layers
   - The test was still running when compaction occurred (RTN evaluation and rotation_rtn levels pending)
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection via globals patching
- `examples/test_rotation_quantization.py`: NEW — Tests rotation + W4A16 RTN quantization, compares baseline/rtn_only/rotation_rtn

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`: 
  - Added `is_pow2()`, `get_hadamard_K()`, `matmul_hadU()` (butterfly Hadamard)
  - **CRITICAL FIX:** `untie_word_embeddings_if_needed()` now sets `model.config.tie_word_embeddings = False` (line ~365)
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`: Complete rewrite — generic R3 via monkeypatch, R4 with validation
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`: All `print` → `logger`, added dimension validation, rich logging

Work completed:
- [x] Fix R1 accuracy degradation bug (tie_word_embeddings)
- [x] Verify all rotation levels preserve accuracy (hellaswag, piqa — 500 samples)
- [x] Analyze Quark's rotation+quantization architecture
- [x] Create rotation+quantization test script
- [ ] Complete running test_rotation_quantization.py (was in progress)
- [ ] Write Quark comparison script (determined to be impractical — see technical details)

Current state:
- All mock tests pass (7/7 in test_rotation_levels.py)
- All reference tests pass (test_quark_comparison.py, test_reference_equivalence.py)
- Real model evaluation: rotation preserves accuracy perfectly (Δ ≤ 0.005 on 500 samples)
- Rotation+quantization test was running (baseline_fp16 and rtn_only complete, rotation_rtn pending)
</work_done>

<technical_details>
**Critical Bug Found & Fixed — tie_word_embeddings:**
- When `model.config.tie_word_embeddings = True` (as in Qwen3-0.6B), HuggingFace's `model.tie_weights()` re-ties lm_head to embed_tokens
- lm_eval's HFLM wrapper calls this during initialization
- After our rotation, embed_tokens has `W @ R1` but lm_head has `(W * γ_final) @ R1` — they MUST be different
- Fix: set `model.config.tie_word_embeddings = False` after untying
- Quark avoids this issue because they don't use lm_eval's HFLM; they have their own eval path

**RMSNorm Fusion + R1 Math:**
- RMSNorm(x@R1) = (x@R1) / RMS(x) when gamma=1 (since orthogonal R preserves norm)
- But RMSNorm(x@R1) with non-trivial gamma ≠ RMSNorm(x) @ R1 (gamma doesn't commute with rotation)
- Therefore fuse_rmsnorm (absorb gamma into weights) is REQUIRED for R1 correctness
- Without fusion: cosine_sim drops to 0.12 (completely wrong)

**Quark's PTQ Architecture:**
- `quantize_model()` → Step 2: `apply_advanced_quant_algo()` applies algorithms sequentially
- `RotationProcessor.apply()`: untie → R1 (fuse_norm + fuse_rotation) → R2 → R3 → R4
- Then: prepare quantization model → calibrate → freeze
- Rotation is config-driven via `scaling_layers` JSON (specifies prev_modules, norm_module, next_modules per layer)
- Quark supports `online_r1_rotation` (with InputRotationWrapper) — different from our offline-only approach

**Quark Comparison Script — Why Impractical:**
- Quark's quantization requires `LLMTemplate`, `QConfig`, `QLayerConfig` infrastructure
- W4A16 equivalent would need specific scheme configuration (int4 weight-only)
- The template system is deeply coupled to model architectures
- Quark's `untie_parameters` uses accelerate's `find_tied_parameters`
- Not worth the complexity for a comparison test

**Auto-Round's RTN Quantization:**
- `AutoRound(model, scheme="W4A16", iters=0)` — iters=0 means RTN (round-to-nearest, no optimization)
- Can accept pre-loaded model object (not just model name string)
- Uses bfloat16 internally for quantization tuning
- `device_map` parameter needs specific GPU ID (e.g., "cuda:7"), not just "cuda"

**Qwen3-0.6B Architecture Details:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q heads, 8 kv heads)
- intermediate_size=3072 (NOT power-of-2 → R4 uses block Hadamard with K=1024, blocks=3)
- Has q_norm and k_norm (per-head RMSNorm after q/k projection) — not affected by R1
- tie_word_embeddings=True in config (embed_tokens shares weight with lm_head)
- 28 transformer layers, model_type="qwen3"

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- In newer transformers: `dtype=` is correct (not `torch_dtype=`; the latter is deprecated)
- auto-round installed editable: `pip install -e .` in /data/lkk/quarot/auto-round
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Core rotation math: rotate_in_channels_, rotate_out_channels_, fuse_rmsnorm_in_model, matmul_hadU
  - **CRITICAL FIX at line ~365:** untie_word_embeddings_if_needed now sets config.tie_word_embeddings=False
  - Also contains: deterministic_hadamard_matrix, random_hadamard_matrix, is_pow2, get_hadamard_K, apply_hadamard_to_linear, get_model_arch_info

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()`
  - Key methods: `_fuse_offline_rotations()` (~line 547), `_validate_dimensions()`, `_init_rotation_matrices()`, `_train_rotations()`
  - Order: untie → fuse_rmsnorm → trainable_smooth → init_rotations → train → fuse_offline → register_online_hooks

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - Online hook registration for R3 and R4
  - R3: Generic monkeypatch via `add_qk_rotation_after_rope()`
  - R4: forward_pre_hook on down_proj with block Hadamard

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - Architecture-generic R3 injection: copies forward, patches globals to replace apply_rotary_pos_emb
  - QKRotationWrapper: wraps apply_rotary_pos_emb to apply Hadamard after RoPE

- `/data/lkk/quarot/auto-round/examples/test_rotation_quantization.py`
  - NEW: Tests rotation + W4A16 RTN quantization
  - Compares baseline_fp16, rtn_only, rotation_rtn, rotation_fp16
  - Uses auto-round's AutoRound(iters=0) for RTN mode

- `/data/lkk/quarot/auto-round/examples/test_qwen3_rotation_eval.py`
  - Real model lm_eval evaluation of all rotation levels
  - Has the dtype bug (uses `dtype=` which now works in newer transformers)

- `/data/lkk/quarot/auto-round/examples/test_rotation_levels.py`
  - Comprehensive mock validation (7 tests, all pass)

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - Quark's RotationProcessor: r1(), r2(), r3(), r4(), apply(), post_process_trained_rotation()
  - Reference for architecture and conventions

- `/data/lkk/quarot/Quark/examples/torch/language_modeling/rotation/train_rotation.py`
  - Quark's full rotation training + quantization pipeline
  - Shows the two-phase approach: train rotation → post_process → quantize → freeze
</important_files>

<next_steps>
Remaining work:
1. **Complete test_rotation_quantization.py evaluation** — The script was running successfully. Need to check final results showing whether rotation helps RTN W4A16 accuracy.

2. **Potential issues with rotation+quantization test:**
   - The `quantize_rotated_model_rtn_w4a16()` function passes the pre-rotated model object to AutoRound. Need to verify AutoRound handles pre-modified models correctly (especially with untied embeddings, fused norms, and online R3/R4 hooks).
   - The `device_map=device` parameter showed a warning "cuda in device_map does not match any modules" — may need adjustment.

3. **Quark comparison:** Determined impractical due to deep infrastructure coupling. The user accepted this.

4. **Potential improvements:**
   - The test script could be extended with more quantization schemes (W8A16, etc.)
   - Could add AutoRound with iters>0 (optimization-based quantization) to show rotation+optimized quant

Immediate next action:
- Run `test_rotation_quantization.py` to completion and verify results
- If rotation_rtn fails (e.g., AutoRound can't handle online R3/R4 hooks), may need to disable R3/R4 for quantization or handle them differently
</next_steps>