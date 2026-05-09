<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark to Intel's auto-round framework. This session focused on three major workstreams: (1) completing a 3-phase refactoring to integrate SpinQuant into auto-round's unified rotation registry/pipeline, (2) fixing a critical bug where non-power-of-2 dimensions (like intermediate_size=3072 in Qwen3-0.6B) couldn't use R3/R4 rotations due to missing known Hadamard matrices, and (3) collecting accuracy benchmark results across rotation×quantization scheme combinations. The approach has been to align with Quark's architecture while fitting auto-round's registry/config patterns.
</overview>

<history>
1. Completed Phase 2 of refactoring: Training logic deduplication
   - Cleaned up unused imports in `trainer.py` (removed `copy`, `F`, `math`, `AdamAndSGDG`, several rotation_utils)
   - Fixed `self.optimizer` type hint (was `Optional[AdamAndSGDG]`, changed to plain `None` init)
   - Verified all imports work, regression tests pass (test_reference_equivalence, test_quark_comparison)
   - Marked Phase 2 todo as done

2. Completed Phase 3 of refactoring: Mixin simplification
   - Rewrote `spinquant_mixin.py` with deprecation warnings on `__init__`, `preprocess_with_spinquant()`, and `patch_compressor_for_spinquant()`
   - Mixin now delegates to `apply_rotation()` internally instead of directly using `SpinQuantPreprocessor`
   - Added comprehensive migration examples in module docstring showing new `rotation_configs=["quarot"]` API
   - Verified deprecation warnings fire correctly, regression tests pass
   - Updated `spinquant_refactoring_proposal.md` with implementation status (all 3 phases complete)

3. Background accuracy tests completed (RTN iters=0 and Tuning iters=200)
   - RTN results: 13 configs tested, R1+R2+R3+R4 configs all failed with "Cannot find suitable Hadamard decomposition for n=3"
   - Tuning results: same pattern — R3+R4 configs failed
   - Key finding: Tuning (iters=200) consistently outperforms RTN by ~3-5% avg across all rotation+scheme combos
   - R1+R2+W4A16+Tuning achieved 0.4856 avg (close to FP16 baseline 0.5029)
   - Results saved to `examples/results_rtn/results.json` and `examples/results_tuning/results.json`

4. User corrected: Qwen3-0.6B intermediate_size is 3072, not 1536
   - Investigated: 3072 = 1024 × 3, and our `get_hadamard_K` only tried power-of-2 factors
   - Discovered Quark has `KNOWN_HADAMARD_MATRICES` dict (4281-line `hadamard.py`) with pre-computed Hadamard matrices for sizes 12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172
   - For 3072: Quark decomposes as 3072 = 12 × 256 (K=12 known Hadamard, 256 butterfly)
   - Our code was missing these entirely

5. Ported known Hadamard matrices from Quark and fixed multiple bugs
   - Created `known_hadamard.py` (4150 lines) with all 11 pre-computed matrices
   - Rewrote `get_hadamard_K()` to use `KNOWN_HADAMARD_MATRICES` lookup (matching Quark's approach)
   - Fixed `matmul_hadU()`: updated auto-detect logic to use `get_hadamard_K`, fixed normalization from `√(n/K)` to `√n` (Quark-aligned)
   - Fixed `random_hadamard_matrix()` to use `get_hadamard_K` + `matmul_hadU` (works for non-pow2 sizes now)
   - Fixed `_fuse_r4_rotation()` in preprocessor: was using old pow2-only K logic with `apply_hadamard_to_linear`, now uses `matmul_hadU` (same as hook)
   - Fixed `_validate_r4_dimensions()`: now uses `get_hadamard_K` try/except instead of manual pow2 loop
   - Fixed `_init_rotation_matrices()` R4 section: uses `get_hadamard_K` instead of manual pow2 loop
   - Simplified `register_spinquant_hooks()` R4 section: from ~100 lines with 6 branches down to ~30 lines using `matmul_hadU` uniformly
   - Fixed device mismatch in all `torch.kron` calls (known Hadamard matrices return CPU tensors)
   - Fixed NameError (`K` → `had_K_val`) in R4 hook log message
   - Verified: All 5 rotation combos work on Qwen3-0.6B with small numerical diffs (max_diff: R1=0.04, R1+R2=0.05, R1+R2+R3=0.05, R1+R2+R4=0.06, R1+R2+R3+R4=0.06)

6. User asked about prior fallback behavior for non-pow2 dimensions
   - Confirmed: fallback logic (warning + disable rotation) is preserved for all R1-R4
   - R4 now uses `get_hadamard_K` try/except (covers many more sizes), only falls back for truly unsupported sizes
   - R1/R2/R3 still check power-of-2 only — could be relaxed since `matmul_hadU`/`get_hadamard_K` now handles non-pow2
   - Showed dimension coverage: 768, 1536, 3072, 5120, 7168, 11008, 14336 all now supported

7. User asked about test script and API doc compatibility
   - Reviewed `test_rotation_scheme_matrix.py` — uses `SpinQuantPreprocessor` directly (still works, not broken)
   - Reviewed `spinquant_quarot_api_guide.md` — needs updates for new unified API
   - User requested: create new versions (don't overwrite old ones), reference old logic
   - Was about to assess what needs rewriting when compaction triggered
</history>

<work_done>
Files created this session:
- `auto_round/algorithms/transforms/spinquant/known_hadamard.py` — 4150 lines, ported from Quark, contains 11 pre-computed Hadamard matrices for non-pow2 sizes

Files modified this session:
- `auto_round/algorithms/transforms/spinquant/trainer.py`:
  - Cleaned unused imports (removed copy, F, math, AdamAndSGDG, several rotation_utils functions)
  - Changed `self.optimizer: Optional[AdamAndSGDG] = None` → `self.optimizer = None`
- `auto_round/compressors_new/spinquant_mixin.py`:
  - Complete rewrite with deprecation warnings, delegates to `apply_rotation()`, migration examples in docstring
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`:
  - `get_hadamard_K()`: rewritten to use `KNOWN_HADAMARD_MATRICES` + `scipy.linalg.hadamard` (matching Quark)
  - `matmul_hadU()`: auto-detect uses `get_hadamard_K`, normalization fixed to `/√n`
  - `random_hadamard_matrix()`: rewritten to use `get_hadamard_K` + `matmul_hadU` (supports non-pow2)
  - All `torch.kron` calls: added `device="cpu"` alignment
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - `_fuse_r4_rotation()`: rewritten to use `matmul_hadU` instead of `apply_hadamard_to_linear`
  - `_validate_r4_dimensions()`: uses `get_hadamard_K` try/except instead of pow2 loop
  - `_init_rotation_matrices()` R4 section: uses `get_hadamard_K` instead of pow2 loop
  - `torch.kron` calls: added `device="cpu"` alignment
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`:
  - R4 hook registration: simplified from ~100 lines/6 branches to ~30 lines using `matmul_hadU`
  - Fixed NameError: `K` → `had_K_val` in log message
  - `torch.kron` calls: added `device="cpu"` alignment
- `docs/spinquant_refactoring_proposal.md`:
  - Updated status from "Draft" to "✅ Implemented (Phase 1–3 complete)"
  - Added §11 "Implementation Status" with per-phase completion tables

Work completed:
- [x] Phase 1: Registry integration (SpinQuantConfig → BaseRotationConfig, SpinQuantRotation registered, config aliases)
- [x] Phase 2: Training logic deduplication (training_core.py, preprocessor/trainer refactored)
- [x] Phase 3: Mixin simplification (deprecated, delegates to apply_rotation)
- [x] Port known Hadamard matrices from Quark (enables non-pow2 dimensions)
- [x] Fix R4 offline/online Hadamard mismatch bug
- [x] Fix matmul_hadU normalization bug
- [x] Verify R1+R2+R3+R4 works on Qwen3-0.6B (all 5 combos, max_diff < 0.07)
- [x] Collect RTN and Tuning accuracy results (13 configs × 5 tasks each)
- [ ] Create new test script (v2) incorporating unified API and R3+R4 support
- [ ] Update API guide documentation for new unified pipeline
- [ ] Relax R1/R2/R3 pow2-only validation (now technically supported via get_hadamard_K)
</work_done>

<technical_details>
**Known Hadamard matrices (critical discovery)**:
- Quark's `hadamard.py` contains pre-computed Hadamard matrices for 11 non-power-of-2 sizes: {12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172}
- These enable decomposition: n = K × 2^m where K is a known Hadamard size
- Example: 3072 = 12 × 256, 11008 = 172 × 64, 14336 = 28 × 512
- Without these, any model with non-pow2 intermediate_size couldn't use R4
- Coverage: virtually all mainstream LLM dimensions are now supported (768, 1536, 3072, 5120, 7168, 11008, 14336)

**Root cause of R4 28x diff on Qwen3-0.6B**:
- `_fuse_r4_rotation()` (offline) used old pow2-only K logic → K=1024 for 3072
- Hook (online) used updated `get_hadamard_K()` → K=12 for 3072
- Different rotations applied offline vs online → massive mismatch
- Fix: both paths now use `matmul_hadU` which internally calls `get_hadamard_K`

**Normalization fix in matmul_hadU**:
- Old code: divided by `√(n/K)` assuming pre-normalized K-block from `deterministic_hadamard_matrix`
- New code: divides by `√n` matching Quark, since `get_hadamard_K` now returns unnormalized matrices (from scipy/KNOWN_HADAMARD)

**Refactoring architecture (completed)**:
- `SpinQuantConfig(BaseRotationConfig)` with `algorithm="spinquant"` enables registry dispatch
- `@BaseRotation.register("spinquant") class SpinQuantRotation` → `apply_to_model()` delegates to `SpinQuantPreprocessor`
- String aliases: "quarot" → `SpinQuantConfig(trainable_rotation=False, trainable_smooth=False)`, "spinquant" → `SpinQuantConfig()`
- `BaseCompressor._apply_rotations()` iterates `rotation_configs` → `apply_rotation()` → registry dispatch
- `training_core.py` contains 7 shared primitives used by both preprocessor and trainer

**Accuracy findings (Qwen3-0.6B, 5 tasks: hellaswag, piqa, winogrande, lambada_openai, mmlu)**:
- FP16 baseline: avg 0.5029
- Best combo: R1+R2+W4A16+Tuning = 0.4856 avg (96.6% of FP16)
- Tuning (iters=200) consistently +3-5% over RTN (iters=0)
- R1 rotation helps lambada significantly
- R3+R4 were blocked by Hadamard decomposition error — now fixed but not yet re-benchmarked

**Fallback behavior (still in place)**:
- R1: pow2 check → warning + disable (could be relaxed)
- R2: pow2 check on head_dim → warning + disable
- R3: pow2 check on head_dim → warning + disable
- R4: `get_hadamard_K` try/except → warning + disable (broadest coverage now)

**Environment**: 8x NVIDIA L20, CUDA, Python 3.12, auto-round at /data/lkk/quarot/auto-round, Quark at /data/lkk/quarot/Quark, GPUs 0-3 busy
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/known_hadamard.py`
  - **Created this session**: 4150 lines, ported from Quark
  - Contains `KNOWN_HADAMARD_MATRICES` dict mapping sizes (12,20,...,172) to generator functions
  - Used by `get_hadamard_K()` for non-power-of-2 Hadamard decomposition

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - **Major changes**: `get_hadamard_K` (line ~50), `matmul_hadU` (line ~89), `random_hadamard_matrix` (line ~36)
  - All three functions rewritten to use `KNOWN_HADAMARD_MATRICES` and align with Quark
  - Also contains `deterministic_hadamard_matrix`, `apply_hadamard_to_linear`, `InputRotationWrapperHadamard`

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Core SpinQuantConfig (line ~63) and SpinQuantPreprocessor (line ~218)
  - **Changed**: `_fuse_r4_rotation` (line ~839) — uses `matmul_hadU` now
  - **Changed**: `_validate_r4_dimensions` (line ~346) — uses `get_hadamard_K` try/except
  - **Changed**: `_init_rotation_matrices` R4 section (line ~488) — uses `get_hadamard_K`
  - 8-step preprocess() pipeline: untie → fuse_rmsnorm → trainable_norms → init_matrices → train → fuse_offline → hooks → cleanup

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - `register_spinquant_hooks()` — registers R3 monkeypatch + R4 forward_pre_hooks
  - **Simplified**: R4 section from ~100 lines to ~30 lines using unified `matmul_hadU`
  - R3 uses `add_wrapper_after_function_call_in_method` for post-RoPE rotation

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/trainer.py`
  - **Changed**: Cleaned unused imports, fixed optimizer type hint
  - RotationTrainer with callbacks, checkpointing; uses training_core shared primitives

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/training_core.py`
  - Created in prior checkpoint: shared training primitives (~250 lines)
  - compute_rotation_loss, move_batch_to_device, check_orthogonality, clone_model_for_reference, create_dual_optimizer, run_training_loop

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/algorithm.py`
  - Created in prior checkpoint: `@BaseRotation.register("spinquant") class SpinQuantRotation`
  - Bridges spinquant into auto-round's unified registry

- `/data/lkk/quarot/auto-round/auto_round/compressors_new/spinquant_mixin.py`
  - **Rewritten**: Deprecated with warnings, delegates to `apply_rotation()`, migration examples
  - Backward compatibility preserved

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/base.py`
  - Modified in prior checkpoint: `_ensure_registry_populated()` includes "spinquant" in lazy load
  - Contains BaseRotation (ABC + registry), BaseRotationConfig

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/__init__.py`
  - Modified in prior checkpoint: `normalize_rotation_config()` handles spinquant dicts/strings
  - Contains `apply_rotation()` unified entry point

- `/data/lkk/quarot/auto-round/auto_round/compressors_new/entry.py`
  - Modified in prior checkpoint: _CONFIG_ALIASES has "spinquant" and "quarot" entries

- `/data/lkk/quarot/auto-round/examples/test_rotation_scheme_matrix.py`
  - Existing test script (not modified this session), uses SpinQuantPreprocessor directly
  - Still functional but doesn't use new unified API; user wants a v2 version

- `/data/lkk/quarot/auto-round/docs/spinquant_quarot_api_guide.md`
  - Existing API guide, needs updates for unified pipeline API

- `/data/lkk/quarot/auto-round/docs/spinquant_refactoring_proposal.md`
  - **Updated**: Status changed to "Implemented", added §11 completion tables

- `/data/lkk/quarot/auto-round/examples/results_rtn/results.json` and `results_tuning/results.json`
  - Accuracy benchmark results: 13 configs × 5 tasks each (RTN and Tuning modes)
</important_files>

<next_steps>
Immediate (user's last request, in progress):
1. **Create new test script v2** — User asked to create a new version of `test_rotation_scheme_matrix.py` (not overwrite) that works with the updated code. Key changes needed:
   - R3+R4 now works on Qwen3-0.6B (was blocked before)
   - Could use unified API (`apply_rotation(model, "quarot")`) instead of direct `SpinQuantPreprocessor`
   - Should add R1+R2+R3 as a rotation level (was missing, only had R1+R2+R3+R4)
   - Reference old script's logic for structure, modes, output formatting

2. **Update API documentation** — User asked to update docs if needed. `spinquant_quarot_api_guide.md` needs:
   - New unified API examples: `rotation_configs=["quarot"]`, `apply_rotation(model, "quarot")`
   - Updated dimension support info (non-pow2 now works)
   - Update "What's Supported" table (R3+R4 now works on more models)

Remaining technical work:
3. **Relax R1/R2/R3 pow2 validation** — Now that `get_hadamard_K` supports non-pow2 via known matrices, R1/R2/R3 validation could be broadened to use `get_hadamard_K` try/except instead of strict pow2 check. R1 uses `matmul_hadU` in online mode which already handles non-pow2.

4. **Re-run accuracy benchmarks with R3+R4** — Previous results had R3+R4 as errors. Now fixed, should re-benchmark to get complete accuracy matrix.

5. **Phase 4 (optional, deferred)** — Config style alignment (Pydantic migration for SpinQuantConfig). Intentionally deferred.
</next_steps>