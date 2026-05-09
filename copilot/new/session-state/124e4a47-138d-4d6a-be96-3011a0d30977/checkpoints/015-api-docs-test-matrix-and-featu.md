<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark to Intel's auto-round framework, and comparing three frameworks (auto-round, Quark, llm-compressor). This session focused on: (1) updating the three-way comparison document with rotation matrix init alignment changes, (2) creating a comprehensive API & Usage guide for auto-round's QuaRot/SpinQuant features, (3) creating a batch testing script for rotation × quantization scheme accuracy matrix, (4) adding feature status warnings (✅/⚠️/❌) throughout code and docs for incomplete features, (5) creating a test catalog document, and (6) launching parallel accuracy tests (RTN vs Tuning).
</overview>

<history>
1. User asked to update `three_way_framework_comparison.md` with the code changes from previous session (random_r1/r2 config addition, R3 init fix)
   - Updated §1.2 table: auto-round now shows configurable `random_r1/r2` (default deterministic), matching Quark
   - Removed stale "R3 note" about unused random buffer
   - Rewrote §1.3 diagram to show three init paths (deterministic/random/trainable)
   - Updated §1.4 framework support summary
   - Expanded §14.4 into side-by-side Quark/auto-round config comparison table
   - Added new §14.7 "Bug Fix: auto-round R3 Init and Config Alignment"

2. User asked to pause three-way comparison work and create: (a) API & Usage documentation for auto-round's QuaRot/SpinQuant features, (b) a batch testing script for rotation × quantization scheme combinations
   - Launched two explore agents to catalog the full API surface and quantization schemes
   - Both agents completed successfully with comprehensive findings

3. Created API & Usage guide (`docs/spinquant_quarot_api_guide.md`, 848 lines)
   - 12 sections: Overview, Core Concepts, Quick Start, API Reference, Configuration Guide, Usage Patterns, Rotation Levels Explained, Quantization Scheme Compatibility, Online vs Offline Rotation, Advanced Topics, Troubleshooting, Examples
   - Covers SpinQuantConfig, SpinQuantPreprocessor, RotationTrainer, hooks, utilities

4. Created batch testing script (`examples/test_rotation_scheme_matrix.py`, 688→732 lines)
   - Tests all rotation × scheme combinations with lm_eval evaluation
   - 5 rotation levels × 11 schemes, supports RTN and GPTQ-style
   - Outputs JSON, CSV, formatted matrix table
   - Created shell wrapper `run_rotation_scheme_matrix.sh` with 7 preset modes

5. User asked about rotation + quantization being decoupled
   - Confirmed: rotation is preprocessing (stage 1), quantization is independent (stage 2)
   - AutoRound sees a standard nn.Module after rotation — any iters value works

6. User asked to run all combinations and produce a results table
   - Checked GPU availability (4-7 free)
   - Ran sanity check on GPU 6 (passed: W4A16 none=0.4000 on limit=10)
   - Launched two parallel tests:
     - GPU 4: RTN (iters=0), 4 rotations × 3 schemes, 4 tasks, full eval
     - GPU 5: Tuning (iters=200), same matrix
   - Both running in background (shellIds: rtn-test, tuning-test)

7. User asked to add feature status warnings in code and docs for incomplete features, better mode explanations in test script, and rotation_size config support
   - Added ⚠️ warnings to 4 source files: `__init__.py`, `preprocessor.py`, `trainer.py`, `training.py`
   - Enhanced test script docstring with 7-mode explanations, added rotation-size help
   - Updated shell script with `ROTATION_SIZE` env var support and detailed mode comments
   - Updated API doc feature table with Notes column, added ❌ items (save/load, pre-trained matrices)

8. User asked for a document cataloging all test scripts
   - Launched explore agent to scan all 19 test/example scripts
   - Created `docs/test_catalog.md` (370+ lines) with:
     - 5 unit tests, 5 integration tests, benchmark matrix, framework comparison, examples
     - Test hierarchy diagram (Level 1-4)
     - Quick run guide, dependency table, feature coverage status

9. User asked how many eval tasks were in the running tests
   - Confirmed 4 tasks: hellaswag, piqa, winogrande, lambada_openai
   
10. User suggested adding mmlu
    - Checked progress: RTN had 2 combos done, Tuning had 1 combo done
    - At this point compaction was triggered
</history>

<work_done>
Files created this session:
- `docs/spinquant_quarot_api_guide.md` (857 lines): Comprehensive API & Usage guide
- `docs/test_catalog.md` (370+ lines): Complete test script catalog with run commands
- `examples/test_rotation_scheme_matrix.py` (732 lines): Batch rotation × scheme accuracy matrix script
- `examples/run_rotation_scheme_matrix.sh` (164 lines): Shell wrapper with 7 preset modes

Files modified this session:
- `docs/three_way_framework_comparison.md`: Updated §1.2/1.3/1.4 tables, removed stale R3 note, expanded §14.4, added §14.7
- `auto_round/algorithms/transforms/spinquant/__init__.py`: Added feature status warnings (✅ QuaRot / ⚠️ SpinQuant / ⚠️ save-load), reordered __all__
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`: Added SpinQuantConfig docstring with feature status, trainable_rotation warning comment
- `auto_round/algorithms/transforms/spinquant/trainer.py`: Added ⚠️ experimental warnings to RotationTrainerConfig and RotationTrainer docstrings
- `auto_round/algorithms/transforms/spinquant/training.py`: Added ⚠️ experimental warning to module docstring

Work in progress:
- [ ] **RTN accuracy test running** on GPU 4 (shellId: rtn-test) — 4 rotations × 3 schemes × 4 tasks, ~2 completed
- [ ] **Tuning accuracy test running** on GPU 5 (shellId: tuning-test) — same matrix with iters=200, ~1 completed
- [ ] User wants to add `mmlu` task — tests need to be restarted with 5 tasks

Log files:
- `/data/lkk/quarot/auto-round/examples/log_rtn.txt`
- `/data/lkk/quarot/auto-round/examples/log_tuning.txt`
- Results dirs: `results_rtn/`, `results_tuning/`

Completed:
- [x] Updated three_way_framework_comparison.md with init alignment changes
- [x] Created API & Usage guide
- [x] Created batch test script + shell wrapper
- [x] Added feature status warnings to code and docs
- [x] Created test catalog document
- [x] Launched parallel accuracy tests (still running)
</work_done>

<technical_details>
**Rotation + Quantization Decoupling:**
- Rotation (SpinQuantPreprocessor) is a preprocessing step that modifies model weights + registers hooks
- AutoRound sees a standard nn.Module — completely unaware rotation happened
- Any quantization mode works: RTN (iters=0), GPTQ-style (iters=200), any scheme
- This means all rotation × scheme × iters combinations are valid

**Test Script Architecture:**
- `test_rotation_scheme_matrix.py` — main Python script with 5 rotation levels × 11 schemes
- Rotation levels: none, R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
- Schemes: W4A16, W3A16, W2A16, W8A16, MXFP4, NVFP4, MXFP8, INT8, INT4, FP8_STATIC, FP8_BLOCK
- Common subsets: COMMON_SCHEMES=W4A16,MXFP4,NVFP4; WEIGHT_ONLY; WEIGHT_ACT
- Shell wrapper has 7 modes: quick, full, full-matrix, weight-only, weight-act, random, gptq
- Supports `ROTATION_SIZE` env var and `--rotation-size` CLI flag
- Outputs: JSON (full results), CSV (spreadsheet), formatted matrix table

**Feature Status Warnings Pattern:**
- ✅ QuaRot (fixed Hadamard, trainable_rotation=False) — production ready
- ⚠️ SpinQuant training (trainable_rotation=True) — experimental, not validated end-to-end
- ⚠️ Model save/load after rotation — not implemented (hooks not serialized)
- ❌ Pre-trained rotation matrices — not shipped

**FP16 Baseline (Qwen3-0.6B, full eval):**
- hellaswag: 0.4732
- piqa: 0.6763
- (winogrande, lambada_openai still evaluating)

**Environment:**
- 8x NVIDIA L20, CUDA, Python 3.12
- auto-round at /data/lkk/quarot/auto-round (editable install)
- Quark at /data/lkk/quarot/Quark
- llm-compressor at /data/lkk/quarot/llm-compressor (editable install)
- GPUs 0-3 busy; using GPUs 4-7

**Key Config Alignment with Quark (from previous session):**
- `random_r1: bool = False`, `random_r2: bool = False` added to SpinQuantConfig
- R3 init fixed from random_hadamard_matrix to deterministic_hadamard_matrix
- Defaults match Quark: deterministic Hadamard for all rotation levels
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/docs/spinquant_quarot_api_guide.md`
  - **Created this session**: Comprehensive API & Usage guide (857 lines)
  - 12 sections covering all QuaRot/SpinQuant features, config presets, usage patterns
  - Feature status table with ✅/⚠️/❌ markers

- `/data/lkk/quarot/auto-round/docs/test_catalog.md`
  - **Created this session**: Complete test/example script catalog (370+ lines)
  - 19 scripts cataloged with what/why/how/dependencies
  - Test hierarchy diagram, quick run guide, feature coverage table

- `/data/lkk/quarot/auto-round/examples/test_rotation_scheme_matrix.py`
  - **Created this session**: Main batch accuracy testing script (732 lines)
  - Tests rotation × scheme × quant-mode combinations
  - Outputs JSON/CSV/table; 7 preset modes via shell wrapper

- `/data/lkk/quarot/auto-round/examples/run_rotation_scheme_matrix.sh`
  - **Created this session**: Shell wrapper (164 lines) with ROTATION_SIZE support
  - 7 modes: quick, full, full-matrix, weight-only, weight-act, random, gptq

- `/data/lkk/quarot/auto-round/docs/three_way_framework_comparison.md`
  - **Modified this session**: Updated §1.2/1.3/1.4/14.4, added §14.7
  - ~1470 lines, 14 main sections

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/__init__.py`
  - **Modified this session**: Added feature status warnings, reordered __all__
  - 126 lines

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - **Modified this session**: Added SpinQuantConfig docstring with feature status
  - SpinQuantConfig at line ~62, preprocess() at line ~206

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/trainer.py`
  - **Modified this session**: Added ⚠️ experimental warnings
  - RotationTrainerConfig at line ~73, RotationTrainer at line ~192

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/training.py`
  - **Modified this session**: Added ⚠️ experimental warning to module docstring
</important_files>

<next_steps>
Immediate (user's latest request):
1. **Add `mmlu` task to running tests** — User wants 5 tasks (hellaswag, piqa, winogrande, lambada_openai, mmlu). Need to stop current tests (shellIds: rtn-test, tuning-test) and restart with `--tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu"`. RTN had ~2 combos done, tuning ~1, so early enough to restart.

2. **Wait for tests to complete, then summarize results into a table** — User explicitly asked to "run all data and summarize into a table". After tests finish:
   - Read results from `results_rtn/results.json` and `results_tuning/results.json`
   - Create a combined comparison table: RTN vs Tuning × Rotation × Scheme
   - Save as a results document

Pending from earlier sessions:
- User will report three-way framework comparison accuracy results (may have issues)
- Rotation config persistence (save/load) — design documented, implementation deferred
- SpinQuant training loop validation on real models — deferred until training is completed
</next_steps>