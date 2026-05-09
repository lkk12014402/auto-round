<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work has expanded to include comparing three frameworks (auto-round, Quark, llm-compressor) for rotation + MXFP4 quantization. The current phase focused on: (1) creating a three-way comparison test script, (2) fixing the transformation summary table's hook counting bug, (3) analyzing llm-compressor's rotation implementation, and (4) writing comprehensive documentation comparing all three frameworks' R1-R4 implementations.
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
    - Created `docs/quark_vs_autoround_wrapper_architecture.md`

20. User asked to update rotation save/load docs for all three matrix types
    - Identified 3 types: Hadamard (deterministic), QuaRot random, SpinQuant trained
    - Updated `docs/rotation_save_load_solution.md`

21. User asked about Quark rotation acceleration kernels → No custom kernels exist
22. User asked about rotation + different quantization scheme compatibility → scheme-agnostic
    - Created `docs/rotation_scheme_compatibility.md`

23. User reported R1+R2+R3+R4 MXFP4 test crash
    - **Root cause:** R4 hook used `matmul_hadU(x, K=128)` on 3072-dim input → butterfly halves to 96 < 128 → dimension mismatch
    - **Fix:** Added block rotation detection when `r4_size < intermediate_size`
    - Verified: cosine_sim=1.0, hellaswag=0.4900

24. User asked to create test scripts for W4A16 and NVFP4 schemes with rotation
    - Created `examples/test_rotation_schemes.py` and `run_rotation_scheme_tests.sh`
    - Smoke tested W4A16+R1 and NVFP4+R1 — both pass

25. User asked to analyze llm-compressor's rotation implementation and create three-way comparison
    - Launched 3 explore agents to analyze llm-compressor + compressed-tensors codebases
    - Installed llm-compressor (`pip install -e .`)
    - Verified all rotation levels (R1, R1+R2, R1+R2+R4, R1+R2+R3+R4) work with MXFP4 in llm-compressor

26. Created three-way comparison test script (`test_three_way_comparison.py`)
    - Fixed Quark config issues: removed invalid `rotation_mode`/`target_modules`, added `middle_layers`, used `LLMTemplate.get_config()`, fixed `_data_loader=None`, changed `.process()` to `.apply()`
    - Smoke tested all three frameworks with R1+MXFP4 — all pass
    - Created `run_three_way_comparison.sh` convenience script

27. User reported transformation summary table showed 0 modules for R1 hooks
    - **Root cause:** `_apply_online_r1()` registered hooks via `module.register_forward_pre_hook()` but didn't store handles in `self._hook_handles`. Summary code only counted from `self._hook_handles` which was empty for R1-only.
    - **Fix:** Added `self._r1_hook_handles` list; `_apply_online_r1()` stores handles there; rewrote counting logic
    - Verified: R1-only → 140 modules, R1+R2+R3+R4 → 196 total (140 R1 + 28 R3 + 28 R4) ✓

28. Created `docs/llm_compressor_rotation_analysis.md` — architecture comparison across all three frameworks

29. User asked for detailed three-way framework comparison document
    - Created `docs/three_way_framework_comparison.md` — comprehensive 10-section document covering rotation matrices, code details, R1 online/offline comparison, inference paths, save/load mechanisms, performance characteristics

30. User pointed out R2 and R3 coverage was sparse in the document
    - Launched explore agent to gather detailed R2/R3 implementation code from all three frameworks
    - Agent completed with comprehensive R2/R3 analysis including code, math, save/load, inference paths
    - **Was about to update the document when compaction triggered**
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection (QKRotationWrapper)
- `examples/test_rotation_quantization.py`: Rotation + W4A16 RTN test
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison
- `examples/run_full_comparison.sh`: Full R1→R1+R2+R3+R4 comparison script
- `examples/test_wrapper_save_load.py`: 7-test suite for InputRotationWrapperHadamard
- `examples/test_rotation_schemes.py`: W4A16 + NVFP4 rotation test
- `examples/run_rotation_scheme_tests.sh`: Shell script for scheme tests
- `examples/test_three_way_comparison.py`: **Three-way comparison script (auto-round, Quark, llm-compressor)**
- `examples/run_three_way_comparison.sh`: Shell script for three-way comparison
- `docs/quark_rotation_analysis.md`: Complete Quark rotation analysis
- `docs/online_r1_wrapper_and_rotation_size.md`: Wrapper comparison and rotation_size docs
- `docs/online_vs_offline_r1_after_quantization.md`: Why online/offline R1 aren't interchangeable
- `docs/r3_r4_online_rotation.md`: R3/R4 documentation
- `docs/input_wrapper_vs_embed_fusion.md`: Mathematical proof input wrapper ≠ embed fusion
- `docs/wrapper_vs_hook_refactoring_analysis.md`: 4 approaches analysis
- `docs/compressors_new_decoupling_analysis.md`: Pipeline decoupling analysis
- `docs/quark_vs_autoround_wrapper_architecture.md`: FX tracing vs type checking architecture
- `docs/rotation_save_load_solution.md`: Complete 3-type persistence scheme
- `docs/rotation_scheme_compatibility.md`: Rotation + quantization scheme compatibility
- `docs/llm_compressor_rotation_analysis.md`: llm-compressor architecture analysis
- `docs/three_way_framework_comparison.md`: **Comprehensive three-way comparison (needs R2/R3 expansion)**

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`: `InputRotationWrapperHadamard`, `apply_hadamard_to_linear()`, `matmul_hadU()`
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - `_apply_online_r1()`: hook-based online R1, stores handles in `self._r1_hook_handles`
  - `_make_online_r1_hook()`: static method for rotation hooks
  - `_fuse_r2_rotation()`, `_fuse_r4_rotation()`: offline fusion
  - `_print_transformation_summary()`: **Fixed hook counting** — uses `self._r1_hook_handles` for R1 count
  - Added `self._r1_hook_handles: list[Any] = []` at line ~192
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`: R4 block rotation fix for `r4_size < intermediate_size`, hook-based R3/R4
- `auto_round/algorithms/transforms/spinquant/__init__.py`: Updated exports

Work completed:
- [x] Hook-based online R1 rotation (compatible with auto-round quantization)
- [x] Hook-based R4 rotation with block rotation support
- [x] R4 block rotation fix for r4_size < intermediate_size
- [x] Fixed transformation summary table hook counting (R1 was showing 0)
- [x] Three-way comparison test script (auto-round, Quark, llm-compressor)
- [x] All architecture analysis documents
- [x] W4A16 + NVFP4 rotation test scripts
- [x] llm-compressor installation and smoke testing
- [ ] **Update three_way_framework_comparison.md with detailed R2/R3 sections** — explore agent completed, update pending
- [ ] Test scripts for W4A16/NVFP4 + rotation combinations — created, user will test
- [ ] Rotation config persistence (save/load) — design documented, implementation deferred
</work_done>

<technical_details>
**Normalization Convention (CRITICAL):**
- Quark's `_get_hadamard_K()` returns UNNORMALIZED matrices (±1 values)
- Auto-round's `get_hadamard_K()` returns NORMALIZED matrices (H/√N, orthogonal)
- llm-compressor's `deterministic_hadamard_matrix()` returns ±1, normalizes by √N during apply in `HadamardTransform.forward()`
- Must NOT re-normalize auto-round's matrices — caused double-normalization bug (hellaswag=0.2628)

**Online R1 vs Offline R1:**
- Offline R1: global change-of-basis. Modifies ALL layers, fuses RMSNorm. No hook needed at inference. Quantization quality suffers because ALL weight distributions change.
- Online R1: local rotation per target module. Only modifies q/k/v/gate/up weights + registers forward_pre_hook. Better quantization quality but needs hook at inference.
- After quantization: online ↔ offline NOT interchangeable: (1) RMSNorm gamma not fused, (2) Q(W@H) ≠ Q(W)@H
- auto-round and Quark use online R1; llm-compressor uses offline R1

**Why InputRotationWrapper breaks auto-round (6 coupling points):**
1. `wrapper_block()` uses `type(m) in SUPPORTED_LAYER_TYPES` — wrapper not recognized
2. `named_modules()` recurses into wrapper → finds inner nn.Linear with wrong path
3. `WrapperLinear.__init__` only recognizes nn.Linear/Conv1D
4. `WrapperLinear.forward` bypasses wrapper's rotation with F.linear
5. `WrapperWALayer.forward` — after MXFP4 packing, weight=[0] → crash
6. `pack_layer()` type check fails

**Why hooks work in auto-round:**
- `WrapperLinear.forward()` explicitly runs `self.orig_layer._forward_pre_hooks` (line 503-518)
- `WrapperWALayer.__init__()` steals hooks from orig_layer (line 557), runs them in forward (line 572-575)

**R4 Block Rotation Bug (fixed):**
- When `r4_rotation_size=128` but `intermediate_size=3072`, `matmul_hadU(x, K=128)` called on full 3072 dim
- Butterfly: 3072→1536→768→384→192→96, stops at 96 < 128 → dimension mismatch
- Fix: detect `r4_size < intermediate_size`, reshape to blocks first, apply Hadamard per block

**Summary Table Hook Counting Bug (fixed this session):**
- `_apply_online_r1()` registered hooks via `module.register_forward_pre_hook()` but didn't store handles
- Summary only counted `self._hook_handles` (which only has R3/R4 from `register_spinquant_hooks()`)
- Fix: Added `self._r1_hook_handles` list, stored handles there, rewrote counting logic

**Three-Way Framework Architecture Differences:**
| Feature | auto-round | Quark | llm-compressor |
|---------|-----------|-------|----------------|
| Graph tracing | Module tree + `type()` | `torch.fx` export | Module tree + regex |
| R1 mode | Online (hooks) | Online (wrappers) | Offline (fused) |
| R3 mechanism | Monkeypatch `apply_rotary_pos_emb` | Monkeypatch `apply_rotary_pos_emb` | QuantizedAttentionImpl + KVCache hooks |
| Save R1/R4 | ❌ hooks lost | ✅ `input_rotation` buffer | ✅ TransformBase submodule |
| Save R3 | ❌ monkeypatch lost | ❌ not supported in export | ✅ submodule saved |
| Quantization API | `AutoRound(scheme=...)` | `ModelQuantizer(config)` | `oneshot(recipe=[...])` |

**R2 Implementation (all frameworks):**
- R2 is head-wise rotation: applies Hadamard per-head on v_proj output channels and o_proj input channels
- Always offline (fused into weights) — no inference overhead
- All three frameworks save/load R2 transparently (weights contain rotation)
- auto-round: `_fuse_r2_rotation()` uses `apply_hadamard_to_linear()` with einsum block ops
- Quark: `r2()` uses `rotate_out_channels_()` / `rotate_in_channels_()`
- llm-compressor: `_multihead_matmul()` with `head_dim` parameter in TransformScheme

**R3 Implementation (all frameworks):**
- R3 is Q/K rotation after RoPE — must be online (runtime cost)
- auto-round: monkeypatches `apply_rotary_pos_emb` → `QKRotationWrapper` intercepts and applies H to Q,K
- Quark: similar monkeypatch approach
- llm-compressor: replaces attention implementation with `QuantizedAttentionImpl` + `QuantizedKVCache` hooks via `Q_ATTN`/`K_CACHE` TransformLocation
- R3 save/load: only llm-compressor can fully persist R3 (submodule-based); auto-round/Quark lose R3 on save

**Three Rotation Matrix Types:**
| Type | Init | Must persist matrix |
|------|------|-------------------|
| Hadamard | deterministic Sylvester construction | No, just save rotation_size |
| QuaRot random | random ±1 diagonal × Hadamard | Yes (float64, N×N) |
| SpinQuant trained | Cayley/Stiefel optimization | Yes (float64, N×N) |

**Quark Three-Way Test Script Config Issues (fixed):**
- `RotationConfig` doesn't accept `rotation_mode` or `target_modules` params
- `scaling_layers` must include `middle_layers` key (not just `first_layer`/`last_layer`/`layers`)
- `RotationProcessor.__init__` requires `_data_loader` arg (pass `None` for non-trainable)
- Method is `.apply()` not `.process()`
- Must use `LLMTemplate.get_config(scheme="mxfp4", algorithm=["rotation"], algo_configs=...)` pattern

**llm-compressor Save/Load Mechanism:**
- `TransformConfig` saved inside `config.json` under `quantization_config.transform_config`
- Online transforms (R4) saved as `TransformBase` nn.Module submodules with tied weights
- `CompressedTensorsHfQuantizer` reads config on load, re-applies transforms
- Uses `_register_tied_transform_weights()` for save_pretrained deduplication

**Qwen3-0.6B Architecture:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q, 8 kv)
- intermediate_size=3072 (= 3 × 1024, NOT pow2)
- tie_word_embeddings=True in config

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- auto-round installed editable at /data/lkk/quarot/auto-round
- Quark at /data/lkk/quarot/Quark
- llm-compressor installed editable at /data/lkk/quarot/llm-compressor
- compressed-tensors 0.15.1a20260503 (dev version from llm-compressor)
- GPUs 0-3 often busy; use GPUs 4-7 for testing

**R1-R4 Fusion Summary:**
| Rotation | Offline fuseable? | Needs inference hook? |
|----------|------------------|----------------------|
| R1 offline | ✅ fully | ❌ |
| R1 online | weight side fused | ✅ forward_pre_hook |
| R2 | ✅ fully (V out + O in) | ❌ |
| R3 | ❌ (after RoPE) | ✅ monkeypatch or attention hook |
| R4 | down_proj input fused (inverse) | ✅ forward_pre_hook on down_proj |
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()` (line ~220)
  - `_apply_online_r1()` (line ~642): hook-based online R1, stores handles in `self._r1_hook_handles`
  - `_make_online_r1_hook()`: static method for rotation hooks
  - `_fuse_r2_rotation()`: offline R2 head-wise fusion
  - `_fuse_r4_rotation()`: offline R4 weight inverse fusion
  - `_fuse_offline_rotations()` (line ~735): offline R1 path
  - `_print_transformation_summary()`: **Fixed this session** — now correctly counts R1/R3/R4 hooks
  - `SpinQuantConfig` dataclass (line 63): all config fields
  - `self._r1_hook_handles` (line ~192): **Added this session** for R1 hook tracking

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - R3 monkeypatch + R4 hook registration
  - R4 block rotation for r4_size < intermediate_size
  - `register_spinquant_hooks()` and `remove_spinquant_hooks()`

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - `get_hadamard_K()`, `matmul_hadU()`, `rotate_in_channels_()`, `apply_hadamard_to_linear()`
  - `InputRotationWrapperHadamard` (utility only, not used in main flow)

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - `QKRotationWrapper`: R3 injection after RoPE
  - `add_qk_rotation_after_rope()`: architecture-generic R3 registration

- `/data/lkk/quarot/auto-round/examples/test_three_way_comparison.py`
  - **Created this session** — three-way comparison test (auto-round, Quark, llm-compressor)
  - `run_autoround()`, `run_quark()`, `run_llmcompressor()` runner functions
  - `_get_quark_rotation_config()` with full scaling_layers including middle_layers
  - Quark uses `LLMTemplate.get_config()` + `ModelQuantizer.quantize_model()` pattern
  - All smoke tested successfully

- `/data/lkk/quarot/auto-round/examples/test_rotation_schemes.py`
  - **Created this session** — W4A16 + NVFP4 rotation test
  - Smoke tested with R1 — both schemes pass

- `/data/lkk/quarot/auto-round/docs/three_way_framework_comparison.md`
  - **Created this session** — comprehensive 10-section comparison document
  - **Needs update:** R2 and R3 sections are sparse, user requested expansion
  - Explore agent completed with detailed R2/R3 code analysis, update pending

- `/data/lkk/quarot/auto-round/docs/llm_compressor_rotation_analysis.md`
  - **Created this session** — llm-compressor architecture analysis

- `/data/lkk/quarot/auto-round/auto_round/wrapper.py`
  - `WrapperLinear` (line 62): auto-round's quantization wrapper
  - Line 503-518: forward() runs orig_layer._forward_pre_hooks — WHY hooks work
  - Line 768-769: `wrapper_block()` type(m) in SUPPORTED_LAYER_TYPES

- `/data/lkk/quarot/auto-round/auto_round/export/utils.py`
  - Line 374: `clean_list` actively REMOVES `rotation_configs` from save

- `/data/lkk/quarot/llm-compressor/src/llmcompressor/modifiers/transform/spinquant/base.py`
  - SpinQuantModifier: recipe-based rotation modifier
  - `_create_r1_scheme()` through `_create_r4_scheme()`: scheme creation for each rotation level
  - `on_start()`: center_embeddings → fuse_norm_linears → apply_transform_config

- `/data/lkk/quarot/compressed-tensors/src/compressed_tensors/transform/factory/base.py`
  - TransformFactory: registry pattern for creating transforms
  - `_apply_to_module()`: handles INPUT/OUTPUT hooks, WEIGHT_INPUT/OUTPUT fusion, Q_ATTN/K_CACHE hooks

- `/data/lkk/quarot/compressed-tensors/src/compressed_tensors/transform/apply.py`
  - `apply_transform_config()`: entry point for applying transforms to model
  - `_register_tied_transform_weights()`: deduplication for save_pretrained

- `/data/lkk/quarot/compressed-tensors/src/compressed_tensors/compressors/model_compressors/model_compressor.py`
  - `update_config()`: saves transform_config to config.json
  - `from_compression_config()`: loads transform_config on model load

- `/data/lkk/quarot/auto-round/examples/run_three_way_comparison.sh`
  - **Created this session** — convenience script for three-way comparison

- `/tmp/1778302945840-copilot-tool-output-sl942v.txt`
  - Explore agent output with detailed R2/R3 implementation analysis from all three frameworks
  - Contains complete code for R2/R3 methods, math derivations, save/load analysis
  - **Use this to update the three_way_framework_comparison.md document**
</important_files>

<next_steps>
Remaining work:
- **Update `docs/three_way_framework_comparison.md` with detailed R2/R3 sections** — user explicitly requested this. Explore agent output is at `/tmp/1778302945840-copilot-tool-output-sl942v.txt` with complete R2/R3 code analysis from all three frameworks. Need to add:
  - R2 section: head-wise rotation math, code details per framework, how block-diagonal multiply works, save/load (all offline/transparent)
  - R3 section: post-RoPE Q/K rotation, three different implementation approaches (monkeypatch vs QuantizedAttentionImpl/KVCache), inference path, save/load (only llm-compressor persists R3)
  - Update comparison tables to be more balanced across R1-R4
- User will test the three-way comparison script and report logs
- User will test W4A16/NVFP4 rotation scheme scripts
- Rotation config persistence (save/load) — design documented, implementation deferred per user
</next_steps>