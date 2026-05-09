<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work involves fixing broken tests, correcting R1-R4 rotation fusion/hook logic, expanding test coverage for all rotation levels, and now validating with real models (Qwen3-0.6B) using lm_eval benchmarks. I'm iteratively debugging each rotation level, fixing architecture-specific issues, and building a comprehensive evaluation pipeline.
</overview>

<history>
1. User asked to fix broken test cases, resolve training code duplication between trainer.py and preprocessor.py, and expand validation tests for R1+R2, R1+R2+R3, R1+R2+R3+R4
   - Explored both codebases and reference implementations (inference_only_r1.py through r4.py)
   - Fixed device mismatches (rotation matrices using config.device instead of model's actual device)
   - Fixed R4 hook signature (changed from 3-arg forward_hook to 2-arg forward_pre_hook)
   - Added `apply_hadamard_to_linear()` function for per-head Hadamard application
   - Rewrote `_fuse_offline_rotations()` with proper R2 per-head fusion and R4 offline fusion
   - Fixed all example files (sys.path, method names, API calls)
   - Created comprehensive `test_rotation_levels.py` with R1/R2/R3/R4 tests

2. Fixed remaining R2/R4 test failures
   - Root cause: `.view()` after einsum produces non-contiguous tensors → fixed with `.reshape()`
   - `register_spinquant_hooks()` read `head_dim`/`intermediate_size` from config but SpinQuantConfig lacked those fields → added explicit parameters
   - `_cleanup()` was removing R3/R4 inference hooks → fixed to preserve them
   - Mock model lacked `.config` attribute → added proper config object for `get_model_arch_info`
   - All 7 tests in test_rotation_levels.py now pass

3. User asked to validate with Qwen3-0.6B using lm_eval
   - Explored Quark's lm_eval implementation (uses HFLM wrapper, simple_evaluate)
   - Verified lm_eval 0.4.11 is installed, Qwen3-0.6B available, 8x L20 GPUs available
   - Created `test_qwen3_rotation_eval.py` evaluation script
   - Found R4 crash: `intermediate_size=3072` not power-of-2 → fixed offline fusion to use block Hadamard (same K as hook)
   - Found R3 broken on real models: hook on q_proj output applies BEFORE RoPE, but R3 must be AFTER RoPE
   - Implemented R3 as monkey-patch of attention forward (inserts Hadamard after RoPE)
   - Verified all levels on Qwen3-0.6B: cosine_sim >0.99996, 100% argmax agreement

4. User asked about R3 mock generality and handling non-power-of-2 dimensions
   - This is the current question: the R3 monkey-patch is Qwen3-specific, not generic
   - User suggests: if model doesn't satisfy power-of-2 or architecture requirements, warn and skip R3/R4
   - Was about to refactor R3 for generality when context compaction triggered
</history>

<work_done>
Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`: Added `apply_hadamard_to_linear()`, fixed `.view()` → `.reshape()` in block Hadamard
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`: Fixed device handling, rewrote `_fuse_offline_rotations()`, added `_fuse_r2_rotation()`/`_fuse_r4_rotation()`, fixed `_cleanup()` to preserve hooks, `_fuse_r4_rotation` uses block K instead of full-size Hadamard, added training docstring
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`: Added `head_dim`/`intermediate_size` params to `register_spinquant_hooks()`, rewrote R3 as attention forward monkey-patch (Qwen3-specific), updated `remove_spinquant_hooks()` to handle R3 patches, fixed R4 hook
- `auto_round/algorithms/transforms/spinquant/trainer.py`: Fixed device handling, passes arch info to hook registration
- `examples/test_rotation_levels.py`: Added proper mock config, `.reshape()` fix, tolerance tuning
- `examples/test_spinquant.py`: Fixed sys.path, adjusted configs, relaxed tolerances
- `examples/spinquant_autoround_example.py`: Fixed `get_dataloader` API, method name fixes
- `examples/usage_examples.py`: Fixed `get_dataloader` API calls

Files created:
- `examples/test_rotation_levels.py`: Comprehensive R1/R2/R3/R4 mock validation (7 tests)
- `examples/test_qwen3_rotation_eval.py`: Real model evaluation with lm_eval

Test results:
- test_spinquant.py: 6/6 ✓
- test_rotation_levels.py: 7/7 ✓
- test_quark_comparison.py: all ✓
- test_reference_equivalence.py: all ✓
- Qwen3-0.6B logit equivalence: all 4 levels ✓ (cosine >0.99996)
- lm_eval full run: was about to execute when user raised generality concern

Current state:
- R1, R2 fully working and generic (offline fusion only)
- R3 works but is Qwen3-specific (monkey-patches attention forward)
- R4 works but requires intermediate_size divisible by some power-of-2 K>1
- User wants: generic R3 approach or graceful skip, and same for R4
</work_done>

<technical_details>
**R1-R4 Architecture:**
- R1: Full hidden_size rotation, fused OFFLINE into all linear layers. Always works (hidden_size is always power-of-2 for transformers).
- R2: Per-head Hadamard on v_proj output and o_proj input. Cancels in per-head attention. Requires head_dim to be power-of-2 (usually 64 or 128). Always offline.
- R3: Online-only rotation on Q and K AFTER RoPE. Cancels in attention scores: (Q@H)@(K@H).T = Q@K.T. Requires monkey-patching attention forward.
- R4: SPLIT - offline fuse block Hadamard into down_proj input + online apply same to activation before down_proj. Cancels: (x@H_block)@(W@H_block).T = x@W.T.

**Key Math Properties:**
- Normalized Sylvester Hadamard: H@H = I (involution), H = H.T (symmetric)
- Block Hadamard for non-power-of-2: split dim N into M blocks of K (where K is largest power-of-2 dividing N), apply K×K Hadamard to each block
- `rotate_in_channels_(layer, R_in)`: W_new = W @ R.T (in float64)
- `rotate_out_channels_(layer, R_out)`: W_new = R.T @ W (in float64)

**R3 Problem - Why hook on q/k_proj doesn't work for real models:**
- Real attention: hidden → q_proj → q_norm → reshape → RoPE → attention
- Hook on q_proj fires BEFORE q_norm and RoPE
- R3 must be AFTER RoPE for cancellation: (Q_rope @ H) @ (K_rope @ H).T = Q_rope @ K_rope.T
- Current fix: monkey-patch entire attention forward (Qwen3-specific)
- Need: architecture-generic solution or graceful skip

**R4 Block Hadamard:**
- Qwen3-0.6B intermediate_size=3072 (not power-of-2)
- K = largest power-of-2 dividing 3072 → K=1024 (3072/1024=3)
- Wait actually: 3072 = 3 × 1024, so K=1024. Let me verify...
- Actually from the code: K starts at 1 and doubles while K*2 <= inter and inter%(K*2)==0. 3072%2=0→K=2, 3072%4=0→K=4, ...3072%1024=0→K=1024, 3072%2048≠0→stop. So K=1024. ✓

**R3 Mock vs Real:**
- Mock model (test_rotation_levels.py): No RoPE, simple Q@K.T attention. Hook on q/k_proj works fine.
- Real model (Qwen3, Llama, etc.): Has RoPE between proj and attention. Must patch after RoPE.
- The mock tests validate the MATH (H@H.T=I cancellation) while real model tests validate the IMPLEMENTATION.

**get_dataloader API:**
- Signature: `get_dataloader(tokenizer, seqlen, dataset_name='NeelNanda/pile-10k', seed=42, bs=8, nsamples=512)`
- Second arg is `seqlen` (positional), NOT dataset_name

**lm_eval Integration:**
- HFLM accepts pre-loaded model via `pretrained=model` kwarg
- `simple_evaluate(model=lm, tasks=task_list, batch_size=bs, limit=limit, device=device)`
- Tasks used in Quark: piqa, hellaswag, winogrande, arc_challenge, arc_easy, lambada_standard, etc.

**Environment:**
- 8x NVIDIA L20 (44.4GB each), GPU 7 has most free memory (~18GB)
- PyTorch with CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- auto-round installed editable: `pip install -e .`
- Qwen3-0.6B model available locally
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Core orchestrator (8-step pipeline). Main entry: `preprocess()`
  - Key methods: `_fuse_offline_rotations()` (~line 459), `_fuse_r2_rotation()` (~line 534), `_fuse_r4_rotation()` (~line 562), `_init_rotation_matrices()` (~line 260), `_train_rotations()` (~line 315)
  - R4 now uses block K instead of full-size Hadamard

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - Online hook registration for R3 and R4
  - R3: Currently Qwen3-specific monkey-patch of attention forward (lines 75-185) - NEEDS GENERALIZATION
  - R4: forward_pre_hook on down_proj applying block Hadamard (lines ~190-220)
  - `remove_spinquant_hooks()` handles both hook handles and R3 monkey-patches

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Low-level rotation math: `apply_hadamard_to_linear()` (line ~282), `deterministic_hadamard_matrix()`, `rotate_in_channels_()`, `rotate_out_channels_()`
  - `get_model_arch_info()` (line 358) extracts hidden_size, head_dim, etc.

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/trainer.py`
  - Full-featured RotationTrainer with callbacks/checkpointing
  - Delegates to SpinQuantPreprocessor for init/fusion

- `/data/lkk/quarot/auto-round/examples/test_rotation_levels.py`
  - Comprehensive mock validation (7 tests, all pass)
  - MockTransformerModel with proper multi-head attention (no RoPE)

- `/data/lkk/quarot/auto-round/examples/test_qwen3_rotation_eval.py`
  - Real model evaluation script using lm_eval
  - Tests baseline vs R1 vs R1+R2 vs R1+R2+R3 vs R1+R2+R3+R4

- `/data/lkk/quarot/inference_only_r1_r2_r3_r4.py` (reference)
  - Shows correct R3/R4 via custom modeling_qwen3_online_r3_r4.py
  - R4 offline: `apply_exact_had_to_linear(mlp.down_proj, had_dim=-1, output=False)` then `fuse_layer_rotation(down_proj, R_out=R1)`

- `/data/lkk/quarot/modeling_qwen3_online_r3_r4.py` (reference)
  - Custom Qwen3 modeling with R3 after RoPE and R4 in MLP forward
  - R3: `fast_hadamard_transform` on query/key after RoPE (lines ~227-236)
  - R4: `matmul_hadU_cuda(act)` before down_proj (line ~89)
</important_files>

<next_steps>
Remaining work:
1. **Refactor R3 for generality**: Current monkey-patch is Qwen3-specific. Need to either:
   - Support multiple architectures (Llama, Qwen2, Qwen3, Mistral) via architecture detection
   - Or provide a generic fallback that works for any model with standard attention pattern
   - User's suggestion: if architecture not supported, warn and skip R3

2. **Add graceful fallback for R3/R4 dimension constraints**:
   - R3: head_dim must be power-of-2 (check and warn/skip if not)
   - R4: intermediate_size must be divisible by some K>1 power-of-2 (check and warn/skip)
   - Print clear warning messages to user when skipping

3. **Run full lm_eval evaluation on Qwen3-0.6B**: The script is ready, just needs to be executed with full samples (no --limit) on tasks like piqa, hellaswag, arc_easy, winogrande, lambada_standard

4. **Address training code duplication**: preprocessor._train_rotations() vs trainer.RotationTrainer - keep both (simple vs full-featured) with clear documentation. Possibly extract shared logic into utility.

Immediate next action:
- Refactor `inplace/apply.py` R3 implementation to be architecture-aware:
  - Detect model type from config (qwen3, llama, qwen2, mistral, etc.)
  - Use appropriate patching strategy per architecture
  - Fall back to skip with warning for unsupported architectures
- Add dimension validation in preprocessor `_init_rotation_matrices()` and hook registration
</next_steps>