<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The work involves fixing broken tests, making R3/R4 architecture-generic (not Qwen3-specific), adding proper logging like Quark, validating with real models (Qwen3-0.6B) using lm_eval benchmarks, and ensuring rotation equivalence. I've been iteratively debugging each rotation level, refactoring for generality, and now investigating why R1 rotation causes significant accuracy degradation (~10-14%) on real eval tasks despite showing near-perfect logit cosine similarity (>0.99999).
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
   - Implemented R3 as Qwen3-specific monkey-patch of attention forward
   - Verified all levels: cosine_sim >0.99996, 100% argmax agreement

3. User asked about R3 mock generality and handling non-power-of-2 dimensions
   - Discussed making R3 architecture-generic vs graceful skip

4. User asked to avoid `fast_hadamard_transform`, use auto-round's existing implementations, add Quark-style logging, and referenced Quark's log file
   - Studied Quark's log output (model info, rotation progress bars, training params, loss)
   - Studied Quark's architecture: `monkeypatch.py` for R3 (replaces `apply_rotary_pos_emb` in forward globals), `hadamard.py` for `matmul_hadU` (butterfly algorithm), `QKRotation` wrapper
   - Implemented `matmul_hadU` in rotation_utils.py (butterfly + block Hadamard, no external deps)
   - Created `monkeypatch.py` with architecture-generic R3 (works for any model calling `apply_rotary_pos_emb`)
   - Rewrote `inplace/apply.py` with generic R3, proper logging, dimension validation
   - Updated `preprocessor.py` with Python logging, dimension validation, rich messages
   - All tests pass (mock + real model logit equivalence: cosine >0.99999)

5. User reported R1 accuracy degradation in full lm_eval (hellaswag -14%, piqa -5.5%) and asked to investigate
   - The log shows float16 dtype was used for evaluation
   - User asks: is it our implementation bug or float64/float32 casting issue?
   - User wants a Quark-based test case (using Quark's code directly) for comparison, referencing `quantize_quark.py`
   - **This is where we are now - need to investigate**
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection via globals patching (from Quark's approach)

Files modified:
- `auto_round/algorithms/transforms/spinquant/rotation_utils.py`: Added `is_pow2()`, `get_hadamard_K()`, `matmul_hadU()` (butterfly Hadamard), updated `__all__`
- `auto_round/algorithms/transforms/spinquant/inplace/apply.py`: Complete rewrite — generic R3 via monkeypatch, R4 with validation, proper logging, graceful skip for unsupported architectures
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`: All `print` → `logger`, added `_validate_dimensions()`, rich logging (arch info, rotation config, fusion progress), removed dead cleanup code

Files unchanged but verified working:
- `examples/test_rotation_levels.py`: 7/7 pass
- `examples/test_spinquant.py`: 6/6 pass
- `examples/test_reference_equivalence.py`: all pass
- `examples/test_quark_comparison.py`: all pass
- `examples/test_qwen3_rotation_eval.py`: logit equivalence passes, but full lm_eval shows R1 accuracy drop

Current issue being investigated:
- R1 rotation on Qwen3-0.6B with float16 shows ~10-14% accuracy degradation on hellaswag/piqa
- Logit cosine similarity is >0.99999 (near-perfect), but accuracy drops significantly
- Hypothesis: float16 precision during rotation fusion (float64→float16 casting) loses information
- Need to compare against Quark's implementation on the same model to isolate the issue
</work_done>

<technical_details>
**R1-R4 Architecture:**
- R1: Full hidden_size rotation, fused OFFLINE into all linear layers. W_new = R1^T @ W @ R1 effectively.
- R2: Per-head Hadamard on v_proj output and o_proj input. Cancels in per-head attention. Offline only.
- R3: Online-only rotation on Q and K AFTER RoPE. Cancels in attention scores: (Q@H)@(K@H).T = Q@K.T
- R4: Offline fuse block Hadamard into down_proj input + online apply same to activation. Cancels: (x@H)@(W@H).T = x@W.T

**R3 Generic Implementation (from Quark):**
- `monkeypatch.py`: `add_wrapper_after_function_call_in_method()` copies the forward method's `__func__`, modifies its `__globals__` to replace `apply_rotary_pos_emb` with `QKRotationWrapper`, then rebinds
- Works for any HuggingFace model that calls `apply_rotary_pos_emb` in attention forward (Llama, Qwen2, Qwen3, Mistral, Phi, Gemma)
- If model doesn't have this function in globals, warns and skips R3

**matmul_hadU (butterfly Hadamard):**
- Efficient O(n log n) Hadamard via recursive butterfly: split tensor into pairs, sum/diff
- For non-pow2 dimensions, use block K: butterfly handles n/K part, explicit K×K Hadamard handles the rest
- Our implementation matches Quark's `_matmul_hadU` from `hadamard.py`

**Dimension Validation:**
- R1: hidden_size must be power-of-2 (auto-disabled if not)
- R2/R3: head_dim must be power-of-2 (auto-disabled if not)
- R4: intermediate_size must have a power-of-2 factor > 1 (auto-disabled if not)
- Qwen3-0.6B: hidden=1024(✓), head_dim=128(✓), intermediate=3072 → K=1024, blocks=3

**Key Issue - R1 Accuracy Degradation:**
- Logit cosine_sim > 0.99999 but lm_eval accuracy drops ~10-14%
- Test was run with `--dtype float16`
- The rotation fusion does: W_float64 = R1^T @ W_float64 @ ... then casts back to float16
- Problem likely: tiny logit differences (max_diff ~0.01) get amplified in token selection for long sequences
- Quark uses bfloat16 and also does rotation in float32/float64 then casts back
- Need to verify: does Quark also show this issue? Or is our fusion math slightly wrong?

**Quark's R1 Fusion Convention:**
- `fuse_layer_rotation_ref`: W = W.T, then R_in @ W (or W @ R_out), then W = W.T back
- Our convention: `rotate_in_channels_(layer, R_in)` does W_new = W @ R_in.T (in float64)
- `rotate_out_channels_(layer, R_out)` does W_new = R_out.T @ W (in float64)
- These should be mathematically equivalent but need to verify the exact R1/R1_inv usage

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- Qwen3-0.6B available locally
- auto-round installed editable: `pip install -e .`

**Quark's quantize_quark.py location:**
- `/data/lkk/quarot/Quark/examples/torch/language_modeling/llm_ptq/quantize_quark.py`
- Can be used as template for a Quark-based rotation-only test (no quantization)
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Core orchestrator (8-step pipeline), main entry: `preprocess()`
  - Key: `_fuse_offline_rotations()` (~line 547), `_validate_dimensions()` (~line 237), `_init_rotation_matrices()` (~line 322)
  - Now uses Python logging module throughout

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - Online hook registration for R3 and R4
  - R3: Generic monkeypatch via `add_qk_rotation_after_rope()` 
  - R4: forward_pre_hook on down_proj with block Hadamard
  - Graceful dimension validation and skip with warnings

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - NEW: Architecture-generic R3 injection
  - `add_wrapper_after_function_call_in_method()`: copies forward, patches globals
  - `QKRotationWrapper`: wraps apply_rotary_pos_emb to apply Hadamard after RoPE

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Low-level rotation math, now includes `matmul_hadU`, `is_pow2`, `get_hadamard_K`
  - `rotate_in_channels_()` and `rotate_out_channels_()` — these are the R1 fusion functions that may be the source of the accuracy issue

- `/data/lkk/quarot/auto-round/examples/test_qwen3_rotation_eval.py`
  - Real model evaluation script using lm_eval
  - Shows R1 accuracy degradation in float16

- `/data/lkk/quarot/auto-round/examples/test_rotation_levels.py`
  - Comprehensive mock validation (7 tests, all pass)

- `/data/lkk/quarot/Quark/examples/torch/language_modeling/llm_ptq/quantize_quark.py`
  - Quark's example for quantization — user wants to adapt this for rotation-only comparison

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - Quark's RotationProcessor with r1(), r2(), r3(), r4() methods
  - R1 uses `rotate_in_channels_`/`rotate_out_channels_` + scaling layers pattern

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/hadamard.py`
  - Quark's `matmul_hadU`, `_matmul_hadU`, `_get_hadamard_K`
  - Reference for our butterfly implementation
</important_files>

<next_steps>
Remaining work:
1. **Investigate R1 accuracy degradation** (CURRENT PRIORITY):
   - The key question: is it our rotation fusion math that's wrong, or float16 precision?
   - Write a test using Quark's code directly (no quantization) to apply R1 and evaluate
   - Compare: same model, same rotation matrix, Quark vs our implementation → same accuracy?
   - If Quark also degrades: it's a precision issue (need bfloat16 or float32 eval)
   - If only ours degrades: our fusion math has a bug

2. **Write Quark-based comparison test**:
   - Adapt `quantize_quark.py` to do rotation-only (no quantization)
   - Use Quark's RotationProcessor with same config on Qwen3-0.6B
   - Compare lm_eval accuracy: Quark-rotated vs our-rotated vs baseline

3. **Specific debugging steps**:
   - Check if the issue is dtype: run our test with `--dtype float32` or `--dtype bfloat16`
   - Check if `fuse_rmsnorm` is the culprit (it's lossy in float16)
   - Verify our `rotate_in_channels_` / `rotate_out_channels_` math matches Quark's exactly
   - Look at the Quark log: they use `bfloat16`, not `float16`

Immediate next actions:
- Read `quantize_quark.py` to understand Quark's API for rotation-only
- Write a side-by-side test: load Qwen3-0.6B, apply rotation with BOTH Quark and our code, compare logits
- Run our eval with `--dtype bfloat16` to see if that fixes the accuracy
</next_steps>