<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase (`/data/lkk/quarot/Quark`) to Intel's auto-round framework (`/data/lkk/quarot/auto-round`). The core work involves implementing online R1 rotation (matching Quark's default behavior) to fix a critical accuracy degradation when combining rotation with MXFP4 quantization. My approach has been to analyze Quark's implementation deeply, implement the equivalent in auto-round, and validate through lm_eval benchmarks.
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

3. User asked to use auto-round's existing implementations, add Quark-style logging
   - Implemented `matmul_hadU` in rotation_utils.py (butterfly algorithm)
   - Created `monkeypatch.py` with architecture-generic R3
   - Updated preprocessor with Python logging module throughout

4. User reported R1 accuracy degradation (hellaswag -14%)
   - **Root cause:** `untie_word_embeddings_if_needed()` didn't set `model.config.tie_word_embeddings = False`
   - lm_eval's HFLM calls `model.tie_weights()` which re-tied weights, overwriting rotated lm_head
   - **Fix:** Added `model.config.tie_word_embeddings = False` after untying

5. User asked for rotation + quantization test scripts and MXFP4 comparison with Quark
   - Created `test_rotation_quantization.py` (W4A16 RTN), `test_quark_rotation_mxfp4.py`, `test_autoround_rotation_mxfp4.py`
   - Added `rotation_size` configurability (Phase 1), matching Quark's semantics
   - Created `run_comparison_tests.sh`

6. User reported rotation+MXFP4 accuracy much worse than MXFP4-only in auto-round
   - Quark: rotation_mxfp4 ≈ mxfp4_only (hellaswag 0.4046 vs 0.4059)
   - Auto-round: rotation_mxfp4 << mxfp4_only (hellaswag 0.3307 vs 0.4054)
   - Created `debug_rotation_mxfp4.py` — confirmed rotation itself causes quant degradation

7. Diagnosed root cause: offline R1 vs online R1
   - **Offline R1** (our impl): rotates ALL weights (embed, q/k/v, o_proj, gate/up, down_proj, lm_head) + fuses RMSNorm gamma
   - **Online R1** (Quark default): only rotates target_modules (q/k/v, gate/up) weights + registers activation hooks. Does NOT touch prev_modules (o_proj, down_proj, embed_tokens), does NOT fuse RMSNorm
   - Hadamard rotation changes weight distribution → harms MXFP4's group_size=32 shared-exponent quantization

8. Implemented online R1 rotation matching Quark's behavior (THIS SESSION)
   - Changed `online_r1_rotation` default to `True`
   - Added `_apply_online_r1()` method and `_make_online_r1_hook()` closure
   - Modified `preprocess()` to skip RMSNorm fusion and untie when online R1
   - Updated transformation summary table for online R1
   - **Verified (limit=200):** Online R1 Δ=-0.03/+0.04 vs Offline R1 Δ=-0.085/-0.08
   - Fixed missing `import math` bug

9. User ran full eval with rotation_size=128 — results are poor (hellaswag=0.2628, piqa=0.5261)
   - This is WORSE than both MXFP4-only (~0.40/0.59) and our limit=200 test (~0.43/0.63)
   - **Issue is unresolved** — needs investigation
</history>

<work_done>
Files created:
- `auto_round/algorithms/transforms/spinquant/monkeypatch.py`: Architecture-generic R3 injection
- `examples/test_rotation_quantization.py`: Rotation + W4A16 RTN test
- `examples/test_quark_rotation_mxfp4.py`: Quark rotation + MXFP4 comparison
- `examples/test_autoround_rotation_mxfp4.py`: Auto-round rotation + MXFP4 comparison
- `examples/run_comparison_tests.sh`: Shell script for all comparison scenarios
- `examples/debug_rotation_mxfp4.py`: 6-step diagnostic script
- `docs/r3_r4_online_rotation.md`: R3/R4 documentation

Files modified (this session):
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - Added `import math`
  - Changed `online_r1_rotation` default from `False` to `True`
  - Modified `preprocess()`: skips RMSNorm fusion and untie_embeddings when online R1
  - Added `_make_online_r1_hook()` module-level function (closure for forward_pre_hook)
  - Added `_apply_online_r1()` method: rotates target modules + registers hooks
  - Updated `_print_transformation_summary()` for online R1 display
  - Added `get_hadamard_K` and `matmul_hadU` to imports
- `examples/test_autoround_rotation_mxfp4.py`: Added `--online-r1`/`--offline-r1` args, updated `apply_rotation()`
- `examples/test_rotation_quantization.py`: Updated `apply_rotation()` to accept `online_r1` param
- `examples/test_qwen3_rotation_eval.py`: Updated `apply_rotation()` to accept `online_r1` param

Current state:
- [x] Online R1 implementation complete
- [x] All 7 mock tests pass
- [x] Quick test (limit=200) showed online R1 dramatically better than offline R1
- [ ] **PROBLEM: Full eval with rotation_size=128 gives very bad accuracy (hellaswag=0.2628)**
  - This was the user's last run, showing online R1 + R2 + MXFP4 + rotation_size=128 gives ~0.26 hellaswag
  - My earlier limit=200 test used rotation_size=None (full 1024), NOT 128
  - The issue may be related to rotation_size=128 (block rotation) interacting badly with online R1
</work_done>

<technical_details>
**Online R1 Implementation (matching Quark):**
- `apply_online_r1()` in Quark (rotation.py line 879): for each target_module:
  1. If rotation_size == in_features: `matmul_hadU(weight)` (butterfly)
  2. Else: `rotate_in_channels_(layer, rotation=hadamard_K / sqrt(rotation_size))`
  3. Wrap with `InputRotationWrapperHadamard` (hook applies same Hadamard to activation)
- The math: `y = H(x) @ (W@H)^T = x @ (H @ H^T) @ W^T = x @ W^T` (identity, since H orthogonal)
- Online R1 does NOT fuse RMSNorm, does NOT modify prev_modules, does NOT touch lm_head
- Quark skips `last_layer` entries when `online_r1_rotation=True` (line 652)

**Why Online R1 Fixes MXFP4 Degradation:**
- Offline R1: changes weight distribution of ALL layers → MXFP4's group_size=32 quantization suffers
- Online R1: only target_modules (q/k/v, gate/up) have rotated weights; prev_modules (o_proj, down_proj) untouched → quantization of most critical layers preserved

**CRITICAL UNRESOLVED ISSUE — rotation_size=128 gives terrible accuracy:**
- My limit=200 test used `rotation_size=None` (→ full 1024) and got good results
- User's full eval used `rotation_size=128` and got hellaswag=0.2628 (terrible)
- This suggests block rotation (128×128 blocks, 8 blocks per hidden_size=1024) has a problem
- Possible causes:
  1. The block Hadamard in the hook doesn't match the block rotation on the weight
  2. When rotation_size < in_features, the hook uses a different code path (block rotation via matrix multiply) that may have a bug
  3. The `_make_online_r1_hook` block rotation path normalizes differently than the weight rotation

**Key code path for block rotation in `_apply_online_r1`:**
```python
if r1_size == in_features:
    # Full rotation via matmul_hadU (butterfly)
    module.weight.data = matmul_hadU(W, hadamard_K, K).to(dtype)
else:
    # Block rotation: build Hadamard and use rotate_in_channels_
    R_block = had_K_local.to(float64) / math.sqrt(r1_size)
    rotate_in_channels_(module, R_in=R_block)
```

And the hook for block rotation:
```python
rot_mat = (had_K / sqrt(rotation_size)).float()
def hook(module, args):
    x_reshaped = x.reshape(*shape[:-1], -1, rotation_size)
    x_rotated = (x_reshaped @ R).reshape(shape)
```

**Potential Bug:** `rotate_in_channels_` does `W @ R.T`, but the hook does `x @ R`. For the cancellation:
- Weight: W_new = W @ R.T (from rotate_in_channels_)
- Hook: x_new = x @ R
- Result: y = x_new @ W_new.T = (x @ R) @ (W @ R.T).T = (x @ R) @ R @ W.T = x @ W.T ✓ (if R orthogonal)

Wait — but `R_block = had_K / sqrt(r1_size)` is passed to `rotate_in_channels_` which does `W @ R.T`. And the hook uses `rot_mat = had_K / sqrt(r1_size)` and does `x @ rot_mat`. For a normalized Hadamard H (H@H=I), R=H, so W@H.T = W@H (symmetric) and x@H. Then (x@H)@(W@H).T = x@H@H@W.T = x@W.T ✓.

But WAIT: `had_K_local` from `get_hadamard_K(128)` returns `(had, K=1)` for pow2 sizes. The had is a NORMALIZED 128×128 Hadamard (divided by sqrt(128)). Then `R_block = had / sqrt(128)` = had / sqrt(128) = original_H / sqrt(128) / sqrt(128) = H_unnorm / 128. This is DOUBLE NORMALIZATION! 

This is likely the bug causing rotation_size=128 to fail.

**Environment:**
- 8x NVIDIA L20 (44.4GB each), CUDA, Python 3.12, transformers 4.57.6, lm_eval 0.4.11
- auto-round installed editable at /data/lkk/quarot/auto-round
- Quark at /data/lkk/quarot/Quark

**Qwen3-0.6B Architecture:**
- hidden_size=1024, head_dim=128, num_heads=16 (GQA: 16 q, 8 kv)
- intermediate_size=3072, 28 layers
- tie_word_embeddings=True in config
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator: 8-step pipeline in `preprocess()` (line ~188)
  - SpinQuantConfig dataclass with `online_r1_rotation=True` default (line ~74)
  - `_apply_online_r1()` at line ~690: the new online R1 implementation
  - `_make_online_r1_hook()` at line ~55: closure for activation Hadamard hooks
  - `_fuse_offline_rotations()` at line ~733: the old offline R1 path
  - **SUSPECTED BUG:** `_apply_online_r1` block rotation path at line ~734: `R_block = had_K_local.to(torch.float64) / math.sqrt(r1_size)` — this double-normalizes because `get_hadamard_K` already returns normalized matrix

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - `get_hadamard_K(n)` at line 50: returns NORMALIZED Hadamard (divided by sqrt(n)) for pow2
  - `matmul_hadU()` at line 75: butterfly algorithm, also normalizes (divides by sqrt(n/K))
  - `rotate_in_channels_()` at line 172: does `W @ R.T` (block or full)
  - `deterministic_hadamard_matrix()` at line 24: returns H/sqrt(N) (NORMALIZED)

- `/data/lkk/quarot/auto-round/examples/test_autoround_rotation_mxfp4.py`
  - The test script user ran showing bad accuracy with rotation_size=128
  - `apply_rotation()` at line 73: passes rotation_size and online_r1 to SpinQuantConfig

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - `apply_online_r1()` at line 879: reference implementation
  - Line 905: `hadamard_K, K = _get_hadamard_K(rotation_size)` — Quark's version returns UNNORMALIZED matrix
  - Line 913: `matmul_hadU(weight, hadamard_K, K)` — Quark's matmul_hadU does normalization
  - Line 925: `rotate_in_channels_(layer, rotation=hadamard_K / sqrt(rotation_size))` — explicitly normalizes

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation_utils.py`
  - `_get_hadamard_K()` — returns UNNORMALIZED Hadamard matrices (just ±1 values)
  - `HadamardTransform.forward()` at line 426: uses matmul_hadU when full, rotate_with_size when block
  - `rotate_with_size()` at line 56: does `x @ rotation_matrix` (rotation_matrix must be pre-normalized)
</important_files>

<next_steps>
**IMMEDIATE — Fix rotation_size=128 accuracy bug:**

The user's full eval shows hellaswag=0.2628 with rotation_size=128, which is catastrophically bad. The likely cause is **double normalization** in the block rotation path:

1. `get_hadamard_K(128)` returns a NORMALIZED matrix (divided by sqrt(128))
2. `_apply_online_r1` then does `R_block = had_K_local / math.sqrt(r1_size)` — dividing by sqrt(128) AGAIN
3. The hook also uses this double-normalized matrix
4. Result: the rotation and hook don't cancel → model output is wrong

**Fix plan:**
1. Check Quark's `_get_hadamard_K` — it likely returns UNNORMALIZED (just ±1) matrices
2. In our `_apply_online_r1()`, remove the extra `/ math.sqrt(r1_size)` since our `get_hadamard_K` already normalizes
3. OR: change `get_hadamard_K` to return unnormalized (matching Quark) and normalize in the callers
4. Also fix the hook's `rot_mat = had_K / sqrt(rotation_size)` line

**Verification steps:**
1. Test with rotation_size=128: should get cosine_sim≈1.0 vs baseline (currently broken)
2. Test with rotation_size=None (full 1024): should still work (was fine in limit=200 test)
3. Re-run full eval with the fix

**Other pending work:**
- Port non-pow2 Hadamard matrices from Quark (TODO)
- Phase 2: RotationLinear training wrapper (future)
- Phase 3: Integration with auto-round tuning loop (future)
</next_steps>