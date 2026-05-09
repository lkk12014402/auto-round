<overview>
The user is migrating SpinQuant/QuaRot rotation implementations from AMD's Quark codebase to Intel's auto-round framework, and comparing three frameworks (auto-round, Quark, llm-compressor) for rotation + quantization. This session focused on: (1) expanding the three-way comparison document with detailed R2/R3/R4 sections, (2) adding online vs offline R1 accuracy impact analysis, (3) analyzing QuaRot vs SpinQuant algorithm differences, (4) fixing rotation matrix initialization to align with Quark (adding `random_r1/r2` config, fixing R3 to use deterministic Hadamard), and (5) updating documentation throughout.
</overview>

<history>
1. User asked to expand R2 and R3 coverage in `three_way_framework_comparison.md` (continuing from previous checkpoint where explore agent had completed R2/R3 analysis)
   - Read explore agent output from `/tmp/1778302945840-copilot-tool-output-sl942v.txt`
   - Added comprehensive §4 (R2, 5 subsections), §5 (R3, 6 subsections), §6 (R4, 6 subsections) with math, code details per framework, comparison tables
   - Renumbered all remaining sections (old §4→§7 through old §10→§13)
   - Fixed all sub-section numbering throughout the document
   - Document grew from ~505 to 1055 lines

2. User reported llm-compressor accuracy was much worse and asked why; requested online vs offline R1 accuracy impact be added to docs
   - Launched explore agent to analyze all three frameworks' R1-R4 online/offline modes
   - Expanded §3 from 24 to 144 lines with detailed accuracy impact data (offline R1 → hellaswag=0.2628 catastrophic)
   - Added new §3A "Complete Online/Offline Summary" with 4 sub-tables covering all R1-R4 × all frameworks
   - Documented why each rotation level is online or offline (mathematical reasons)
   - Confirmed auto-round R1 can be both online and offline via `online_r1_rotation` config

3. User asked whether llm-compressor supports trainable rotation matrices
   - Found `validate_not_implemented()` in llm-compressor's `base.py` line 115-118 that raises `NotImplementedError` for both `learnable=True` and `randomize=True`
   - Confirmed only deterministic Hadamard works in llm-compressor

4. User asked why auto-round doesn't support random Hadamard and what's the difference between QuaRot and deterministic
   - Discovered auto-round DOES support random Hadamard — `_init_rotation_matrices()` calls `random_hadamard_matrix()` for R1/R2/R3
   - Previous comparison table was wrong — I had confused "computation engine" (`matmul_hadU` uses deterministic internally) with "matrix choice" (R1/R2 actually use random)
   - Explained the three matrix types: deterministic H, random H×D (QuaRot), trainable (SpinQuant)
   - Corrected the §1.2 table in the document

5. User asked why the previous comparison table was wrong
   - Explained the confusion: `deterministic_hadamard_matrix()` appears frequently because it's the building block for both butterfly algorithm AND random Hadamard construction
   - The actual rotation matrices for R1/R2 were `random_hadamard_matrix()` = H × D
   - R4 must use deterministic because butterfly algorithm needs Sylvester recursive structure

6. User asked to add R3/R4 always-deterministic explanation to docs
   - Updated document: merged R3/R4 explanation, added Quark code evidence (R3 `matmul_hadU(q)` no args, R4 `random=False`)
   - Fixed auto-round R3 in §1.2 table from "Random" to "Deterministic" (runtime uses `matmul_hadU` which is deterministic)
   - Added pattern summary: "R1/R2 offline → can use Random; R3/R4 online → must use Deterministic"

7. User asked about QuaRot vs SpinQuant differences in Quark and llm-compressor
   - Deep analysis of Quark's `RotationConfig`: found `random_r1=False` (default!), `random_r2=False`, `trainable=False`
   - Found Quark defaults are deterministic, NOT random (corrected previous assumption)
   - Analyzed full SpinQuant training pipeline: RotationLinear wrappers → Cayley SGD → post_process_trained_rotation
   - Added new §14 "QuaRot vs SpinQuant: Algorithm Differences" (6 subsections, ~120 lines)
   - Fixed §1.2 and §1.4 tables to show Quark defaults as deterministic

8. User asked about SpinQuant mode rotation matrix initialization
   - Confirmed: SpinQuant mode (`trainable=True`) initializes from `get_rotation_matrix(random=self.random_r1)` then wraps as `nn.Parameter`
   - Default `random_r1=False` → deterministic Hadamard initialization, then trained via Cayley SGD
   - Final matrix is learned orthogonal (no longer Hadamard structure)

9. User asked if auto-round QuaRot can only initialize with random
   - Confirmed: auto-round code always used `random_hadamard_matrix()` for non-trainable R1/R2
   - No config toggle existed (unlike Quark's `random_r1/r2`)
   - SpinQuant mode used `torch.eye()` (identity init) — different from Quark (Hadamard init)

10. User asked to fix the initialization and align with Quark's config
    - Added `random_r1: bool = False` and `random_r2: bool = False` to `SpinQuantConfig`
    - Updated `_init_rotation_matrices()` to respect new config: deterministic or random based on flag
    - Fixed R3 init from `random_hadamard_matrix()` to `deterministic_hadamard_matrix()` (matching runtime behavior)
    - Tested all modes: deterministic R1 (cos=0.9997), random R1 (cos=0.9997), R1+R2+R3 both modes, R1+R2+R3+R4 with rotation_size=128 — all pass

11. User asked to update the document with all these changes
    - Was about to update when compaction triggered
</history>

<work_done>
Files modified this session:
- `auto_round/algorithms/transforms/spinquant/preprocessor.py`:
  - Added `random_r1: bool = False` and `random_r2: bool = False` to `SpinQuantConfig` (lines ~79-86)
  - Rewrote `_init_rotation_matrices()` (lines ~403-490):
    - R1: respects `random_r1` flag (deterministic or random Hadamard)
    - R2: respects `random_r2` flag
    - R3: changed from `random_hadamard_matrix()` to `deterministic_hadamard_matrix()` (fixes bug)
    - R4: unchanged (already deterministic)
    - SpinQuant trainable mode: unchanged (`torch.eye()` init)
  - Updated log messages to show "Deterministic Hadamard" or "Random Hadamard"

- `docs/three_way_framework_comparison.md`:
  - §1 rewritten (4 subsections → 7 subsections): Three matrix types, per-framework per-rotation table, computation engine vs matrix choice distinction, framework support summary, normalization, algorithm, rotation size
  - §3 expanded (24 → 144 lines): per-module comparison, accuracy impact with real data, auto-round/Quark both-mode support, llm-compressor offline-only
  - §3A added (new section): Complete online/offline summary tables, mathematical reasons, auto-round switchability
  - §4 added (R2 detailed, 5 subsections): math, code per framework, comparison table
  - §5 added (R3 detailed, 6 subsections): math, monkeypatch vs attention swapping, comparison table, why after RoPE
  - §6 added (R4 detailed, 6 subsections): math, block rotation, code per framework, comparison table
  - §7-§13 renumbered from old §4-§10 with sub-section numbers fixed
  - §14 added (QuaRot vs SpinQuant, 6 subsections): algorithm differences, math, framework mapping, Quark configs, training pipeline, usage guidance
  - Fixed §1.2 table: Quark defaults corrected to `random_r1=False` (was incorrectly `True`)
  - Fixed §1.2 table: auto-round R3 corrected to "Deterministic" (was incorrectly "Random")
  - Document grew from ~505 → 1431 lines

Work completed:
- [x] Expanded R2/R3/R4 sections in three-way comparison doc
- [x] Added online vs offline R1 accuracy impact analysis
- [x] Analyzed llm-compressor trainable rotation support (NotImplementedError)
- [x] Fixed rotation matrix type confusion (computation engine vs matrix choice)
- [x] Added QuaRot vs SpinQuant algorithm comparison section
- [x] Added `random_r1/r2` config fields aligned with Quark
- [x] Fixed R3 init bug (was random, should be deterministic)
- [x] Tested all rotation modes (deterministic, random, R1-R4 combos)
- [ ] **PENDING: Update docs/three_way_framework_comparison.md with the code changes** (user asked, compaction triggered before execution)

Tests verified:
- Deterministic R1: cosine_sim=0.999650 ✅
- Random R1: cosine_sim=0.999650 ✅
- R1+R2+R3 deterministic: cosine_sim=0.999646 ✅
- R1+R2+R3 random: cosine_sim=0.999646 ✅
- R1+R2+R3+R4 (rotation_size=128) deterministic: cosine_sim=0.999944 ✅
</work_done>

<technical_details>
**Three Types of Rotation Matrices:**
| Type | Formula | Deterministic? | Must Persist? |
|------|---------|----------------|---------------|
| Deterministic Hadamard | H = Sylvester(N)/√N | Yes | No |
| Random Hadamard (QuaRot) | R = H × D, D = diag(±1) random | No | Yes (save D or R) |
| Trainable (SpinQuant) | R = Cayley(A) optimized | No | Yes (save trained R) |

**Which Matrix Type Per Rotation Level (after fixes):**
| Framework | R1 | R2 | R3 | R4 |
|-----------|----|----|----|----|
| auto-round | Configurable (`random_r1`), default deterministic | Configurable (`random_r2`), default deterministic | **Always deterministic** | **Always deterministic** |
| Quark | Configurable (`random_r1`), default `False` | Configurable (`random_r2`), default `False` | Always deterministic | Always deterministic (`random=False`) |
| llm-compressor | Deterministic only | Deterministic only | Deterministic only | Deterministic only |

**Why R3/R4 Must Be Deterministic:**
- Both are online (run every inference step)
- Use `matmul_hadU()` butterfly algorithm: O(N·logN)
- Butterfly exploits Sylvester `[[H,H],[H,-H]]` recursive structure
- Random diagonal D destroys this structure → falls back to O(N²) dense matmul
- Quark R3: `matmul_hadU(q)` no args = default deterministic
- Quark R4: `get_rotation_matrix(size, random=False)` explicit

**Computation Engine vs Matrix Choice (Key Confusion Source):**
- `deterministic_hadamard_matrix()` appears everywhere because it's:
  1. The butterfly algorithm's internal building block
  2. The base for constructing random Hadamard (H × D)
- This does NOT mean the rotation matrix itself is deterministic
- R1/R2 matrices can be random; R3/R4 matrices must be deterministic

**Auto-round R3 Init Bug (Fixed):**
- `_init_rotation_matrices()` called `random_hadamard_matrix()` for R3 buffer
- But `QKRotationWrapper.forward()` calls `matmul_hadU(q)` with no args → uses deterministic butterfly internally
- The stored random matrix was never actually used at runtime
- Fix: changed init to `deterministic_hadamard_matrix()`

**Quark Default Correction:**
- `RotationConfig.random_r1 = False` (not True as previously assumed)
- `RotationConfig.random_r2 = False`
- `RotationConfig.trainable = False`
- `RotationConfig.online_r1_rotation = None` (defaults to False if r1=True)
- Quark example configs don't set `random_r1/r2` → use deterministic default

**QuaRot vs SpinQuant in Quark:**
- QuaRot: `trainable=False` + optionally `random_r1=True` → fixed rotation, no training
- SpinQuant: `trainable=True` → init from deterministic H (or random), train via Cayley SGD
- Key: `trainable=True` + `r3=True` raises `NotImplementedError` in Quark
- SpinQuant training: RotationLinear wrapper → QDQ on activations → Cayley SGD → post_process_trained_rotation() fuses R into weights

**llm-compressor Limitations:**
- `learnable=True` → `NotImplementedError`
- `randomize=True` → `NotImplementedError`
- Only deterministic Hadamard works
- Only offline R1 (no online mode)

**Online R1 vs Offline R1 Accuracy (Qwen3-0.6B, hellaswag, MXFP4):**
| Config | Accuracy |
|--------|----------|
| FP16 baseline | ~0.4250 |
| Online R1 + MXFP4 | ~0.4550 |
| Offline R1 + MXFP4 | ~0.2628 (catastrophic) |
| MXFP4 only | ~0.4200 |

**Online/Offline Summary:**
| Rotation | Can be online? | Can be offline? | Why? |
|----------|---------------|----------------|------|
| R1 | ✅ (hook) | ✅ (global fuse) | Both mathematically equivalent in FP |
| R2 | ❌ | ✅ only | V/O directly connected, fully fuseable |
| R3 | ✅ only | ❌ | Must be after RoPE (position-dependent) |
| R4 | ✅ (activation) + ✅ (weight inverse) | Hybrid always | Activation runtime-dependent |

**Normalization Convention (CRITICAL - unchanged):**
- auto-round: `get_hadamard_K()` returns normalized H/√N
- Quark: `_get_hadamard_K()` returns unnormalized ±1
- Must NOT double-normalize

**Environment:**
- 8x NVIDIA L20, CUDA, Python 3.12
- auto-round at /data/lkk/quarot/auto-round (editable install)
- Quark at /data/lkk/quarot/Quark
- llm-compressor at /data/lkk/quarot/llm-compressor (editable install)
- GPUs 0-3 often busy; use GPUs 4-7
</technical_details>

<important_files>
- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/preprocessor.py`
  - Main orchestrator for rotation pipeline
  - **Modified this session:** Added `random_r1`, `random_r2` config fields (lines ~79-86)
  - **Modified this session:** Rewrote `_init_rotation_matrices()` (lines ~403-490) to respect random config and fix R3 deterministic init
  - Key methods: `preprocess()` (line ~240), `_apply_online_r1()` (line ~660+), `_fuse_offline_rotations()` (line ~750+)
  - `SpinQuantConfig` dataclass at line 63

- `/data/lkk/quarot/auto-round/docs/three_way_framework_comparison.md`
  - **Heavily modified this session:** Grew from ~505 to 1431 lines
  - 14 main sections covering all aspects of three-framework comparison
  - §1 (Rotation Matrices, 7 subsections), §3 (R1 Online/Offline, 5 subsections), §3A (Complete Online/Offline Summary)
  - §4 (R2 detailed), §5 (R3 detailed), §6 (R4 detailed)
  - §14 (QuaRot vs SpinQuant, 6 subsections) — new this session
  - **PENDING:** Needs update with the `random_r1/r2` code changes and R3 fix

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/rotation_utils.py`
  - Core utilities: `deterministic_hadamard_matrix()`, `random_hadamard_matrix()`, `matmul_hadU()`, `get_hadamard_K()`
  - `random_hadamard_matrix` = deterministic H × random diag(±1) D

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/monkeypatch.py`
  - `QKRotationWrapper`: R3 injection after RoPE
  - Line 125-126: `matmul_hadU(q.float())` — uses deterministic butterfly, confirming R3 is always deterministic at runtime

- `/data/lkk/quarot/auto-round/auto_round/algorithms/transforms/spinquant/inplace/apply.py`
  - R3 monkeypatch + R4 hook registration
  - `register_spinquant_hooks()` and `remove_spinquant_hooks()`

- `/data/lkk/quarot/Quark/quark/torch/quantization/config/config.py`
  - `RotationConfig` at line 2084: `random_r1=False`, `random_r2=False`, `trainable=False` defaults
  - Line 2271: `trainable=True` + `r3=True` raises `NotImplementedError`

- `/data/lkk/quarot/Quark/quark/torch/algorithm/rotation/rotation.py`
  - Main rotation class with `r1()`, `r2()`, `r3()`, `r4()` methods
  - Line 667: `get_rotation_matrix(size, random=self.random_r1)` for R1
  - Line 1019: `get_rotation_matrix(size, random=False)` for R4 (always deterministic)
  - Line 669-670: SpinQuant trainable R1 becomes `nn.Parameter`

- `/data/lkk/quarot/llm-compressor/src/llmcompressor/modifiers/transform/spinquant/base.py`
  - SpinQuantModifier with `learnable=False`, `randomize=False` fields
  - Line 115-118: `validate_not_implemented()` blocks `learnable=True` and `randomize=True`
</important_files>

<next_steps>
**Immediate (user explicitly requested, was about to execute when compaction triggered):**
- Update `docs/three_way_framework_comparison.md` with the code changes:
  - Update §1.2 table to reflect auto-round now has configurable `random_r1/r2` (default deterministic)
  - Update §1.4 framework support summary
  - Add note about R3 init bug fix
  - Update §14 to mention auto-round's alignment with Quark config

**Pending from earlier:**
- User will test three-way comparison script and report logs
- User will test W4A16/NVFP4 rotation scheme scripts
- Rotation config persistence (save/load) — design documented, implementation deferred
</next_steps>