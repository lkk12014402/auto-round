#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive Rotation × Quantization Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
#
# Full-coverage test matrix across:
#   Models:        Qwen3-0.6B, Qwen3-8B, Llama-3.1-8B-Instruct
#   Rotations:     none, R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
#   Schemes:       W4A16, MXFP4, NVFP4
#   Iters:         0 (RTN), 200 (tuning)
#   Rot. sizes:    16, 32, 128, auto
#   Rot. modes:    hadamard, random (+ per-rotation combos)
#   Tasks:         hellaswag, piqa, winogrande, lambada_openai
#
# Organized into 10 parts, dispatched across 4 or 8 GPUs:
#
#   Part 1:  Deterministic × RTN               (15 combos/model)
#   Part 2:  Deterministic × tuning(200)        (15 combos/model)
#   Part 3:  Rotation size sweep × RTN          (48 combos/model)
#   Part 4:  Rotation size sweep × tuning       (48 combos/model)
#   Part 5:  Random all × RTN                   (12 combos/model)
#   Part 6:  Random all × tuning                (12 combos/model)
#   Part 7:  Per-rotation random combos × RTN   ( 8 combos/model)
#   Part 8:  Per-rotation random combos × tuning( 4 combos/model)
#   Part 9:  Baseline FP16 (no quantization)    ( 6 combo/model)
#   Part 10: Size sweep + random                (12 combos/model)
#   Part 11: FP16 + rotation (rotation lossless)(5 combos/model)
#   Part 12: Wikitext perplexity (key combos)   (10 combos/model)
#
# Usage:
#   bash run_benchmark.sh                              # 4-GPU default
#   bash run_benchmark.sh --gpus 8                     # 8-GPU mode
#   bash run_benchmark.sh --models "Qwen/Qwen3-0.6B"  # single model
#   bash run_benchmark.sh --parts 1,2,5                # selective parts
#   bash run_benchmark.sh --dry-run                    # show plan only
#   bash run_benchmark.sh --wait                       # wait for all jobs
#
# Results:
#   results_benchmark_<timestamp>/
#   ├── <model>_part<N>/                 # per-job output
#   │   └── results.json
#   ├── logs/
#   │   └── <model>_part<N>.log
#   └── summary.json                     # aggregated at the end
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════════════
MODELS="${MODELS:-Qwen/Qwen3-0.6B,Qwen/Qwen3-8B,meta-llama/Llama-3.1-8B-Instruct}"
GPUS="${GPUS:-4}"
PARTS="${PARTS:-1,2,3,4,5,6,7,8,9,10,11,12}"
TASKS="${TASKS:-hellaswag,piqa,winogrande,lambada_openai}"
LIMIT="${LIMIT:-}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-512}"
DRY_RUN=0
WAIT_MODE=0
OUTDIR_OVERRIDE=""
SEED="${SEED:-42}"

# ═══════════════════════════════════════════════════════════════════════════════
# Parse CLI
# ═══════════════════════════════════════════════════════════════════════════════
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)    MODELS="$2";           shift 2;;
        --gpus)      GPUS="$2";             shift 2;;
        --parts)     PARTS="$2";            shift 2;;
        --tasks)     TASKS="$2";            shift 2;;
        --limit)     LIMIT="$2";            shift 2;;
        --nsamples)  NSAMPLES="$2";         shift 2;;
        --seqlen)    SEQLEN="$2";           shift 2;;
        --outdir)    OUTDIR_OVERRIDE="$2";  shift 2;;
        --seed)      SEED="$2";             shift 2;;
        --dry-run)   DRY_RUN=1;             shift;;
        --wait)      WAIT_MODE=1;           shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Parse models
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

# Parse parts into lookup
IFS=',' read -ra PART_ARRAY <<< "$PARTS"
declare -A RUN_PART
for p in "${PART_ARRAY[@]}"; do
    if [[ ! $p =~ ^([1-9]|1[0-2])$ ]]; then
        echo "ERROR: Invalid part: $p (must be 1-12)"
        exit 1
    fi
    RUN_PART[$p]=1
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$OUTDIR_OVERRIDE" ]]; then
    OUTDIR="$OUTDIR_OVERRIDE"
else
    OUTDIR="results_benchmark_${TIMESTAMP}"
fi
LOG_DIR="$OUTDIR/logs"
mkdir -p "$LOG_DIR"

LIMIT_FLAG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# GPU scheduling
# ═══════════════════════════════════════════════════════════════════════════════
declare -a GPU_QUEUE=()
for ((i=0; i<GPUS; i++)); do
    GPU_QUEUE+=($i)
done
NEXT_GPU_IDX=0

get_next_gpu() {
    # Sets CURRENT_GPU (avoids subshell from $() which loses state)
    CURRENT_GPU=${GPU_QUEUE[$NEXT_GPU_IDX]}
    NEXT_GPU_IDX=$(( (NEXT_GPU_IDX + 1) % ${#GPU_QUEUE[@]} ))
}

# ═══════════════════════════════════════════════════════════════════════════════
# Job tracking
# ═══════════════════════════════════════════════════════════════════════════════
declare -a PIDS=()
declare -a LABELS=()
JOB_COUNT=0

# Wait for a GPU slot if all are busy (max concurrent = num GPUs)
wait_for_slot() {
    while (( ${#PIDS[@]} >= GPUS )); do
        # Wait for any child to finish
        local new_pids=()
        local new_labels=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                new_pids+=("${PIDS[$i]}")
                new_labels+=("${LABELS[$i]}")
            else
                wait "${PIDS[$i]}" 2>/dev/null || true
                echo "  ✓ ${LABELS[$i]} (PID ${PIDS[$i]}) — finished"
            fi
        done
        PIDS=("${new_pids[@]}")
        LABELS=("${new_labels[@]}")
        if (( ${#PIDS[@]} >= GPUS )); then
            sleep 5
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════════════════
# Launch function
# ═══════════════════════════════════════════════════════════════════════════════
launch_job() {
    local model="$1"
    local part="$2"
    local desc="$3"
    shift 3
    # Remaining args are the test_save_load_roundtrip.py arguments

    local model_short=$(echo "$model" | sed 's|.*/||')
    local label="${model_short}_part${part}_${desc}"
    local log_file="${LOG_DIR}/${label}.log"
    local out_subdir="${OUTDIR}/${label}"
    get_next_gpu
    local gpu=$CURRENT_GPU

    JOB_COUNT=$((JOB_COUNT + 1))

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY-RUN] GPU:${gpu} | ${label}"
        echo "            python3 test_save_load_roundtrip.py --model $model --device cuda:${gpu} $*"
        return
    fi

    wait_for_slot

    echo "  [Job #${JOB_COUNT}] GPU:${gpu} | ${label} → ${log_file}"

    python3 test_save_load_roundtrip.py \
        --model "$model" \
        --device "cuda:${gpu}" \
        --tasks "$TASKS" \
        --nsamples "$NSAMPLES" \
        --seqlen "$SEQLEN" \
        --seed "$SEED" \
        $LIMIT_FLAG \
        --output-dir "$out_subdir" \
        --cleanup \
        "$@" \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    LABELS+=("$label")
}

# ═══════════════════════════════════════════════════════════════════════════════
# Print plan
# ═══════════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  COMPREHENSIVE ROTATION × QUANTIZATION BENCHMARK"
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Models:       ${MODEL_ARRAY[*]}"
echo "  GPUs:         $GPUS (cuda:0..cuda:$((GPUS-1)))"
echo "  Parts:        $PARTS"
echo "  Tasks:        $TASKS"
echo "  Limit:        ${LIMIT:-full}"
echo "  Nsamples:     $NSAMPLES"
echo "  Seqlen:       $SEQLEN"
echo "  Seed:         $SEED"
echo "  Output:       $OUTDIR"
echo "  Dry-run:      $DRY_RUN"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Launch all jobs
# ═══════════════════════════════════════════════════════════════════════════════

for MODEL in "${MODEL_ARRAY[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
    echo ""
    echo "╔═════════════════════════════════════════════════════════════════╗"
    echo "║  Model: $MODEL"
    echo "╚═════════════════════════════════════════════════════════════════╝"

    # ─────────────────────────────────────────────────────────────────────
    # Part 1: Deterministic Hadamard × RTN (5 rotations × 3 schemes = 15)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[1]:-}" ]]; then
        echo ""
        echo "  ── Part 1: Deterministic × RTN ──"
        launch_job "$MODEL" 1 "det_rtn" \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --quant-iters 0
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 2: Deterministic Hadamard × Tuning (5 × 3 = 15)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[2]:-}" ]]; then
        echo ""
        echo "  ── Part 2: Deterministic × Tuning (iters=200) ──"
        launch_job "$MODEL" 2 "det_tune" \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --quant-iters 200
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 3: Rotation size sweep × RTN (4 rotations × 4 sizes × 3 sch = 48)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[3]:-}" ]]; then
        echo ""
        echo "  ── Part 3: Rotation size sweep × RTN ──"
        launch_job "$MODEL" 3 "size_rtn" \
            --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --rotation-sizes "16,32,128,auto" \
            --quant-iters 0
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 4: Rotation size sweep × Tuning (4 × 4 × 3 = 48)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[4]:-}" ]]; then
        echo ""
        echo "  ── Part 4: Rotation size sweep × Tuning ──"
        launch_job "$MODEL" 4 "size_tune" \
            --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --rotation-sizes "16,32,128,auto" \
            --quant-iters 200
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 5: Random all × RTN (4 rotations × 3 schemes = 12)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[5]:-}" ]]; then
        echo ""
        echo "  ── Part 5: Random (all) × RTN ──"
        launch_job "$MODEL" 5 "rand_all_rtn" \
            --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --rotation-modes "random" \
            --quant-iters 0
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 6: Random all × Tuning (4 × 3 = 12)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[6]:-}" ]]; then
        echo ""
        echo "  ── Part 6: Random (all) × Tuning ──"
        launch_job "$MODEL" 6 "rand_all_tune" \
            --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --rotation-modes "random" \
            --quant-iters 200
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 7: Per-rotation random combos × RTN (8 combos)
    #   - R1+R2, only R2 random         × W4A16,MXFP4,NVFP4 (3)
    #   - R1+R2+R3+R4, R2+R4 random     × W4A16               (1)
    #   - R1+R2+R3+R4, R1+R3 random     × W4A16               (1)
    #   - R1+R2+R3+R4, R1+R2+R3+R4 rand × W4A16,MXFP4,NVFP4 (3)
    #   Split into sub-jobs since --random-rotations can't sweep
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[7]:-}" ]]; then
        echo ""
        echo "  ── Part 7: Per-rotation random × RTN ──"

        # 7a: R1+R2, only R2 random
        launch_job "$MODEL" 7a "rand_R2_rtn" \
            --rotations "R1+R2" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --random-rotations "R2" \
            --quant-iters 0

        # 7b: R1+R2+R3+R4, R2+R4 random
        launch_job "$MODEL" 7b "rand_R2R4_rtn" \
            --rotations "R1+R2+R3+R4" \
            --schemes "W4A16" \
            --random-rotations "R2,R4" \
            --quant-iters 0

        # 7c: R1+R2+R3+R4, R1+R3 random
        launch_job "$MODEL" 7c "rand_R1R3_rtn" \
            --rotations "R1+R2+R3+R4" \
            --schemes "W4A16" \
            --random-rotations "R1,R3" \
            --quant-iters 0

        # 7d: R1+R2+R3+R4, all 4 random
        launch_job "$MODEL" 7d "rand_all4_rtn" \
            --rotations "R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --random-rotations "R1,R2,R3,R4" \
            --quant-iters 0
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 8: Per-rotation random combos × Tuning (4 combos)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[8]:-}" ]]; then
        echo ""
        echo "  ── Part 8: Per-rotation random × Tuning ──"

        # 8a: R1+R2, only R2 random
        launch_job "$MODEL" 8a "rand_R2_tune" \
            --rotations "R1+R2" \
            --schemes "W4A16,MXFP4" \
            --random-rotations "R2" \
            --quant-iters 200

        # 8b: R1+R2+R3+R4, all 4 random
        launch_job "$MODEL" 8b "rand_all4_tune" \
            --rotations "R1+R2+R3+R4" \
            --schemes "W4A16" \
            --random-rotations "R1,R2,R3,R4" \
            --quant-iters 200
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 9: FP16 Baseline — no quantization, optional rotation
    # Measures the ceiling (FP16 accuracy) as reference
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[9]:-}" ]]; then
        echo ""
        echo "  ── Part 9: FP16 Baseline (no quantization) ──"
        launch_job "$MODEL" 9 "fp16_baseline" \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "FP16" \
            --quant-iters 0 \
            --skip-inmemory
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 10: Rotation size + random (4 rotations × 3 sizes = 12)
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[10]:-}" ]]; then
        echo ""
        echo "  ── Part 10: Rotation size × random × RTN ──"
        launch_job "$MODEL" 10 "size_rand" \
            --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16" \
            --rotation-sizes "32,128,auto" \
            --rotation-modes "random" \
            --quant-iters 0
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 11: FP16 + rotation + random (verify rotation is lossless)
    # Compares FP16 accuracy with and without rotation to confirm
    # rotation introduces no accuracy degradation on its own
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[11]:-}" ]]; then
        echo ""
        echo "  ── Part 11: FP16 + rotation + random (lossless check) ──"

        # 11a: FP16 + rotation (hadamard)
        launch_job "$MODEL" 11a "fp16_rot_had" \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "FP16" \
            --quant-iters 0 \
            --skip-inmemory

        # 11b: FP16 + rotation (random)
        launch_job "$MODEL" 11b "fp16_rot_rand" \
            --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "FP16" \
            --rotation-modes "random" \
            --quant-iters 0 \
            --skip-inmemory
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Part 12: Wikitext perplexity on key combinations
    # Uses wikitext2 for stable perplexity metric in addition to accuracy
    # ─────────────────────────────────────────────────────────────────────
    if [[ -n "${RUN_PART[12]:-}" ]]; then
        echo ""
        echo "  ── Part 12: Wikitext perplexity ──"

        # 12a: FP16 baseline + key quant combos with wikitext
        launch_job "$MODEL" 12a "ppl_baseline" \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "FP16,W4A16" \
            --quant-iters 0 \
            --tasks "wikitext" \
            --skip-inmemory

        # 12b: MXFP4/NVFP4 with wikitext
        launch_job "$MODEL" 12b "ppl_fp4" \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "MXFP4,NVFP4" \
            --quant-iters 0 \
            --tasks "wikitext" \
            --skip-inmemory
    fi

done

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  ALL JOBS LAUNCHED"
echo "  Total jobs:    $JOB_COUNT"
echo "  Active PIDs:   ${PIDS[*]:-none}"
echo "  Output dir:    $OUTDIR"
echo "  Log dir:       $LOG_DIR"
echo "  Monitor:       tail -f $LOG_DIR/<job>.log"
echo "═══════════════════════════════════════════════════════════════════════════"

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "  (Dry-run mode — no jobs were actually started)"
    echo ""
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Wait for all jobs
# ═══════════════════════════════════════════════════════════════════════════════
if [[ "$WAIT_MODE" == "1" ]]; then
    echo ""
    echo "Waiting for all ${#PIDS[@]} jobs to finish..."
    FAILED=0
    SUCCEEDED=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        label=${LABELS[$i]}
        if wait "$pid"; then
            echo "  ✓ ${label} (PID ${pid}) — done"
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "  ✗ ${label} (PID ${pid}) — FAILED (exit $?)"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  All jobs finished at: $(date)"
    echo "  Succeeded: $SUCCEEDED / $((SUCCEEDED + FAILED))"
    echo "  Failed:    $FAILED"
    echo "═══════════════════════════════════════════════════════════════"

    # ── Aggregate results ──
    echo ""
    echo "Aggregating results..."

    python3 - "$OUTDIR" <<'PYEOF'
import json, glob, os, sys

outdir = sys.argv[1]

# Collect all results.json files (both in subdirs and directly)
patterns = [
    os.path.join(outdir, "*/results.json"),
    os.path.join(outdir, "*/*/results.json"),
]
rfiles = set()
for pat in patterns:
    rfiles.update(glob.glob(pat))
rfiles = sorted(rfiles)

if not rfiles:
    print("  No results.json files found.")
    sys.exit(0)

all_results = []
for rfile in rfiles:
    rel = os.path.relpath(rfile, outdir)
    part_label = os.path.dirname(rel)
    try:
        with open(rfile) as f:
            data = json.load(f)
        for r in data:
            r["source"] = part_label
        all_results.extend(data)
        print(f"  Loaded {len(data):>3} results from {rel}")
    except Exception as e:
        print(f"  ERROR loading {rel}: {e}")

# Save combined
combined = os.path.join(outdir, "summary.json")
with open(combined, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n  Combined {len(all_results)} results → {combined}")

# ── Print summary table ──
print()

# Discover all tasks
all_tasks = set()
for r in all_results:
    for k in ("metrics_from_disk", "metrics_in_memory"):
        m = r.get(k, {})
        if isinstance(m, dict):
            all_tasks.update(m.keys())
all_tasks = sorted(all_tasks)

# Build header
task_cols = "".join(f" {t[:12]:>12}" for t in all_tasks)
hdr = f"{'Model':<20} {'Rotation':<16} {'Scheme':<8} {'Iters':>5} {'RSize':>5} {'Random':<12}{task_cols}  {'Match':>5} {'Status':<6}"
sep = "-" * len(hdr)

print(hdr)
print(sep)

for r in all_results:
    model = r.get("model", "?").split("/")[-1][:19]
    rot = r.get("rotation", "?")[:15]
    sch = r.get("scheme", "?")[:7]
    it = str(r.get("quant_iters", "?"))
    rs = str(r.get("rotation_size", "auto") or "auto")[:5]

    rand = r.get("random_set")
    if rand:
        rand_str = "+".join(s.upper() for s in sorted(rand))
    elif r.get("rotation_mode") == "random":
        rand_str = "all"
    else:
        rand_str = "-"

    status = r.get("status", "?")[:5]
    match = r.get("roundtrip_match")
    match_str = "✓" if match is True else ("✗" if match is False else "-")

    # Prefer disk metrics, fall back to in_memory
    disk_m = r.get("metrics_from_disk", {}) or {}
    mem_m = r.get("metrics_in_memory", {}) or {}

    task_vals = ""
    for t in all_tasks:
        v = disk_m.get(t)
        if v is None:
            v = mem_m.get(t)
        if v is not None:
            task_vals += f" {v:12.4f}"
        else:
            task_vals += f" {'---':>12}"

    print(f"{model:<20} {rot:<16} {sch:<8} {it:>5} {rs:>5} {rand_str:<12}{task_vals}  {match_str:>5} {status:<6}")

print(sep)
print(f"Total: {len(all_results)} experiments")
print()

# ── Per-model summary ──
from collections import defaultdict
model_stats = defaultdict(lambda: {"total": 0, "success": 0, "error": 0, "match": 0, "mismatch": 0})
for r in all_results:
    m = r.get("model", "?").split("/")[-1]
    model_stats[m]["total"] += 1
    if r.get("status") == "success":
        model_stats[m]["success"] += 1
    elif r.get("status") == "error":
        model_stats[m]["error"] += 1
    if r.get("roundtrip_match") is True:
        model_stats[m]["match"] += 1
    elif r.get("roundtrip_match") is False:
        model_stats[m]["mismatch"] += 1

print("Per-model summary:")
print(f"  {'Model':<25} {'Total':>6} {'OK':>6} {'Err':>6} {'Match':>6} {'Mism':>6}")
for m, s in sorted(model_stats.items()):
    print(f"  {m:<25} {s['total']:>6} {s['success']:>6} {s['error']:>6} {s['match']:>6} {s['mismatch']:>6}")
print()

# ── Export CSV ──
csv_path = os.path.join(outdir, "summary.csv")
import csv
fieldnames = ["model", "rotation", "scheme", "quant_iters", "rotation_size",
              "rotation_mode", "random_set", "status", "roundtrip_match", "source"]
fieldnames.extend([f"disk_{t}" for t in all_tasks])
fieldnames.extend([f"mem_{t}" for t in all_tasks])
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in all_results:
        row = {
            "model": r.get("model", ""),
            "rotation": r.get("rotation", ""),
            "scheme": r.get("scheme", ""),
            "quant_iters": r.get("quant_iters", ""),
            "rotation_size": r.get("rotation_size", ""),
            "rotation_mode": r.get("rotation_mode", ""),
            "random_set": ",".join(sorted(r.get("random_set", []) or [])),
            "status": r.get("status", ""),
            "roundtrip_match": r.get("roundtrip_match", ""),
            "source": r.get("source", ""),
        }
        disk_m = r.get("metrics_from_disk", {}) or {}
        mem_m = r.get("metrics_in_memory", {}) or {}
        for t in all_tasks:
            row[f"disk_{t}"] = disk_m.get(t, "")
            row[f"mem_{t}"] = mem_m.get(t, "")
        writer.writerow(row)
print(f"  CSV exported → {csv_path}")
print()
PYEOF

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  BENCHMARK COMPLETE"
    echo "  Results:    $OUTDIR/summary.json"
    echo "  CSV:        $OUTDIR/summary.csv"
    echo "  Logs:       $LOG_DIR/"
    echo "═══════════════════════════════════════════════════════════════"

    exit $FAILED
fi

# Not waiting — print monitor instructions
echo ""
echo "Jobs are running in background. To wait and aggregate:"
echo "  wait && bash -c 'cd $SCRIPT_DIR && python3 ...'"
echo ""
echo "Or re-run with --wait to block until completion."
echo ""
