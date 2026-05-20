#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Multi-model × Multi-mode Rotation Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
#
# 3 models × 2 modes (full, tuning) = 6 jobs, each on a separate GPU.
# Both modes include: --compare-random --save-load
#   → det vs random Hadamard side-by-side + save/load roundtrip
#
# GPU assignment:
#   cuda:0  Qwen3-0.6B   full
#   cuda:1  Qwen3-0.6B   tuning
#   cuda:2  Qwen3-8B     full
#   cuda:3  Qwen3-8B     tuning
#   cuda:4  Llama-3.1-8B full
#   cuda:5  Llama-3.1-8B tuning
#
# Usage:
#   bash run.sh          # Launch all 6 jobs in background
#   bash run.sh --wait   # Launch and wait for all to finish
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Multi-model Rotation Benchmark"
echo "  Modes include: --compare-random --save-load"
echo "  Log directory:  $LOG_DIR"
echo "  Started at:     $(date)"
echo "═══════════════════════════════════════════════════════════════"

declare -a PIDS=()
declare -a LABELS=()

launch() {
    local gpu=$1 model=$2 mode=$3
    local short_name=$(echo "$model" | sed 's|.*/||')
    local label="${short_name}_${mode}"
    local log_file="${LOG_DIR}/${label}.log"

    echo "  [GPU ${gpu}] ${label} → ${log_file}"

    DEVICE="cuda:${gpu}" \
    MODEL="$model" \
    bash run_rotation_scheme_matrix_v2.sh "$mode" \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    LABELS+=("$label")
}

# ── Launch 6 jobs ────────────────────────────────────────────────────────────
#launch 2 "Qwen/Qwen3-0.6B"                   full
#launch 3 "Qwen/Qwen3-0.6B"                   tuning
#launch 1 "Qwen/Qwen3-0.6B"                   size-sweep
launch 4 "Qwen/Qwen3-8B"                     full
launch 5 "Qwen/Qwen3-8B"                     tuning
launch 6 "meta-llama/Llama-3.1-8B-Instruct"  full
launch 7 "meta-llama/Llama-3.1-8B-Instruct"  tuning

# ── Block-wise (layer-wise) rotation tests ───────────────────────────────────
#launch 0 "Qwen/Qwen3-0.6B"                   layerwise
#launch 1 "Qwen/Qwen3-0.6B"                   layerwise-tuning
#launch 2 "Qwen/Qwen3-0.6B"                   layerwise-compare

# ── Large model multi-GPU tests (uncomment to use) ───────────────────────────
# For 32B/70B/122B models, use run_multi_gpu.sh instead:
#   bash run_multi_gpu.sh --model 32b --mode quick
#   bash run_multi_gpu.sh --model 70b --mode full
#   GPUS_70B="0,1,2,3" bash run_multi_gpu.sh --model 70b

echo ""
echo "  All 6 jobs launched. PIDs: ${PIDS[*]}"
echo "  Monitor: tail -f ${LOG_DIR}/<job>.log"
echo "═══════════════════════════════════════════════════════════════"

# ── Wait if requested ────────────────────────────────────────────────────────
if [[ "${1:-}" == "--wait" ]]; then
    echo ""
    echo "Waiting for all jobs to finish..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        label=${LABELS[$i]}
        if wait "$pid"; then
            echo "  ✓ ${label} (PID ${pid}) — done"
        else
            echo "  ✗ ${label} (PID ${pid}) — FAILED (exit $?)"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Finished at: $(date)"
    echo "  Results: ${FAILED} failed, $((${#PIDS[@]} - FAILED)) succeeded"
    echo "  Logs:    ${LOG_DIR}/"
    echo "  Results: results_v2_*/"
    echo "═══════════════════════════════════════════════════════════════"
    exit $FAILED
fi
