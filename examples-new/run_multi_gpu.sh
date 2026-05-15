#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Multi-GPU Rotation Benchmark for Large Models (32B / 70B / 122B)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Uses model parallelism (accelerate device_map="auto") to distribute large
# models across multiple GPUs. Each job uses ALL specified GPUs for one model.
#
# This is different from run.sh which assigns one GPU per job.
# Here we use multiple GPUs for a single model that doesn't fit in one GPU.
#
# Memory requirements (FP16):
#   - 32B  → ~64GB  → 2× A100-80G or 4× A100-40G
#   - 70B  → ~140GB → 2× A100-80G or 4× A100-40G  
#   - 122B → ~244GB → 4× A100-80G or 8× A100-40G
#
# GPU assignment (default, configurable):
#   GPUS_32B="0,1"        — 2 GPUs for 32B model
#   GPUS_70B="0,1,2,3"    — 4 GPUs for 70B model
#   GPUS_122B="0,1,2,3,4,5,6,7"  — 8 GPUs for 122B model
#
# Modes (same as run_rotation_scheme_matrix_v2.sh):
#   quick       — limit=100, 3 schemes, 2 tasks (~30min per model)
#   full        — no limit, 3 schemes, 5 tasks (~4-8h per model)
#   layerwise   — block-wise rotation (saves CPU RAM for huge models)
#
# Usage:
#   bash run_multi_gpu.sh                           # Run all models sequentially
#   bash run_multi_gpu.sh --model 70b              # Only 70B model
#   bash run_multi_gpu.sh --model 32b --mode full  # 32B full eval
#   bash run_multi_gpu.sh --parallel               # Run models in parallel (needs enough GPUs)
#
# Environment overrides:
#   GPUS_32B="4,5" MODE=quick bash run_multi_gpu.sh --model 32b
#   GPUS_70B="0,1,2,3" bash run_multi_gpu.sh --model 70b
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_multigpu_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── Default GPU assignments ──────────────────────────────────────────────────
GPUS_32B="${GPUS_32B:-0,1}"
GPUS_70B="${GPUS_70B:-0,1,2,3}"
GPUS_122B="${GPUS_122B:-0,1,2,3,4,5,6,7}"

# ── Default models ───────────────────────────────────────────────────────────
MODEL_32B="${MODEL_32B:-Qwen/Qwen3-32B}"
MODEL_70B="${MODEL_70B:-Qwen/Qwen2.5-72B-Instruct}"
MODEL_122B="${MODEL_122B:-Qwen/Qwen3-235B-A22B}"

# ── Default mode and options ─────────────────────────────────────────────────
MODE="${MODE:-quick}"
PARALLEL=false
TARGET_MODEL=""
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            TARGET_MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --layerwise)
            MODE="layerwise"
            shift
            ;;
        --tuning)
            MODE="tuning"
            shift
            ;;
        --limit)
            EXTRA_ARGS="${EXTRA_ARGS} --limit $2"
            shift 2
            ;;
        --tasks)
            EXTRA_ARGS="${EXTRA_ARGS} --tasks $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_multi_gpu.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL    Target model: 32b, 70b, 122b, or full path"
            echo "  --mode MODE      Test mode: quick, full, layerwise, tuning (default: quick)"
            echo "  --parallel       Run models in parallel (needs non-overlapping GPUs)"
            echo "  --layerwise      Shortcut for --mode layerwise"
            echo "  --tuning         Shortcut for --mode tuning"
            echo "  --limit N        Limit eval samples per task"
            echo "  --tasks TASKS    Comma-separated eval tasks"
            echo ""
            echo "Environment variables:"
            echo "  GPUS_32B         GPUs for 32B model (default: 0,1)"
            echo "  GPUS_70B         GPUs for 70B model (default: 0,1,2,3)"
            echo "  GPUS_122B        GPUs for 122B model (default: 0,1,2,3,4,5,6,7)"
            echo "  MODEL_32B        32B model name/path"
            echo "  MODEL_70B        70B model name/path"
            echo "  MODEL_122B       122B model name/path"
            echo "  MODE             Test mode (default: quick)"
            exit 0
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# ── Determine rotation options based on mode ─────────────────────────────────
get_mode_args() {
    local mode=$1
    case "$mode" in
        quick)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3+R4' --schemes 'W4A16,MXFP4,NVFP4' --limit 100 --tasks 'hellaswag,piqa'"
            ;;
        full)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4' --schemes 'W4A16,MXFP4,NVFP4' --compare-random --save-load --tasks 'hellaswag,piqa,winogrande,lambada_openai'"
            ;;
        layerwise)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4' --schemes 'W4A16,MXFP4,NVFP4' --layerwise --save-load --tasks 'hellaswag,piqa,winogrande,lambada_openai'"
            ;;
        tuning)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4' --schemes 'W4A16,MXFP4,NVFP4' --quant-iters 200 --compare-random --save-load --tasks 'hellaswag,piqa,winogrande,lambada_openai'"
            ;;
        layerwise-tuning)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4' --schemes 'W4A16,MXFP4,NVFP4' --layerwise --quant-iters 200 --save-load --tasks 'hellaswag,piqa,winogrande,lambada_openai'"
            ;;
        weight-only)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3+R4' --weight-only --tasks 'hellaswag,piqa,winogrande,lambada_openai'"
            ;;
        *)
            echo "--rotations 'none,R1,R1+R2,R1+R2+R3+R4' --schemes 'W4A16,MXFP4,NVFP4' --limit 100 --tasks 'hellaswag,piqa'"
            ;;
    esac
}

# ── Launch function ──────────────────────────────────────────────────────────
run_model() {
    local gpus=$1 model=$2 label=$3
    local short_name=$(echo "$model" | sed 's|.*/||')
    local log_file="${LOG_DIR}/${short_name}_${MODE}.log"
    local mode_args=$(get_mode_args "$MODE")

    echo "═══════════════════════════════════════════════════════════════"
    echo "  Model:   $model"
    echo "  GPUs:    $gpus"
    echo "  Mode:    $MODE"
    echo "  Log:     $log_file"
    echo "═══════════════════════════════════════════════════════════════"

    # CUDA_VISIBLE_DEVICES limits which GPUs are visible to the process.
    # The device spec "0,1,..." maps to the visible GPUs.
    # We pass the GPU indices directly to --device (AutoRound handles the mapping).
    local cmd="python test_rotation_scheme_matrix_v2.py \
        --model '$model' \
        --device '$gpus' \
        $mode_args \
        $EXTRA_ARGS"

    echo "  Command: $cmd"
    echo ""

    if [ "$PARALLEL" = true ]; then
        eval "$cmd" > "$log_file" 2>&1 &
        echo "  Launched in background (PID: $!)"
        PIDS+=($!)
        LABELS+=("$label")
    else
        echo "  Running (this may take hours for large models)..."
        eval "$cmd" 2>&1 | tee "$log_file"
        echo "  ✓ Done: $label"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Multi-GPU Rotation Benchmark"
echo "  Mode:       $MODE"
echo "  Parallel:   $PARALLEL"
echo "  Log dir:    $LOG_DIR"
echo "  Started:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

declare -a PIDS=()
declare -a LABELS=()

case "${TARGET_MODEL,,}" in
    32b|32)
        run_model "$GPUS_32B" "$MODEL_32B" "32B_${MODE}"
        ;;
    70b|70)
        run_model "$GPUS_70B" "$MODEL_70B" "70B_${MODE}"
        ;;
    122b|122)
        run_model "$GPUS_122B" "$MODEL_122B" "122B_${MODE}"
        ;;
    "")
        # Run all models
        if [ "$PARALLEL" = true ]; then
            echo "Running all models in parallel..."
            echo "  WARNING: Ensure GPU assignments don't overlap!"
            echo ""
            run_model "$GPUS_32B" "$MODEL_32B" "32B_${MODE}"
            run_model "$GPUS_70B" "$MODEL_70B" "70B_${MODE}"
            run_model "$GPUS_122B" "$MODEL_122B" "122B_${MODE}"
        else
            echo "Running all models sequentially..."
            echo ""
            run_model "$GPUS_32B" "$MODEL_32B" "32B_${MODE}"
            echo ""
            run_model "$GPUS_70B" "$MODEL_70B" "70B_${MODE}"
            echo ""
            run_model "$GPUS_122B" "$MODEL_122B" "122B_${MODE}"
        fi
        ;;
    *)
        # Custom model path — use GPUS_70B as default GPU set
        CUSTOM_GPUS="${CUSTOM_GPUS:-$GPUS_70B}"
        run_model "$CUSTOM_GPUS" "$TARGET_MODEL" "custom_${MODE}"
        ;;
esac

# ── Wait for parallel jobs ───────────────────────────────────────────────────
if [ "$PARALLEL" = true ] && [ ${#PIDS[@]} -gt 0 ]; then
    echo ""
    echo "Waiting for ${#PIDS[@]} parallel jobs..."
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
    echo "  Finished: $(date)"
    echo "  Results: ${FAILED} failed, $((${#PIDS[@]} - FAILED)) succeeded"
    echo "  Logs:    ${LOG_DIR}/"
    echo "═══════════════════════════════════════════════════════════════"
    exit $FAILED
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Finished: $(date)"
echo "  Logs:     ${LOG_DIR}/"
echo "  Results:  results_v2_*/"
echo "═══════════════════════════════════════════════════════════════"
