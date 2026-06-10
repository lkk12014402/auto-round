#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline for Qwen3-32B: save model (1 GPU) → eval vLLM (multi-GPU)
#
# Rotation configs:
#   - R1 (default hadamard, rotation_size=hidden_size=5120)
#   - R1+R2 (R2 offline fused)
#   - R1+R2+R3 (R3 online on Q/K)
#   - R1_random (random R1)
#   - R1_size32 / R1_size128 (custom block rotation)
#   - R1+R4_size32 / R1+R4_size128 (R4 with TP-safe block rotation)
#   - R1+R2+R4_size32 / R1+R2+R4_size128
#
# Usage:
#   # Run all configs:
#   bash run_qwen3_32b.sh
#
#   # Run specific configs:
#   bash run_qwen3_32b.sh R1 R1+R2 R1+R4_size128
#
#   # Override GPU/TP settings:
#   SAVE_DEVICE=cuda:0 EVAL_GPUS=0,1,2,3 TP=4 bash run_qwen3_32b.sh
#
#   # Skip save (models already saved):
#   SKIP_SAVE=1 bash run_qwen3_32b.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
MODEL_SHORT="Qwen3-8B"

# Save uses 1 GPU
SAVE_DEVICE=${SAVE_DEVICE:-"cuda:0"}

# vLLM eval uses multiple GPUs
EVAL_GPUS=${EVAL_GPUS:-"2,3"}
TP=${TP:-2}

TASKS=${TASKS:-"piqa,hellaswag,mmlu,gsm8k"}
BATCH_SIZE=${BATCH_SIZE:-"64"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

MODEL_BASE=${MODEL_BASE:-"./rotated_models_${MODEL_SHORT}"}
RESULTS_DIR=${RESULTS_DIR:-"./lm_eval_results_${MODEL_SHORT}"}

SKIP_SAVE=${SKIP_SAVE:-0}

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ─── Rotation Configs ─────────────────────────────────────────────────────────
# NOTE: R4 uses rotation_size=32 or 128 for TP compatibility.
#       Full-size R4 (rotation_size=intermediate_size) requires TP=1.

ALL_CONFIGS=(
    #"R1"
    #"R1+R2"
    #"R1+R2+R3"
    #"R1_random"
    #"R1_size32"
    #"R1_size128"
    "R1+R4_size32"
    "R1+R4_size128"
    #"R1+R2+R4_size32"
    #"R1+R2+R4_size128"
)

# R3 configs — vLLM not supported, skip vLLM eval
R3_CONFIGS=("R1+R2+R3")

# ─── Helper functions ─────────────────────────────────────────────────────────

is_r3_config() {
    local cfg="$1"
    for r3 in "${R3_CONFIGS[@]}"; do
        [ "$cfg" = "$r3" ] && return 0
    done
    return 1
}

find_model_dir() {
    local config_dir="$1"
    if [ -f "${config_dir}/config.json" ]; then
        echo "$config_dir"
        return
    fi
    for sub in "$config_dir"/*/; do
        if [ -f "${sub}/config.json" ]; then
            echo "$sub"
            return
        fi
    done
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Map config name to SpinQuantConfig kwargs for save
get_save_args() {
    local cfg="$1"
    case "$cfg" in
        "R1")
            echo "--r1 --no-r2 --no-r3 --no-r4"
            ;;
        "R1+R2")
            echo "--r1 --r2 --no-r3 --no-r4"
            ;;
        "R1+R2+R3")
            echo "--r1 --r2 --r3 --no-r4"
            ;;
        "R1_random")
            echo "--r1 --no-r2 --no-r3 --no-r4 --random-r1"
            ;;
        "R1_size32")
            echo "--r1 --no-r2 --no-r3 --no-r4 --rotation-size 32"
            ;;
        "R1_size128")
            echo "--r1 --no-r2 --no-r3 --no-r4 --rotation-size 128"
            ;;
        "R1+R4_size32")
            echo "--r1 --no-r2 --no-r3 --r4 --rotation-size 32"
            ;;
        "R1+R4_size128")
            echo "--r1 --no-r2 --no-r3 --r4 --rotation-size 128"
            ;;
        "R1+R2+R4_size32")
            echo "--r1 --r2 --no-r3 --r4 --rotation-size 32"
            ;;
        "R1+R2+R4_size128")
            echo "--r1 --r2 --no-r3 --r4 --rotation-size 128"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ─── Determine which configs to run ──────────────────────────────────────────

if [ $# -ge 1 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

# ─── Print banner ────────────────────────────────────────────────────────────

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║         Qwen3-32B Rotation Evaluation Pipeline                           ║"
echo "╠═══════════════════════════════════════════════════════════════════════════╣"
echo "║  Model:      $MODEL_NAME"
echo "║  Save GPU:   $SAVE_DEVICE"
echo "║  Eval GPUs:  $EVAL_GPUS (TP=$TP)"
echo "║  Tasks:      $TASKS"
echo "║  Batch:      $BATCH_SIZE"
echo "║  Configs:    ${CONFIGS[*]}"
echo "║  Skip save:  $SKIP_SAVE"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=${#CONFIGS[@]}
CURRENT=0
FAILED_CONFIGS=()
START_TIME=$(date +%s)

# ─── FP16 Baseline (vLLM) ────────────────────────────────────────────────────


# ─── Main loop: per-config pipeline ──────────────────────────────────────────

for CONFIG in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║  [$CURRENT/$TOTAL] Config: $CONFIG"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    CONFIG_START=$(date +%s)

    # ── Step 1: Save model (single GPU) ───────────────────────────────────────
    MODEL_DIR="${MODEL_BASE}/${CONFIG}"
    ACTUAL_MODEL_DIR=$(find_model_dir "$MODEL_DIR")

    if [ "$SKIP_SAVE" != "1" ] && [ -z "$ACTUAL_MODEL_DIR" ]; then
        echo ""
        echo "  [$(timestamp)] Step 1/2: Saving model on $SAVE_DEVICE..."

        SAVE_ARGS=$(get_save_args "$CONFIG")
        if [ -z "$SAVE_ARGS" ]; then
            echo "  ❌ Unknown config: $CONFIG"
            FAILED_CONFIGS+=("$CONFIG:unknown")
            continue
        fi

        # Use the save_rotated_models.py script
        CUDA_VISIBLE_DEVICES="3" python save_rotated_models.py \
            --model "$MODEL_NAME" \
            --device "cuda:0" \
            --output-base "$MODEL_BASE" \
            --configs "$CONFIG" \
            --seqlen 512 \
            --nsamples 128 \
            2>&1 | tee "${MODEL_BASE}/${CONFIG}_save.log"

        ACTUAL_MODEL_DIR=$(find_model_dir "$MODEL_DIR")
        if [ -z "$ACTUAL_MODEL_DIR" ]; then
            echo "  ❌ FAILED: Model not saved for $CONFIG"
            FAILED_CONFIGS+=("$CONFIG:save")
            continue
        fi
        echo "  ✅ Save complete: $ACTUAL_MODEL_DIR"
    elif [ -z "$ACTUAL_MODEL_DIR" ]; then
        echo "  ❌ SKIP_SAVE=1 but model not found at $MODEL_DIR"
        FAILED_CONFIGS+=("$CONFIG:not_found")
        continue
    else
        echo "  ⏭️  Model already saved: $ACTUAL_MODEL_DIR"
    fi

    # ── Step 2: vLLM eval (multi-GPU) ────────────────────────────────────────
    if is_r3_config "$CONFIG"; then
        echo ""
        echo "  [$(timestamp)] Step 2/2: vLLM eval — ⏭️ SKIPPED (R3 not yet supported in vLLM)"
    else
        VLLM_RESULT_DIR="${RESULTS_DIR}/${CONFIG}"
        if [ -f "${VLLM_RESULT_DIR}/results.json" ]; then
            echo ""
            echo "  [$(timestamp)] Step 2/2: vLLM eval — ⏭️ already done"
        else
            echo ""
            echo "  [$(timestamp)] Step 2/2: vLLM eval (TP=$TP on GPUs $EVAL_GPUS)..."
            mkdir -p "$VLLM_RESULT_DIR"

            MODEL_ARGS="pretrained=${ACTUAL_MODEL_DIR}"
            MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${TP}"
            MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN}"
            MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=16384"
            MODEL_ARGS="${MODEL_ARGS},max_num_seqs=128"
            MODEL_ARGS="${MODEL_ARGS},add_bos_token=True"
            MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
            MODEL_ARGS="${MODEL_ARGS},dtype=bfloat16"
            MODEL_ARGS="${MODEL_ARGS},enable_prefix_caching=False"
	    MODEL_ARGS="${MODEL_ARGS},max_gen_toks=2048"

            if CUDA_VISIBLE_DEVICES="$EVAL_GPUS" lm_eval \
                --model vllm \
                --model_args "$MODEL_ARGS" \
                --tasks "$TASKS" \
                --batch_size "$BATCH_SIZE" \
                --output_path "$VLLM_RESULT_DIR" \
                2>&1 | tee "${VLLM_RESULT_DIR}.log"; then
                echo "  ✅ vLLM eval complete"
            else
                echo "  ❌ vLLM eval FAILED for $CONFIG"
                FAILED_CONFIGS+=("$CONFIG:vllm")
            fi
        fi
    fi

    # ── Per-config summary ────────────────────────────────────────────────────
    CONFIG_END=$(date +%s)
    CONFIG_ELAPSED=$((CONFIG_END - CONFIG_START))
    CONFIG_MIN=$((CONFIG_ELAPSED / 60))
    CONFIG_SEC=$((CONFIG_ELAPSED % 60))
    echo ""
    echo "  ⏱️  $CONFIG completed in ${CONFIG_MIN}m ${CONFIG_SEC}s"
done

# ─── Final summary ───────────────────────────────────────────────────────────

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                        PIPELINE COMPLETE                                 ║"
echo "╠═══════════════════════════════════════════════════════════════════════════╣"
echo "║  Model:       $MODEL_NAME"
echo "║  Total time:  ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "║  Configs run: $TOTAL"
echo "║  Models dir:  $MODEL_BASE"
echo "║  Results dir: $RESULTS_DIR"

if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo "║"
    echo "║  ❌ FAILURES:"
    for f in "${FAILED_CONFIGS[@]}"; do
        echo "║     - $f"
    done
fi

echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Done! $(timestamp)"
