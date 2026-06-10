#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline: for each rotation config, save model → eval vLLM → eval HF
#
# This processes configs one-by-one so you can see results incrementally and
# catch issues early without wasting time saving all models first.
#
# Usage:
#   # Run all configs:
#   bash run.sh
#
#   # Run specific configs:
#   bash run.sh R1 R1+R4 R1+R2+R3
#
#   # Change settings:
#   CUDA_VISIBLE_DEVICES=6 TASKS="piqa,hellaswag" LIMIT=50 bash run.sh
#
#   # Multi-GPU:
#   CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash run.sh
#
#   # Skip save step (models already saved):
#   SKIP_SAVE=1 bash run.sh
#
#   # Skip specific backends:
#   SKIP_VLLM=1 bash run.sh    # HF only
#   SKIP_HF=1 bash run.sh      # vLLM only
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Configuration ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
NUM_GPUS=${NUM_GPUS:-1}
TASKS=${TASKS:-"piqa,hellaswag,gsm8k"}
BATCH_SIZE=${BATCH_SIZE:-64}
LIMIT=${LIMIT:-100}
DEVICE=${DEVICE:-"cuda:0"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

MODEL_BASE=${MODEL_BASE:-"./rotated_models_Qwen3-0.6B"}
RESULTS_VLLM=${RESULTS_VLLM:-"./lm_eval_results_vllm"}
RESULTS_HF=${RESULTS_HF:-"./lm_eval_results_hf"}

SKIP_SAVE=${SKIP_SAVE:-0}
SKIP_VLLM=${SKIP_VLLM:-0}
SKIP_HF=${SKIP_HF:-0}

# All available configs (order: simple → complex)
ALL_CONFIGS=(
    #"R1"
    #"R1+R2"
    "R1+R4"
    "R1+R2+R4"
    "R1_size128"
    "R1_random"
    "R1+R4_size128"
    "R1+R2+R3"
    "R1+R2+R3+R4"
)

# R3-containing configs (vLLM not supported)
R3_CONFIGS=("R1+R2+R3" "R1+R2+R3+R4")

# ─── Helper functions ─────────────────────────────────────────────────────────

is_r3_config() {
    local cfg="$1"
    for r3 in "${R3_CONFIGS[@]}"; do
        if [ "$cfg" = "$r3" ]; then
            return 0
        fi
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

# ─── Determine which configs to run ──────────────────────────────────────────

if [ $# -ge 1 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

# ─── Print banner ────────────────────────────────────────────────────────────

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║         End-to-End Rotation Evaluation Pipeline                      ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Model:    $MODEL_NAME"
echo "║  GPU:      $CUDA_VISIBLE_DEVICES (TP=$NUM_GPUS)"
echo "║  Tasks:    $TASKS"
echo "║  Limit:    $LIMIT"
echo "║  Configs:  ${CONFIGS[*]}"
echo "║  Skip:     save=$SKIP_SAVE  vllm=$SKIP_VLLM  hf=$SKIP_HF"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=${#CONFIGS[@]}
CURRENT=0
FAILED_CONFIGS=()
START_TIME=$(date +%s)

# ─── FP16 Baseline (HF only) ─────────────────────────────────────────────────

if [ "$SKIP_HF" != "1" ]; then
    FP16_RESULT_DIR="${RESULTS_HF}/FP16_baseline"
    if [ ! -f "${FP16_RESULT_DIR}/results.json" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$(timestamp)] FP16 Baseline — HF eval"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        mkdir -p "$FP16_RESULT_DIR"
        lm_eval \
            --model hf \
            --model_args "pretrained=${MODEL_NAME},dtype=bfloat16,add_bos_token=True" \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --output_path "$FP16_RESULT_DIR" \
            2>&1 | tee "${FP16_RESULT_DIR}.log"
        echo "  ✅ FP16 baseline done"
    else
        echo "  ⏭️  FP16 baseline already evaluated, skipping"
    fi
    echo ""
fi

# ─── Main loop: per-config pipeline ──────────────────────────────────────────

for CONFIG in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║  [$CURRENT/$TOTAL] Config: $CONFIG"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    CONFIG_START=$(date +%s)

    # ── Step 1: Save model ────────────────────────────────────────────────────
    MODEL_DIR="${MODEL_BASE}/${CONFIG}"
    ACTUAL_MODEL_DIR=$(find_model_dir "$MODEL_DIR")

    if [ "$SKIP_SAVE" != "1" ] && [ -z "$ACTUAL_MODEL_DIR" ]; then
        echo ""
        echo "  [$(timestamp)] Step 1/3: Saving model..."
        python save_rotated_models.py \
            --device "$DEVICE" \
            --configs "$CONFIG" \
            --output-base "$MODEL_BASE" \
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

    # ── Step 2: vLLM eval ─────────────────────────────────────────────────────
    if [ "$SKIP_VLLM" != "1" ]; then
        if is_r3_config "$CONFIG"; then
            echo ""
            echo "  [$(timestamp)] Step 2/3: vLLM eval — ⏭️ SKIPPED (R3 not supported)"
        else
            VLLM_RESULT_DIR="${RESULTS_VLLM}/${CONFIG}"
            if [ -f "${VLLM_RESULT_DIR}/results.json" ]; then
                echo ""
                echo "  [$(timestamp)] Step 2/3: vLLM eval — ⏭️ already done"
            else
                echo ""
                echo "  [$(timestamp)] Step 2/3: vLLM eval..."
                mkdir -p "$VLLM_RESULT_DIR"

                MODEL_ARGS="pretrained=${ACTUAL_MODEL_DIR}"
                MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${NUM_GPUS}"
                MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN}"
                MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=32768"
                MODEL_ARGS="${MODEL_ARGS},max_num_seqs=128"
                MODEL_ARGS="${MODEL_ARGS},add_bos_token=True"
                MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
                MODEL_ARGS="${MODEL_ARGS},dtype=bfloat16"
                MODEL_ARGS="${MODEL_ARGS},max_gen_toks=2048"
                MODEL_ARGS="${MODEL_ARGS},enable_prefix_caching=False"

                if lm_eval \
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
    fi

    # ── Step 3: HF eval ───────────────────────────────────────────────────────
    if [ "$SKIP_HF" != "1" ]; then
        HF_RESULT_DIR="${RESULTS_HF}/${CONFIG}"
        if [ -f "${HF_RESULT_DIR}/results.json" ]; then
            echo ""
            echo "  [$(timestamp)] Step 3/3: HF eval — ⏭️ already done"
        else
            echo ""
            echo "  [$(timestamp)] Step 3/3: HF eval..."
            mkdir -p "$HF_RESULT_DIR"

            if lm_eval \
                --model hf \
                --model_args "pretrained=${ACTUAL_MODEL_DIR},dtype=bfloat16,add_bos_token=True" \
                --tasks "$TASKS" \
                --batch_size "$BATCH_SIZE" \
                --output_path "$HF_RESULT_DIR" \
                2>&1 | tee "${HF_RESULT_DIR}.log"; then
                echo "  ✅ HF eval complete"
            else
                echo "  ❌ HF eval FAILED for $CONFIG"
                FAILED_CONFIGS+=("$CONFIG:hf")
            fi
        fi
    fi

    # ── Per-config summary ────────────────────────────────────────────────────
    CONFIG_END=$(date +%s)
    CONFIG_ELAPSED=$((CONFIG_END - CONFIG_START))
    echo ""
    echo "  ⏱️  $CONFIG completed in ${CONFIG_ELAPSED}s"
done

# ─── Final summary ───────────────────────────────────────────────────────────

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_SEC=$((TOTAL_ELAPSED % 60))

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                        PIPELINE COMPLETE                             ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "║  Configs run: $TOTAL"
echo "║  Models dir:  $MODEL_BASE"
echo "║  HF results:  $RESULTS_HF"
echo "║  vLLM results: $RESULTS_VLLM"

if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo "║"
    echo "║  ❌ FAILURES:"
    for f in "${FAILED_CONFIGS[@]}"; do
        echo "║     - $f"
    done
fi

echo "╚═══════════════════════════════════════════════════════════════════════╝"

# ─── Run comparison if both backends were evaluated ───────────────────────────

if [ "$SKIP_VLLM" != "1" ] && [ "$SKIP_HF" != "1" ]; then
    if [ -d "$RESULTS_VLLM" ] && [ -d "$RESULTS_HF" ]; then
        echo ""
        echo "Running comparison..."
        python compare_results.py \
            --hf-dir "$RESULTS_HF" \
            --vllm-dir "$RESULTS_VLLM" \
            2>&1 || echo "  ⚠️  Comparison script failed (non-critical)"
    fi
fi

echo ""
echo "Done! $(timestamp)"
