#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate rotated models with lm_eval using HuggingFace (transformers) backend.
#
# This evaluates each model using the HF pipeline with auto-round's QuantLinear
# forward path (including online rotation hooks rebuilt at load time).
#
# Usage:
#   # Evaluate all saved models:
#   bash eval_hf.sh
#
#   # Evaluate specific model:
#   bash eval_hf.sh ./rotated_models_Qwen3-0.6B/R1+R4
#
#   # Change GPU/tasks/limit:
#   CUDA_VISIBLE_DEVICES=6 TASKS="piqa,hellaswag" LIMIT=50 bash eval_hf.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
TASKS=${TASKS:-"piqa,hellaswag,lambada_openai,mmlu"}
BATCH_SIZE=${BATCH_SIZE:-8}
LIMIT=${LIMIT:-100}
OUTPUT_BASE=${OUTPUT_BASE:-"./lm_eval_results_hf"}
MODEL_BASE=${MODEL_BASE:-"./rotated_models_Qwen3-0.6B"}

mkdir -p "$OUTPUT_BASE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  lm_eval — HuggingFace backend"
echo "  Tasks: $TASKS"
echo "  Limit: $LIMIT"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Model base: $MODEL_BASE"
echo "═══════════════════════════════════════════════════════════════════════"

evaluate_model() {
    local MODEL_PATH="$1"
    local CONFIG_NAME="$2"
    local RESULT_DIR="${OUTPUT_BASE}/${CONFIG_NAME}"

    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo "  Evaluating: $CONFIG_NAME"
    echo "  Model path: $MODEL_PATH"
    echo "  Results: $RESULT_DIR"
    echo "───────────────────────────────────────────────────────────────────────"

    if [ -f "${RESULT_DIR}/results.json" ]; then
        echo "  ⏭️  Results already exist, skipping"
        return
    fi

    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,add_bos_token=True"

    lm_eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --output_path "$RESULT_DIR" \
        2>&1 | tee "${RESULT_DIR}.log"

    echo "  ✅ Done: $CONFIG_NAME"
}

# If a specific model path is given as argument, evaluate just that
if [ $# -ge 1 ]; then
    MODEL_PATH="$1"
    CONFIG_NAME=$(basename "$MODEL_PATH")
    evaluate_model "$MODEL_PATH" "$CONFIG_NAME"
    exit 0
fi

# Otherwise evaluate all models in MODEL_BASE
if [ ! -d "$MODEL_BASE" ]; then
    echo "ERROR: Model base directory not found: $MODEL_BASE"
    echo "Run save_rotated_models.py first."
    exit 1
fi

# Also evaluate FP16 baseline
echo ""
echo "───────────────────────────────────────────────────────────────────────"
echo "  Evaluating: FP16 baseline (Qwen/Qwen3-0.6B)"
echo "───────────────────────────────────────────────────────────────────────"
FP16_RESULT_DIR="${OUTPUT_BASE}/FP16_baseline"
if [ ! -f "${FP16_RESULT_DIR}/results.json" ]; then
    lm_eval \
        --model hf \
        --model_args "pretrained=Qwen/Qwen3-0.6B,dtype=bfloat16,add_bos_token=True" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --output_path "$FP16_RESULT_DIR" \
        2>&1 | tee "${FP16_RESULT_DIR}.log"
else
    echo "  ⏭️  FP16 baseline results already exist, skipping"
fi

# Evaluate each saved rotation config
for MODEL_DIR in "$MODEL_BASE"/*/; do
    # AutoRound creates a model subdirectory (e.g., Qwen3-0.6B-mxfp-w4g32)
    # Find the actual model dir with config.json
    ACTUAL_MODEL_DIR=""
    if [ -f "${MODEL_DIR}/config.json" ]; then
        ACTUAL_MODEL_DIR="$MODEL_DIR"
    else
        # Check subdirectories
        for SUB_DIR in "$MODEL_DIR"/*/; do
            if [ -f "${SUB_DIR}/config.json" ]; then
                ACTUAL_MODEL_DIR="$SUB_DIR"
                break
            fi
        done
    fi

    if [ -z "$ACTUAL_MODEL_DIR" ]; then
        continue
    fi

    CONFIG_NAME=$(basename "$MODEL_DIR")
    evaluate_model "$ACTUAL_MODEL_DIR" "$CONFIG_NAME"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ALL HF EVALUATIONS COMPLETE"
echo "  Results in: $OUTPUT_BASE"
echo "═══════════════════════════════════════════════════════════════════════"
