#!/usr/bin/env bash
# Evaluate ONE quantized model for ONE task group with lm_eval.
#
# Mirrors the provided eval_hf.sh / eval_hf_gsm8k.sh, parameterized so the
# orchestrator can call it uniformly. Results are written under OUTPUT_PATH,
# which the aggregator later scans.
#
# Required env vars:
#   MODEL         path to the quantized model dir (contains config.json)
#   OUTPUT_PATH   directory to write lm_eval results into
#   MODE          "general" | "gsm8k"
#   TASKS         comma-separated lm_eval tasks
# Optional env vars:
#   GPU           CUDA_VISIBLE_DEVICES value            (default 0)
#   BATCH_SIZE    lm_eval batch size                    (default 64)
#   SEED          random seed                           (default 42)
#   DTYPE         model dtype                           (default bfloat16)
#   ENABLE_THINKING  true|false (Qwen3-style)           (default false)
#   EXTRA_ARGS    extra lm_eval flags (e.g. gen_kwargs) (default empty)

set -euo pipefail

: "${MODEL:?MODEL is required}"
: "${OUTPUT_PATH:?OUTPUT_PATH is required}"
: "${MODE:?MODE is required (general|gsm8k)}"
: "${TASKS:?TASKS is required}"

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1

mkdir -p "$OUTPUT_PATH"

# gsm8k uses the chat template, so pass the thinking flag through model_args.
MODEL_ARGS="pretrained=${MODEL},dtype=${DTYPE}"
if [[ "$MODE" == "gsm8k" ]]; then
  MODEL_ARGS="${MODEL_ARGS},enable_thinking=${ENABLE_THINKING}"
fi

echo "[eval] mode=$MODE tasks=$TASKS model=$MODEL gpu=$GPU -> $OUTPUT_PATH"

# shellcheck disable=SC2086  # EXTRA_ARGS must word-split into flags
lm_eval \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --output_path "$OUTPUT_PATH" \
  $EXTRA_ARGS

echo "[eval] finished mode=$MODE -> $OUTPUT_PATH"
