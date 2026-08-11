#!/usr/bin/env bash
# Evaluate ONE quantized model for ONE task group with lm_eval, using the
# **vLLM** backend (much faster than the hf backend for generation).
#
# Mirrors the provided eval_vllm.sh, parameterized so the orchestrator can call
# it uniformly. Results are written under OUTPUT_PATH, which the aggregator
# later scans.
#
# Required env vars:
#   MODEL         path to the quantized model dir (contains config.json)
#   OUTPUT_PATH   directory to write lm_eval results into
#   MODE          "general" | "gsm8k"
#   TASKS         comma-separated lm_eval tasks
# Optional env vars:
#   GPU                     CUDA_VISIBLE_DEVICES value           (default 0)
#   BATCH_SIZE              lm_eval batch size                   (default 64)
#   SEED                    random seed                          (default 42)
#   DTYPE                   model dtype                          (default bfloat16)
#   ENABLE_THINKING         true|false (Qwen3-style, gsm8k only) (default false)
#   EXTRA_ARGS              extra lm_eval flags (chat template…) (default empty)
#   TP_SIZE                 vLLM tensor_parallel_size            (default 1)
#   MAX_MODEL_LEN           vLLM max_model_len                   (default 8192)
#   MAX_NUM_BATCHED_TOKENS  vLLM max_num_batched_tokens          (default 32768)
#   MAX_NUM_SEQS            vLLM max_num_seqs                    (default 128)
#   MAX_GEN_TOKS            generation cap                       (default 2048)
#   GPU_MEMORY_UTILIZATION  vLLM gpu_memory_utilization          (default 0.85)
#   ADD_BOS_TOKEN           true|false                           (default True)

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

TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ADD_BOS_TOKEN="${ADD_BOS_TOKEN:-True}"

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL=1

mkdir -p "$OUTPUT_PATH"

# Base vLLM model_args (mirrors eval_vllm.sh).
MODEL_ARGS="pretrained=${MODEL}"
MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${TP_SIZE}"
MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN}"
MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
MODEL_ARGS="${MODEL_ARGS},max_num_seqs=${MAX_NUM_SEQS}"
MODEL_ARGS="${MODEL_ARGS},add_bos_token=${ADD_BOS_TOKEN}"
MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
MODEL_ARGS="${MODEL_ARGS},dtype=${DTYPE}"
MODEL_ARGS="${MODEL_ARGS},max_gen_toks=${MAX_GEN_TOKS}"
MODEL_ARGS="${MODEL_ARGS},enable_prefix_caching=False"

# gsm8k uses the chat template; forward the Qwen3-style thinking flag.
if [[ "$MODE" == "gsm8k" ]]; then
  MODEL_ARGS="${MODEL_ARGS},enable_thinking=${ENABLE_THINKING}"
fi

echo "[eval] backend=vllm mode=$MODE tasks=$TASKS model=$MODEL gpu=$GPU -> $OUTPUT_PATH"

# shellcheck disable=SC2086  # EXTRA_ARGS / MODEL_ARGS must word-split into flags
lm_eval \
  --model vllm \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --output_path "$OUTPUT_PATH" \
  $EXTRA_ARGS

echo "[eval] finished mode=$MODE -> $OUTPUT_PATH"
