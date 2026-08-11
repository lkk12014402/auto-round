#!/usr/bin/env bash
# Multi-model calibration study on 8x L20.
#
# 临时选模型/方案：只改本文件顶部的变量即可，无需编辑 configs/experiments.py。
# 模型的 GPU 分配 / device_map / vllm_extra 等仍在 experiments.py 的 MODELS 注册表里定义；
# 这里只是从注册表中“挑选本次要跑的子集”。
#
# 用 --parallel 时，run_experiments.py 会为每个被选中的模型起一个子进程，
# 把 CUDA_VISIBLE_DEVICES 固定到该模型的 gpus，模型内部各 cell 顺序执行。
# 大模型分片（quant device_map="auto"，vLLM tensor_parallel_size=len(gpus)）；
# Qwen3.6 推理模型的 reasoning_parser=qwen3,language_model_only=True 由 vllm_extra 携带。
cd "$(dirname "$0")"

export VLLM_QDQ=1
export VLLM_AR_MXFP4_MODULAR_MOE=1
export VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
export VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
export VLLM_ENABLE_STATIC_MOE=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_ENABLE_AR_EXT=1

# ------------------------------- 可编辑区 -------------------------------
# 本次要跑的模型名（空=注册表里全部）。名字必须与 experiments.py MODELS[*].name 一致。
# 例：MODELS=(Qwen3-0.6B Qwen3-8B)
MODELS=(Qwen3-0.6B)

# 本次要跑的量化方案（空=全部，见 experiments.py SCHEMES）。例：SCHEMES=(MXFP4 W4A16)
SCHEMES=(MXFP4)

# 本次要跑的标定配方 key（空=全部，见 experiments.py RECIPES）。例：RECIPES=(pile10k ultrachat)
RECIPES=()

# 多模型并行（每个模型独占各自的 GPU）。单模型时留空也可。
PARALLEL=1

# 传给 run_experiments.py 的其它参数，例如 --skip-eval / --skip-quant / --force
EXTRA_FLAGS=()

# 日志文件（顶层聚合日志；每模型日志在 runs/logs/parallel__<model>.log）
LOG=test_parallel.log
# ----------------------------------------------------------------------

# gated 模型访问所需（如无需要可删）。
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HOME=/home/hshen/lkk/calib_feat/calib_experiments/

# 组装命令行参数。
ARGS=(scripts/run_experiments.py)
[[ "$PARALLEL" == "1" ]] && ARGS+=(--parallel)
[[ ${#MODELS[@]}  -gt 0 ]] && ARGS+=(--models  "${MODELS[@]}")
[[ ${#SCHEMES[@]} -gt 0 ]] && ARGS+=(--schemes "${SCHEMES[@]}")
[[ ${#RECIPES[@]} -gt 0 ]] && ARGS+=(--recipes "${RECIPES[@]}")
[[ ${#EXTRA_FLAGS[@]} -gt 0 ]] && ARGS+=("${EXTRA_FLAGS[@]}")

echo "cmd: python -u ${ARGS[*]}"

# -u => unbuffered python，让顶层重定向日志实时写入。
nohup python -u "${ARGS[@]}" >> "$LOG" 2>&1 &

echo "started PID $! -> tail -f $(pwd)/$LOG"
if [[ "$PARALLEL" == "1" && ${#MODELS[@]} -ne 1 ]]; then
  echo "per-model logs: $(pwd)/runs/logs/parallel__<model>.log"
else
  echo "per-cell logs:  $(pwd)/runs/logs/<exp_id>__quant.log , <exp_id>__eval-<mode>.log"
fi
