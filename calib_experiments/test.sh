#!/usr/bin/env bash
# Smoke test: quantize + eval Qwen3-0.6B with MXFP4 across all recipes.
# Logs stream live to t_qwen0.6B.log; the chosen GPU is honored.
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=2

# -u => unbuffered python so the top-level redirect log fills in live.
nohup python -u scripts/run_experiments.py --schemes MXFP4 \
  >> t_qwen0.6B.log 2>&1 &

echo "started PID $! -> tail -f $(pwd)/t_qwen0.6B.log"
