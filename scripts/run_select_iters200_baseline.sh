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


export https_proxy="http://proxy.ims.intel.com:911"
export http_proxy="http://proxy.ims.intel.com:911"
export HF_HOME="/data/HF_models"


SELECTIVE_ROTATION_LOG_LEVEL=DEBUG python test_rotation_module_compare.py --device cuda:4 --variants none --model Qwen/Qwen3-8B --quant-iters 200 --tasks mmlu,gsm8k,piqa,hellaswag >> iters200.log 2>&1 &
