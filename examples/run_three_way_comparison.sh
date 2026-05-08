#!/bin/bash
# Three-Way Comparison: Auto-Round vs Quark vs llm-compressor
# Rotation + MXFP4 RTN on Qwen3-0.6B
#
# Usage:
#   bash run_three_way_comparison.sh [GPU_ID] [LIMIT]
#   bash run_three_way_comparison.sh 4 100    # Quick test
#   bash run_three_way_comparison.sh 4        # Full eval

set -euo pipefail
cd "$(dirname "$0")"

GPU_ID=${1:-4}
LIMIT=${2:-""}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs_comparison"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${TIMESTAMP}_three_way_comparison.log"

LIMIT_ARGS=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

echo "================================================================"
echo "  Three-Way Comparison: Rotation + MXFP4"
echo "  Frameworks: Auto-Round, Quark, llm-compressor"
echo "  Model: Qwen/Qwen3-0.6B"
echo "  GPU: $GPU_ID | Limit: ${LIMIT:-full}"
echo "  Time: $(date)"
echo "================================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_three_way_comparison.py \
    --device cuda:0 \
    --rotations "R1,R1+R2,R1+R2+R4,R1+R2+R3+R4" \
    --frameworks "autoround,quark,llmcompressor" \
    --include-baselines \
    $LIMIT_ARGS \
    2>&1 | tee "$LOGFILE"

echo ""
echo "Log saved to: $LOGFILE"
