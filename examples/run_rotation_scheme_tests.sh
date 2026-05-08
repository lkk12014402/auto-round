#!/bin/bash
# Run rotation × quantization scheme tests on Qwen3-0.6B
# Tests: W4A16 (INT4) and NVFP4 with all rotation levels (R1→R1+R2+R3+R4)
#
# Usage:
#   bash run_rotation_scheme_tests.sh [GPU_ID] [LIMIT]
#   bash run_rotation_scheme_tests.sh 7 100    # Quick test
#   bash run_rotation_scheme_tests.sh 7        # Full eval

set -euo pipefail
cd "$(dirname "$0")"

GPU_ID=${1:-7}
LIMIT=${2:-""}
DEVICE="cuda:${GPU_ID}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs_comparison"
mkdir -p "$LOGDIR"

LIMIT_ARGS=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

echo "================================================================"
echo "  Rotation × Quantization Scheme Tests"
echo "  Device: $DEVICE"
echo "  Limit:  ${LIMIT:-full}"
echo "  Time:   $(date)"
echo "================================================================"

# ── W4A16 (INT4 weight-only) ──────────────────────────────────────
echo ""
echo ">>> Testing W4A16 (INT4) with all rotation levels..."
LOGFILE="${LOGDIR}/${TIMESTAMP}_W4A16_rotations.log"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_rotation_schemes.py \
    --device cuda:0 \
    --schemes W4A16 \
    --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
    --include-baselines \
    $LIMIT_ARGS \
    2>&1 | tee "$LOGFILE"

echo "W4A16 log: $LOGFILE"

# ── NVFP4 ─────────────────────────────────────────────────────────
echo ""
echo ">>> Testing NVFP4 with all rotation levels..."
LOGFILE="${LOGDIR}/${TIMESTAMP}_NVFP4_rotations.log"

CUDA_VISIBLE_DEVICES=$GPU_ID python test_rotation_schemes.py \
    --device cuda:0 \
    --schemes NVFP4 \
    --rotations "R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
    --include-baselines \
    $LIMIT_ARGS \
    2>&1 | tee "$LOGFILE"

echo "NVFP4 log: $LOGFILE"

echo ""
echo "================================================================"
echo "  All tests completed at $(date)"
echo "  Logs in: $LOGDIR/"
echo "================================================================"
