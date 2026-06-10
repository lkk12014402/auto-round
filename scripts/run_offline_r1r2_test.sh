#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Offline R1+R2 Rotation Equivalence & Save/Load Roundtrip Test
# ═══════════════════════════════════════════════════════════════════════════════
#
# Verifies:
#   1. Offline R1+R2 rotation does NOT change bf16 model accuracy
#   2. Rotation + quantization save/load roundtrip is lossless
#
# Usage:
#   bash run_offline_r1r2_test.sh                     # Quick test (limit=100)
#   bash run_offline_r1r2_test.sh full                # Full evaluation
#   DEVICE=cuda:2 bash run_offline_r1r2_test.sh       # Specify GPU
#   MODEL=meta-llama/Llama-3.2-1B bash run_offline_r1r2_test.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODE="${1:-quick}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
LOG_FILE="offline_r1r2_${MODEL_SHORT}_${MODE}_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════"
echo "  Offline R1+R2 Equivalence & Roundtrip Test"
echo "  Model:  $MODEL"
echo "  Device: $DEVICE"
echo "  Mode:   $MODE"
echo "  Log:    $LOG_FILE"
echo "═══════════════════════════════════════════════════════════════"

case "$MODE" in
    quick)
        python test_offline_r1r2_equivalence.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --limit 100 \
            --tasks "hellaswag,piqa" \
            --schemes "W4A16" \
            --batch-size 8 \
	    --keep-models \
            2>&1 | tee "$LOG_FILE"
        ;;
    full)
        python test_offline_r1r2_equivalence.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            --schemes "W4A16,MXFP4" \
            --batch-size 8 \
            2>&1 | tee "$LOG_FILE"
        ;;
    full-schemes)
        python test_offline_r1r2_equivalence.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --tasks "hellaswag,piqa,winogrande" \
            --schemes "W4A16,W3A16,MXFP4,NVFP4,FP8_STATIC" \
            --batch-size 8 \
            2>&1 | tee "$LOG_FILE"
        ;;
    tuning)
        python test_offline_r1r2_equivalence.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --tasks "hellaswag,piqa" \
            --schemes "W4A16" \
            --quant-iters 200 \
            --batch-size 8 \
            2>&1 | tee "$LOG_FILE"
        ;;
    equiv-only)
        # Only test rotation equivalence, skip quantization roundtrip
        python test_offline_r1r2_equivalence.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            --skip-roundtrip \
            --batch-size 8 \
            2>&1 | tee "$LOG_FILE"
        ;;
    roundtrip-only)
        # Only test roundtrip, skip equivalence
        python test_offline_r1r2_equivalence.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --limit 100 \
            --tasks "hellaswag,piqa" \
            --schemes "W4A16" \
            --skip-equivalence \
            --batch-size 8 \
            2>&1 | tee "$LOG_FILE"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available: quick, full, full-schemes, tuning, equiv-only, roundtrip-only"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════════"
