#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Rotation × Quantization Scheme Accuracy Matrix (v2 — unified API)
# ═══════════════════════════════════════════════════════════════════════════════
#
# v2 improvements over v1:
#   - Uses unified apply_rotation() API (BaseRotation registry dispatch)
#   - Non-power-of-2 dimensions work natively (known Hadamard matrices)
#   - rotation_size rarely needed (auto-detected from model dims)
#   - R1+R2+R3 added as rotation level
#   - mmlu added to default evaluation tasks
#
# Tests all combinations of rotation levels with quantization schemes
# and evaluates accuracy via lm_eval.
#
# ⚠️  This script tests QuaRot (fixed Hadamard rotation) only.
#     SpinQuant (trainable rotation) training loop is experimental.
#
# Environment variables:
#   DEVICE         GPU device (default: cuda:7)
#   MODEL          Model name/path (default: Qwen/Qwen3-0.6B)
#   ROTATION_SIZE  Block rotation size (default: auto — rarely needed with v2)
#
# Modes:
#   quick       — limit=100, 3 common schemes (W4A16, MXFP4, NVFP4), 2 tasks
#   full        — No limit, 3 common schemes, 5 tasks (incl. mmlu)
#   full-matrix — No limit, ALL 11 schemes × ALL 5 rotation levels
#   weight-only — W2/W3/W4/W8 A16
#   weight-act  — MXFP4/NVFP4/INT8/MXFP8
#   random      — Same as full but random Hadamard (H×D)
#   tuning      — All 5 rotations × common schemes, iters=200 (auto-round tuning)
#   tuning-matrix — ALL 5 rotations × ALL 11 schemes, iters=200 (full tuning matrix)
#
# Usage:
#   bash run_rotation_scheme_matrix_v2.sh                  # Quick test
#   bash run_rotation_scheme_matrix_v2.sh full              # Full eval
#   bash run_rotation_scheme_matrix_v2.sh full-matrix       # All combos
#   bash run_rotation_scheme_matrix_v2.sh tuning            # With tuning
#   DEVICE=cuda:4 bash run_rotation_scheme_matrix_v2.sh full
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cuda:7}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
ROTATION_SIZE="${ROTATION_SIZE:-}"
MODE="${1:-quick}"

# Build rotation-size flag
RS_FLAG=""
if [ -n "$ROTATION_SIZE" ]; then
    RS_FLAG="--rotation-size $ROTATION_SIZE"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Rotation × Scheme Matrix Test (v2 — unified API)"
echo "  Model:          $MODEL"
echo "  Device:         $DEVICE"
echo "  Mode:           $MODE"
echo "  rotation_size:  ${ROTATION_SIZE:-auto (known Hadamard)}"
echo "═══════════════════════════════════════════════════════════════"

case "$MODE" in
    quick)
        echo "Running quick test (limit=100, common schemes, 2 tasks)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --limit 100 \
            --tasks "hellaswag,piqa"
        ;;

    full)
        echo "Running full eval (no limit, common schemes, 5 tasks)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu"
        ;;

    full-matrix)
        echo "Running full matrix (all rotations × all schemes, no limit)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --full-matrix \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu"
        ;;

    weight-only)
        echo "Running weight-only schemes (W2/W3/W4/W8)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --weight-only \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu" \
            ${2:+--limit $2}
        ;;

    weight-act)
        echo "Running weight+activation schemes (MXFP4/NVFP4/INT8/MXFP8)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --weight-act \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu" \
            ${2:+--limit $2}
        ;;

    random)
        echo "Running with random Hadamard..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --random-hadamard \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu" \
            ${2:+--limit $2}
        ;;

    tuning)
        echo "Running with auto-round tuning (iters=200), common schemes..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --quant-iters 200 \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu" \
            ${2:+--limit $2}
        ;;

    tuning-matrix)
        echo "Running full tuning matrix (all rotations × all schemes, iters=200)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --full-matrix \
            --quant-iters 200 \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: quick, full, full-matrix, weight-only, weight-act, random, tuning, tuning-matrix"
        exit 1
        ;;
esac

echo ""
echo "Done! Check the results_v2_* directory for JSON/CSV outputs."
