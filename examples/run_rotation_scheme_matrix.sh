#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Rotation × Quantization Scheme Accuracy Matrix
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests all combinations of rotation levels (R1, R1+R2, R1+R2+R3, R1+R2+R3+R4)
# with multiple quantization schemes and evaluates accuracy via lm_eval.
#
# Results saved to timestamped directory with JSON, CSV, and config.
#
# ⚠️  This script tests QuaRot (fixed Hadamard rotation) only.
#     SpinQuant (trainable rotation) is experimental and not included.
#
# Environment variables:
#   DEVICE         GPU device (default: cuda:7)
#   MODEL          Model name/path (default: Qwen/Qwen3-0.6B)
#   ROTATION_SIZE  Block rotation size (default: auto from model dims)
#                  Set to 128 for models with non-power-of-2 dimensions.
#
# Modes:
#   quick       — limit=100, 3 common schemes (W4A16, MXFP4, NVFP4), 2 tasks
#                 Good for ~15min sanity check to verify everything works.
#   full        — No limit, 3 common schemes, 4 tasks
#                 Full accuracy numbers for common configurations. ~2 hours.
#   full-matrix — No limit, ALL 11 schemes × ALL 5 rotation levels
#                 Exhaustive test covering every combination. ~8+ hours.
#   weight-only — W2A16, W3A16, W4A16, W8A16 with rotation
#                 Focus on weight-only quantization (activations stay FP16).
#                 R1/R2 usually matter most here.
#   weight-act  — MXFP4, NVFP4, INT8, MXFP8 with rotation
#                 Weight+Activation quantization — R3/R4 help most here
#                 because they smooth activations before quantization.
#   random      — Same as full but with random Hadamard (H×D)
#                 Compares random vs deterministic rotation quality.
#                 Random may better scatter outliers but is not reproducible.
#   gptq        — W4A16 with iters=200 GPTQ-style optimization
#                 Tests rotation combined with optimized quantization.
#                 Slower but higher accuracy than RTN (iters=0).
#
# Usage:
#   bash run_rotation_scheme_matrix.sh                  # Quick test (limit=100)
#   bash run_rotation_scheme_matrix.sh full              # Full eval (no limit)
#   bash run_rotation_scheme_matrix.sh full-matrix       # All rotations × all schemes
#   bash run_rotation_scheme_matrix.sh weight-only       # W2/W3/W4/W8 only
#   bash run_rotation_scheme_matrix.sh weight-only 100   # W2/W3/W4/W8, limit=100
#   bash run_rotation_scheme_matrix.sh weight-act        # MXFP4/NVFP4/INT8 only
#   ROTATION_SIZE=128 bash run_rotation_scheme_matrix.sh full  # With block rotation
#   DEVICE=cuda:4 bash run_rotation_scheme_matrix.sh full      # Custom GPU
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
echo "  Rotation × Scheme Matrix Test"
echo "  Model:          $MODEL"
echo "  Device:         $DEVICE"
echo "  Mode:           $MODE"
echo "  rotation_size:  ${ROTATION_SIZE:-auto}"
echo "═══════════════════════════════════════════════════════════════"

case "$MODE" in
    quick)
        echo "Running quick test (limit=100, common schemes)..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --limit 100 \
            --tasks "hellaswag,piqa"
        ;;

    full)
        echo "Running full eval (no limit, common schemes)..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --tasks "hellaswag,piqa,winogrande,lambada_openai"
        ;;

    full-matrix)
        echo "Running full matrix (all rotations × all schemes, no limit)..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --full-matrix \
            --tasks "hellaswag,piqa,winogrande,lambada_openai"
        ;;

    weight-only)
        echo "Running weight-only schemes (W2/W3/W4/W8)..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --weight-only \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    weight-act)
        echo "Running weight+activation schemes (MXFP4/NVFP4/INT8/MXFP8)..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --weight-act \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    random)
        echo "Running with random Hadamard..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --random-hadamard \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    gptq)
        echo "Running with GPTQ-style optimization (iters=200)..."
        python test_rotation_scheme_matrix.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG \
            --rotations "none,R1,R1+R2" \
            --schemes "W4A16" \
            --quant-iters 200 \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: quick, full, full-matrix, weight-only, weight-act, random, gptq"
        exit 1
        ;;
esac

echo ""
echo "Done! Check the results_matrix_* directory for JSON/CSV outputs."
