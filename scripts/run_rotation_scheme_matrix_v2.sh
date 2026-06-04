#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Rotation × Quantization Scheme Accuracy Matrix (v2 — pipeline API)
# ═══════════════════════════════════════════════════════════════════════════════
#
# v2 improvements over v1:
#   - Uses AutoRound(rotation_config=...) pipeline integration
#     (rotation is applied automatically at Phase 4.5 before quantization)
#   - Supports "quarot"/"spinquant" string shorthands and SpinQuantConfig objects
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
#   DEVICE           GPU device (default: cuda:7)
#   MODEL            Model name/path (default: Qwen/Qwen3-0.6B)
#   ROTATION_SIZE    Block rotation size (default: auto — rarely needed with v2)
#   ROTATION_SIZES   Comma-separated sizes to sweep, e.g. "16,32,64,128,auto"
#
# Modes:
#   quick       — limit=100, 3 common schemes (W4A16, MXFP4, NVFP4), 2 tasks
#   full        — No limit, 3 common schemes, 5 tasks (incl. mmlu)
#   full-matrix — No limit, ALL 11 schemes × ALL 5 rotation levels
#   weight-only — W2/W3/W4/W8 A16
#   weight-act  — MXFP4/NVFP4/INT8/MXFP8
#   random      — Same as full but random Hadamard (H×D)
#   size-sweep  — Sweep rotation_sizes (16,32,64,128,auto) × rotations × schemes
#   tuning      — All 5 rotations × common schemes, iters=200 (auto-round tuning)
#   tuning-matrix — ALL 5 rotations × ALL 11 schemes, iters=200 (full tuning matrix)
#   layerwise   — Block-wise rotation (applied per-block via lifecycle hooks), RTN
#   layerwise-tuning — Block-wise rotation + iters=200 auto-round tuning
#   layerwise-compare — Full-model vs block-wise comparison (validates equivalence)
#
# Usage:
#   bash run_rotation_scheme_matrix_v2.sh                  # Quick test
#   bash run_rotation_scheme_matrix_v2.sh full              # Full eval
#   bash run_rotation_scheme_matrix_v2.sh full-matrix       # All combos
#   bash run_rotation_scheme_matrix_v2.sh size-sweep        # rotation_size sweep
#   bash run_rotation_scheme_matrix_v2.sh tuning            # With tuning
#   bash run_rotation_scheme_matrix_v2.sh layerwise         # Block-wise rotation
#   bash run_rotation_scheme_matrix_v2.sh layerwise-tuning  # Block-wise + tuning
#   bash run_rotation_scheme_matrix_v2.sh layerwise-compare # Full vs block-wise
#   DEVICE=cuda:4 bash run_rotation_scheme_matrix_v2.sh full
#   DEVICE="0,1,2,3" bash run_rotation_scheme_matrix_v2.sh full  # Multi-GPU
#   DEVICE=auto bash run_rotation_scheme_matrix_v2.sh full       # All GPUs
# ═══════════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Configuration ────────────────────────────────────────────────────────────
# DEVICE supports:
#   Single GPU:  "cuda:6"      — use one specific GPU
#   Multi-GPU:   "0,1,2,3"    — use multiple GPUs (accelerate device_map)
#                "auto"        — use all available GPUs
DEVICE="${DEVICE:-cuda:6}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
ROTATION_SIZE="${ROTATION_SIZE:-}"
ROTATION_SIZES="${ROTATION_SIZES:-}"
MODE="${1:-quick}"

# Build rotation-size flags
RS_FLAG=""
if [ -n "$ROTATION_SIZE" ]; then
    RS_FLAG="--rotation-size $ROTATION_SIZE"
fi
RSS_FLAG=""
if [ -n "$ROTATION_SIZES" ]; then
    RSS_FLAG="--rotation-sizes $ROTATION_SIZES"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Rotation × Scheme Matrix Test (v2 — pipeline API)"
echo "  Model:            $MODEL"
echo "  Device:           $DEVICE"
echo "  Mode:             $MODE"
echo "  rotation_size:    ${ROTATION_SIZE:-auto (known Hadamard)}"
echo "  rotation_sizes:   ${ROTATION_SIZES:-(not set)}"
echo "═══════════════════════════════════════════════════════════════"

case "$MODE" in
    quick)
        echo "Running quick test (limit=100, common schemes, 2 tasks)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --limit 100 \
            --tasks "hellaswag,piqa"
        ;;

    full)
        echo "Running full eval (no limit, common schemes, 4 tasks, det vs random, save/load)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --compare-random \
            --save-load \
            --tasks "piqa"
        ;;

    full-matrix)
        echo "Running full matrix (all rotations × all schemes, no limit)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --full-matrix \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu"
        ;;

    weight-only)
        echo "Running weight-only schemes (W2/W3/W4/W8)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --weight-only \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu" \
            ${2:+--limit $2}
        ;;

    weight-act)
        echo "Running weight+activation schemes (MXFP4/NVFP4/INT8/MXFP8)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --weight-act \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu" \
            ${2:+--limit $2}
        ;;

    random)
        echo "Note: 'random' mode is deprecated — use 'full' which now includes --compare-random."
        echo "Running random-only for backward compatibility..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --random-hadamard \
            --save-load \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    size-sweep)
        echo "Running rotation_size sweep with det vs random (${ROTATION_SIZES:-16,64,128,auto})..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            --rotation-sizes "${ROTATION_SIZES:-16,64,128,auto}" \
            --rotations "none,R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --compare-random \
            --save-load \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    tuning)
        echo "Running with auto-round tuning (iters=200), det vs random, save/load..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --quant-iters 200 \
            --compare-random \
            --save-load \
            --tasks "hellaswag,piqa" \
            ${2:+--limit $2}
        ;;

    tuning-matrix)
        echo "Running full tuning matrix (all rotations × all schemes, iters=200)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --full-matrix \
            --quant-iters 200 \
            --tasks "hellaswag,piqa,winogrande,lambada_openai,mmlu"
        ;;

    layerwise)
        echo "Running block-wise (layer-wise) rotation, RTN, save/load..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --layerwise \
            --save-load \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    layerwise-tuning)
        echo "Running block-wise rotation + auto-round tuning (iters=200), save/load..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4,NVFP4" \
            --layerwise \
            --quant-iters 200 \
            --save-load \
            --tasks "hellaswag,piqa,winogrande,lambada_openai" \
            ${2:+--limit $2}
        ;;

    layerwise-compare)
        echo "Running full-model vs block-wise comparison (validates equivalence)..."
        echo "  Step 1/2: Full-model rotation (baseline)..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4" \
            --save-load \
            --tasks "hellaswag,piqa" \
            --output-dir "_compare_fullmodel_${TIMESTAMP}" \
            ${2:+--limit $2}
        echo "  Step 2/2: Block-wise rotation..."
        python test_rotation_scheme_matrix_v2.py \
            --model "$MODEL" \
            --device "$DEVICE" \
            $RS_FLAG $RSS_FLAG \
            --rotations "R1,R1+R2,R1+R2+R3+R4" \
            --schemes "W4A16,MXFP4" \
            --layerwise \
            --save-load \
            --tasks "hellaswag,piqa" \
            --output-dir "_compare_layerwise_${TIMESTAMP}" \
            ${2:+--limit $2}
        echo ""
        echo "Compare results in _compare_fullmodel_${TIMESTAMP}/ vs _compare_layerwise_${TIMESTAMP}/"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: quick, full, full-matrix, weight-only, weight-act, random, size-sweep, tuning, tuning-matrix, layerwise, layerwise-tuning, layerwise-compare"
        exit 1
        ;;
esac

echo ""
echo "Done! Check the results_v2_* directory for JSON/CSV outputs."
