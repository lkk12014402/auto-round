#!/bin/bash
# =============================================================================
# Quark vs Auto-Round: Rotation + Quantization Comparison Tests
#
# This script runs all comparison scenarios between Quark and Auto-Round
# on Qwen/Qwen3-0.6B with lm_eval evaluation.
#
# Usage:
#   bash examples/run_comparison_tests.sh                    # Run all scenarios
#   bash examples/run_comparison_tests.sh --scenario 1       # Run specific scenario
#   bash examples/run_comparison_tests.sh --scenario 1,2     # Run multiple scenarios
#   bash examples/run_comparison_tests.sh --limit 200        # Quick test with 200 samples
#   bash examples/run_comparison_tests.sh --tasks piqa       # Single task
#   bash examples/run_comparison_tests.sh --model Qwen/Qwen3-0.6B --dry-run
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TASKS="${TASKS:-piqa,hellaswag}"
LIMIT="${LIMIT:-}"            # empty = full eval
BATCH_SIZE="${BATCH_SIZE:-8}"
LOG_DIR="${LOG_DIR:-examples/logs}"
DRY_RUN=false
SCENARIOS_TO_RUN="all"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";       shift 2 ;;
        --tasks)      TASKS="$2";       shift 2 ;;
        --limit)      LIMIT="$2";       shift 2 ;;
        --batch-size) BATCH_SIZE="$2";  shift 2 ;;
        --log-dir)    LOG_DIR="$2";     shift 2 ;;
        --scenario)   SCENARIOS_TO_RUN="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;     shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL        Model name (default: Qwen/Qwen3-0.6B)"
            echo "  --tasks TASKS        Comma-separated lm_eval tasks (default: piqa,hellaswag)"
            echo "  --limit N            Limit samples per task (default: full eval)"
            echo "  --batch-size N       Evaluation batch size (default: 8)"
            echo "  --log-dir DIR        Log directory (default: examples/logs)"
            echo "  --scenario 1,2,3..   Run specific scenarios (default: all)"
            echo "  --dry-run            Print commands without executing"
            echo ""
            echo "Scenarios:"
            echo "  1  MXFP4, R1+R2, rotation_size=128  (Quark vs Auto-Round)"
            echo "  2  MXFP4, R1+R2, full rotation      (Quark vs Auto-Round)"
            echo "  3  W4A16 RTN, R1+R2                  (Auto-Round only)"
            echo "  4  Rotation only (no quant)           (Auto-Round only)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

# ── Helper ───────────────────────────────────────────────────────────────────
run_cmd() {
    local desc="$1"
    local logfile="$2"
    shift 2
    local cmd="$*"

    echo ""
    echo "┌──────────────────────────────────────────────────────────────────"
    echo "│ $desc"
    echo "│ Log: $logfile"
    echo "│ CMD: $cmd"
    echo "└──────────────────────────────────────────────────────────────────"

    if $DRY_RUN; then
        echo "  [DRY-RUN] Skipped."
        return 0
    fi

    eval "$cmd" 2>&1 | tee "$logfile"
    echo ""
    echo "  ✓ Done. Log saved to $logfile"
}

should_run() {
    local scenario="$1"
    if [[ "$SCENARIOS_TO_RUN" == "all" ]]; then
        return 0
    fi
    echo ",$SCENARIOS_TO_RUN," | grep -q ",$scenario,"
}

# ── Print banner ─────────────────────────────────────────────────────────────
echo "======================================================================"
echo " Quark vs Auto-Round Comparison Tests"
echo "======================================================================"
echo "  Model:      $MODEL"
echo "  Tasks:      $TASKS"
echo "  Limit:      ${LIMIT:-full}"
echo "  Batch size: $BATCH_SIZE"
echo "  Log dir:    $LOG_DIR"
echo "  Scenarios:  $SCENARIOS_TO_RUN"
echo "  Timestamp:  $TIMESTAMP"
echo "  Dry-run:    $DRY_RUN"
echo "======================================================================"

# =============================================================================
# Scenario 1: MXFP4 — Quark default (R1+R2, rotation_size=128)
#
# Quark 默认: rotation_size=128, online_r1=True
# Auto-Round: rotation_size=128, online_r1=True (matching Quark)
# 量化: MXFP4 (W4A4)
# =============================================================================
if should_run 1; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Scenario 1: MXFP4 + R1+R2, rotation_size=128                 ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    run_cmd \
        "[Scenario 1a] Quark: MXFP4 + R1+R2 (rotation_size=128)" \
        "$LOG_DIR/${TIMESTAMP}_s1a_quark_mxfp4_r128.log" \
        python examples/test_quark_rotation_mxfp4.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline_fp16,mxfp4_only,rotation_mxfp4 \
            --rotation-size 128 \
            $LIMIT_ARG

    run_cmd \
        "[Scenario 1b] Auto-Round: MXFP4 + R1+R2 (rotation_size=128)" \
        "$LOG_DIR/${TIMESTAMP}_s1b_autoround_mxfp4_r128.log" \
        python examples/test_autoround_rotation_mxfp4.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline_fp16,mxfp4_only,rotation_mxfp4 \
            --rotation-size 128 \
            $LIMIT_ARG
fi

# =============================================================================
# Scenario 2: MXFP4 — Full rotation (R1+R2, rotation_size=full dim)
#
# Quark: rotation_size=1024 (= hidden_size), online_r1=True
# Auto-Round: rotation_size=None (= full hidden_size=1024)
# =============================================================================
if should_run 2; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Scenario 2: MXFP4 + R1+R2, full rotation (size=1024)         ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    run_cmd \
        "[Scenario 2a] Quark: MXFP4 + R1+R2 (rotation_size=1024)" \
        "$LOG_DIR/${TIMESTAMP}_s2a_quark_mxfp4_r1024.log" \
        python examples/test_quark_rotation_mxfp4.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline_fp16,rotation_mxfp4 \
            --rotation-size 1024 \
            $LIMIT_ARG

    run_cmd \
        "[Scenario 2b] Auto-Round: MXFP4 + R1+R2 (full rotation)" \
        "$LOG_DIR/${TIMESTAMP}_s2b_autoround_mxfp4_full.log" \
        python examples/test_autoround_rotation_mxfp4.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline_fp16,rotation_mxfp4 \
            $LIMIT_ARG
fi

# =============================================================================
# Scenario 3: W4A16 RTN — Auto-Round only (Quark 不支持 W4A16)
#
# 对比: rotation_size=128 vs full rotation
# =============================================================================
if should_run 3; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Scenario 3: W4A16 RTN + R1+R2 (Auto-Round only)              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    run_cmd \
        "[Scenario 3a] Auto-Round: W4A16 RTN + R1+R2 (rotation_size=128)" \
        "$LOG_DIR/${TIMESTAMP}_s3a_autoround_w4a16_r128.log" \
        python examples/test_rotation_quantization.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline_fp16,rtn_only,rotation_rtn \
            --no-r3 --no-r4 \
            --rotation-size 128 \
            $LIMIT_ARG

    run_cmd \
        "[Scenario 3b] Auto-Round: W4A16 RTN + R1+R2 (full rotation)" \
        "$LOG_DIR/${TIMESTAMP}_s3b_autoround_w4a16_full.log" \
        python examples/test_rotation_quantization.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline_fp16,rtn_only,rotation_rtn \
            --no-r3 --no-r4 \
            $LIMIT_ARG
fi

# =============================================================================
# Scenario 4: Rotation only (no quantization) — 验证旋转等价性
#
# 旋转不应该改变精度 (应与 baseline 完全一致)
# =============================================================================
if should_run 4; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Scenario 4: Rotation only, no quantization                    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    run_cmd \
        "[Scenario 4a] Auto-Round: Rotation only R1+R2 (rotation_size=128)" \
        "$LOG_DIR/${TIMESTAMP}_s4a_rotation_only_r128.log" \
        python examples/test_qwen3_rotation_eval.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline,r1r2 \
            --rotation-size 128 \
            $LIMIT_ARG

    run_cmd \
        "[Scenario 4b] Auto-Round: Rotation only R1+R2 (full rotation)" \
        "$LOG_DIR/${TIMESTAMP}_s4b_rotation_only_full.log" \
        python examples/test_qwen3_rotation_eval.py \
            --model "$MODEL" \
            --device cuda:0 \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --levels baseline,r1r2 \
            $LIMIT_ARG
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "======================================================================"
echo " All requested scenarios completed!"
echo " Logs saved to: $LOG_DIR/${TIMESTAMP}_*.log"
echo ""
echo " To compare results, search for 'COMPARISON TABLE' in each log:"
echo "   grep -A 20 'COMPARISON TABLE' $LOG_DIR/${TIMESTAMP}_*.log"
echo "======================================================================"
