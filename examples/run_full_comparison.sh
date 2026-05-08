#!/bin/bash
# =============================================================================
# Run full comparison: Auto-Round vs Quark
# Rotation levels: R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
# Quantization: MXFP4 + RTN
# Model: Qwen/Qwen3-0.6B
#
# NOTE: Quark R3 does not support custom rotation_size. When R3 is enabled,
#       rotation_size=None is used for Quark (R1 uses hidden_size=1024,
#       R4 uses intermediate_size=3072 with built-in non-pow2 Hadamard).
#       Auto-Round always uses the specified rotation_size (128 by default).
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="Qwen/Qwen3-0.6B"
TASKS="piqa,hellaswag"
LIMIT=""  # empty = full eval; set to e.g. "200" for quick test
BATCH_SIZE=8
ROTATION_SIZE=128
NSAMPLES=128
SEQLEN=512

# Devices — run auto-round and quark on different GPUs in parallel
DEVICE_AR="cuda:6"
DEVICE_QK="cuda:7"

LOGDIR="$SCRIPT_DIR/logs_comparison"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="$2"; shift 2;;
        --device-ar) DEVICE_AR="$2"; shift 2;;
        --device-qk) DEVICE_QK="$2"; shift 2;;
        --rotation-size) ROTATION_SIZE="$2"; shift 2;;
        --tasks) TASKS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "============================================================"
echo " Rotation + MXFP4 Comparison: Auto-Round vs Quark"
echo "============================================================"
echo " Model:         $MODEL"
echo " Tasks:         $TASKS"
echo " Limit:         ${LIMIT:-full}"
echo " Rotation size: $ROTATION_SIZE"
echo " Device (AR):   $DEVICE_AR"
echo " Device (QK):   $DEVICE_QK"
echo " Log dir:       $LOGDIR"
echo "============================================================"

# =====================================================================
# Step 1: Run baselines (FP16 + MXFP4-only) for both frameworks
# =====================================================================
echo ""
echo ">>> [1/3] Running baselines (FP16, MXFP4-only)..."

# Auto-Round baseline
AR_BASE_LOG="$LOGDIR/${TIMESTAMP}_autoround_baseline.log"
echo "  Auto-Round baseline → $AR_BASE_LOG"
python test_autoround_rotation_mxfp4.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --device "$DEVICE_AR" \
    --nsamples $NSAMPLES --seqlen $SEQLEN \
    --batch_size $BATCH_SIZE \
    --levels "baseline_fp16,mxfp4_only" \
    --no-r1 --no-r2 \
    $LIMIT_ARG \
    > "$AR_BASE_LOG" 2>&1 &
AR_BASE_PID=$!

# Quark baseline
QK_BASE_LOG="$LOGDIR/${TIMESTAMP}_quark_baseline.log"
echo "  Quark baseline → $QK_BASE_LOG"
python test_quark_rotation_mxfp4.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --device "$DEVICE_QK" \
    --num_calib_data $NSAMPLES --seqlen $SEQLEN \
    --batch_size $BATCH_SIZE \
    --levels "baseline_fp16,mxfp4_only" \
    --no-r1 --no-r2 \
    $LIMIT_ARG \
    > "$QK_BASE_LOG" 2>&1 &
QK_BASE_PID=$!

wait $AR_BASE_PID $QK_BASE_PID
echo "  ✓ Baselines complete"

# =====================================================================
# Step 2: Run each rotation level (sequentially to avoid OOM)
# =====================================================================
echo ""
echo ">>> [2/3] Running rotation levels..."

# --- R1 only ---
echo ""
echo "  --- R1 only ---"

AR_LOG="$LOGDIR/${TIMESTAMP}_autoround_R1.log"
echo "    Auto-Round (R1) → $AR_LOG"
python test_autoround_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_AR" \
    --nsamples $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --no-r2 --rotation-size $ROTATION_SIZE --online-r1 \
    $LIMIT_ARG \
    > "$AR_LOG" 2>&1 &
AR_PID=$!

QK_LOG="$LOGDIR/${TIMESTAMP}_quark_R1.log"
echo "    Quark (R1) → $QK_LOG"
python test_quark_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_QK" \
    --num_calib_data $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --no-r2 --rotation-size $ROTATION_SIZE \
    $LIMIT_ARG \
    > "$QK_LOG" 2>&1 &
QK_PID=$!

wait $AR_PID $QK_PID
echo "    ✓ R1 complete"

# --- R1+R2 ---
echo ""
echo "  --- R1+R2 ---"

AR_LOG="$LOGDIR/${TIMESTAMP}_autoround_R1+R2.log"
echo "    Auto-Round (R1+R2) → $AR_LOG"
python test_autoround_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_AR" \
    --nsamples $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --r2 --rotation-size $ROTATION_SIZE --online-r1 \
    $LIMIT_ARG \
    > "$AR_LOG" 2>&1 &
AR_PID=$!

QK_LOG="$LOGDIR/${TIMESTAMP}_quark_R1+R2.log"
echo "    Quark (R1+R2) → $QK_LOG"
python test_quark_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_QK" \
    --num_calib_data $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --r2 --rotation-size $ROTATION_SIZE \
    $LIMIT_ARG \
    > "$QK_LOG" 2>&1 &
QK_PID=$!

wait $AR_PID $QK_PID
echo "    ✓ R1+R2 complete"

# --- R1+R2+R3 ---
# NOTE: Quark R3 does not support custom rotation_size, so it uses None
# (R1 uses full hidden_size=1024, R4 would use intermediate_size=3072)
echo ""
echo "  --- R1+R2+R3 ---"

AR_LOG="$LOGDIR/${TIMESTAMP}_autoround_R1+R2+R3.log"
echo "    Auto-Round (R1+R2+R3) → $AR_LOG"
python test_autoround_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_AR" \
    --nsamples $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --r2 --enable-r3 --rotation-size $ROTATION_SIZE --online-r1 \
    $LIMIT_ARG \
    > "$AR_LOG" 2>&1 &
AR_PID=$!

QK_LOG="$LOGDIR/${TIMESTAMP}_quark_R1+R2+R3.log"
echo "    Quark (R1+R2+R3, rotation_size=None due to R3 limitation) → $QK_LOG"
python test_quark_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_QK" \
    --num_calib_data $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --r2 --enable-r3 --rotation-size $ROTATION_SIZE \
    $LIMIT_ARG \
    > "$QK_LOG" 2>&1 &
QK_PID=$!

wait $AR_PID $QK_PID
echo "    ✓ R1+R2+R3 complete"

# --- R1+R2+R3+R4 ---
echo ""
echo "  --- R1+R2+R3+R4 ---"

AR_LOG="$LOGDIR/${TIMESTAMP}_autoround_R1+R2+R3+R4.log"
echo "    Auto-Round (R1+R2+R3+R4) → $AR_LOG"
python test_autoround_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_AR" \
    --nsamples $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --r2 --enable-r3 --enable-r4 --rotation-size $ROTATION_SIZE --online-r1 \
    $LIMIT_ARG \
    > "$AR_LOG" 2>&1 &
AR_PID=$!

QK_LOG="$LOGDIR/${TIMESTAMP}_quark_R1+R2+R3+R4.log"
echo "    Quark (R1+R2+R3+R4, rotation_size=None due to R3 limitation) → $QK_LOG"
python test_quark_rotation_mxfp4.py \
    --model "$MODEL" --tasks "$TASKS" --device "$DEVICE_QK" \
    --num_calib_data $NSAMPLES --seqlen $SEQLEN --batch_size $BATCH_SIZE \
    --levels "rotation_mxfp4" \
    --r1 --r2 --enable-r3 --enable-r4 --rotation-size $ROTATION_SIZE \
    $LIMIT_ARG \
    > "$QK_LOG" 2>&1 &
QK_PID=$!

wait $AR_PID $QK_PID
echo "    ✓ R1+R2+R3+R4 complete"

# =====================================================================
# Step 3: Parse results and print summary table
# =====================================================================
echo ""
echo ">>> [3/3] Collecting results..."
echo ""

python3 - "$LOGDIR" "$TIMESTAMP" <<'PYEOF'
import sys, os, re, glob

logdir = sys.argv[1]
timestamp = sys.argv[2]

def parse_log(filepath):
    """Extract acc_norm values from a log file."""
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            m = re.search(r'(\w+):\s+acc_norm\s*=\s*([\d.]+)', line)
            if m:
                metrics[m.group(1)] = float(m.group(2))
    return metrics

# Collect all results
results = {}
log_files = sorted(glob.glob(os.path.join(logdir, f"{timestamp}_*.log")))

for lf in log_files:
    basename = os.path.basename(lf)
    parts = basename.replace(timestamp + "_", "").replace(".log", "")
    metrics = parse_log(lf)
    if metrics:
        results[parts] = metrics

# Print summary
print("=" * 100)
print("FULL COMPARISON: Auto-Round vs Quark — Rotation + MXFP4 (Qwen3-0.6B)")
print("=" * 100)

# Determine all tasks
all_tasks = sorted(set(t for m in results.values() for t in m.keys()))

# Header
header = f"{'Configuration':<40}"
for task in all_tasks:
    header += f" | {task:<12}"
print(header)
print("-" * len(header))

# Print rows in logical order
order = [
    ("autoround_baseline", "Auto-Round: FP16 baseline"),
    ("quark_baseline", "Quark: FP16 baseline"),
    ("", ""),  # separator
    ("autoround_R1", "Auto-Round: R1 + MXFP4"),
    ("quark_R1", "Quark: R1 + MXFP4"),
    ("", ""),
    ("autoround_R1+R2", "Auto-Round: R1+R2 + MXFP4"),
    ("quark_R1+R2", "Quark: R1+R2 + MXFP4"),
    ("", ""),
    ("autoround_R1+R2+R3", "Auto-Round: R1+R2+R3 + MXFP4"),
    ("quark_R1+R2+R3", "Quark: R1+R2+R3 + MXFP4"),
    ("", ""),
    ("autoround_R1+R2+R3+R4", "Auto-Round: R1+R2+R3+R4 + MXFP4"),
    ("quark_R1+R2+R3+R4", "Quark: R1+R2+R3+R4 + MXFP4"),
]

for key, label in order:
    if not key:
        print()
        continue
    # Find matching key in results (baseline logs may have both fp16 and mxfp4)
    if key in results:
        metrics = results[key]
    else:
        # Try partial match
        matches = [k for k in results if key in k]
        if matches:
            metrics = results[matches[0]]
        else:
            row = f"{label:<40}"
            for task in all_tasks:
                row += f" | {'N/A':<12}"
            print(row)
            continue
    row = f"{label:<40}"
    for task in all_tasks:
        val = metrics.get(task)
        if val is not None:
            row += f" | {val:<12.4f}"
        else:
            row += f" | {'N/A':<12}"
    print(row)

# Print delta table
print()
print("Δ vs MXFP4-only (positive = rotation helps):")
print("-" * 80)
ar_base = results.get("autoround_baseline", {})
qk_base = results.get("quark_baseline", {})

for key, label in order:
    if not key or "baseline" in key:
        continue
    if key not in results:
        continue
    metrics = results[key]
    # Determine which baseline to compare against
    base = ar_base if "autoround" in key else qk_base
    row = f"  {label:<38}"
    for task in all_tasks:
        val = metrics.get(task)
        base_val = base.get(task)
        if val is not None and base_val is not None:
            delta = val - base_val
            row += f" | {delta:+.4f}     "
        else:
            row += f" | {'N/A':<12}"
    print(row)

print("=" * 100)
print(f"\nNOTE: Quark R3 does not support custom rotation_size.")
print(f"      When R3 is enabled, Quark uses rotation_size=None internally")
print(f"      (R1 uses hidden_size, R4 uses intermediate_size with non-pow2 Hadamard).")
print(f"\nLogs saved in: {logdir}/")
PYEOF

echo ""
echo "Done! All logs in: $LOGDIR/"
