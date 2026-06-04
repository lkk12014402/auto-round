#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Full Rotation × Scheme × Iters Matrix Test
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests all combinations of:
#   Rotations: none, R1, R1+R2, R1+R2+R3, R1+R2+R3+R4
#   Schemes:   W4A16, MXFP4, NVFP4
#   Iters:     0 (RTN), 200 (auto-round tuning)
#   Random:    several per-rotation random combinations
#
# Usage:
#   bash run_full_matrix.sh                    # default: Qwen3-0.6B, cuda:0
#   bash run_full_matrix.sh --model <model>    # custom model
#   bash run_full_matrix.sh --device cuda:1    # custom device
#   bash run_full_matrix.sh --limit 200        # custom eval limit
#   bash run_full_matrix.sh --parts 1,2        # only run Part 1 and Part 2
#   bash run_full_matrix.sh --parts 3          # only run Part 3 (random RTN)
#   bash run_full_matrix.sh --outdir mydir     # custom output directory
#
# Output saved to: full_matrix_<model>_<timestamp>/
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-50}"
TASKS="${TASKS:-hellaswag}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-512}"
PARTS="${PARTS:-1,2,3,4}"
OUTDIR_OVERRIDE=""

# ── Parse CLI overrides ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2";   shift 2;;
        --device)  DEVICE="$2";  shift 2;;
        --limit)   LIMIT="$2";   shift 2;;
        --tasks)   TASKS="$2";   shift 2;;
        --nsamples) NSAMPLES="$2"; shift 2;;
        --seqlen)  SEQLEN="$2";  shift 2;;
        --parts)   PARTS="$2";   shift 2;;
        --outdir)  OUTDIR_OVERRIDE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Parse --parts into a lookup set
IFS=',' read -ra PART_ARRAY <<< "$PARTS"
declare -A RUN_PART
for p in "${PART_ARRAY[@]}"; do
    RUN_PART[$p]=1
done

MODEL_SHORT=$(basename "$MODEL")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "$OUTDIR_OVERRIDE" ]]; then
    OUTDIR="$OUTDIR_OVERRIDE"
else
    OUTDIR="full_matrix_${MODEL_SHORT}_${TIMESTAMP}"
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "  FULL MATRIX TEST"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Model:     $MODEL"
echo "  Device:    $DEVICE"
echo "  Limit:     $LIMIT"
echo "  Tasks:     $TASKS"
echo "  Nsamples:  $NSAMPLES"
echo "  Seqlen:    $SEQLEN"
echo "  Parts:     $PARTS"
echo "  Output:    $OUTDIR"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

COMMON_ARGS="--model $MODEL --device $DEVICE --limit $LIMIT --tasks $TASKS --nsamples $NSAMPLES --seqlen $SEQLEN"

run_test() {
    local desc="$1"
    shift
    local logfile="$OUTDIR/$(echo "$desc" | tr ' /+' '___').log"
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  ▶ $desc"
    echo "  Log: $logfile"
    echo "────────────────────────────────────────────────────────────────"
    python3 test_save_load_roundtrip.py $COMMON_ARGS "$@" \
        --output-dir "$OUTDIR/$desc" \
        --cleanup \
        2>&1 | tee "$logfile"
    echo ""
}

mkdir -p "$OUTDIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Deterministic Hadamard — RTN (iters=0)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -n "${RUN_PART[1]:-}" ]]; then
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Part 1: Deterministic Hadamard × RTN (iters=0)                 ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

run_test "hadamard_rtn" \
    --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
    --schemes "W4A16,MXFP4,NVFP4" \
    --quant-iters 0
else
echo "  Skipping Part 1 (not in --parts $PARTS)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Deterministic Hadamard — Tuning (iters=200)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -n "${RUN_PART[2]:-}" ]]; then
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Part 2: Deterministic Hadamard × Tuning (iters=200)            ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

run_test "hadamard_iters200" \
    --rotations "none,R1,R1+R2,R1+R2+R3,R1+R2+R3+R4" \
    --schemes "W4A16,MXFP4,NVFP4" \
    --quant-iters 200
else
echo "  Skipping Part 2 (not in --parts $PARTS)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Random combinations — RTN (iters=0)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -n "${RUN_PART[3]:-}" ]]; then
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Part 3: Random rotation combos × RTN (iters=0)                 ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

# Combo 1: R1+R2, only R2 random
run_test "random_R2_rtn" \
    --rotations "R1+R2" \
    --schemes "W4A16,MXFP4,NVFP4" \
    --random-rotations "R2" \
    --quant-iters 0

# Combo 2: R1+R2, both random
run_test "random_R1R2_rtn" \
    --rotations "R1+R2" \
    --schemes "W4A16,MXFP4,NVFP4" \
    --random-rotations "R1,R2" \
    --quant-iters 0

# Combo 3: R1+R2+R3+R4, R2+R4 random
run_test "random_R2R4_rtn" \
    --rotations "R1+R2+R3+R4" \
    --schemes "W4A16" \
    --random-rotations "R2,R4" \
    --quant-iters 0

# Combo 4: R1+R2+R3+R4, all random
run_test "random_all_rtn" \
    --rotations "R1+R2+R3+R4" \
    --schemes "W4A16" \
    --random-rotations "R1,R2,R3,R4" \
    --quant-iters 0

else
echo "  Skipping Part 3 (not in --parts $PARTS)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Part 4: Random combinations — Tuning (iters=200)
# ═══════════════════════════════════════════════════════════════════════════════
if [[ -n "${RUN_PART[4]:-}" ]]; then
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Part 4: Random rotation combos × Tuning (iters=200)            ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

# Combo 5: R1+R2, only R2 random, with tuning
run_test "random_R2_iters200" \
    --rotations "R1+R2" \
    --schemes "W4A16,MXFP4" \
    --random-rotations "R2" \
    --quant-iters 200

# Combo 6: R1+R2+R3+R4, all random, with tuning
run_test "random_all_iters200" \
    --rotations "R1+R2+R3+R4" \
    --schemes "W4A16" \
    --random-rotations "R1,R2,R3,R4" \
    --quant-iters 200

else
echo "  Skipping Part 4 (not in --parts $PARTS)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ALL PARTS COMPLETE"
echo "  Output directory: $OUTDIR"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Collect all results.json into one summary
echo "Collecting results..."
python3 -c "
import json, glob, os, sys

outdir = '$OUTDIR'
all_results = []
for rfile in sorted(glob.glob(os.path.join(outdir, '*/results.json'))):
    part = os.path.basename(os.path.dirname(rfile))
    with open(rfile) as f:
        data = json.load(f)
    for r in data:
        r['part'] = part
    all_results.extend(data)

# Save combined
combined = os.path.join(outdir, 'all_results.json')
with open(combined, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'Combined {len(all_results)} results → {combined}')

# Print summary table
print()
hdr = f\"\"\"{'Rotation':<20} {'Scheme':<10} {'Iters':>5} {'Random':<15} {'Disk':>8} {'Mem':>8} {'Match':>6} {'Status':<8}\"\"\"
print(hdr)
print('-' * len(hdr))
for r in all_results:
    rot = r.get('rotation', '?')
    sch = r.get('scheme', '?')
    it = r.get('quant_iters', '?')
    rand = r.get('random_set')
    if rand:
        rand_str = '+'.join(s.upper() for s in rand)
    elif r.get('rotation_mode') == 'random':
        rand_str = 'all'
    else:
        rand_str = '-'
    status = r.get('status', '?')
    match = r.get('roundtrip_match')
    match_str = '✓' if match is True else ('✗' if match is False else '-')
    disk_m = r.get('metrics_from_disk', {})
    mem_m = r.get('metrics_inmemory', {})
    # Use first task
    tasks = list(disk_m.keys()) or list(mem_m.keys()) or ['?']
    t = tasks[0]
    disk_v = f\"{disk_m.get(t, 0):.4f}\" if t in disk_m else '-'
    mem_v = f\"{mem_m.get(t, 0):.4f}\" if t in mem_m else '-'
    print(f\"{rot:<20} {sch:<10} {it:>5} {rand_str:<15} {disk_v:>8} {mem_v:>8} {match_str:>6} {status:<8}\")
print()
" 2>&1 || echo "(summary script failed, check individual results.json files)"

echo "Done! Full results in: $OUTDIR/"
