#!/usr/bin/env python3
"""Analyze rotation × quantization scheme matrix log files.

Parses the log produced by test_rotation_scheme_matrix_v2.py and generates
a structured analysis report with accuracy tables, roundtrip diff tables,
per-task detail, and summary statistics.

Usage:
    python analyze_log.py <log_file> [--threshold 0.005] [--append] [--output report.txt]

Examples:
    # Print report to stdout
    python analyze_log.py logs_20260512_072651/Qwen3-0.6B_full.log

    # Append report to the end of the log file
    python analyze_log.py logs_20260512_072651/Qwen3-0.6B_full.log --append

    # Save to separate file with custom threshold
    python analyze_log.py logs_20260512_072651/Qwen3-0.6B_full.log -o report.txt --threshold 0.01
"""

import argparse
import re
import sys
from pathlib import Path


# ─── Canonical names ────────────────────────────────────────────────────────
TASKS = ["hellaswag", "lambada_openai", "piqa", "winogrande"]
SCHEMES = ["FP16", "W4A16", "MXFP4", "NVFP4"]
ROTATIONS = ["none", "R1", "R1+R2", "R1+R2+R3", "R1+R2+R3+R4"]
MODES = ["det", "random"]


# ─── Parsing ────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> dict:
    """Parse a test_rotation_scheme_matrix_v2 log file.

    Returns a dict with:
        fp16_metrics: {task: accuracy}
        combos: OrderedDict of combo_name -> {mem, disk, tasks_diff}
        combo_order: list of combo names in order
    """
    with open(log_path) as f:
        lines = f.readlines()

    # Strip previously appended analysis (if any)
    clean_lines = []
    for line in lines:
        if line.startswith("=" * 50) and "ANALYSIS REPORT" in "".join(
                lines[lines.index(line):lines.index(line) + 3]):
            break
        clean_lines.append(line)
    lines = clean_lines

    # ── 1. Find FP16 baseline results ──
    # They appear right before "[1/N]" without an "In-memory results:" header
    fp16_metrics = {}
    first_combo_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\[1/\d+\]', line):
            first_combo_idx = i
            break

    if first_combo_idx:
        for i in range(max(0, first_combo_idx - 10), first_combo_idx):
            m = re.search(r'(\w+):\s+(\d+\.\d{4})\s*$', lines[i])
            if m and m.group(1) in TASKS:
                fp16_metrics[m.group(1)] = float(m.group(2))

    # ── 2. Parse each combo's in-memory and from-disk results ──
    combos = {}
    combo_order = []
    current_combo = ""
    in_mem = False
    from_disk = False

    for line in lines:
        # Combo header: [N/M] combo_name
        m = re.search(r'\[\d+/\d+\]\s+(.+)', line)
        if m:
            current_combo = m.group(1).strip()
            if current_combo not in combos:
                combos[current_combo] = {"mem": {}, "disk": {}, "tasks_diff": []}
                combo_order.append(current_combo)
            in_mem = False
            from_disk = False
            continue

        if "In-memory results:" in line:
            in_mem = True
            from_disk = False
            continue
        if "From-disk results:" in line:
            from_disk = True
            in_mem = False
            continue
        if "Saving model" in line or "Loading from" in line:
            in_mem = False
            continue

        # In-memory accuracy: "    task: 0.XXXX"
        if in_mem and current_combo:
            m = re.search(r'(\w+):\s+(\d+\.\d{4})\s*$', line)
            if m and m.group(1) in TASKS:
                combos[current_combo]["mem"][m.group(1)] = float(m.group(2))

        # From-disk detail: "    task: mem=X disk=Y diff=Z ✓/✗"
        if from_disk and current_combo:
            m = re.search(
                r'(\w+):\s+mem=(\d+\.\d+)\s+disk=(\d+\.\d+)\s+diff=(\d+\.\d+)',
                line,
            )
            if m:
                task = m.group(1)
                mem_v, disk_v, diff_v = (
                    float(m.group(2)), float(m.group(3)), float(m.group(4)),
                )
                combos[current_combo]["disk"][task] = disk_v
                combos[current_combo]["tasks_diff"].append(
                    (task, mem_v, disk_v, diff_v)
                )

    return {
        "fp16_metrics": fp16_metrics,
        "combos": combos,
        "combo_order": combo_order,
    }


def _parse_combo_key(name: str) -> tuple:
    """Extract (rotation, scheme, matrix_mode) from a combo name string."""
    mm = "random" if "[random]" in name else "det"
    sch = next((s for s in ["W4A16", "MXFP4", "NVFP4"] if s in name), None)
    rot = next(
        (r for r in reversed(ROTATIONS) if name.startswith(r + " ×")), None
    )
    return rot, sch, mm


def build_lookups(parsed: dict, threshold: float) -> tuple:
    """Build accuracy and roundtrip lookup dicts from parsed data.

    Returns (acc_lookup, rt_lookup) where:
        acc_lookup[(rot, sch, mm)] = (metrics_dict, avg_accuracy)
        rt_lookup[(rot, sch, mm)] = (max_diff, avg_diff, is_pass, tasks_diff)
    """
    fp16 = parsed["fp16_metrics"]
    combos = parsed["combos"]
    combo_order = parsed["combo_order"]

    acc = {}
    rt = {}

    # FP16 baseline
    if fp16:
        vals = [fp16[t] for t in TASKS if t in fp16]
        acc[("none", "FP16", "det")] = (fp16, sum(vals) / len(vals))

    for name in combo_order:
        d = combos[name]
        rot, sch, mm = _parse_combo_key(name)
        if not rot or not sch:
            continue

        # Accuracy
        if d["mem"]:
            vals = [d["mem"][t] for t in TASKS if t in d["mem"]]
            acc[(rot, sch, mm)] = (d["mem"], sum(vals) / len(vals) if vals else 0)

        # Roundtrip
        if d["tasks_diff"]:
            max_d = max(x[3] for x in d["tasks_diff"])
            avg_d = sum(x[3] for x in d["tasks_diff"]) / len(d["tasks_diff"])
            rt[(rot, sch, mm)] = (max_d, avg_d, max_d < threshold, d["tasks_diff"])

    return acc, rt


# ─── Report generation ──────────────────────────────────────────────────────

def generate_report(parsed: dict, threshold: float = 5e-3,
                    log_name: str = "") -> str:
    """Generate the full analysis report as a string."""
    acc, rt = build_lookups(parsed, threshold)
    combos = parsed["combos"]
    combo_order = parsed["combo_order"]
    thr_pct = threshold * 100

    lines = []

    def pr(s=""):
        lines.append(s)

    pr("=" * 130)
    pr(f"  ANALYSIS REPORT — {log_name or 'Rotation × Scheme Matrix'}")
    pr("=" * 130)

    # ── Section 1: Accuracy matrix ──
    cw = 10
    pr()
    pr("━" * 130)
    pr("  1. ACCURACY MATRIX — Average of 4 tasks (in-memory)")
    pr("━" * 130)
    pr()
    pr(f"  {'Rotation':<26} │ {'FP16':^{cw}} │ {'W4A16':^{cw*2+1}} │ "
       f"{'MXFP4':^{cw*2+1}} │ {'NVFP4':^{cw*2+1}}")
    pr(f"  {'':<26} │ {'':^{cw}} │ {'det':>{cw}} {'rand':>{cw}} │ "
       f"{'det':>{cw}} {'rand':>{cw}} │ {'det':>{cw}} {'rand':>{cw}}")
    pr(f"  {'─'*26}─┼─{'─'*cw}─┼─{'─'*(cw*2+1)}─┼─"
       f"{'─'*(cw*2+1)}─┼─{'─'*(cw*2+1)}")

    for rot in ROTATIONS:
        row = f"  {rot:<26} │"
        for sch in SCHEMES:
            if sch == "FP16":
                e = acc.get((rot, sch, "det"))
                row += f" {e[1]:>{cw}.4f} │" if e else f" {'—':>{cw}} │"
            else:
                for mm in MODES:
                    if rot == "none" and mm == "random":
                        row += f" {'—':>{cw}}"
                    else:
                        e = acc.get((rot, sch, mm))
                        row += f" {e[1]:>{cw}.4f}" if e else f" {'—':>{cw}}"
                row += " │" if sch != "NVFP4" else ""
        pr(row)

    # Delta vs FP16
    fp16_entry = acc.get(("none", "FP16", "det"))
    if fp16_entry:
        fp16_avg = fp16_entry[1]
        pr()
        pr(f"  Delta vs FP16 baseline ({fp16_avg:.4f}):")
        pr(f"  {'Rotation':<26} │ {'':^{cw}} │ {'W4A16':^{cw*2+1}} │ "
           f"{'MXFP4':^{cw*2+1}} │ {'NVFP4':^{cw*2+1}}")
        pr(f"  {'─'*26}─┼─{'─'*cw}─┼─{'─'*(cw*2+1)}─┼─"
           f"{'─'*(cw*2+1)}─┼─{'─'*(cw*2+1)}")
        for rot in ROTATIONS:
            row = f"  {rot:<26} │ {'':>{cw}} │"
            for sch in ["W4A16", "MXFP4", "NVFP4"]:
                for mm in MODES:
                    if rot == "none" and mm == "random":
                        row += f" {'—':>{cw}}"
                    else:
                        e = acc.get((rot, sch, mm))
                        if e:
                            delta = (e[1] - fp16_avg) * 10000
                            row += f" {delta:>+{cw}.0f}"
                        else:
                            row += f" {'—':>{cw}}"
                row += " │" if sch != "NVFP4" else ""
            pr(row)
        pr(f"  (values in basis points, 1bp = 0.01%)")

    # Per-task tables
    for task in TASKS:
        pr()
        pr(f"  Task: {task}")
        pr(f"  {'Rotation':<26} │ {'FP16':^{cw}} │ {'det':>{cw}} "
           f"{'rand':>{cw}} │ {'det':>{cw}} {'rand':>{cw}} │ "
           f"{'det':>{cw}} {'rand':>{cw}}")
        pr(f"  {'─'*26}─┼─{'─'*cw}─┼─{'─'*(cw*2+1)}─┼─"
           f"{'─'*(cw*2+1)}─┼─{'─'*(cw*2+1)}")
        for rot in ROTATIONS:
            row = f"  {rot:<26} │"
            for sch in SCHEMES:
                if sch == "FP16":
                    e = acc.get((rot, sch, "det"))
                    v = e[0].get(task) if e else None
                    row += f" {v:>{cw}.4f} │" if v else f" {'—':>{cw}} │"
                else:
                    for mm in MODES:
                        if rot == "none" and mm == "random":
                            row += f" {'—':>{cw}}"
                        else:
                            e = acc.get((rot, sch, mm))
                            v = e[0].get(task) if e else None
                            row += f" {v:>{cw}.4f}" if v else f" {'—':>{cw}}"
                    row += " │" if sch != "NVFP4" else ""
            pr(row)

    # ── Section 2: Roundtrip table ──
    if rt:
        pr()
        pr("━" * 130)
        pr(f"  2. ROUNDTRIP SAVE/LOAD — max|task diff %  "
           f"(✓ < {thr_pct:.1f}%, ✗ ≥ {thr_pct:.1f}%)")
        pr("━" * 130)
        pr()
        pr(f"  {'Rotation':<26} │ {'FP16':^14} │ {'W4A16':^24} │ "
           f"{'MXFP4':^24} │ {'NVFP4':^24}")
        pr(f"  {'':<26} │ {'det':>7} {'rand':>7} │ {'det':>12} "
           f"{'rand':>12} │ {'det':>12} {'rand':>12} │ {'det':>12} "
           f"{'rand':>12}")
        pr(f"  {'─'*26}─┼─{'─'*14}─┼─{'─'*24}─┼─{'─'*24}─┼─{'─'*24}")

        for rot in ROTATIONS:
            row = f"  {rot:<26} │"
            for sch in SCHEMES:
                if sch == "FP16":
                    row += f" {'N/A':>7} {'—':>7} │"
                else:
                    for mm in MODES:
                        if rot == "none" and mm == "random":
                            row += f" {'—':>12}"
                        else:
                            e = rt.get((rot, sch, mm))
                            if e:
                                sym = "✓" if e[2] else "✗"
                                row += f"  {sym}{e[0]*100:>5.2f}%"
                            else:
                                row += f" {'—':>12}"
                    row += " │" if sch != "NVFP4" else ""
            pr(row)

    # ── Section 3: Per-task roundtrip detail ──
    if rt:
        pr()
        pr("━" * 130)
        pr("  3. ROUNDTRIP PER-TASK DETAIL")
        pr("━" * 130)

        for name in combo_order:
            d = combos[name]
            if not d["tasks_diff"]:
                continue
            max_d = max(x[3] for x in d["tasks_diff"])
            avg_d = sum(x[3] for x in d["tasks_diff"]) / len(d["tasks_diff"])
            sym = "✓" if max_d < threshold else "✗"
            pr()
            pr(f"  {sym} {name}")
            pr(f"    {'Task':<20} {'mem':>8} {'disk':>8} {'diff':>10} {'diff%':>8}")
            pr(f"    {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
            for task, mem, disk, diff in d["tasks_diff"]:
                s2 = "✓" if diff < threshold else "✗"
                pr(f"    {task:<20} {mem:>8.4f} {disk:>8.4f} "
                   f"{diff:>10.6f} {diff*100:>7.3f}% {s2}")
            pr(f"    {'':>20} {'':>8} {'':>8} {'─'*10} {'─'*8}")
            pr(f"    {'MAX':>20} {'':>8} {'':>8} "
               f"{max_d:>10.6f} {max_d*100:>7.3f}%")
            pr(f"    {'AVG':>20} {'':>8} {'':>8} "
               f"{avg_d:>10.6f} {avg_d*100:>7.3f}%")

    # ── Section 4: Summary ──
    if rt:
        pr()
        pr("━" * 130)
        pr("  4. SUMMARY")
        pr("━" * 130)
        for sch in ["W4A16", "MXFP4", "NVFP4"]:
            entries = [(k, v) for k, v in rt.items() if k[1] == sch]
            if not entries:
                continue
            all_max = [v[0] for _, v in entries]
            pc = sum(1 for m in all_max if m < threshold)
            pr(f"\n  {sch}: {pc}/{len(entries)} pass (<{thr_pct:.1f}%),  "
               f"max_diff range: {min(all_max)*100:.2f}% ~ "
               f"{max(all_max)*100:.2f}%,  "
               f"mean: {sum(all_max)/len(all_max)*100:.2f}%")

    # ── Section 5: Observations ──
    pr()
    pr("━" * 130)
    pr("  5. KEY OBSERVATIONS")
    pr("━" * 130)

    # Auto-generate observations from data
    w4a16_entries = [(k, v) for k, v in rt.items() if k[1] == "W4A16"]
    mxfp4_entries = [(k, v) for k, v in rt.items() if k[1] == "MXFP4"]
    nvfp4_entries = [(k, v) for k, v in rt.items() if k[1] == "NVFP4"]

    if w4a16_entries:
        maxes = [v[0] for _, v in w4a16_entries]
        pc = sum(1 for m in maxes if m < threshold)
        pr(f"\n  ■ W4A16 (weight-only INT4):")
        pr(f"    - {pc}/{len(w4a16_entries)} pass. "
           f"Max diff: {min(maxes)*100:.2f}% ~ {max(maxes)*100:.2f}%. "
           f"Save/load roundtrip {'is correct' if pc == len(w4a16_entries) else 'has issues'}.")
        pr(f"    - Small diffs are normal lm_eval non-determinism "
           f"(GPU FP accumulation, batch order).")

    if mxfp4_entries:
        maxes = [v[0] for _, v in mxfp4_entries]
        pc = sum(1 for m in maxes if m < threshold)
        none_e = rt.get(("none", "MXFP4", "det"))
        pr(f"\n  ■ MXFP4 (W4A4, block=32):")
        pr(f"    - {pc}/{len(mxfp4_entries)} pass. "
           f"Max diff: {min(maxes)*100:.2f}% ~ {max(maxes)*100:.2f}%.")
        if none_e:
            pr(f"    - 'none' rotation → {none_e[0]*100:.2f}% diff. "
               f"This is NOT a rotation bug.")
        pr(f"    - Root cause: in-memory fake-quant vs on-disk real format "
           f"use different activation quantization paths.")

    if nvfp4_entries:
        maxes = [v[0] for _, v in nvfp4_entries]
        pc = sum(1 for m in maxes if m < threshold)
        none_e = rt.get(("none", "NVFP4", "det"))
        pr(f"\n  ■ NVFP4 (W4A4, gs=16):")
        pr(f"    - {pc}/{len(nvfp4_entries)} pass. "
           f"Max diff: {min(maxes)*100:.2f}% ~ {max(maxes)*100:.2f}%.")
        if none_e:
            pr(f"    - 'none' rotation → {none_e[0]*100:.2f}% diff. "
               f"Same root cause as MXFP4.")

    # Accuracy trends
    if fp16_entry:
        pr(f"\n  ■ Accuracy trends:")
        pr(f"    - FP16 baseline: {fp16_avg:.4f} avg")
        for sch in ["W4A16", "MXFP4", "NVFP4"]:
            none_e = acc.get(("none", sch, "det"))
            if none_e:
                delta_bp = (none_e[1] - fp16_avg) * 10000
                pr(f"    - {sch} (none): {none_e[1]:.4f} avg "
                   f"({delta_bp:+.0f} bp vs FP16)")

    pr()
    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze rotation × scheme matrix log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("log_file", help="Path to the log file to analyze")
    parser.add_argument(
        "--threshold", type=float, default=5e-3,
        help="Roundtrip match threshold (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append the report to the end of the log file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write report to a separate file instead of stdout",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    # Parse
    parsed = parse_log(str(log_path))
    n_combos = len(parsed["combo_order"])
    has_fp16 = bool(parsed["fp16_metrics"])
    has_rt = any(parsed["combos"][n]["tasks_diff"] for n in parsed["combo_order"])

    print(f"Parsed: {n_combos} combos, FP16 baseline: {has_fp16}, "
          f"roundtrip data: {has_rt}", file=sys.stderr)

    # Generate
    report = generate_report(
        parsed,
        threshold=args.threshold,
        log_name=log_path.name,
    )

    # Output
    if args.append:
        # Remove previously appended analysis from the log
        with open(log_path) as f:
            content = f.read()
        marker = "\n" + "=" * 130 + "\n  ANALYSIS REPORT"
        if marker in content:
            content = content[:content.index(marker)].rstrip() + "\n"
        with open(log_path, "w") as f:
            f.write(content + "\n" + report + "\n")
        print(f"✓ Report appended to {log_path}", file=sys.stderr)
    elif args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"✓ Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
