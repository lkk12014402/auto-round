#!/usr/bin/env python
"""Aggregate lm_eval results across the calibration-dataset study.

Scans ``<workdir>/eval/<exp_id>/<mode>/`` trees (each carrying an
``exp_meta.json`` written by the orchestrator plus lm_eval ``results*.json``),
extracts metrics, and produces:

    <workdir>/report/results_long.csv       tidy long table (one row per metric)
    <workdir>/report/summary.csv            pivot: recipe x task (primary metric)
    <workdir>/report/report.md              human-readable report with deltas

For every (model, scheme) block the report shows each recipe's primary metric
per task, the delta vs the pile-10k baseline, an average across tasks, the best
recipe per task, and the quantization wall-clock time.

    python aggregate_results.py --workdir calib_experiments/runs
"""

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "configs"))

import experiments as E  # noqa: E402


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def newest_results_json(group_dir):
    """Return the most recent lm_eval results json under ``group_dir``."""
    hits = glob.glob(os.path.join(group_dir, "**", "results*.json"), recursive=True)
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p))
    return hits[-1]


def pick_primary_metric(task, metrics: dict):
    """Choose the primary metric value for a task from its lm_eval metric dict.

    Tries the configured key, then a prefix match (before the comma), then the
    first non-stderr numeric metric.
    """
    want = E.PRIMARY_METRICS.get(task)
    if want and want in metrics:
        return want, metrics[want]
    if want:
        prefix = want.split(",")[0]
        for k, v in metrics.items():
            if k.split(",")[0] == prefix and isinstance(v, (int, float)):
                return k, v
    for k, v in metrics.items():
        if "stderr" in k or "alias" in k:
            continue
        if isinstance(v, (int, float)):
            return k, v
    return None, None


def quant_elapsed(workdir, exp_id):
    meta = load_json(os.path.join(workdir, "models", exp_id, "quant_meta.json"))
    if meta:
        return meta.get("elapsed_sec")
    return None


def collect(workdir):
    """Return (long_rows, cells).

    long_rows: list of dicts, one per (exp, task, metric).
    cells: dict keyed by (model, scheme, recipe) -> {task: (metric, value)},
           plus meta.
    """
    long_rows = []
    cells = {}
    eval_root = os.path.join(workdir, "eval")
    for meta_path in glob.glob(os.path.join(eval_root, "*", "*", "exp_meta.json")):
        meta = load_json(meta_path)
        if not meta:
            continue
        group_dir = os.path.dirname(meta_path)
        rjson_path = newest_results_json(group_dir)
        if not rjson_path:
            continue
        rjson = load_json(rjson_path)
        if not rjson or "results" not in rjson:
            continue

        key = (meta["model"], meta["scheme"], meta["recipe"])
        cell = cells.setdefault(
            key,
            {
                "model": meta["model"],
                "scheme": meta["scheme"],
                "recipe": meta["recipe"],
                "dataset": meta.get("dataset", ""),
                "note": meta.get("recipe_note", ""),
                "baseline": bool(meta.get("baseline")),
                "tasks": {},
                "quant_sec": quant_elapsed(workdir, meta["exp_id"]),
            },
        )

        for task, metrics in rjson["results"].items():
            if not isinstance(metrics, dict):
                continue
            # Record every numeric metric in the long table.
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)) and "stderr" not in mk and mk != "alias":
                    long_rows.append(
                        {
                            "model": meta["model"],
                            "scheme": meta["scheme"],
                            "recipe": meta["recipe"],
                            "dataset": meta.get("dataset", ""),
                            "note": meta.get("recipe_note", ""),
                            "mode": meta.get("mode", ""),
                            "task": task,
                            "metric": mk,
                            "value": mv,
                        }
                    )
            # Record the primary metric for the summary (skip mmlu subtasks).
            if task in E.PRIMARY_METRICS:
                pk, pv = pick_primary_metric(task, metrics)
                if pv is not None:
                    cell["tasks"][task] = (pk, pv)
    return long_rows, cells


def write_long_csv(long_rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["model", "scheme", "recipe", "dataset", "note", "mode", "task", "metric", "value"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(long_rows, key=lambda r: (r["model"], r["scheme"], r["task"], r["recipe"])):
            w.writerow(r)


def ordered_tasks():
    """Report task columns in the order declared by PRIMARY_METRICS."""
    return list(E.PRIMARY_METRICS.keys())


def recipe_order():
    return [r["key"] for r in E.RECIPES]


def write_summary_and_report(cells, workdir):
    report_dir = os.path.join(workdir, "report")
    os.makedirs(report_dir, exist_ok=True)
    tasks = ordered_tasks()

    # Group cells by (model, scheme).
    blocks = {}
    for (model, scheme, recipe), cell in cells.items():
        blocks.setdefault((model, scheme), {})[recipe] = cell

    # ---- summary.csv (pivot, with deltas vs baseline) ----
    summary_path = os.path.join(report_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "scheme", "recipe", "dataset"]
        for t in tasks:
            header += [t, f"{t}_delta"]
        header += ["avg", "avg_delta", "quant_sec"]
        w.writerow(header)

        for (model, scheme) in sorted(blocks):
            block = blocks[(model, scheme)]
            base_key = E.baseline_recipe_key()
            base = block.get(base_key)
            base_vals = {t: base["tasks"].get(t, (None, None))[1] for t in tasks} if base else {}
            base_avg = _avg([base_vals.get(t) for t in tasks]) if base else None

            for rk in recipe_order():
                if rk not in block:
                    continue
                cell = block[rk]
                row = [model, scheme, rk, cell["dataset"]]
                vals = []
                for t in tasks:
                    v = cell["tasks"].get(t, (None, None))[1]
                    vals.append(v)
                    bv = base_vals.get(t)
                    delta = (v - bv) if (v is not None and bv is not None) else None
                    row += [_fmt(v), _fmt(delta, signed=True)]
                avg = _avg(vals)
                avg_delta = (avg - base_avg) if (avg is not None and base_avg is not None) else None
                row += [_fmt(avg), _fmt(avg_delta, signed=True), cell.get("quant_sec")]
                w.writerow(row)

    # ---- report.md ----
    report_path = os.path.join(report_dir, "report.md")
    lines = []
    lines.append("# Calibration Dataset Study — Results\n")
    lines.append(
        f"- Samples (nsamples): **{E.NSAMPLES}**  |  seqlen: **{E.SEQLEN}**  |  "
        f"iters: **{E.ITERS}**  |  seed: **{E.SEED}**\n"
    )
    lines.append(f"- Baseline recipe: **{E.baseline_recipe_key()}** (pile-10k)\n")
    lines.append("- Metrics are the primary metric per task; Δ is vs the baseline recipe.\n")
    lines.append("- Higher is better for all reported metrics. **Bold** = best per task within a block.\n")

    for (model, scheme) in sorted(blocks):
        block = blocks[(model, scheme)]
        lines.append(f"\n## {model} — {scheme}\n")

        # best per task (for bolding)
        best = {}
        for t in tasks:
            vv = [(rk, block[rk]["tasks"].get(t, (None, None))[1]) for rk in block]
            vv = [(rk, v) for rk, v in vv if v is not None]
            if vv:
                best[t] = max(vv, key=lambda x: x[1])[0]

        base_key = E.baseline_recipe_key()
        base = block.get(base_key)
        base_vals = {t: base["tasks"].get(t, (None, None))[1] for t in tasks} if base else {}
        base_avg = _avg([base_vals.get(t) for t in tasks]) if base else None

        header = ["recipe"] + tasks + ["avg", "Δavg", "quant(s)", "note"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for rk in recipe_order():
            if rk not in block:
                continue
            cell = block[rk]
            cols = [rk + (" *(baseline)*" if cell["baseline"] else "")]
            vals = []
            for t in tasks:
                v = cell["tasks"].get(t, (None, None))[1]
                vals.append(v)
                bv = base_vals.get(t)
                cell_txt = _fmt(v)
                if v is not None and best.get(t) == rk:
                    cell_txt = f"**{cell_txt}**"
                if v is not None and bv is not None and rk != base_key:
                    cell_txt += f" ({_fmt(v - bv, signed=True)})"
                cols.append(cell_txt)
            avg = _avg(vals)
            avg_txt = _fmt(avg)
            if avg is not None and base_avg is not None and rk != base_key:
                avg_txt += f" ({_fmt(avg - base_avg, signed=True)})"
            cols.append(avg_txt)
            cols.append(_fmt(avg - base_avg, signed=True) if (avg is not None and base_avg is not None) else "-")
            cols.append(str(cell.get("quant_sec") or "-"))
            cols.append(cell.get("note", ""))
            lines.append("| " + " | ".join(cols) + " |")

        # Best-recipe callout
        ranked = []
        for rk in block:
            avg = _avg([block[rk]["tasks"].get(t, (None, None))[1] for t in tasks])
            if avg is not None:
                ranked.append((rk, avg))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked:
            top = ", ".join(f"{rk} ({avg:.4f})" for rk, avg in ranked[:3])
            lines.append(f"\n**Top recipes by avg:** {top}\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return summary_path, report_path


def _avg(values):
    vv = [v for v in values if isinstance(v, (int, float))]
    return sum(vv) / len(vv) if vv else None


def _fmt(v, signed=False):
    if v is None:
        return "-"
    if signed:
        return f"{v:+.4f}"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description="Aggregate calibration-study results.")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "runs"))
    args = ap.parse_args()

    long_rows, cells = collect(args.workdir)
    if not cells:
        print(f"No results found under {os.path.join(args.workdir, 'eval')}. "
              "Run experiments first.")
        return

    long_path = os.path.join(args.workdir, "report", "results_long.csv")
    write_long_csv(long_rows, long_path)
    summary_path, report_path = write_summary_and_report(cells, args.workdir)

    print("Wrote:")
    print(f"  {long_path}   ({len(long_rows)} metric rows)")
    print(f"  {summary_path}")
    print(f"  {report_path}")
    print("\n--- report.md preview ---\n")
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
