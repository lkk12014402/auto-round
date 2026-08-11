#!/usr/bin/env python
"""Orchestrate the full calibration-dataset study.

Iterates every (model, scheme, recipe) cell from ``configs/experiments.py``,
quantizes the model with that calibration recipe, then evaluates it with
lm_eval (all EVAL_GROUPS). Designed to be resumable and selectable.

Layout produced under --workdir (default: calib_experiments/runs):
    models/<exp_id>/                     quantized model + quant_meta.json
    eval/<exp_id>/<mode>/                lm_eval results + exp_meta.json
    logs/<exp_id>__{quant,eval-<mode>}.log
    run_manifest.csv                     one row per (cell, stage) with status

Examples
--------
    # Dry-run: print the plan only
    python run_experiments.py --dry-run

    # Run everything
    python run_experiments.py

    # Only MXFP4, only baseline + two code recipes
    python run_experiments.py --schemes MXFP4 --recipes pile10k opencode swe

    # Re-evaluate without re-quantizing
    python run_experiments.py --skip-quant
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "configs"))

import experiments as E  # noqa: E402

RUN_QUANTIZE = os.path.join(HERE, "run_quantize.py")
RUN_EVAL = os.path.join(HERE, "run_eval.sh")


def _sel(value, allowed):
    """True if ``value`` passes the (possibly empty) allow-list."""
    return (not allowed) or (value in allowed)


def _tee(cmd, log_path, env=None):
    """Run cmd, streaming output to console AND a log file. Returns exit code.

    Forces unbuffered child output (``PYTHONUNBUFFERED=1``) and flushes both
    sinks per line so logs appear live even when our own stdout is redirected
    to a file / running in the background.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    child_env = dict(os.environ if env is None else env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env, text=True, bufsize=1
        )
        captured = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            captured.append(line)
        proc.wait()
    return proc.returncode, "".join(captured)


def _find_results_json(path):
    """Return True if an lm_eval results json already exists under ``path``."""
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.startswith("results") and f.endswith(".json"):
                return True
    return False


def manifest_writer(workdir):
    path = os.path.join(workdir, "run_manifest.csv")
    exists = os.path.exists(path)
    fh = open(path, "a", newline="")
    writer = csv.writer(fh)
    if not exists:
        writer.writerow(
            ["timestamp", "exp_id", "model", "scheme", "recipe", "stage", "status", "elapsed_sec", "detail"]
        )
    return fh, writer


def _run_parallel(args, plan_models, logs_dir):
    """Launch one child ``run_experiments.py`` per model, in parallel.

    each child is filtered to a single model (``--models <name>``) WITHOUT
    ``--parallel``, and has ``CUDA_VISIBLE_DEVICES`` pinned to that model's
    physical GPUs. Cells of the same model still run sequentially inside its
    child. Per-model stdout/stderr is streamed to ``parallel__<model>.log``.
    """
    # Flags to forward verbatim to each child (everything except --parallel and
    # --models, which we set per child).
    passthrough = []
    if args.schemes:
        passthrough += ["--schemes"] + list(args.schemes)
    if args.recipes:
        passthrough += ["--recipes"] + list(args.recipes)
    if args.force:
        passthrough += ["--force"]
    if args.skip_quant:
        passthrough += ["--skip-quant"]
    if args.skip_eval:
        passthrough += ["--skip-eval"]
    passthrough += ["--workdir", args.workdir]

    procs = []
    for model in plan_models:
        mgpus = E.gpus_csv(model)
        log_path = os.path.join(logs_dir, f"parallel__{model['name']}.log")
        cmd = [sys.executable, os.path.abspath(__file__), "--models", model["name"]] + passthrough
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=mgpus, PYTHONUNBUFFERED="1")
        log_fh = open(log_path, "w")
        print(f"[parallel] launch {model['name']:25s} GPUs=[{mgpus}] -> {log_path}")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, text=True)
        procs.append((model["name"], proc, log_fh))

    print(f"[parallel] {len(procs)} model subprocess(es) running; waiting for completion...")
    results = {}
    for name, proc, log_fh in procs:
        proc.wait()
        log_fh.close()
        results[name] = proc.returncode
        print(f"[parallel] {name} finished rc={proc.returncode}")

    failed = {k: v for k, v in results.items() if v != 0}
    print("\n=== Parallel run summary ===")
    for name, rc in results.items():
        print(f"  {name:25s} rc={rc} {'OK' if rc == 0 else 'FAILED'}")
    if failed:
        print(f"[parallel] {len(failed)} model(s) failed: {', '.join(failed)}")
    return


def main():
    ap = argparse.ArgumentParser(description="Run the calibration-dataset study.")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--models", nargs="*", default=[], help="Filter by model name(s).")
    ap.add_argument("--schemes", nargs="*", default=[], help="Filter by scheme(s).")
    ap.add_argument("--recipes", nargs="*", default=[], help="Filter by recipe key(s).")
    ap.add_argument("--skip-quant", action="store_true", help="Do not quantize (eval only).")
    ap.add_argument("--skip-eval", action="store_true", help="Do not evaluate (quantize only).")
    ap.add_argument("--force", action="store_true", help="Redo stages even if outputs exist.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    ap.add_argument(
        "--gpu",
        default=None,
        help="GPU id for quant + eval. Precedence: --gpu > $CUDA_VISIBLE_DEVICES > config GPU.",
    )
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Run each model in its own subprocess on its configured GPUs (config MODELS[*].gpus), "
        "in parallel. Cells of the SAME model still run sequentially within that subprocess.",
    )
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    args = ap.parse_args()

    # Make our own stdout line-buffered so logs show up live under redirection
    # (e.g. `python run_experiments.py >> run.log 2>&1 &`).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Resolve GPU: explicit flag wins, else respect an exported
    # CUDA_VISIBLE_DEVICES, else fall back to the config default. This is what
    # makes `export CUDA_VISIBLE_DEVICES=2` in a wrapper script take effect.
    gpu = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") or E.GPU
    args.gpu = gpu
    print(f"[setup] using GPU (CUDA_VISIBLE_DEVICES) = {gpu}", flush=True)

    models_dir = os.path.join(args.workdir, "models")
    eval_dir = os.path.join(args.workdir, "eval")
    logs_dir = os.path.join(args.workdir, "logs")
    for d in (models_dir, eval_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    # Build the plan.
    plan = []
    for model, scheme, recipe in E.iter_experiments():
        if not _sel(model["name"], args.models):
            continue
        if not _sel(scheme, args.schemes):
            continue
        if not _sel(recipe["key"], args.recipes):
            continue
        plan.append((model, scheme, recipe))

    print(f"=== Calibration study plan: {len(plan)} cells ===")
    for model, scheme, recipe in plan:
        eid = E.exp_id(model["name"], scheme, recipe["key"])
        gpus = E.gpus_csv(model)
        print(f"  {eid:45s} gpus=[{gpus}] dataset={recipe['dataset']}")

    # Warn about GPU-assignment problems (overlaps / out-of-range).
    plan_models = []
    for model, _s, _r in plan:
        if model["name"] not in [m["name"] for m in plan_models]:
            plan_models.append(model)
    for w in E.validate_gpu_assignments(plan_models):
        print(f"[gpu-warn] {w}")

    if args.dry_run:
        print("\n(dry-run) nothing executed.")
        return

    # -------------------- Parallel: one subprocess per model --------------------
    # Each child re-invokes this script filtered to a single model, with
    # CUDA_VISIBLE_DEVICES pinned to that model's physical GPUs. Cells of the
    # same model still run sequentially inside its child.
    if args.parallel and len(plan_models) > 1:
        return _run_parallel(args, plan_models, logs_dir)

    fh, manifest = manifest_writer(args.workdir)

    def record(eid, model, scheme, recipe, stage, status, elapsed, detail=""):
        manifest.writerow(
            [time.strftime("%Y-%m-%d %H:%M:%S"), eid, model["name"], scheme, recipe["key"],
             stage, status, elapsed, detail]
        )
        fh.flush()

    for model, scheme, recipe in plan:
        eid = E.exp_id(model["name"], scheme, recipe["key"])
        out_model_dir = os.path.join(models_dir, eid)
        model_path_file = os.path.join(out_model_dir, "model_path.txt")

        # Physical GPUs this model owns (may be several for a sharded model).
        model_gpus_csv = E.gpus_csv(model) or str(args.gpu)
        model_tp = E.tp_size(model)

        # ---------------- Quantize ----------------
        model_path = None
        if not args.skip_quant:
            already = os.path.exists(model_path_file)
            if already and not args.force:
                model_path = open(model_path_file).read().strip()
                print(f"[skip quant] {eid} (exists: {model_path})")
                record(eid, model, scheme, recipe, "quantize", "skipped", 0, model_path)
            else:
                cmd = [
                    sys.executable, RUN_QUANTIZE,
                    "--model", model["path"],
                    "--scheme", scheme,
                    "--dataset", recipe["dataset"],
                    "--output-dir", out_model_dir,
                    "--nsamples", str(E.NSAMPLES),
                    "--seqlen", str(E.SEQLEN),
                    "--iters", str(E.ITERS),
                    "--seed", str(E.SEED),
                    "--device-map", str(model.get("device_map", "auto")),
                ]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=model_gpus_csv)
                print(f"\n[quantize] {eid} on GPUs [{model_gpus_csv}] device_map={model.get('device_map', 'auto')}")
                t0 = time.time()
                rc, out = _tee(cmd, os.path.join(logs_dir, f"{eid}__quant.log"), env=env)
                dt = round(time.time() - t0, 2)
                if rc != 0:
                    record(eid, model, scheme, recipe, "quantize", "failed", dt, f"rc={rc}")
                    print(f"[quantize] FAILED {eid} (rc={rc})")
                    if not args.continue_on_error:
                        break
                    continue
                for line in out.splitlines():
                    if line.startswith("MODEL_PATH="):
                        model_path = line.split("=", 1)[1].strip()
                record(eid, model, scheme, recipe, "quantize", "done", dt, model_path or "")
        else:
            if os.path.exists(model_path_file):
                model_path = open(model_path_file).read().strip()

        # ---------------- Evaluate ----------------
        if args.skip_eval:
            continue
        if not model_path or not os.path.exists(model_path):
            print(f"[eval] SKIP {eid}: no quantized model found.")
            record(eid, model, scheme, recipe, "eval", "skipped", 0, "no model")
            continue

        for group in E.EVAL_GROUPS:
            mode = group["mode"]
            group_out = os.path.join(eval_dir, eid, mode)
            os.makedirs(group_out, exist_ok=True)

            # Write meta the aggregator uses to map results -> (model,scheme,recipe).
            with open(os.path.join(group_out, "exp_meta.json"), "w") as f:
                json.dump(
                    {
                        "exp_id": eid,
                        "model": model["name"],
                        "model_path": model_path,
                        "scheme": scheme,
                        "recipe": recipe["key"],
                        "recipe_note": recipe.get("note", ""),
                        "dataset": recipe["dataset"],
                        "baseline": bool(recipe.get("baseline")),
                        "mode": mode,
                        "tasks": group["tasks"],
                        "nsamples": E.NSAMPLES,
                        "seqlen": E.SEQLEN,
                        "iters": E.ITERS,
                    },
                    f,
                    indent=2,
                )

            if _find_results_json(group_out) and not args.force:
                print(f"[skip eval] {eid}/{mode} (results exist)")
                record(eid, model, scheme, recipe, f"eval-{mode}", "skipped", 0, "")
                continue

            env = dict(
                os.environ,
                MODEL=model_path,
                OUTPUT_PATH=group_out,
                MODE=mode,
                TASKS=group["tasks"],
                GPU=model_gpus_csv,
                TP_SIZE=str(model_tp),
                BATCH_SIZE=str(group.get("batch_size", 64)),
                SEED=str(E.SEED),
                ENABLE_THINKING=str(model.get("enable_thinking", False)).lower(),
                EXTRA_ARGS=group.get("extra_args", ""),
                VLLM_EXTRA=str(model.get("vllm_extra", "")),
            )
            print(f"[eval] {eid}/{mode} on GPUs [{model_gpus_csv}] tp={model_tp}")
            t0 = time.time()
            rc, _ = _tee(
                ["bash", RUN_EVAL], os.path.join(logs_dir, f"{eid}__eval-{mode}.log"), env=env
            )
            dt = round(time.time() - t0, 2)
            status = "done" if rc == 0 else "failed"
            record(eid, model, scheme, recipe, f"eval-{mode}", status, dt, f"rc={rc}")
            if rc != 0:
                print(f"[eval] FAILED {eid}/{mode} (rc={rc})")
                if not args.continue_on_error:
                    break

    fh.close()
    print("\n=== All done. Manifest: %s ===" % os.path.join(args.workdir, "run_manifest.csv"))
    print("Aggregate with:  python scripts/aggregate_results.py --workdir %s" % args.workdir)


if __name__ == "__main__":
    main()
