# Calibration Dataset Study for AutoRound Quantization

Compare how the **calibration dataset** affects the accuracy of quantized LLMs,
so a new default can replace the license-encumbered `pile-10k`.

Everything is driven by one config file: **`configs/experiments.py`**.

## What it compares

- **Baseline:** `pile-10k` (current default; every recipe is compared against it).
- **Candidate datasets** (pure-text LLM calibration sets, all registered in
  `auto_round/calib_dataset.py`):
  `ultrachat_200k`, `IF_multi_constraints_upto5`, `Nemotron-SFT-Math-v3`,
  `Nemotron-SFT-OpenCode-v1`, `Nemotron-RL-Agentic-SWE-Pivot-v1`.
- **Recipes:** each dataset alone + several ratio mixtures (see `RECIPES`).
- **Schemes:** `MXFP4` (priority) and `W4A16`.

All quantization settings are **aligned with the auto_round defaults**
(`nsamples=128`, `seqlen=2048`, `iters=200`, `seed=42`, `batch_size=8`) so the
**calibration dataset is the only variable**. Chat/instruct/code datasets use
`concat=true` so short samples are packed into `seqlen`-long sequences (pile
documents are already long enough); otherwise `filter_func` would drop samples
shorter than `seqlen` and starve the calibration set.

## Files

| File | Purpose |
|------|---------|
| `configs/experiments.py` | Single source of truth: models, schemes, recipes, eval config. **Edit this.** |
| `scripts/run_quantize.py` | Quantize ONE model with ONE recipe; saves model + `quant_meta.json`. |
| `scripts/run_eval.sh`     | Evaluate ONE model for ONE task group via `lm_eval` (mirrors your eval_hf*.sh). |
| `scripts/run_experiments.py` | Orchestrator: loops all cells, quantize → eval, resumable. |
| `scripts/aggregate_results.py` | Parse all results → `results_long.csv`, `summary.csv`, `report.md`. |

## Prerequisites

```bash
# AutoRound (the local checkout, with the 4 new datasets already registered)
cd /home/hshen/lkk/calib_feat/auto-round && pip install -e .
# Evaluation harness
pip install lm-eval
```

The new datasets are downloaded from the HuggingFace Hub on first use, so ensure
network/HF access (set `HF_TOKEN` if a dataset is gated).

## Usage

All commands run from `calib_experiments/`.

```bash
# 1. Inspect the plan (nothing runs)
python scripts/run_experiments.py --dry-run

# 2. Run the priority scheme first (quantize + evaluate every recipe)
python scripts/run_experiments.py --schemes MXFP4

# 3. Then the secondary scheme
python scripts/run_experiments.py --schemes W4A16

# 4. Aggregate into CSVs + markdown report
python scripts/aggregate_results.py
```

Useful selectors / flags for `run_experiments.py`:

```bash
--models Qwen3-0.6B            # restrict to given model name(s)
--schemes MXFP4               # restrict to given scheme(s)
--recipes pile10k opencode    # restrict to given recipe key(s)
--skip-quant                  # only (re)evaluate existing models
--skip-eval                   # only quantize
--force                       # redo stages even if outputs exist
--gpu 6                       # GPU id for quant + eval
--dry-run                     # print plan and exit
```

The orchestrator is **resumable**: a finished quantization (detected via
`model_path.txt`) or a finished eval group (results json present) is skipped
unless `--force` is given.

## Outputs (under `runs/`)

```
runs/
  models/<exp_id>/                  quantized model + quant_meta.json + model_path.txt
  eval/<exp_id>/<mode>/             lm_eval results + exp_meta.json  (mode = general|gsm8k)
  logs/<exp_id>__{quant,eval-*}.log per-step logs
  run_manifest.csv                  one row per (cell, stage): status + wall time
  report/
    results_long.csv                tidy: one row per (model,scheme,recipe,task,metric)
    summary.csv                     pivot: recipe x task primary metric + Δ vs baseline
    report.md                       readable report: Δ vs pile-10k, best recipe, quant time
```

`exp_id = <model>__<scheme>__<recipe>` (e.g. `Qwen3-0.6B__MXFP4__opencode`).

## Reading the report

`report.md` has one table per `(model, scheme)` block. For each recipe it shows
the primary metric per task, the delta vs the `pile10k` baseline in parentheses,
an average across tasks, and the quantization time. The best value per task is
**bold**, and the top-3 recipes by average are called out below each table.

Primary metrics (configurable in `PRIMARY_METRICS`):
`piqa` → `acc_norm`, `hellaswag` → `acc_norm`, `mmlu` → `acc`,
`gsm8k` → `exact_match (strict-match)`.

## Adding a model / recipe / task

- **Model:** add an entry to `MODELS` (set `device_map="auto"` for large models).
  The large-model example is present but commented out.
- **Recipe:** add to `RECIPES` with a `dataset` DSL string. For mixtures, make the
  `num=` values sum to `NSAMPLES`.
- **Task:** add to the relevant `EVAL_GROUPS` entry and (optionally) to
  `PRIMARY_METRICS` so it appears in the summary.

## Notes / caveats

- `Nemotron-SFT-OpenCode-v1` has no `train` split; its loader defaults to the
  `general` split. Override via DSL `...:split=agent_skills` if desired.
- `github-code-clean` (used by auto_round's code auto-selection) needs
  `datasets<=3.6.0`; the datasets used here are parquet-based and unaffected.
- Because we pass `dataset=` explicitly for every recipe, PR #2107's automatic
  code-dataset selection never interferes — the `pile10k` recipe is a true
  apples-to-apples baseline.
- For a fair FP reference, you can additionally evaluate the unquantized model
  and compare; that is outside this harness but easy to add as another recipe
  pointing at the original model.
