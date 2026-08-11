"""Central experiment configuration for the calibration-dataset study.

Everything the orchestrator, quantizer and evaluator need is declared here so
there is a single source of truth.  Edit this file to add/remove models,
schemes or calibration recipes.

Study goal
----------
Pile-10k has licensing concerns, so we want to pick a new default calibration
set.  We compare, per (model, scheme), how different calibration datasets and
mixtures affect the accuracy of the quantized model.  ``pile-10k`` is the
baseline every recipe is compared against.

Candidate datasets (all pure-text LLM calibration sets, registered in
``auto_round/calib_dataset.py``):
  * pile-10k                             (BASELINE, current default)
  * ultrachat_200k                       (general chat, already supported)
  * IF_multi_constraints_upto5           (instruction following)
  * Nemotron-SFT-Math-v3                 (math reasoning)
  * Nemotron-SFT-OpenCode-v1             (coding)
  * Nemotron-RL-Agentic-SWE-Pivot-v1     (agentic SWE - may help recover agent
                                          ability after quantization)
"""

# ---------------------------------------------------------------------------
# Global knobs
# ---------------------------------------------------------------------------

# Total calibration samples per run. Aligned with the auto_round default
# (base.py: nsamples defaults to 128) so the calibration DATASET is the only
# variable vs the pile-10k baseline. Recipe mixtures below use ``num=`` that
# sums to this value so the ratios are exact.
NSAMPLES = 128

# Calibration sequence length. Aligned with the auto_round default (2048).
# NOTE: filter_func in auto_round drops samples shorter than SEQLEN, so short
# chat/instruct/code samples must be concatenated (``concat=true``) to build
# enough fixed-length sequences; pile documents are already long enough.
SEQLEN = 2048

# SignRound optimization iterations (0 => RTN, no tuning).
ITERS = 200

# Random seed for reproducibility (calibration shuffling + tuning).
SEED = 42

# ---------------------------------------------------------------------------
# Models under test.
#   name       : short id used in output paths (keep filesystem-safe)
#   path       : HF hub id or local path
#   device_map : passed to AutoRound (0 / "auto" / dict). Big models -> "auto".
#   enable_thinking : Qwen3-style flag forwarded to lm_eval model_args for gsm8k
# ---------------------------------------------------------------------------

# Each model owns a *disjoint* set of physical GPU ids via ``gpus`` so that
# multiple models can be quantized+evaluated in parallel (one process per model,
# see run_experiments.py --parallel). Fields:
#   gpus            : list of physical GPU ids this model runs on. len>1 => the
#                     model is sharded across cards (quant device_map="auto",
#                     vLLM tensor_parallel_size=len(gpus)).
#   device_map      : AutoRound device_map. "auto" spreads over the model's gpus
#                     (recommended for big models). A single id also works.
#   enable_thinking : Qwen3-style flag forwarded to vLLM/lm_eval for gsm8k.
#   vllm_extra      : extra comma-separated vLLM model_args appended verbatim at
#                     eval time (e.g. reasoning_parser / language_model_only for
#                     Qwen3.6). Leave "" for plain models.
#
# 8x L20 example allocation (edit to taste): a small model on 1 card, a large
# reasoning model sharded on 4 cards, etc. Keep the gpu sets disjoint.
MODELS = [
    {
        "name": "Qwen3-0.6B",
        "path": "Qwen/Qwen3-0.6B",
        "gpus": [6],
        "device_map": "auto",
        "enable_thinking": False,
        "vllm_extra": "",
    },
    # ---- Example large Qwen3.6 reasoning model sharded over 4 cards ----
    # {
    #     "name": "Qwen3.6-30B",
    #     "path": "Qwen/Qwen3.6-30B-A3B",
    #     "gpus": [1, 2, 3, 4],
    #     "device_map": "auto",
    #     "enable_thinking": False,
    #     # Qwen3.6 needs these at vLLM inference time:
    #     "vllm_extra": "reasoning_parser=qwen3,language_model_only=True",
    # },
    # ---- Another mid model on 2 cards ----
    # {
    #     "name": "Qwen3-8B",
    #     "path": "Qwen/Qwen3-8B",
    #     "gpus": [5, 6],
    #     "device_map": "auto",
    #     "enable_thinking": False,
    #     "vllm_extra": "",
    # },
]

# ---------------------------------------------------------------------------
# Quantization schemes. MXFP4 is the priority; W4A16 is secondary.
# Order matters: the orchestrator runs them top-to-bottom.
# ---------------------------------------------------------------------------

SCHEMES = [
    "MXFP4",
    "W4A16",
]

# ---------------------------------------------------------------------------
# Calibration recipes.
#   key    : short id used in output paths / report columns
#   dataset: the auto_round dataset DSL string (comma-separated sources,
#            each optionally annotated with :num= / :concat= / :split= ...)
#   note   : human description for the report
#
# The ``num=`` values in each mixture sum to NSAMPLES (512).
# ``concat=true`` is used for chat/instruct/code sets so short samples are
# packed into SEQLEN-long sequences (pile documents are long enough already).
# ---------------------------------------------------------------------------

RECIPES = [
    # ---- baseline ----
    {
        "key": "pile10k",
        "dataset": "pile-10k",
        "note": "BASELINE: current default NeelNanda/pile-10k",
        "baseline": True,
    },
    # ---- single-dataset ----
    {
        "key": "ultrachat",
        "dataset": "ultrachat_200k:concat=true",
        "note": "General chat only",
    },
    {
        "key": "if_multi",
        "dataset": "IF_multi_constraints_upto5:concat=true",
        "note": "Instruction following only",
    },
    {
        "key": "math",
        "dataset": "Nemotron-SFT-Math-v3:concat=true",
        "note": "Math reasoning only",
    },
    {
        "key": "opencode",
        "dataset": "Nemotron-SFT-OpenCode-v1:concat=true",
        "note": "Coding only",
    },
    {
        "key": "swe",
        "dataset": "Nemotron-RL-Agentic-SWE-Pivot-v1:concat=true",
        "note": "Agentic SWE only",
    },
    # ---- mixtures ----
    {
        "key": "mix_equal5",
        "dataset": (
            "ultrachat_200k:concat=true:num=26,"
            "IF_multi_constraints_upto5:concat=true:num=26,"
            "Nemotron-SFT-Math-v3:concat=true:num=26,"
            "Nemotron-SFT-OpenCode-v1:concat=true:num=25,"
            "Nemotron-RL-Agentic-SWE-Pivot-v1:concat=true:num=25"
        ),
        "note": "Equal ~20% mix of all 5 candidate datasets (sum=128)",
    },
    {
        "key": "mix_chat_if",
        "dataset": (
            "ultrachat_200k:concat=true:num=64,"
            "IF_multi_constraints_upto5:concat=true:num=64"
        ),
        "note": "General + instruction following (50/50, sum=128)",
    },
    {
        "key": "mix_code_reason",
        "dataset": (
            "Nemotron-SFT-OpenCode-v1:concat=true:num=43,"
            "Nemotron-SFT-Math-v3:concat=true:num=43,"
            "Nemotron-RL-Agentic-SWE-Pivot-v1:concat=true:num=42"
        ),
        "note": "Code + math + SWE (equal thirds, sum=128)",
    },
    {
        "key": "mix_code_heavy",
        "dataset": (
            "Nemotron-SFT-OpenCode-v1:concat=true:num=64,"
            "Nemotron-RL-Agentic-SWE-Pivot-v1:concat=true:num=32,"
            "Nemotron-SFT-Math-v3:concat=true:num=32"
        ),
        "note": "Code-heavy: OpenCode 50% + SWE 25% + Math 25% (sum=128)",
    },
    {
        "key": "mix_balanced",
        "dataset": (
            "ultrachat_200k:concat=true:num=50,"
            "IF_multi_constraints_upto5:concat=true:num=26,"
            "Nemotron-SFT-Math-v3:concat=true:num=26,"
            "Nemotron-SFT-OpenCode-v1:concat=true:num=26"
        ),
        "note": "Chat-leaning balanced mix, no SWE (sum=128)",
    },
]

# ---------------------------------------------------------------------------
# Evaluation configuration (mirrors the provided eval_hf.sh / eval_hf_gsm8k.sh)
#   mode "general": multiple-choice / likelihood tasks, no chat template
#   mode "gsm8k"  : generative, chat template + few-shot-as-multiturn
# ---------------------------------------------------------------------------

EVAL_GROUPS = [
    {
        "mode": "general",
        "tasks": "piqa,hellaswag,mmlu",
        "batch_size": 64,
        "extra_args": "",  # extra lm_eval flags
    },
    {
        "mode": "gsm8k",
        "tasks": "gsm8k",
        "batch_size": 64,
        "extra_args": "--apply_chat_template --fewshot_as_multiturn",
    },
]

# Primary metric used in the summary report per task (key found in lm_eval
# results json, matched by prefix). All metrics are still stored in the CSV.
PRIMARY_METRICS = {
    "piqa": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "mmlu": "acc,none",
    "gsm8k": "exact_match,strict-match",
}

# Fallback GPU used when a model has no explicit ``gpus`` list (single-GPU mode
# / non-parallel runs).
GPU = "0"

# Total physical GPUs available on the box (for validation only).
NUM_GPUS = 8


# ---------------------------------------------------------------------------
# GPU-allocation helpers
# ---------------------------------------------------------------------------

def model_gpus(model):
    """Return the list of physical GPU ids assigned to ``model``.

    Falls back to the global ``GPU`` (parsed as comma list) when the model has
    no explicit ``gpus`` field.
    """
    g = model.get("gpus")
    if g is None:
        return [int(x) for x in str(GPU).split(",") if x != ""]
    if isinstance(g, int):
        return [g]
    return [int(x) for x in g]


def gpus_csv(model):
    """CUDA_VISIBLE_DEVICES string for ``model`` (e.g. "1,2,3,4")."""
    return ",".join(str(x) for x in model_gpus(model))


def tp_size(model):
    """vLLM tensor_parallel_size = number of GPUs the model is sharded over."""
    return max(1, len(model_gpus(model)))


def validate_gpu_assignments(models=None):
    """Sanity-check per-model GPU assignments.

    Returns a list of human-readable warnings (overlapping cards, ids out of
    range). Overlaps are allowed but flagged, since parallel runs on the same
    card will contend for memory.
    """
    models = models if models is not None else MODELS
    warnings = []
    owner = {}
    for m in models:
        for gid in model_gpus(m):
            if gid < 0 or gid >= NUM_GPUS:
                warnings.append(
                    f"model {m['name']!r} uses GPU {gid} which is outside 0..{NUM_GPUS - 1}"
                )
            if gid in owner:
                warnings.append(
                    f"GPU {gid} assigned to both {owner[gid]!r} and {m['name']!r} "
                    f"(parallel runs will contend)"
                )
            else:
                owner[gid] = m["name"]
    return warnings


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def exp_id(model_name: str, scheme: str, recipe_key: str) -> str:
    """Filesystem-safe unique id for one experiment cell."""
    return f"{model_name}__{scheme}__{recipe_key}"


def baseline_recipe_key() -> str:
    for r in RECIPES:
        if r.get("baseline"):
            return r["key"]
    return RECIPES[0]["key"]


def iter_experiments():
    """Yield (model, scheme, recipe) for every experiment cell."""
    for model in MODELS:
        for scheme in SCHEMES:
            for recipe in RECIPES:
                yield model, scheme, recipe
