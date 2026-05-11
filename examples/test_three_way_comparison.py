"""
Three-Way Comparison: Auto-Round vs Quark vs llm-compressor
Rotation + MXFP4 Quantization Accuracy

Compares QuaRot/SpinQuant rotation (R1, R1+R2, R1+R2+R4, R1+R2+R3+R4)
combined with MXFP4 RTN quantization across three frameworks.
All use deterministic Hadamard rotation (no training).

Auto-Round uses the pipeline API: AutoRound(rotation_config=SpinQuantConfig(...))
which applies rotation automatically at Phase 4.5 before quantization.

Usage:
    # Quick test (limit=100):
    python test_three_way_comparison.py --device cuda:4 --limit 100

    # Full eval:
    python test_three_way_comparison.py --device cuda:4

    # Specific rotation levels:
    python test_three_way_comparison.py --device cuda:4 --rotations R1,R1+R2+R3+R4

    # Specific frameworks:
    python test_three_way_comparison.py --device cuda:4 --frameworks autoround,llmcompressor

    # Auto-round only (no Quark/llm-compressor dependency):
    python test_three_way_comparison.py --device cuda:4 --frameworks autoround
"""

import argparse
import gc
import logging
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Rotation level definitions ──────────────────────────────────────────────
ROTATION_LEVELS = OrderedDict([
    ("R1",          dict(r1=True, r2=False, r3=False, r4=False)),
    ("R1+R2",       dict(r1=True, r2=True,  r3=False, r4=False)),
    ("R1+R2+R4",    dict(r1=True, r2=True,  r3=False, r4=True)),
    ("R1+R2+R3+R4", dict(r1=True, r2=True,  r3=True,  r4=True)),
])


# ═════════════════════════════════════════════════════════════════════════════
# Framework-specific implementations
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, tokenizer, tasks, batch_size=8, limit=None, device="cuda:0"):
    """Common evaluation via lm_eval."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks
    results = simple_evaluate(model=lm, tasks=task_list, batch_size=batch_size,
                              limit=limit, device=device)
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc_norm,none") or task_results.get("acc,none")
        if acc is not None:
            metrics[task_name] = acc
    return metrics


def load_model(model_name, device, dtype=torch.float16):
    """Load a fresh model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model


# ── Auto-Round (pipeline API) ───────────────────────────────────────────────

def run_autoround(model_name, tokenizer, rotation_flags, device, nsamples=128,
                  seqlen=512, rotation_size=128):
    """Apply rotation + MXFP4 quantization using auto-round pipeline API.

    Uses AutoRound(rotation_config=SpinQuantConfig(...)) so rotation is
    applied automatically at Phase 4.5 before quantization.
    """
    from auto_round import AutoRound
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    rotation_config = None
    if rotation_flags:
        rotation_config = SpinQuantConfig(
            **rotation_flags,
            rotation_size=rotation_size,
            online_r1_rotation=True,
            trainable_rotation=False,
            trainable_smooth=False,
        )

    ar = AutoRound(
        model_name,
        tokenizer=tokenizer,
        rotation_config=rotation_config,
        scheme="MXFP4_RCEIL",
        iters=0,
        nsamples=nsamples,
        seqlen=seqlen,
        device_map=device,
    )
    ar.quantize()
    model = ar.model.eval().to(device)
    return model


# ── Quark ───────────────────────────────────────────────────────────────────

def _ensure_quark_path():
    """Add Quark to sys.path if not already importable."""
    try:
        import quark  # noqa: F401
    except ImportError:
        quark_path = os.environ.get("QUARK_PATH", "/data/lkk/quarot/Quark")
        if os.path.isdir(quark_path) and quark_path not in sys.path:
            sys.path.insert(0, quark_path)


def _get_quark_rotation_config(rotation_flags, rotation_size=128):
    """Build Quark RotationConfig from our standardized rotation flags."""
    from quark.torch.quantization.config.config import RotationConfig

    r1 = rotation_flags.get("r1", False)
    r2 = rotation_flags.get("r2", False)
    r3 = rotation_flags.get("r3", False)
    r4 = rotation_flags.get("r4", False)

    # Quark R3 doesn't support custom rotation_size
    effective_rotation_size = None if r3 else rotation_size

    scaling_layers = {
        "first_layer": [
            {
                "prev_modules": ["model.embed_tokens"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
            },
        ],
        "middle_layers": [
            {
                "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
                "target_modules": [
                    "model.layers.layer_id.mlp.up_proj",
                    "model.layers.layer_id.mlp.gate_proj",
                ],
            },
        ],
        "last_layer": [
            {
                "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                "norm_module": "model.norm",
                "next_modules": ["lm_head"],
                "target_modules": ["lm_head"],
            }
        ],
    }

    return RotationConfig(
        backbone="model",
        model_decoder_layers="model.layers",
        v_proj="self_attn.v_proj",
        o_proj="self_attn.o_proj",
        self_attn="self_attn",
        mlp="mlp",
        r1=r1, r2=r2, r3=r3, r4=r4,
        trainable=False,
        online_r1_rotation=True,
        rotation_size=effective_rotation_size,
        scaling_layers=scaling_layers,
    )


def run_quark(model_name, tokenizer, rotation_flags, device, nsamples=128,
              seqlen=512, rotation_size=128):
    """Apply rotation + MXFP4 quantization using Quark."""
    _ensure_quark_path()
    from quark.torch import LLMTemplate, ModelQuantizer
    from quark.torch.utils.llm import get_calib_dataloader

    model = load_model(model_name, device)
    model_type = model.config.model_type
    template = LLMTemplate.get(model_type)

    if rotation_flags:
        rot_config = _get_quark_rotation_config(rotation_flags, rotation_size)
        algo_configs = {"rotation": rot_config}
        quant_config = template.get_config(
            scheme="mxfp4",
            algorithm=["rotation"],
            layer_config={},
            algo_configs=algo_configs,
        )
    else:
        quant_config = template.get_config(
            scheme="mxfp4",
            algorithm=[],
            layer_config={},
        )

    calib_dataloader = get_calib_dataloader(
        dataset_name="pileval_for_awq_benchmark",
        tokenizer=tokenizer,
        batch_size=1,
        num_calib_data=nsamples,
        seqlen=seqlen,
        device=device,
    )

    quantizer = ModelQuantizer(quant_config)
    with torch.no_grad():
        model = quantizer.quantize_model(model, calib_dataloader)
    model = quantizer.freeze(model)
    model.eval()
    return model


# ── llm-compressor ──────────────────────────────────────────────────────────

def run_llmcompressor(model_name, tokenizer, rotation_flags, device, nsamples=128,
                      seqlen=512, rotation_size=128):
    """Apply rotation + MXFP4 quantization using llm-compressor."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform import SpinQuantModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # MXFP4 observer requires float32 scales (compressed-tensors asserts
    # scale_dtype is torch.float32).  Use .to(device) instead of device_map
    # to guarantee all tensors land on the same GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()

    recipe = []

    if rotation_flags:
        rotations = []
        if rotation_flags.get("r1"): rotations.append("R1")
        if rotation_flags.get("r2"): rotations.append("R2")
        if rotation_flags.get("r3"): rotations.append("R3")
        if rotation_flags.get("r4"): rotations.append("R4")
        recipe.append(SpinQuantModifier(
            rotations=rotations,
            transform_type="hadamard",
            transform_block_size=rotation_size,
        ))

    recipe.append(QuantizationModifier(
        targets="Linear", scheme="MXFP4", ignore=["lm_head"]
    ))

    oneshot(model=model, recipe=recipe, pipeline="datafree")
    # oneshot() may scatter tensors across GPUs; consolidate back
    model = model.to(device).eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Main comparison logic
# ═════════════════════════════════════════════════════════════════════════════

FRAMEWORK_RUNNERS = {
    "autoround": run_autoround,
    "quark": run_quark,
    "llmcompressor": run_llmcompressor,
}


def run_single(framework, model_name, tokenizer, rotation_name, rotation_flags,
               args):
    """Run a single framework × rotation combination."""
    label = f"{framework}/{rotation_name}" if rotation_name else f"{framework}/MXFP4_only"
    logger.info(f"{'='*65}")
    logger.info(f"  {label}")
    logger.info(f"{'='*65}")

    t0 = time.time()
    model = None
    try:
        runner = FRAMEWORK_RUNNERS[framework]
        model = runner(
            model_name, tokenizer, rotation_flags,
            device=args.device,
            nsamples=args.nsamples, seqlen=args.seqlen,
            rotation_size=args.rotation_size,
        )

        setup_time = time.time() - t0
        logger.info(f"  Rotation+Quantization done in {setup_time:.1f}s. Evaluating...")

        metrics = evaluate_model(
            model, tokenizer, args.tasks,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )
        for task, acc in sorted(metrics.items()):
            logger.info(f"    {task}: {acc:.4f}")

        return label, metrics

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return label, {"error": str(e)}

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_comparison_table(all_results, tasks, model_name=""):
    """Print a formatted comparison table grouped by rotation level."""
    task_list = sorted(tasks)
    frameworks = []
    for label in all_results:
        fw = label.split("/")[0]
        if fw not in frameworks:
            frameworks.append(fw)

    # Collect rotation levels
    rot_levels = []
    for label in all_results:
        rot = label.split("/", 1)[1] if "/" in label else label
        if rot not in rot_levels:
            rot_levels.append(rot)

    print("\n" + "=" * 100)
    print("THREE-WAY COMPARISON: Rotation + MXFP4 RTN Quantization")
    print(f"Model: {model_name} | Rotation: Hadamard (deterministic, no training)")
    print("=" * 100)

    for task in task_list:
        print(f"\n── {task} {'─'*(85-len(task))}")
        header = f"  {'Rotation Level':<20}"
        for fw in frameworks:
            header += f" | {fw:<15}"
        print(header)
        print("  " + "-" * (20 + len(frameworks) * 18))

        for rot in rot_levels:
            row = f"  {rot:<20}"
            for fw in frameworks:
                label = f"{fw}/{rot}"
                metrics = all_results.get(label, {})
                val = metrics.get(task)
                if val is not None:
                    row += f" | {val:<15.4f}"
                elif "error" in metrics:
                    row += f" | {'ERROR':<15}"
                else:
                    row += f" | {'N/A':<15}"
            print(row)

    # Print errors
    errors = [(k, v) for k, v in all_results.items() if isinstance(v, dict) and "error" in v]
    if errors:
        print(f"\n── ERRORS {'─'*70}")
        for label, v in errors:
            print(f"  {label}: {v['error'][:120]}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Three-Way Comparison: Auto-Round vs Quark vs llm-compressor (Rotation + MXFP4)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tasks", default="piqa,hellaswag")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--rotation-size", type=int, default=128)
    parser.add_argument(
        "--rotations", default="R1,R1+R2,R1+R2+R4,R1+R2+R3+R4",
        help="Comma-separated rotation levels"
    )
    parser.add_argument(
        "--frameworks", default="autoround,quark,llmcompressor",
        help="Comma-separated frameworks to test"
    )
    parser.add_argument("--include-baselines", action="store_true",
                        help="Include MXFP4-only (no rotation) baseline per framework")
    args = parser.parse_args()

    frameworks = [f.strip() for f in args.frameworks.split(",")]
    rotations = [r.strip() for r in args.rotations.split(",")]

    for f in frameworks:
        if f not in FRAMEWORK_RUNNERS:
            print(f"Unknown framework '{f}'. Available: {list(FRAMEWORK_RUNNERS.keys())}")
            sys.exit(1)
    for r in rotations:
        if r not in ROTATION_LEVELS:
            print(f"Unknown rotation '{r}'. Available: {list(ROTATION_LEVELS.keys())}")
            sys.exit(1)

    logger.info(f"Model:      {args.model}")
    logger.info(f"Frameworks: {frameworks}")
    logger.info(f"Rotations:  {rotations}")
    logger.info(f"Tasks:      {args.tasks}")
    logger.info(f"Limit:      {args.limit or 'full'}")
    logger.info(f"Device:     {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    all_results = OrderedDict()
    all_tasks = set()

    # Baselines: MXFP4 only (no rotation)
    if args.include_baselines:
        for fw in frameworks:
            label, metrics = run_single(fw, args.model, tokenizer, "MXFP4_only", None, args)
            all_results[label] = metrics
            if "error" not in metrics:
                all_tasks.update(metrics.keys())

    # Rotation + MXFP4 for each level × framework
    for rot_name in rotations:
        rot_flags = ROTATION_LEVELS[rot_name]
        for fw in frameworks:
            label, metrics = run_single(fw, args.model, tokenizer, rot_name, rot_flags, args)
            all_results[label] = metrics
            if "error" not in metrics:
                all_tasks.update(metrics.keys())

    print_comparison_table(all_results, all_tasks, model_name=args.model)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logdir = "logs_comparison"
    os.makedirs(logdir, exist_ok=True)
    result_file = f"{logdir}/{timestamp}_three_way_comparison.txt"
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Frameworks: {frameworks}\n")
        f.write(f"Rotations: {rotations}\n")
        f.write(f"Limit: {args.limit or 'full'}\n")
        f.write(f"Rotation size: {args.rotation_size}\n\n")
        for label, metrics in all_results.items():
            f.write(f"{label}: {metrics}\n")
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
