#!/usr/bin/env python
"""Quantize ONE model with ONE calibration recipe and save it.

This is the atomic quantization step used by ``run_experiments.py``. It can
also be run standalone for a single cell, e.g.:

    python run_quantize.py \
        --model Qwen/Qwen3-0.6B \
        --scheme MXFP4 \
        --dataset "pile-10k" \
        --output-dir /path/to/out \
        --nsamples 128 --seqlen 2048 --iters 200

On success it:
  * saves the quantized model under ``--output-dir``
  * locates the actual saved model directory (the folder containing
    ``config.json``) and writes it to ``<output-dir>/model_path.txt``
  * writes ``<output-dir>/quant_meta.json`` with all settings + wall time
  * prints ``MODEL_PATH=<path>`` on stdout for the orchestrator to capture
"""

import argparse
import json
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Work around a torch lazy circular-import bug that surfaces when auto_round's
# default torch.compile fires during block quantization:
#
#     ImportError: cannot import name '_normalize_function_or_error'
#                  from 'torch.fx.operator_schemas'
#
# `torch._subclasses.schema_check_mode` imports that symbol at module top-level,
# but the module is only lazily imported (via _disable_current_modes()) during
# dynamo compilation. If `torch.fx.operator_schemas` happens to still be mid-
# import at that moment (the symbol is defined late in the file), the import
# fails even though the symbol exists. Eagerly importing both modules here, in a
# clean order and before auto_round/dynamo run, fully resolves them up front and
# breaks the circular-import race. See analysis in the calibration study notes.
try:
    import torch.fx.operator_schemas  # noqa: F401
    import torch._subclasses.schema_check_mode  # noqa: F401
except Exception:
    # Never let the workaround itself break quantization; torch may relayout
    # these internals in future versions.
    pass


def find_saved_model_dir(output_dir: str):
    """Return the directory under ``output_dir`` that holds the saved model.

    AutoRound may nest the model in a subfolder (e.g.
    ``output_dir/Qwen3-0.6B-mxfp-w4g32/``). We locate the deepest directory
    containing a ``config.json`` and model weight shards.
    """
    candidates = []
    for root, _dirs, files in os.walk(output_dir):
        if "config.json" not in files:
            continue
        has_weights = any(
            f.endswith((".safetensors", ".bin", ".pt", ".gguf")) for f in files
        )
        # Prefer dirs that actually contain weights; fall back to any config dir.
        candidates.append((has_weights, len(root), root))
    if not candidates:
        return None
    # Prefer (has_weights=True), then the deepest path.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def main():
    ap = argparse.ArgumentParser(description="Quantize one model with one calibration recipe.")
    ap.add_argument("--model", required=True, help="HF hub id or local path.")
    ap.add_argument("--scheme", required=True, help="Quantization scheme, e.g. MXFP4 / W4A16.")
    ap.add_argument("--dataset", required=True, help="auto_round calibration dataset DSL string.")
    ap.add_argument("--output-dir", required=True, help="Directory to save the quantized model.")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device-map", default="0", help='AutoRound device_map: "0" / "auto" / json dict.')
    ap.add_argument("--format", default="llm_compressor", help="Export format.")
    ap.add_argument("--low-gpu-mem-usage", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # device_map may be an int-like string, "auto", or a json dict.
    device_map = args.device_map
    if isinstance(device_map, str):
        if device_map.strip().startswith("{"):
            device_map = json.loads(device_map)
        elif device_map.isdigit():
            device_map = int(device_map)

    meta = {
        "model": args.model,
        "scheme": args.scheme,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "iters": args.iters,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "format": args.format,
        "output_dir": os.path.abspath(args.output_dir),
        "status": "running",
    }
    meta_path = os.path.join(args.output_dir, "quant_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    from auto_round import AutoRound

    t0 = time.time()
    try:
        ar = AutoRound(
            args.model,
            scheme=args.scheme,
            dataset=args.dataset,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            iters=args.iters,
            seed=args.seed,
            batch_size=args.batch_size,
            device_map=device_map,
            low_gpu_mem_usage=args.low_gpu_mem_usage,
        )
        ar.quantize_and_save(output_dir=args.output_dir, format=args.format)
    except Exception:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc()
        meta["elapsed_sec"] = round(time.time() - t0, 2)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        traceback.print_exc()
        sys.exit(1)

    elapsed = round(time.time() - t0, 2)
    model_dir = find_saved_model_dir(args.output_dir) or args.output_dir

    meta["status"] = "done"
    meta["elapsed_sec"] = elapsed
    meta["model_path"] = os.path.abspath(model_dir)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.output_dir, "model_path.txt"), "w") as f:
        f.write(os.path.abspath(model_dir) + "\n")

    print(f"[quantize] done in {elapsed}s")
    print(f"MODEL_PATH={os.path.abspath(model_dir)}")


if __name__ == "__main__":
    main()
