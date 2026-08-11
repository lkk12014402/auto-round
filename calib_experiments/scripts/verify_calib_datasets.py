#!/usr/bin/env python
"""Verify auto_round calibration datasets & dataloader correctness.

For each requested calibration dataset (using the same auto_round DSL the study
uses, e.g. ``ultrachat_200k:concat=true``), this builds the *actual* calibration
dataloader via ``auto_round.calib_dataset.get_dataloader`` and dumps, for a few
samples:

  * input_ids  : shape / dtype / min-max id / seqlen correctness
  * attention_mask : shape / dtype / #ones (valid tokens)
  * BOS/EOS analysis : does the seq start with BOS? end with EOS? how many EOS
                       occur inside (i.e. document separators for concat packing)?
  * decoded text preview (head + tail) so you can eyeball the packing/formatting

This is a *diagnostic* tool — it downloads real datasets (streaming), so the
first run per dataset may take a while.

Examples
--------
    # Default: the 6 study datasets, Qwen tokenizer, tiny sample budget
    python scripts/verify_calib_datasets.py

    # A single dataset, more samples, longer decoded preview
    python scripts/verify_calib_datasets.py \
        --datasets "ultrachat_200k:concat=true" \
        --nsamples 8 --bs 2 --preview 2 --decode-chars 400

    # Non-concat baseline
    python scripts/verify_calib_datasets.py --datasets "pile-10k"

    # List every registered dataset name and exit
    python scripts/verify_calib_datasets.py --list
"""

import argparse
import os
import sys
import traceback

import torch

# Pre-import to dodge a torch lazy circular-import bug (see run_quantize.py).
try:
    import torch.fx.operator_schemas  # noqa: F401
    import torch._subclasses.schema_check_mode  # noqa: F401
except Exception:
    pass

from transformers import AutoTokenizer

import auto_round.calib_dataset as cd


# The datasets the calibration study actually uses (short DSL forms).
DEFAULT_DATASETS = [
    "pile-10k",                                   # baseline, no concat
    "ultrachat_200k:concat=true",
    "IF_multi_constraints_upto5:concat=true",
    "Nemotron-SFT-Math-v3:concat=true",
    "Nemotron-SFT-OpenCode-v1:concat=true",
    "Nemotron-RL-Agentic-SWE-Pivot-v1:concat=true",
]

SEP = "=" * 88
SUB = "-" * 88


def list_registered():
    print("Registered calibration datasets (CALIB_DATASETS keys):")
    for k in cd.CALIB_DATASETS.keys():
        print(f"  {k}")


def analyze_sequence(ids, attn, tokenizer, seqlen):
    """Return a dict of per-sequence diagnostics."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    ids_list = ids.tolist()
    n = len(ids_list)
    eos_count = ids_list.count(eos_id) if eos_id is not None else 0
    bos_count = ids_list.count(bos_id) if bos_id is not None else 0
    valid = int(attn.sum().item())
    return {
        "len": n,
        "len_ok": n == seqlen,
        "dtype_ids": str(ids.dtype),
        "dtype_attn": str(attn.dtype),
        "min_id": int(ids.min().item()),
        "max_id": int(ids.max().item()),
        "vocab_ok": int(ids.max().item()) < tokenizer.vocab_size,
        "attn_valid": valid,
        "attn_all_ones": valid == n,
        "starts_with_bos": (bos_id is not None and ids_list[0] == bos_id),
        "ends_with_eos": (eos_id is not None and ids_list[-1] == eos_id),
        "eos_count": eos_count,
        "bos_count": bos_count,
        "pad_count": (ids_list.count(pad_id) if pad_id is not None else 0),
    }


def dump_dataset(dsl, tokenizer, seqlen, nsamples, bs, preview, decode_chars):
    print(SEP)
    print(f"DATASET DSL : {dsl}")
    print(f"tokenizer   : {tokenizer.name_or_path}  "
          f"(bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
          f"pad={tokenizer.pad_token_id}, vocab={tokenizer.vocab_size})")
    print(f"seqlen={seqlen}  nsamples={nsamples}  bs={bs}")
    print(SUB)

    try:
        dl = cd.get_dataloader(tokenizer, seqlen, dataset_name=dsl,
                               seed=42, bs=bs, nsamples=nsamples)
    except SystemExit as e:
        print(f"[FAIL] get_dataloader called sys.exit({e.code}) — likely a "
              f"network/dataset load error.")
        return
    except Exception:
        print("[FAIL] get_dataloader raised:")
        traceback.print_exc()
        return

    total_seqs = 0
    n_batches = 0
    all_len_ok = True
    all_vocab_ok = True
    dtypes_ids = set()
    dtypes_attn = set()
    eos_inside_total = 0
    shown = 0

    for bi, batch in enumerate(dl):
        if batch is None:
            print(f"[batch {bi}] collate returned None (all samples filtered out)")
            continue
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        n_batches += 1
        total_seqs += ids.shape[0]
        print(f"[batch {bi}] input_ids={tuple(ids.shape)} {ids.dtype}   "
              f"attention_mask={tuple(attn.shape)} {attn.dtype}")

        for si in range(ids.shape[0]):
            info = analyze_sequence(ids[si], attn[si], tokenizer, seqlen)
            dtypes_ids.add(info["dtype_ids"])
            dtypes_attn.add(info["dtype_attn"])
            all_len_ok &= info["len_ok"]
            all_vocab_ok &= info["vocab_ok"]
            # EOS occurring anywhere except the final position = internal separators.
            inside = info["eos_count"] - (1 if info["ends_with_eos"] else 0)
            eos_inside_total += max(inside, 0)

            if shown < preview:
                shown += 1
                print(SUB)
                print(f"  sample #{shown}  "
                      f"len={info['len']} (ok={info['len_ok']})  "
                      f"ids.dtype={info['dtype_ids']}  attn.dtype={info['dtype_attn']}")
                print(f"    token id range: [{info['min_id']}, {info['max_id']}]  "
                      f"vocab_ok={info['vocab_ok']}")
                print(f"    attention: valid={info['attn_valid']}/{info['len']}  "
                      f"all_ones={info['attn_all_ones']}  pad_count={info['pad_count']}")
                print(f"    BOS: starts_with_bos={info['starts_with_bos']}  "
                      f"bos_count={info['bos_count']}")
                print(f"    EOS: ends_with_eos={info['ends_with_eos']}  "
                      f"eos_count={info['eos_count']}  "
                      f"eos_inside(separators)={max(info['eos_count'] - (1 if info['ends_with_eos'] else 0), 0)}")
                head_ids = ids[si][:24].tolist()
                tail_ids = ids[si][-24:].tolist()
                print(f"    first 24 ids: {head_ids}")
                print(f"    last  24 ids: {tail_ids}")
                if decode_chars > 0:
                    txt = tokenizer.decode(ids[si].tolist(), skip_special_tokens=False)
                    head = txt[:decode_chars].replace("\n", "\\n")
                    tail = txt[-decode_chars:].replace("\n", "\\n")
                    print(f"    decoded[head {decode_chars}c]: {head}")
                    print(f"    decoded[tail {decode_chars}c]: {tail}")

    print(SUB)
    print(f"SUMMARY [{dsl}]")
    print(f"  batches={n_batches}  total_sequences={total_seqs}")
    print(f"  ids dtypes seen : {sorted(dtypes_ids)}")
    print(f"  attn dtypes seen: {sorted(dtypes_attn)}")
    print(f"  all seq len == {seqlen} : {all_len_ok}")
    print(f"  all token ids < vocab   : {all_vocab_ok}")
    print(f"  total EOS-as-separator inside sequences: {eos_inside_total}  "
          f"(0 => documents concatenated with NO EOS boundary)")
    if total_seqs == 0:
        print("  [WARN] produced 0 calibration sequences — dataset empty or all filtered.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                    help="HF tokenizer id or path (default Qwen/Qwen3-0.6B).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="auto_round dataset DSL string(s). Default = the 6 study datasets.")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=4,
                    help="Calibration samples to request (kept small for speed).")
    ap.add_argument("--bs", type=int, default=2, help="Dataloader batch size.")
    ap.add_argument("--preview", type=int, default=2,
                    help="How many sequences to dump in detail per dataset.")
    ap.add_argument("--decode-chars", type=int, default=300,
                    help="Chars of decoded head/tail to print (0 to disable decoding).")
    ap.add_argument("--list", action="store_true",
                    help="List every registered dataset name and exit.")
    args = ap.parse_args()

    if args.list:
        list_registered()
        return

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    print(f"Will verify {len(datasets)} dataset(s): {datasets}\n")

    for dsl in datasets:
        dump_dataset(dsl, tokenizer, args.seqlen, args.nsamples, args.bs,
                     args.preview, args.decode_chars)

    print(SEP)
    print("Done. Review the SUMMARY blocks above for correctness "
          "(seqlen, dtype, vocab range, BOS/EOS, separators).")


if __name__ == "__main__":
    main()
