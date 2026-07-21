# Architecture-aware MXFP mixed-precision policies

`mixed_mxfp_policy` adds reusable MXFP defaults for Hugging Face language models without depending primarily on model-family-specific module-name substrings.

## Policies

- `dense_mxfp4`: selected transformer-block linear targets use `MXFP4`.
- `moe_conservative`: routed expert linears use `MXFP4`; other selected transformer-block linears use `MXFP8`; MoE routers stay in floating point; attention and linear-attention projections stay in floating point.
- `moe_balanced`: routed expert linears use `MXFP4`; other selected transformer-block linears use `MXFP8`; MoE routers stay in floating point.

If you leave `scheme="W4A16"` (the default), AutoRound automatically swaps it to the matching MXFP base scheme required by the selected policy. Passing an incompatible explicit base scheme raises an error.

## CLI

Dense:

```bash
auto_round /path/to/model \
  --mixed_mxfp_policy dense_mxfp4 \
  --format llm_compressor \
  --output_dir ./dense-mxfp4
```

MoE conservative:

```bash
auto_round /path/to/model \
  --mixed_mxfp_policy moe_conservative \
  --format llm_compressor \
  --output_dir ./moe-mxfp4-mxfp8-conservative
```

MoE balanced:

```bash
auto_round /path/to/model \
  --mixed_mxfp_policy moe_balanced \
  --format llm_compressor \
  --output_dir ./moe-mxfp4-mxfp8-balanced
```

## API

```python
from auto_round import AutoRound

ar = AutoRound(
    "/path/to/model",
    mixed_mxfp_policy="moe_conservative",
)
ar.quantize_and_save("./qmodel", format="llm_compressor")
```

## Precedence

Policy defaults are merged before `set_layer_config()` expands the final configuration.

- Explicit `layer_config` overrides policy defaults, including policy-generated FP router/attention skips.
- Explicit `ignore_layers` also overrides policy defaults.
- Existing lm-head handling, embedding handling, shape-alignment skips, selected-block filtering, and multimodal block selection still apply.

## Detection notes

- MoE expert collections are detected structurally from repeated expert submodules plus an associated router/gate sibling, rather than from exact module paths like `mlp.experts` or `self_attn`.
- Router/gate detection first uses expert-collection relationships and tensor shapes; limited class/name hints are only fallbacks.
- Conservative attention preservation uses attention-module structure and common attention metadata, with limited class/name fallbacks when structural metadata is missing.
- Dense gated MLP layers such as `gate_proj` are not treated as MoE routers unless they are structurally tied to an expert collection.

## Limitations / fallback guidance

- Unknown or custom `trust_remote_code` models may not expose enough structure for automatic detection. In that case, keep using explicit `layer_config` and/or `ignore_layers`.
- Model-free MXFP policy detection builds a lightweight meta model from config; if that fails, AutoRound falls back to the explicit scheme and user overrides only.
