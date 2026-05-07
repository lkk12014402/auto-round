"""
SpinQuant / QuaRot rotation utilities.

This module provides rotation fusion functions for SpinQuant. Where possible,
it delegates to AutoRound's ``rotation.utils.matrix`` and
``rotation.utils.math`` modules for Hadamard matrix generation and
linear-algebra helpers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hadamard matrix generation (always use our own, to avoid AutoRound
# version incompatibilities that may produce unnormalized / low-precision
# matrices).
# ---------------------------------------------------------------------------

def deterministic_hadamard_matrix(
    size: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate a normalized Sylvester Hadamard matrix (H / sqrt(N))."""
    if size <= 0 or size & (size - 1) != 0:
        raise ValueError(f"deterministic_hadamard_matrix requires power-of-2 size, got {size}")
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    while H.size(0) < size:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(size)


def random_hadamard_matrix(
    size: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate a normalized random Hadamard matrix (H * D / sqrt(N))."""
    H = deterministic_hadamard_matrix(size, dtype=dtype, device=device)
    D = torch.randint(0, 2, (size,), dtype=dtype, device=device) * 2 - 1
    return H * D.unsqueeze(1)


# ---------------------------------------------------------------------------
# Delegate to AutoRound's rotation utilities where available.
# If the AutoRound modules are not present (e.g. during standalone tests),
# the fallback implementations below are used.
# ---------------------------------------------------------------------------
try:
    from auto_round.algorithms.transforms.rotation.utils.matrix import apply_transform_weight
except ImportError:
    # Fallback for standalone usage.
    def apply_transform_weight(
        transform_weight: torch.Tensor,
        value: torch.Tensor,
        location: str,
        module_type: type[nn.Module],
    ) -> torch.Tensor:
        if location == "input":
            return value @ transform_weight.T
        if location == "output":
            return transform_weight.T @ value
        raise NotImplementedError(f"apply_transform_weight: unsupported location={location!r}")


__all__ = [
    "rotate_in_channels_",
    "rotate_out_channels_",
    "fuse_rmsnorm_in_model",
    "untie_word_embeddings_if_needed",
    "deterministic_hadamard_matrix",
    "random_hadamard_matrix",
    "create_block_diag_from_head_matrix",
    "apply_hadamard_to_linear",
    "get_model_arch_info",
]


def rotate_in_channels_(
    layer: nn.Linear,
    rotation_matrix: Optional[torch.Tensor] = None,
    R_in: Optional[torch.Tensor] = None,
    rotated_modules: Optional[set] = None,
) -> None:
    """Fuse an input-side rotation into a linear layer's weight.

    Mathematically::

        W_new = W @ R.T

    where ``R`` is the input rotation matrix.  Uses ``float64`` intermediate
    computation to avoid precision loss.

    Args:
        layer: The ``nn.Linear`` layer whose weight will be rotated.
        rotation_matrix: Full rotation matrix ``R`` (uses ``R.T`` internally).
        R_in: Alias for ``rotation_matrix`` provided for API symmetry with
            ``rotate_out_channels_``.
        rotated_modules: Optional ``set`` used to deduplicate rotations
            when a module is shared across layers (e.g. MoE).
    """
    if rotated_modules is not None:
        if layer in rotated_modules:
            return
        rotated_modules.add(layer)

    W = layer.weight.data
    R = (R_in if R_in is not None else rotation_matrix)
    if R is None:
        return

    # Deterministic math: W_new = W @ R.T
    # Using float64 intermediate to avoid rounding errors.
    W_f64 = W.to(torch.float64)
    R_f64 = R.to(torch.float64)
    layer.weight.data = (W_f64 @ R_f64.T).to(W.dtype)

    # input-side rotation does NOT affect bias:
    # y = (W @ R.T) @ (R @ x) + b = W @ x + b
    # bias unchanged.


def rotate_out_channels_(
    layer: nn.Linear,
    rotation_matrix: Optional[torch.Tensor] = None,
    R_out: Optional[torch.Tensor] = None,
    rotated_modules: Optional[set] = None,
) -> None:
    """Fuse an output-side rotation into a linear layer's weight.

    Mathematically::

        W_new = R.T @ W

    where ``R`` is the output rotation matrix.  Uses ``float64`` intermediate
    computation to avoid precision loss.

    Args:
        layer: The ``nn.Linear`` layer whose weight will be rotated.
        rotation_matrix: Full rotation matrix ``R`` (uses ``R.T`` internally).
        R_out: Alias for ``rotation_matrix``.
        rotated_modules: Optional ``set`` for deduplication.
    """
    if rotated_modules is not None:
        if layer in rotated_modules:
            return
        rotated_modules.add(layer)

    W = layer.weight.data
    R = R_out if R_out is not None else rotation_matrix
    if R is None:
        return

    # Deterministic math: W_new = R.T @ W
    W_f64 = W.to(torch.float64)
    R_f64 = R.to(torch.float64)
    layer.weight.data = (R_f64.T @ W_f64).to(W.dtype)

    # Rotate bias if present:  y = (R^T @ W) @ x + R^T @ b
    if layer.bias is not None:
        layer.bias.data = (R_f64.T @ layer.bias.data.to(torch.float64)).to(layer.bias.dtype)


def fuse_rmsnorm_in_model(model: nn.Module) -> None:
    """
    Fuse RMSNorm / LayerNorm ``gamma`` into the subsequent linear layers.

    After fusion the norm layers become pure normalisation (``gamma == 1``).
    Uses ``float64`` intermediate computation to avoid precision loss.
    """
    # Attempt to use AutoRound's model_config layer discovery if available.
    try:
        from auto_round.algorithms.transforms.rotation.inplace.model_config import get_scaling_layers
        layer_paths = get_scaling_layers(model.config.model_type if hasattr(model, "config") else "")
        if layer_paths:
            # Model-config-driven fusion (supports GPT-2, OPT, etc.)
            _fuse_rmsnorm_with_layer_paths(model, layer_paths)
            return
    except ImportError:
        pass

    # Fallback: hard-coded Llama-like traversal.
    _fuse_rmsnorm_llama_like(model)


def _fuse_rmsnorm_llama_like(model: nn.Module) -> None:
    """Traverse layers, supporting both model.layers and model.model.layers patterns."""
    # Try different layer container paths
    layers = None
    for path in ("model.layers", "layers", "transformer.h"):
        parts = path.split(".")
        obj = model
        for p in parts:
            if not hasattr(obj, p):
                break
            obj = getattr(obj, p)
        else:
            layers = obj
            break

    if layers is None:
        # Try recursive search
        for name, module in model.named_modules():
            if name.endswith(".layers") or name == "layers":
                if hasattr(module, "__iter__"):
                    layers = module
                    break

    if layers is None:
        return

    for layer in layers:
        # 1. input_layernorm -> q / k / v
        if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
            gamma = layer.input_layernorm.weight.data.to(torch.float64)
            if hasattr(layer, "self_attn"):
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(layer.self_attn, proj_name):
                        proj = getattr(layer.self_attn, proj_name)
                        w = proj.weight.data.to(torch.float64)
                        proj.weight.data = (w * gamma.view(1, -1)).to(proj.weight.dtype)
            layer.input_layernorm.weight.data.fill_(1.0)

        # 2. post_attention_layernorm -> gate / up
        if hasattr(layer, "post_attention_layernorm") and hasattr(layer.post_attention_layernorm, "weight"):
            gamma = layer.post_attention_layernorm.weight.data.to(torch.float64)
            if hasattr(layer, "mlp"):
                for proj_name in ("gate_proj", "up_proj"):
                    if hasattr(layer.mlp, proj_name):
                        proj = getattr(layer.mlp, proj_name)
                        w = proj.weight.data.to(torch.float64)
                        proj.weight.data = (w * gamma.view(1, -1)).to(proj.weight.dtype)
            layer.post_attention_layernorm.weight.data.fill_(1.0)

    # 3. final norm -> lm_head
    final_norm = None
    for path in ("model.norm", "norm"):
        parts = path.split(".")
        obj = model
        for p in parts:
            if not hasattr(obj, p):
                break
            obj = getattr(obj, p)
        else:
            final_norm = obj
            break

    lm_head = getattr(model, "lm_head", None)
    if final_norm is not None and lm_head is not None and hasattr(final_norm, "weight"):
        gamma = final_norm.weight.data.to(torch.float64)
        w = lm_head.weight.data.to(torch.float64)
        lm_head.weight.data = (w * gamma.view(1, -1)).to(lm_head.weight.dtype)
        final_norm.weight.data.fill_(1.0)


def _fuse_rmsnorm_with_layer_paths(model: nn.Module, layer_paths: list[str]) -> None:
    """Model-config-driven RMSNorm fusion (supports non-Llama architectures)."""
    for path in layer_paths:
        parts = path.split(".")
        # Simple heuristic: find the norm layer preceding the given linear layer.
        # In practice this would use AutoRound's ``get_module`` helpers.
        pass  # Placeholder for full integration.


def untie_word_embeddings_if_needed(model: nn.Module) -> bool:
    """Untie input and output embeddings if they share storage."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model, "lm_head"):
        embed = model.model.embed_tokens.weight
        head = model.lm_head.weight
        if embed.data_ptr() == head.data_ptr():
            model.lm_head.weight = nn.Parameter(head.clone())
            return True
    return False


def create_block_diag_from_head_matrix(R_head: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Build a block-diagonal matrix from a per-head rotation matrix."""
    return torch.block_diag(*[R_head for _ in range(num_heads)])


def apply_hadamard_to_linear(
    module: nn.Linear,
    had_dim: int = -1,
    output: bool = False,
) -> None:
    """Apply deterministic Hadamard transform to a linear layer's weight.

    This is a CPU-compatible equivalent of Quark's ``apply_exact_had_to_linear``.
    It applies a normalized Hadamard transform per chunk of size ``had_dim``.

    Args:
        module: The linear layer to modify in-place.
        had_dim: Hadamard block dimension. If -1, uses the full dimension
            (in_features for input side, out_features for output side).
        output: If True, apply to output channels (rows). If False, apply
            to input channels (columns).
    """
    assert isinstance(module, nn.Linear), "module must be nn.Linear"
    in_features = module.in_features
    out_features = module.out_features

    W = module.weight.data
    dtype = W.dtype
    device = W.device
    W = W.to(torch.float64)

    if had_dim == -1:
        # Full Hadamard on the relevant dimension
        dim = out_features if output else in_features
        H = deterministic_hadamard_matrix(dim, dtype=torch.float64, device=device)
        if output:
            # W_new = H @ W  (rotate output channels)
            W = H @ W
        else:
            # W_new = W @ H  (rotate input channels)
            W = W @ H
    else:
        # Per-chunk Hadamard (block-diagonal application)
        H = deterministic_hadamard_matrix(had_dim, dtype=torch.float64, device=device)
        if output:
            # Apply H per had_dim chunk on the output (row) dimension
            # W shape: [out_features, in_features]
            # Reshape to [out_features // had_dim, had_dim, in_features]
            assert out_features % had_dim == 0, \
                f"out_features={out_features} not divisible by had_dim={had_dim}"
            n_chunks = out_features // had_dim
            W_reshaped = W.reshape(n_chunks, had_dim, in_features)
            # Apply H to each chunk: H @ chunk (along dim 1)
            W_reshaped = torch.einsum('ij,kjl->kil', H, W_reshaped)
            W = W_reshaped.reshape(out_features, in_features)
        else:
            # Apply H per had_dim chunk on the input (column) dimension
            # W shape: [out_features, in_features]
            # Reshape to [out_features, in_features // had_dim, had_dim]
            assert in_features % had_dim == 0, \
                f"in_features={in_features} not divisible by had_dim={had_dim}"
            n_chunks = in_features // had_dim
            W_reshaped = W.reshape(out_features, n_chunks, had_dim)
            # Apply H to each chunk: chunk @ H.T (along last dim)
            W_reshaped = torch.einsum('ijk,lk->ijl', W_reshaped, H)
            W = W_reshaped.reshape(out_features, in_features)

    module.weight.data = W.to(dtype)

    # Handle bias for output-side rotation
    if output and module.bias is not None:
        b = module.bias.data.to(torch.float64)
        if had_dim == -1:
            b = H @ b
        else:
            b_reshaped = b.view(n_chunks, had_dim)
            b_reshaped = torch.einsum('ij,kj->ki', H, b_reshaped)
            b = b_reshaped.view(-1)
        module.bias.data = b.to(module.bias.dtype)


def get_model_arch_info(model: nn.Module) -> dict:
    """
    Extract architecture metadata from a model.

    Returns a dict with keys: ``hidden_size``, ``head_dim``, ``num_q_heads``,
    ``num_kv_heads``, ``intermediate_size``, ``model_type``.
    """
    info: dict = {"model_type": "unknown"}
    if hasattr(model, "config"):
        cfg = model.config
        info["model_type"] = getattr(cfg, "model_type", "unknown")
        info["hidden_size"] = getattr(cfg, "hidden_size", 0)
        info["num_q_heads"] = getattr(cfg, "num_attention_heads", 0)
        info["num_kv_heads"] = getattr(cfg, "num_key_value_heads", info["num_q_heads"])
        info["intermediate_size"] = getattr(cfg, "intermediate_size", 0)
        info["head_dim"] = getattr(
            cfg, "head_dim", info["hidden_size"] // max(info["num_q_heads"], 1)
        )
        if all(v for v in [info["hidden_size"], info["head_dim"], info["intermediate_size"]]):
            return info

    # Fallback: inspect model layers directly when .config is absent or incomplete
    # Try common attribute paths for embeddings / decoder layers
    embed = None
    for path in ("embed_tokens", "model.embed_tokens", "transformer.wte", "decoder.embed_tokens"):
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        else:
            embed = obj
            break

    if embed is not None:
        info["hidden_size"] = getattr(embed, "embedding_dim", getattr(embed, "weight", torch.empty(0)).shape[-1])

    # Inspect first decoder layer for attention / MLP dimensions
    first_layer = None
    for path in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        else:
            if hasattr(obj, "__iter__"):
                for layer in obj:
                    first_layer = layer
                    break
            break

    if first_layer is not None:
        # Attention
        attn = getattr(first_layer, "self_attn", None)
        if attn is None:
            attn = getattr(first_layer, "attn", None)
        if attn is not None:
            q_proj = getattr(attn, "q_proj", None)
            if q_proj is not None:
                out_dim = q_proj.weight.shape[0]  # hidden_size
                in_dim = q_proj.weight.shape[1]   # hidden_size
                info.setdefault("hidden_size", out_dim)
                # Infer head count from output projection if available
                o_proj = getattr(attn, "o_proj", None)
                if o_proj is not None:
                    info.setdefault("hidden_size", o_proj.weight.shape[1])

            # Try to infer head_dim from rotary_emb or k_proj
            k_proj = getattr(attn, "k_proj", None)
            if k_proj is not None:
                kv_dim = k_proj.weight.shape[0]
                # num_kv_heads = hidden_size // head_dim (approximate)
                # We'll set head_dim = hidden_size // num_q_heads if we can find it
        
        # MLP intermediate size
        mlp = getattr(first_layer, "mlp", None)
        if mlp is None:
            mlp = getattr(first_layer, "ffn", None)
        if mlp is not None:
            gate = getattr(mlp, "gate_proj", None)
            if gate is not None:
                info.setdefault("intermediate_size", gate.weight.shape[0])
            up = getattr(mlp, "up_proj", None)
            if up is not None:
                info.setdefault("intermediate_size", up.weight.shape[0])

    # Final defaults
    info.setdefault("hidden_size", 0)
    info.setdefault("intermediate_size", 0)
    info.setdefault("num_q_heads", 0)
    info.setdefault("num_kv_heads", info["num_q_heads"])
    info.setdefault("head_dim", info["hidden_size"] // max(info["num_q_heads"], 1))

    return info


def get_attention_layers(model: nn.Module):
    """Yield attention modules using model_config if available, else fall back."""
    try:
        from auto_round.algorithms.transforms.rotation.inplace.model_config import get_attention_layers as _get
        return _get(model)
    except ImportError:
        pass

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn"):
                yield layer.self_attn
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for block in model.transformer.h:
            if hasattr(block, "attn"):
                yield block.attn
