"""
SpinQuant in-place application utilities.

This module provides ``apply_spinquant_in_place`` and hook registration
that follow the same patterns used by AutoRound's
``auto_round.algorithms.transforms.rotation.inplace`` package.

R3 rotation uses the architecture-generic monkeypatch approach from QuaRot/Quark:
we replace ``apply_rotary_pos_emb`` in the attention forward's globals with a
wrapper that applies Hadamard after RoPE. This works for any HuggingFace model
(Llama, Qwen2, Qwen3, Mistral, Phi, Gemma, etc.).

R4 rotation uses a forward_pre_hook on ``down_proj`` that applies block Hadamard.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    is_pow2,
    matmul_hadU,
)

logger = logging.getLogger("auto_round.spinquant")


def register_spinquant_hooks(
    model: nn.Module,
    config: Any,
    compute_device: Optional[torch.device] = None,
    head_dim: int = 0,
    intermediate_size: int = 0,
    r4_rotation_size: int = 0,
) -> list[Any]:
    """Register online rotation hooks for SpinQuant R3 (Q/K) and R4 (MLP activation).

    R3 uses the architecture-generic monkeypatch approach: replaces
    ``apply_rotary_pos_emb`` in attention forward's globals with a wrapper
    that applies normalized Hadamard to Q and K after RoPE.

    R4 registers a forward_pre_hook on each ``down_proj`` that applies block
    Hadamard to the activation before the linear layer.

    Args:
        model: The transformer model to patch.
        config: A ``SpinQuantConfig`` instance (or any object with ``r3``,
            ``r4`` booleans).
        compute_device: Device for hook computation.
        head_dim: Per-head dimension for R3 rotation. If 0, tries config.head_dim.
        intermediate_size: MLP intermediate dimension for R4. If 0, tries config.intermediate_size.
        r4_rotation_size: Override R4 rotation size (for custom rotation_size).
            If 0, falls back to intermediate_size.

    Returns:
        A list of hook handles that can be used to remove the hooks later.
    """
    handles: list[Any] = []

    if compute_device is None:
        compute_device = next(model.parameters()).device

    # Resolve dimensions from explicit args or config fallback
    if head_dim <= 0:
        head_dim = getattr(config, "head_dim", 0)
    if intermediate_size <= 0:
        intermediate_size = getattr(config, "intermediate_size", 0)

    # ------------------------------------------------------------------
    # R3: Q/K rotation after RoPE (head_dim Hadamard)
    # Uses monkeypatch to wrap apply_rotary_pos_emb in attention forward.
    # ------------------------------------------------------------------
    if getattr(config, "r3", False) and head_dim > 0:
        # Validate head_dim is power-of-2
        if not is_pow2(head_dim):
            logger.warning(
                f"[SpinQuant] R3 requires head_dim to be a power of 2, but got head_dim={head_dim}. "
                f"Skipping R3 rotation. Model accuracy may be affected."
            )
        else:
            from auto_round.algorithms.transforms.spinquant.monkeypatch import (
                QKRotationWrapper,
                add_qk_rotation_after_rope,
            )

            r3_count = 0
            for name, module in model.named_modules():
                if name.endswith("self_attn") and hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                    try:
                        wrapper = add_qk_rotation_after_rope(
                            module,
                            rope_function_name="apply_rotary_pos_emb",
                        )
                        wrapper.set_hadamard(
                            torch.empty(0),  # matmul_hadU auto-computes
                            head_dim,
                        )
                        handles.append(("r3_monkeypatch", name, module, wrapper))
                        r3_count += 1
                    except ValueError as e:
                        if r3_count == 0:
                            # First layer failed - likely unsupported architecture
                            logger.warning(
                                f"[SpinQuant] R3 monkeypatch failed for '{name}': {e}. "
                                f"This model architecture may not support R3 rotation. Skipping R3."
                            )
                            break
                        else:
                            logger.warning(f"[SpinQuant] R3 monkeypatch failed for '{name}': {e}")

            if r3_count > 0:
                logger.info(f"[SpinQuant] R3: Applied Hadamard(head_dim={head_dim}) after RoPE on {r3_count} attention layers")

    # ------------------------------------------------------------------
    # R4: MLP activation rotation (intermediate_size Hadamard)
    # ------------------------------------------------------------------
    if getattr(config, "r4", False) and intermediate_size > 0:
        # Use r4_rotation_size if provided, otherwise fall back to intermediate_size
        r4_size = r4_rotation_size if r4_rotation_size > 0 else intermediate_size
        # Find the largest power-of-2 K that divides r4_size.
        K = 1
        while K * 2 <= r4_size and r4_size % (K * 2) == 0:
            K *= 2

        if K <= 1:
            logger.warning(
                f"[SpinQuant] R4 requires rotation size to be divisible by a power of 2 > 1, "
                f"but r4_rotation_size={r4_size} has no such factor. Skipping R4 rotation."
            )
        else:
            from auto_round.algorithms.transforms.spinquant.rotation_utils import (
                deterministic_hadamard_matrix,
                get_hadamard_K,
            )

            # Determine if we need block rotation (r4_size < intermediate_size)
            need_block_rotation = (r4_size < intermediate_size)

            if need_block_rotation:
                # Block rotation: reshape input to blocks of r4_size, apply Hadamard per block
                if is_pow2(r4_size):
                    # Each block is power-of-2: use fast butterfly on each block
                    had_K_mat, had_K_val = get_hadamard_K(r4_size)
                    had_K_mat = had_K_mat.to(device=compute_device, dtype=torch.float32)

                    def _make_r4_hook(had_mat, k_val, rot_size):
                        def hook(module, args):
                            x = args[0]
                            shape = x.shape
                            # Reshape: [..., intermediate_size] → [..., num_blocks, r4_size]
                            x = x.reshape(*shape[:-1], -1, rot_size)
                            # Apply Hadamard to last dimension of each block
                            x = matmul_hadU(x, hadamard_K=had_mat.to(x.device), K=k_val)
                            return (x.reshape(shape),) + args[1:]
                        return hook

                    hook_had_K = had_K_mat
                    hook_K = had_K_val
                else:
                    # Each block is non-power-of-2: build explicit rotation matrix
                    had_K_mat, had_K_val = get_hadamard_K(r4_size)
                    R_block = had_K_mat.to(torch.float64)
                    if R_block.shape[0] != r4_size:
                        had_1, _ = get_hadamard_K(r4_size // had_K_val)
                        R_block = torch.kron(had_K_mat.to(torch.float64), had_1.to(torch.float64))
                    R_block_f32 = R_block.float().to(compute_device)

                    def _make_r4_hook(had_mat, k_val, rot_size):
                        def hook(module, args):
                            x = args[0]
                            dtype = x.dtype
                            shape = x.shape
                            R = R_block_f32.to(x.device, dtype=x.dtype)
                            x = x.reshape(*shape[:-1], -1, rot_size)
                            x = (x @ R).reshape(shape).to(dtype)
                            return (x,) + args[1:]
                        return hook

                    hook_had_K = had_K_mat.to(device=compute_device, dtype=torch.float32)
                    hook_K = had_K_val
            else:
                # Full-dimension rotation: r4_size == intermediate_size
                had_K_mat, had_K_val = get_hadamard_K(r4_size)
                had_K_mat = had_K_mat.to(device=compute_device, dtype=torch.float32)

                if is_pow2(r4_size):
                    # Full butterfly Hadamard on entire dimension
                    def _make_r4_hook(had_mat, k_val, rot_size):
                        def hook(module, args):
                            x = args[0]
                            x = matmul_hadU(x, hadamard_K=had_mat.to(x.device), K=k_val)
                            return (x,) + args[1:]
                        return hook
                else:
                    # Block Hadamard for non-power-of-2 full dimension
                    R_block = had_K_mat.to(torch.float64)
                    if R_block.shape[0] != r4_size:
                        had_1, _ = get_hadamard_K(r4_size // had_K_val)
                        R_block = torch.kron(had_K_mat.to(torch.float64), had_1.to(torch.float64))
                    R_block_f32 = R_block.float().to(compute_device)

                    def _make_r4_hook(had_mat, k_val, rot_size):
                        def hook(module, args):
                            x = args[0]
                            dtype = x.dtype
                            shape = x.shape
                            R = R_block_f32.to(x.device, dtype=x.dtype)
                            x = x.reshape(*shape[:-1], -1, rot_size)
                            x = (x @ R).reshape(shape).to(dtype)
                            return (x,) + args[1:]
                        return hook

                hook_had_K = had_K_mat
                hook_K = had_K_val

            r4_count = 0
            for name, module in list(model.named_modules()):
                if "down_proj" in name and isinstance(module, nn.Linear):
                    hook = _make_r4_hook(hook_had_K, hook_K, r4_size)
                    handle = module.register_forward_pre_hook(hook)
                    handles.append(handle)
                    r4_count += 1

            logger.info(
                f"[SpinQuant] R4: Registered forward_pre_hook(rotation_size={r4_size}, K={K}, "
                f"block_rotation={need_block_rotation}) on {r4_count} down_proj layers"
                + (f" (r4_rotation_size={r4_size})" if r4_size != intermediate_size else "")
            )

    return handles


def remove_spinquant_hooks(handles: list[Any]) -> None:
    """Safely remove all SpinQuant hook handles and R3 monkeypatches."""
    for h in handles:
        try:
            if isinstance(h, tuple) and h[0] == "r3_monkeypatch":
                # Restore original forward method globals
                _, name, module, wrapper = h
                # The monkeypatch replaced the forward method; we need to restore
                # by removing the patched method (falls back to class method)
                if hasattr(module, "forward") and not isinstance(module.forward, type(module).forward):
                    try:
                        delattr(module, "forward")
                    except AttributeError:
                        pass
            elif isinstance(h, tuple) and h[0] == "r3_patch":
                # Legacy: restore original forward
                _, name, module = h
                if hasattr(module, "_spinquant_original_forward"):
                    module.forward = module._spinquant_original_forward
                    delattr(module, "_spinquant_original_forward")
            else:
                # Standard hook handle (R4 forward_pre_hook, etc.)
                h.remove()
        except Exception:
            pass


def apply_spinquant_in_place(
    model: nn.Module,
    config: Any,
    dataloader: Optional[Any] = None,
) -> nn.Module:
    """Apply SpinQuant rotations to a model **in-place**.

    This is the SpinQuant equivalent of AutoRound's
    ``apply_in_place`` in ``rotation.inplace.apply``.

    Steps:
        1. Fuse RMSNorm gamma into linear weights.
        2. Optionally replace RMSNorm with TrainableRMSNorm.
        3. Initialise rotation matrices.
        4. (If trainable) run training loop.
        5. Fuse offline rotations into weights.
        6. Register online hooks (R3 / R4).

    Args:
        model: The model to modify.
        config: ``SpinQuantConfig`` instance.
        dataloader: Calibration data (required when training).

    Returns:
        The modified ``model`` (same object, mutated in-place).
    """
    from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantPreprocessor

    preprocessor = SpinQuantPreprocessor(model, config)
    return preprocessor.preprocess(dataloader)

