"""
SpinQuant in-place application utilities.

This module provides ``apply_spinquant_in_place`` and hook registration
that follow the same patterns used by AutoRound's
``auto_round.algorithms.transforms.rotation.inplace`` package.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

# Attempt to reuse AutoRound's hook infrastructure when available.
try:
    from auto_round.algorithms.transforms.rotation.inplace.hooks import (
        BUILTIN_ROTATION_PRESETS,
        _get_or_create_random_hadamard,
        get_or_create_random_hadamard,
        register_online_hooks,
    )
    _HAS_AUTOROUND_HOOKS = True
except ImportError:
    _HAS_AUTOROUND_HOOKS = False

    def _get_or_create_random_hadamard(size, device, **kwargs):
        from auto_round.algorithms.transforms.spinquant.rotation_utils import random_hadamard_matrix
        return random_hadamard_matrix(size, device=device)

    BUILTIN_ROTATION_PRESETS: set[str] = set()


def register_spinquant_hooks(
    model: nn.Module,
    config: Any,
    compute_device: Optional[torch.device] = None,
    head_dim: int = 0,
    intermediate_size: int = 0,
) -> list[Any]:
    """
    Register online rotation hooks for SpinQuant R3 (Q/K) and R4 (MLP activation).

    This function mirrors ``register_online_hooks`` from AutoRound's
    ``rotation.inplace.hooks`` but extends it with SpinQuant-specific
    rotations.

    Args:
        model: The transformer model to patch.
        config: A ``SpinQuantConfig`` instance (or any object with ``r3``,
            ``r4`` booleans).
        compute_device: Device for hook computation.
        head_dim: Per-head dimension for R3 rotation. If 0, tries config.head_dim.
        intermediate_size: MLP intermediate dimension for R4. If 0, tries config.intermediate_size.

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
    # ------------------------------------------------------------------
    if getattr(config, "r3", False) and head_dim > 0:
        R3_head = _get_or_create_random_hadamard(head_dim, compute_device)

        def _make_r3_hook(rotation_matrix: torch.Tensor):
            def hook(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
                # output shape: (..., num_heads, seq_len, head_dim) or similar
                orig_shape = output.shape
                num_groups = orig_shape[-1] // head_dim
                x = output.view(-1, num_groups, head_dim).to(torch.float64)
                R = rotation_matrix.to(torch.float64)
                x = torch.matmul(x, R.t()).to(output.dtype)
                return x.view(orig_shape)
            return hook

        for name, module in model.named_modules():
            if "q_proj" in name or "k_proj" in name:
                h = module.register_forward_hook(_make_r3_hook(R3_head))
                handles.append(h)

    # ------------------------------------------------------------------
    # R4: MLP activation rotation (intermediate_size Hadamard)
    # ------------------------------------------------------------------
    if getattr(config, "r4", False) and intermediate_size > 0:
        inter = intermediate_size
        # Find the largest power-of-2 K that divides intermediate_size.
        K = 1
        while K * 2 <= inter and inter % (K * 2) == 0:
            K *= 2

        if K > 1:
            from auto_round.algorithms.transforms.spinquant.rotation_utils import (
                deterministic_hadamard_matrix,
            )
            had_K = deterministic_hadamard_matrix(K, dtype=torch.float32, device=compute_device)

            def _make_r4_hook(had: torch.Tensor, K_val: int):
                def hook(module: nn.Module, input: Any) -> Any:
                    # forward_pre_hook: modifies input before the module runs
                    act = input[0] if isinstance(input, tuple) else input
                    orig_shape = act.shape
                    N = orig_shape[-1]
                    M = N // K_val
                    x = act.view(-1, M, K_val).to(torch.float64)
                    H = had.to(torch.float64)
                    x = torch.matmul(x, H.t()).to(act.dtype)
                    result = x.view(orig_shape)
                    if isinstance(input, tuple):
                        return (result,) + input[1:]
                    return result
                return hook

            for name, module in model.named_modules():
                if "down_proj" in name:
                    h = module.register_forward_pre_hook(_make_r4_hook(had_K, K))
                    handles.append(h)

    return handles


def remove_spinquant_hooks(handles: list[Any]) -> None:
    """Safely remove a list of hook handles."""
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def apply_spinquant_in_place(
    model: nn.Module,
    config: Any,
    dataloader: Optional[Any] = None,
) -> nn.Module:
    """
    Apply SpinQuant rotations to a model **in-place**.

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

