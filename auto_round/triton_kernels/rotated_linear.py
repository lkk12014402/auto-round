# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RotatedMXFP4Linear: Drop-in replacement for hook-based rotation + MXFP4QuantLinear.

This module fuses online rotation (x @ R) and MXFP4 weight dequantization into
a single forward pass, eliminating:
  - Python hook callback overhead (major bottleneck in autoregressive generation)
  - Intermediate tensor allocation for dequantized weights
  - Multiple kernel launches

Usage:
    # Convert from hook-based MXFP4QuantLinear:
    rotated_linear = RotatedMXFP4Linear.from_mxfp4_quant_linear(
        quant_linear,
        rotation_matrix=R,  # [K, K] or None
    )

    # Or create directly:
    rotated_linear = RotatedMXFP4Linear(
        packed_weight=packed_w,   # [N, K/2] uint8
        weight_scale=scale,       # [N, K/32] uint8
        rotation_matrix=R,        # [K, K] tensor or None
        bias=bias,
    )
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm


class RotatedMXFP4Linear(nn.Module):
    """Fused rotation + MXFP4 dequant + GEMM linear layer.

    Replaces the combination of:
      1. forward_pre_hook doing ``x = x @ R`` (online rotation)
      2. MXFP4QuantLinear.forward() doing dequant + F.linear

    With a single module that uses the Triton fused dequant+GEMM kernel.

    Attributes:
        packed_weight: [out_features, in_features/2] uint8 packed FP4 weights
        weight_scale: [out_features, in_features/32] uint8 e8m0 scales
        rotation_matrix: [in_features, in_features] rotation matrix or None
        bias: Optional [out_features] bias
    """

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rotation_matrix: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 32,
        apply_act_quant: bool = False,
        fp32_precision: str = "ieee",
    ):
        super().__init__()
        self.group_size = group_size
        self.apply_act_quant = apply_act_quant
        self.fp32_precision = fp32_precision

        # Register as buffers (not parameters - no gradient needed)
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)

        if rotation_matrix is not None:
            self.register_buffer("rotation_matrix", rotation_matrix)
        else:
            self.rotation_matrix = None

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        # Derived dimensions
        self.out_features = packed_weight.shape[0]
        self.in_features = packed_weight.shape[1] * 2

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass: optional rotation + fused dequant + GEMM.

        Args:
            input: [..., in_features] activation tensor.

        Returns:
            [..., out_features] output tensor.
        """
        # Apply rotation if configured
        if self.rotation_matrix is not None:
            input = input @ self.rotation_matrix

        # Use Triton fused dequant + GEMM
        output = triton_mxfp4_gemm(
            input=input,
            packed_weight=self.packed_weight,
            weight_scale=self.weight_scale,
            bias=self.bias,
            group_size=self.group_size,
            fp32_precision=self.fp32_precision,
        )

        return output

    @classmethod
    def from_mxfp4_quant_linear(
        cls,
        quant_linear: nn.Module,
        rotation_matrix: Optional[torch.Tensor] = None,
    ) -> "RotatedMXFP4Linear":
        """Create from an existing MXFP4QuantLinear module.

        Args:
            quant_linear: An MXFP4QuantLinear instance with packed_weight and weight_scale.
            rotation_matrix: Optional rotation matrix [K, K] (e.g., from a removed hook).

        Returns:
            A new RotatedMXFP4Linear module.
        """
        # Extract packed weight and scale from the quant module
        if hasattr(quant_linear, "weight_packed"):
            packed_weight = quant_linear.weight_packed
        elif hasattr(quant_linear, "weight"):
            packed_weight = quant_linear.weight
        else:
            raise ValueError("Cannot find packed weight in quant_linear module")

        weight_scale = quant_linear.weight_scale
        bias = getattr(quant_linear, "bias", None)
        group_size = getattr(quant_linear, "group_size", 32)

        return cls(
            packed_weight=packed_weight,
            weight_scale=weight_scale,
            rotation_matrix=rotation_matrix,
            bias=bias.data if bias is not None else None,
            group_size=group_size,
        )

    def extra_repr(self) -> str:
        has_rot = self.rotation_matrix is not None
        has_bias = self.bias is not None
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, rotation={has_rot}, bias={has_bias}"
        )


def convert_model_to_rotated_mxfp4(
    model: nn.Module,
    rotation_map: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Convert all MXFP4QuantLinear modules in a model to RotatedMXFP4Linear.

    This function:
      1. Finds all MXFP4QuantLinear modules
      2. Removes associated rotation hooks
      3. Replaces with RotatedMXFP4Linear (fused Triton kernel)

    Args:
        model: Model containing MXFP4QuantLinear modules.
        rotation_map: Optional dict mapping module name -> rotation matrix.
            If not provided, attempts to extract rotation from registered hooks.

    Returns:
        Modified model with RotatedMXFP4Linear modules.
    """
    if rotation_map is None:
        rotation_map = {}

    # Collect rotation matrices from hooks before removing them
    rotation_from_hooks = _extract_rotation_from_hooks(model)
    rotation_map = {**rotation_from_hooks, **rotation_map}

    # Find and replace MXFP4QuantLinear modules
    replacements = {}
    for name, module in model.named_modules():
        if type(module).__name__ in ("MXFP4QuantLinear", "MXINT4QuantLinear"):
            rot_matrix = rotation_map.get(name, None)
            try:
                new_module = RotatedMXFP4Linear.from_mxfp4_quant_linear(
                    module, rotation_matrix=rot_matrix
                )
                replacements[name] = new_module
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to convert {name} to RotatedMXFP4Linear: {e}"
                )

    # Apply replacements
    for name, new_module in replacements.items():
        _set_module_by_name(model, name, new_module)

    # Remove any remaining rotation hooks
    _remove_rotation_hooks(model)

    return model


def _extract_rotation_from_hooks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract rotation matrices from forward_pre_hooks.

    Looks for hooks that store rotation matrices as closures or buffers.
    """
    rotations = {}

    for name, module in model.named_modules():
        # Check for rotation buffer (set by serialize.py rebuild)
        if hasattr(module, "_spinquant_rotation_matrix"):
            rotations[name] = module._spinquant_rotation_matrix
            continue

        # Check hooks for rotation closure variables
        for hook_id, hook in module._forward_pre_hooks.items():
            if hasattr(hook, "__closure__") and hook.__closure__:
                for cell in hook.__closure__:
                    try:
                        val = cell.cell_contents
                        if isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[0] == val.shape[1]:
                            # Likely a rotation matrix (square, 2D)
                            if val.shape[0] == module.in_features if hasattr(module, "in_features") else False:
                                rotations[name] = val
                                break
                    except (ValueError, AttributeError):
                        continue

    return rotations


def _remove_rotation_hooks(model: nn.Module):
    """Remove spinquant-tagged rotation hooks from all modules."""
    for module in model.modules():
        hooks_to_remove = []
        for hook_id, hook in module._forward_pre_hooks.items():
            # Check for spinquant tag
            if getattr(hook, "_spinquant_hook", False):
                hooks_to_remove.append(hook_id)
            # Also check function name for rotation-related hooks
            elif hasattr(hook, "__name__") and "rotation" in hook.__name__.lower():
                hooks_to_remove.append(hook_id)
            elif hasattr(hook, "__qualname__") and "rotation" in hook.__qualname__.lower():
                hooks_to_remove.append(hook_id)

        for hook_id in hooks_to_remove:
            del module._forward_pre_hooks[hook_id]


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a named submodule in the model."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# =============================================================================
# Lightweight patching: use Triton GEMM inside existing MXFP4QuantLinear
# =============================================================================


def patch_mxfp4_forward_triton(model: nn.Module) -> int:
    """Patch MXFP4QuantLinear.forward() to use Triton fused dequant+GEMM.

    This is the simplest integration path:
    - Keeps existing rotation hooks/patches (Hadamard R1, etc.) unchanged
    - Only replaces the expensive dequant+F.linear with triton_mxfp4_gemm
    - No module replacement, no structural changes

    Args:
        model: Model containing MXFP4QuantLinear modules.

    Returns:
        Number of modules patched.
    """
    from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm

    patched_classes = set()
    n_patched = 0

    for _, module in model.named_modules():
        cls = type(module)
        if cls.__name__ == "MXFP4QuantLinear" and cls not in patched_classes:
            _patch_mxfp4_class_triton(cls)
            patched_classes.add(cls)
        if cls.__name__ == "MXFP4QuantLinear":
            n_patched += 1

    return n_patched


def _patch_mxfp4_class_triton(cls: type) -> None:
    """Replace MXFP4QuantLinear.forward with Triton-accelerated version.

    This replaces the full forward (including spinquant rotation logic),
    so it should be called AFTER rebuild_spinquant_online().
    """
    if getattr(cls, "_triton_forward_patched", False):
        return

    from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm

    @torch.inference_mode()
    def triton_forward(self, x):
        # Apply spinquant rotation from buffers (same as serialize.py patch)
        if hasattr(self, "spinquant_r1_type"):
            from auto_round.algorithms.transforms.spinquant.serialize import (
                _apply_rotation_from_buffer,
                _R1_PREFIX,
            )
            x = _apply_rotation_from_buffer(self, x, _R1_PREFIX)

        if hasattr(self, "spinquant_r4_type"):
            from auto_round.algorithms.transforms.spinquant.serialize import (
                _apply_rotation_from_buffer,
                _R4_PREFIX,
            )
            x = _apply_rotation_from_buffer(self, x, _R4_PREFIX)

        # Input quantization if configured
        if not self.pre_dequantized_input:
            x = self.qdq_input(x)

        # Use Triton fused dequant+GEMM instead of Python dequant + F.linear
        if self.pre_dequantized:
            return torch.nn.functional.linear(x, self.weight, self.bias)

        bias = self.bias.data if self.bias is not None else None
        output = triton_mxfp4_gemm(
            x.float(),
            self.weight_packed,
            self.weight_scale,
            bias=bias,
            group_size=self.group_size,
            fp32_precision="ieee",
        )
        return output.to(x.dtype)

    cls.forward = triton_forward
    cls._triton_forward_patched = True
