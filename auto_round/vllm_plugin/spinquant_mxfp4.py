# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
vLLM quantization plugin for SpinQuant/QuaRot MXFP4 models.

Supports:
  - R1 online rotation (Hadamard butterfly or full matrix) on q/k/v/gate/up
  - R4 online rotation (Hadamard butterfly or full matrix) on down_proj
  - R3 online rotation (Q/K after RoPE — head_dim rotation)
  - MXFP4 packed weight dequant + GEMM (Triton fused kernel)

Weight format (from auto-round):
  - weight_packed: [N, K//2] uint8 — two E2M1 FP4 values per byte
  - weight_scale:  [N, K//32] uint8 — e8m0 shared exponents
  - spinquant_r1_type: int — 0=hadamard, 1=random, 2=trained
  - spinquant_r1_size: int — rotation dimension for R1
  - spinquant_r4_type: int — 0=hadamard, 1=random, 2=trained
  - spinquant_r4_size: int — rotation dimension for R4
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

logger = logging.getLogger(__name__)

# Rotation type constants (must match auto_round serialization)
ROTATION_TYPE_HADAMARD = 0
ROTATION_TYPE_RANDOM = 1
ROTATION_TYPE_TRAINED = 2


# =============================================================================
# Configuration
# =============================================================================


@register_quantization_config("spinquant_mxfp4")
class SpinQuantMXFP4Config(QuantizationConfig):
    """Configuration for SpinQuant/QuaRot MXFP4 quantization in vLLM."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 32,
        online_r1: bool = True,
        online_r4: bool = False,
        online_r3: bool = False,
        r1_type: str = "hadamard",
        r4_type: str = "hadamard",
        r3_type: str = "hadamard",
        rotation_size: int | None = None,
        hidden_size: int = 0,
        head_dim: int = 128,
        intermediate_size: int = 0,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.online_r1 = online_r1
        self.online_r4 = online_r4
        self.online_r3 = online_r3
        self.r1_type = r1_type
        self.r4_type = r4_type
        self.r3_type = r3_type
        self.rotation_size = rotation_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size

    def __repr__(self) -> str:
        parts = [
            f"bits={self.bits}",
            f"group_size={self.group_size}",
            f"online_r1={self.online_r1}",
            f"r1_type={self.r1_type}",
        ]
        if self.online_r4:
            parts.append(f"online_r4={self.online_r4}, r4_type={self.r4_type}")
        if self.online_r3:
            parts.append(f"online_r3={self.online_r3}, r3_type={self.r3_type}")
        return f"SpinQuantMXFP4Config({', '.join(parts)})"

    @classmethod
    def get_name(cls) -> str:
        return "spinquant_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Volta+

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SpinQuantMXFP4Config":
        """Parse from model's quantization_config."""
        sq_config = config.get("spinquant_config", {})
        return cls(
            bits=config.get("bits", 4),
            group_size=config.get("group_size", 32),
            online_r1=sq_config.get("online_r1_rotation", True),
            online_r4=sq_config.get("r4", False),
            online_r3=sq_config.get("r3", False),
            r1_type="random" if sq_config.get("random_r1", False) else "hadamard",
            r4_type="random" if sq_config.get("random_r4", False) else "hadamard",
            r3_type="random" if sq_config.get("random_r3", False) else "hadamard",
            rotation_size=sq_config.get("rotation_size", None),
            hidden_size=sq_config.get("hidden_size", 0),
            head_dim=sq_config.get("head_dim", 128),
            intermediate_size=sq_config.get("intermediate_size", 0),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
    ) -> str | None:
        """Auto-detect spinquant_mxfp4 models from their config."""
        # If user explicitly requested it
        if user_quant == "spinquant_mxfp4":
            return "spinquant_mxfp4"
        # Auto-detect: has spinquant_config with online rotation + mxfp data type
        sq = hf_quant_cfg.get("spinquant_config", {})
        if not sq or not sq.get("online_r1_rotation", False):
            return None
        data_type = hf_quant_cfg.get("data_type", "").lower()
        if "mxfp" in data_type or "mx_fp" in data_type:
            return "spinquant_mxfp4"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "SpinQuantMXFP4LinearMethod | None":
        if isinstance(layer, LinearBase):
            return SpinQuantMXFP4LinearMethod(self)
        return None


# =============================================================================
# Linear Method
# =============================================================================


class SpinQuantMXFP4LinearMethod(LinearMethodBase):
    """Linear method implementing SpinQuant R1/R4 rotation + MXFP4 weight dequant + GEMM."""

    def __init__(self, quant_config: SpinQuantMXFP4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size

        # Packed MXFP4 weight: [N, K//2] uint8
        weight_packed = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_packed,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 2,
            } | extra_weight_attrs,
        )
        layer.register_parameter("weight_packed", weight_packed)

        # Scale: [N, K//group_size] uint8 (e8m0)
        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_scale,
            {
                "input_dim": 1,
                "output_dim": 0,
            } | extra_weight_attrs,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # R1 rotation metadata (buffers, loaded from checkpoint)
        # These are scalar int32 buffers
        spinquant_r1_type = Parameter(
            torch.zeros((), dtype=torch.int32), requires_grad=False
        )
        set_weight_attrs(spinquant_r1_type, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r1_type", spinquant_r1_type)

        spinquant_r1_size = Parameter(
            torch.zeros((), dtype=torch.int32), requires_grad=False
        )
        set_weight_attrs(spinquant_r1_size, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r1_size", spinquant_r1_size)

        # R1 rotation matrix (for trained/random rotations loaded from checkpoint)
        # For Hadamard type, this won't be in the checkpoint and stays zeros (unused).
        # Shape: [rot_size, rot_size] — use config rotation_size or input_size as default.
        rot_size = self.quant_config.rotation_size or input_size_per_partition
        spinquant_r1_matrix = Parameter(
            torch.zeros(rot_size, rot_size, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            spinquant_r1_matrix, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("spinquant_r1_matrix", spinquant_r1_matrix)

        # R4 rotation metadata (for down_proj layers — MLP activation rotation)
        # Same structure as R1: type + size + optional matrix
        spinquant_r4_type = Parameter(
            torch.zeros((), dtype=torch.int32), requires_grad=False
        )
        set_weight_attrs(spinquant_r4_type, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r4_type", spinquant_r4_type)

        spinquant_r4_size = Parameter(
            torch.zeros((), dtype=torch.int32), requires_grad=False
        )
        set_weight_attrs(spinquant_r4_size, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r4_size", spinquant_r4_size)

        # R4 rotation matrix (for trained/random R4 rotations)
        r4_rot_size = self.quant_config.rotation_size or input_size_per_partition
        spinquant_r4_matrix = Parameter(
            torch.zeros(r4_rot_size, r4_rot_size, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            spinquant_r4_matrix, {"ignore_warning": True} | extra_weight_attrs
        )
        layer.register_parameter("spinquant_r4_matrix", spinquant_r4_matrix)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Pre-compute full rotation matrices for compile-friendly inference."""
        # --- R1 rotation ---
        self._process_rotation(layer, prefix="r1")
        # --- R4 rotation ---
        self._process_rotation(layer, prefix="r4")

    def _process_rotation(self, layer: torch.nn.Module, prefix: str) -> None:
        """Process rotation parameters for a given prefix (r1 or r4).

        Builds the full rotation matrix from checkpoint metadata and stores
        it as layer._<prefix>_rotation_matrix for use in apply().
        """
        type_attr = f"spinquant_{prefix}_type"
        size_attr = f"spinquant_{prefix}_size"
        matrix_attr = f"spinquant_{prefix}_matrix"
        result_attr = f"_{prefix}_rotation_matrix"
        rot_size_attr = f"_{prefix}_rot_size"

        if not hasattr(layer, type_attr):
            setattr(layer, result_attr, None)
            return

        rot_type = int(getattr(layer, type_attr).item())
        rot_size = int(getattr(layer, size_attr).item())

        if rot_size == 0:
            setattr(layer, result_attr, None)
            # Free placeholder matrix
            if hasattr(layer, matrix_attr):
                delattr(layer, matrix_attr)
            return

        device = layer.weight_packed.device

        if rot_type == ROTATION_TYPE_HADAMARD:
            R = _build_full_hadamard(rot_size, device)
        elif rot_type in (ROTATION_TYPE_RANDOM, ROTATION_TYPE_TRAINED):
            if hasattr(layer, matrix_attr):
                mat = getattr(layer, matrix_attr).data
                if mat.any():
                    if mat.shape[0] >= rot_size:
                        R = mat[:rot_size, :rot_size].to(device=device, dtype=torch.float32)
                    else:
                        logger.warning(
                            f"{matrix_attr} shape {mat.shape} < rot_size {rot_size}, "
                            f"falling back to random orthogonal matrix"
                        )
                        R = _generate_random_orthogonal(rot_size, device)
                else:
                    logger.warning(
                        f"rot_type={rot_type} (random/trained) but no matrix in checkpoint "
                        f"for {prefix}, generating random orthogonal matrix"
                    )
                    R = _generate_random_orthogonal(rot_size, device)
            else:
                R = _generate_random_orthogonal(rot_size, device)
        else:
            R = None

        if R is not None:
            setattr(layer, result_attr, R.to(torch.float16))
            setattr(layer, rot_size_attr, rot_size)
        else:
            setattr(layer, result_attr, None)

        # Free the original parameter to save memory
        if hasattr(layer, matrix_attr):
            delattr(layer, matrix_attr)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward: apply R1/R4 rotation + MXFP4 dequant + GEMM.

        R1 applies to q/k/v/gate/up projections (ColumnParallel, input not sharded).
        R4 applies to down_proj (RowParallel, input sharded by TP).
        Both are compile-friendly matmuls with pre-computed matrices.
        """
        # Step 1: Apply R1 rotation (for q/k/v/gate/up layers)
        if self.quant_config.online_r1:
            x = self._apply_rotation(layer, x, "_r1_rotation_matrix", "_r1_rot_size")

        # Step 2: Apply R4 rotation (for down_proj layers)
        if self.quant_config.online_r4:
            x = self._apply_rotation(layer, x, "_r4_rotation_matrix", "_r4_rot_size")

        # Step 3: MXFP4 dequant + GEMM (custom op)
        output = torch.ops.auto_round.spinquant_mxfp4_linear(
            x, layer.weight_packed, layer.weight_scale, self.quant_config.group_size
        )

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _apply_rotation(
        layer: torch.nn.Module,
        x: torch.Tensor,
        matrix_attr: str,
        size_attr: str,
    ) -> torch.Tensor:
        """Apply rotation if the layer has a pre-computed rotation matrix."""
        R = getattr(layer, matrix_attr, None)
        if R is None:
            return x
        R = R.to(dtype=x.dtype)
        rot_size = getattr(layer, size_attr)
        in_features = x.shape[-1]
        if rot_size == in_features:
            return x @ R
        else:
            shape = x.shape
            x = x.reshape(*shape[:-1], -1, rot_size)
            return (x @ R).reshape(shape)


# =============================================================================
# Hadamard utilities (used at load time only, not in forward path)
# =============================================================================


try:
    from auto_round.algorithms.transforms.spinquant.rotation_utils import (
        get_hadamard_K as _auto_round_get_hadamard_K,
    )
    _HAS_AUTO_ROUND_HAD = True
except ImportError:
    _HAS_AUTO_ROUND_HAD = False
    _auto_round_get_hadamard_K = None


def get_hadamard_K(n: int) -> tuple[torch.Tensor, int]:
    """Get Hadamard matrix H_K and block size K for dimension n."""
    if _HAS_AUTO_ROUND_HAD:
        return _auto_round_get_hadamard_K(n)
    if n & (n - 1) == 0:
        return torch.tensor([[1.0]]), 1
    raise ValueError(
        f"Dimension {n} is not a power of 2. Install auto-round for full Hadamard support."
    )


def _build_full_hadamard(n: int, device: torch.device) -> torch.Tensor:
    """Build full [n, n] normalized Hadamard matrix on device.

    Called once at model load time, NOT in the forward path.
    """
    had_K, K = get_hadamard_K(n)
    had_K = had_K.to(device=device, dtype=torch.float32)

    # Build butterfly Hadamard as explicit matrix
    H = torch.eye(n, device=device, dtype=torch.float32)
    size = n
    while size > K:
        half = size // 2
        # Construct butterfly matrix for this level
        B = torch.zeros(n, n, device=device, dtype=torch.float32)
        for start in range(0, n, size):
            for i in range(half):
                B[start + i, start + i] = 1.0
                B[start + i, start + half + i] = 1.0
                B[start + half + i, start + i] = 1.0
                B[start + half + i, start + half + i] = -1.0
        H = B @ H
        size = half

    # Apply H_K block-diagonally
    if K > 1:
        K_block = torch.block_diag(*[had_K for _ in range(n // K)])
        H = K_block @ H

    H = H / math.sqrt(n)
    return H


def _generate_random_orthogonal(n: int, device: torch.device) -> torch.Tensor:
    """Generate a random orthogonal matrix [n, n] via QR decomposition.

    Used as fallback when rot_type is random/trained but no matrix is in checkpoint.
    Deterministic per (n, device) — uses a fixed seed for reproducibility across TP ranks.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42 + n)  # deterministic across ranks
    rand_mat = torch.randn(n, n, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(rand_mat)
    return Q.to(device=device)


# =============================================================================
# Custom Op: MXFP4 Linear (opaque to torch.compile, no graph break)
# =============================================================================

try:
    from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm as _triton_mxfp4_gemm
    _HAS_TRITON_MXFP4 = True
except (ImportError, RuntimeError):
    _HAS_TRITON_MXFP4 = False
    _triton_mxfp4_gemm = None


def _spinquant_mxfp4_linear_impl(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Actual MXFP4 dequant + GEMM implementation."""
    if _HAS_TRITON_MXFP4:
        output = _triton_mxfp4_gemm(
            x.float(),
            weight_packed,
            weight_scale,
            bias=None,
            group_size=group_size,
            fp32_precision="tf32",
        )
        return output.to(x.dtype)

    # Fallback: Python dequant
    return _mxfp4_dequant_linear_fallback(x, weight_packed, weight_scale, group_size)


def _spinquant_mxfp4_linear_fake(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    N = weight_packed.shape[0]
    return torch.empty(*x.shape[:-1], N, dtype=x.dtype, device=x.device)


def _mxfp4_dequant_linear_fallback(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Pure PyTorch fallback for MXFP4 dequant + linear."""
    N, K_half = weight_packed.shape
    K = K_half * 2

    # Unpack: low nibble = even cols, high nibble = odd cols
    low = (weight_packed & 0x0F).to(torch.int32)
    high = ((weight_packed >> 4) & 0x0F).to(torch.int32)
    unpacked = torch.stack([low, high], dim=-1).reshape(N, K)

    # E2M1 decode
    E2M1_LUT = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=x.device,
    )
    sign = torch.where(unpacked >= 8, -1.0, 1.0)
    mag_idx = unpacked & 0x07
    abs_val = E2M1_LUT[mag_idx]
    fp_values = (abs_val * sign).to(x.dtype)

    # Apply scale
    scale_float = torch.pow(
        2.0,
        weight_scale.to(torch.int32).float() - 127.0,
    ).to(x.dtype)
    fp_values = fp_values.reshape(N, -1, group_size)
    scale_float = scale_float.unsqueeze(-1)
    w_dequant = (fp_values * scale_float).reshape(N, K)

    return torch.nn.functional.linear(x, w_dequant, None)


# Register custom op with vLLM's library system
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.platforms import current_platform

# Create our own library for the custom op
_auto_round_lib = torch.library.Library("auto_round", "DEF")

direct_register_custom_op(
    op_name="spinquant_mxfp4_linear",
    op_func=_spinquant_mxfp4_linear_impl,
    mutates_args=[],
    fake_impl=_spinquant_mxfp4_linear_fake,
    target_lib=_auto_round_lib,
    dispatch_key=current_platform.dispatch_key,
)
