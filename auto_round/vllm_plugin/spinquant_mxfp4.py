# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
vLLM quantization plugin for SpinQuant/QuaRot MXFP4 models.

Supports:
  - R1 online rotation (Hadamard butterfly or full matrix) on q/k/v/gate/up
  - R4 online rotation (Hadamard butterfly or full matrix) on down_proj
  - R3 online rotation (Q/K after RoPE — head_dim rotation)
  - MXFP4 activation QDQ after online rotation
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
import os
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

logger = logging.getLogger("vllm")

MXFP4_BLOCK_SIZE = 32
FP4_E2M1_MAX = 6.0
F32_MIN_NORMAL = 2 ** (-126)

RUNTIME_BACKEND_PACKED_FUSED = "packed_fused"
RUNTIME_BACKEND_QUARK_LIKE_DENSE = "quark_like_dense"
RUNTIME_BACKEND_PREUNPACK_FP8 = "preunpack_fp8"
VALID_RUNTIME_BACKENDS = {
    RUNTIME_BACKEND_PACKED_FUSED,
    RUNTIME_BACKEND_QUARK_LIKE_DENSE,
    RUNTIME_BACKEND_PREUNPACK_FP8,
}

# Rotation type constants (must match auto_round serialization)
ROTATION_TYPE_HADAMARD = 0
ROTATION_TYPE_RANDOM = 1
ROTATION_TYPE_TRAINED = 2

# Rotation runtime modes prepared once at load time.
ROTATION_RUNTIME_NONE = 0
ROTATION_RUNTIME_MATRIX = 1
ROTATION_RUNTIME_HADAMARD = 2


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
        runtime_backend: str = RUNTIME_BACKEND_PACKED_FUSED,
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
        self.runtime_backend = runtime_backend

    def __repr__(self) -> str:
        parts = [
            f"bits={self.bits}",
            f"group_size={self.group_size}",
            f"online_r1={self.online_r1}",
            f"r1_type={self.r1_type}",
            f"runtime_backend={self.runtime_backend}",
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
        instance = cls(
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
            runtime_backend=_resolve_runtime_backend(config),
        )
        # Log only active rotations for this specific model
        active = []
        if instance.online_r1:
            r1_size = instance.rotation_size or instance.hidden_size
            r1_detail = _describe_hadamard(instance.r1_type, r1_size)
            active.append(f"R1(online, {r1_detail})")
        if sq_config.get("r2", False):
            active.append("R2(offline, fused into weights)")
        if instance.online_r3:
            r3_size = instance.head_dim
            r3_detail = _describe_hadamard(instance.r3_type, r3_size)
            active.append(f"R3(online, {r3_detail})")
        if instance.online_r4:
            r4_size = instance.rotation_size or instance.intermediate_size
            r4_detail = _describe_hadamard(instance.r4_type, r4_size)
            active.append(f"R4(online, {r4_detail})")
        rot_str = ", ".join(active) if active else "none"
        logger.info(
            f"[AutoRound] Model quantization: MXFP{instance.bits} (group_size={instance.group_size}) | "
            f"Online rotations: [{rot_str}] | activation_qdq=enabled(even) | "
            f"runtime_backend={instance.runtime_backend} | "
            f"hidden_size={instance.hidden_size}, intermediate_size={instance.intermediate_size}"
        )
        return instance

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
            if self.runtime_backend == RUNTIME_BACKEND_QUARK_LIKE_DENSE:
                return SpinQuantMXFP4QuarkLikeDenseLinearMethod(self)
            if self.runtime_backend == RUNTIME_BACKEND_PREUNPACK_FP8:
                return SpinQuantMXFP4PreunpackFP8LinearMethod(self)
            return SpinQuantMXFP4PackedFusedLinearMethod(self)
        return None


# =============================================================================
# Linear Method
# =============================================================================


class SpinQuantMXFP4LinearMethod(LinearMethodBase):
    """Base linear method implementing shared SpinQuant rotation handling."""

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
        layer.params_dtype = params_dtype

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
        # NOTE: Use shape (1,) instead of () for RowParallelLinear compatibility.
        # RowParallelLinear.weight_loader reshapes scalar tensors from () to (1,)
        # before the shape assertion check.
        spinquant_r4_type = Parameter(
            torch.zeros(1, dtype=torch.int32), requires_grad=False
        )
        set_weight_attrs(spinquant_r4_type, {"ignore_warning": True} | extra_weight_attrs)
        layer.register_parameter("spinquant_r4_type", spinquant_r4_type)

        spinquant_r4_size = Parameter(
            torch.zeros(1, dtype=torch.int32), requires_grad=False
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
        """Prepare runtime rotation state once after checkpoint loading.

        This mirrors Quark's exported online-R1 semantics:
        - packed weights are already in their saved form and are never touched here
        - only activation-side online rotation metadata/state is prepared
        """
        # --- R1 rotation ---
        self._process_rotation(layer, prefix="r1")
        # --- R4 rotation ---
        self._process_rotation(layer, prefix="r4")
        self._prepare_runtime_weight_backend(layer)

    def _process_rotation(self, layer: torch.nn.Module, prefix: str) -> None:
        """Prepare load-time rotation state for a given prefix (r1 or r4)."""
        type_attr = f"spinquant_{prefix}_type"
        size_attr = f"spinquant_{prefix}_size"
        matrix_attr = f"spinquant_{prefix}_matrix"
        result_attr = f"_{prefix}_rotation_matrix"
        rot_size_attr = f"_{prefix}_rot_size"
        runtime_attr = f"_{prefix}_rotation_runtime"
        hadamard_attr = f"_{prefix}_hadamard_K"
        hadamard_factor_attr = f"_{prefix}_hadamard_factor"

        setattr(layer, result_attr, None)
        setattr(layer, rot_size_attr, 0)
        setattr(layer, runtime_attr, ROTATION_RUNTIME_NONE)
        setattr(layer, hadamard_attr, None)
        setattr(layer, hadamard_factor_attr, 0)

        if not hasattr(layer, type_attr):
            return

        rot_type = int(getattr(layer, type_attr).item())
        rot_size = int(getattr(layer, size_attr).item())

        if rot_size == 0:
            # Free placeholder matrix
            if hasattr(layer, matrix_attr):
                delattr(layer, matrix_attr)
            return

        device = layer.weight_packed.device
        in_features = layer.weight_packed.shape[1] * 2

        if rot_type == ROTATION_TYPE_HADAMARD:
            is_po2 = (rot_size & (rot_size - 1) == 0) and rot_size > 0
            if not is_po2:
                try:
                    _, K = get_hadamard_K(rot_size)
                except (ValueError, ImportError):
                    K = "?"
                if not hasattr(self, f"_logged_{prefix}_non_po2"):
                    logger.info(
                        f"[AutoRound] {prefix.upper()} rotation: size={rot_size} "
                        f"(non-power-of-2, K={K}, using matmul_hadU butterfly)"
                    )
                    setattr(self, f"_logged_{prefix}_non_po2", True)
            hadamard_K, K = get_hadamard_K(rot_size)
            hadamard_K = hadamard_K.to(device=device, dtype=torch.float32)
            setattr(layer, rot_size_attr, rot_size)

            if rot_size == in_features:
                setattr(layer, runtime_attr, ROTATION_RUNTIME_HADAMARD)
                setattr(layer, hadamard_attr, hadamard_K)
                setattr(layer, hadamard_factor_attr, K)
                R = None
            elif in_features % rot_size == 0:
                R = _build_block_hadamard(rot_size, hadamard_K, K, device)
            else:
                raise ValueError(
                    f"{prefix.upper()} rotation_size={rot_size} is not compatible "
                    f"with in_features={in_features}"
                )
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
            setattr(layer, runtime_attr, ROTATION_RUNTIME_MATRIX)

        # Free the original parameter to save memory
        if hasattr(layer, matrix_attr):
            delattr(layer, matrix_attr)

    def _prepare_runtime_weight_backend(self, layer: torch.nn.Module) -> None:
        """Prepare load-time weight representation for the selected runtime backend."""
        backend = self.quant_config.runtime_backend
        if backend == RUNTIME_BACKEND_PACKED_FUSED:
            return

        target_dtype = getattr(layer, "params_dtype", torch.bfloat16)
        if backend == RUNTIME_BACKEND_QUARK_LIKE_DENSE:
            weight_dense = _dequant_packed_mxfp4_weight(
                layer.weight_packed.data,
                layer.weight_scale.data,
                self.quant_config.group_size,
                target_dtype=target_dtype,
            )
            layer.register_parameter(
                "weight_dense_qdq",
                Parameter(weight_dense, requires_grad=False),
            )
            _clear_packed_weight_storage(layer)
            return

        if backend == RUNTIME_BACKEND_PREUNPACK_FP8:
            weight_fp8, scale_bf16 = _preunpack_mxfp4_weight(
                layer.weight_packed.data,
                layer.weight_scale.data,
                self.quant_config.group_size,
            )
            layer.register_parameter(
                "weight_unpacked_fp8",
                Parameter(weight_fp8, requires_grad=False),
            )
            layer.register_parameter(
                "weight_scale_bf16",
                Parameter(scale_bf16, requires_grad=False),
            )
            _clear_packed_weight_storage(layer)
            return

        raise ValueError(f"Unsupported SpinQuant runtime backend: {backend}")

    def _prepare_activations(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply online rotation and activation qdq before the backend-specific GEMM path."""
        # Step 1: Apply R1 rotation (for q/k/v/gate/up layers)
        if self.quant_config.online_r1:
            x = self._apply_rotation(layer, x, prefix="r1")

        # Step 2: Apply R4 rotation (for down_proj layers)
        if self.quant_config.online_r4:
            x = self._apply_rotation(layer, x, prefix="r4")

        # Step 3: MXFP4 activation qdq (Quark/vllm-ext aligned)
        x = torch.ops.auto_round.spinquant_mxfp4_act_qdq(x, self.quant_config.group_size)
        return x

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility dispatcher for direct instantiation in tests and local usage."""
        backend = self.quant_config.runtime_backend
        if backend == RUNTIME_BACKEND_QUARK_LIKE_DENSE:
            return SpinQuantMXFP4QuarkLikeDenseLinearMethod(self.quant_config).apply(layer, x, bias)
        if backend == RUNTIME_BACKEND_PREUNPACK_FP8:
            return SpinQuantMXFP4PreunpackFP8LinearMethod(self.quant_config).apply(layer, x, bias)
        return SpinQuantMXFP4PackedFusedLinearMethod(self.quant_config).apply(layer, x, bias)

    @staticmethod
    def _apply_rotation(
        layer: torch.nn.Module,
        x: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """Apply a prepared rotation in the lightest available runtime form."""
        runtime = getattr(layer, f"_{prefix}_rotation_runtime", ROTATION_RUNTIME_NONE)
        rot_size = getattr(layer, f"_{prefix}_rot_size", 0)

        if runtime == ROTATION_RUNTIME_HADAMARD:
            hadamard_K = getattr(layer, f"_{prefix}_hadamard_K", None)
            hadamard_factor = getattr(layer, f"_{prefix}_hadamard_factor", 0)
            if hadamard_K is None or rot_size == 0:
                return x
            return matmul_hadU(
                x,
                hadamard_K=hadamard_K.to(device=x.device, dtype=x.dtype),
                K=hadamard_factor,
            ).to(x.dtype)

        matrix_attr = f"_{prefix}_rotation_matrix"
        R = getattr(layer, matrix_attr, None)
        if R is None:
            return x
        R = R.to(dtype=x.dtype)
        in_features = x.shape[-1]
        if rot_size == in_features:
            return x @ R
        else:
            shape = x.shape
            x = x.reshape(*shape[:-1], -1, rot_size)
            return (x @ R).reshape(shape)

    @staticmethod
    def _apply_dense_linear(
        weight: torch.Tensor,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run dense GEMM while keeping activation/output dtype stable."""
        input_dtype = x.dtype
        compute_dtype = weight.dtype
        if x.dtype != compute_dtype:
            x = x.to(compute_dtype)
        if bias is not None and bias.dtype != compute_dtype:
            bias = bias.to(compute_dtype)
        output = torch.nn.functional.linear(x, weight, bias)
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output


class SpinQuantMXFP4PackedFusedLinearMethod(SpinQuantMXFP4LinearMethod):
    """Packed low-bit runtime: activation qdq + fused weight dequant/GEMM."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._prepare_activations(layer, x)
        output = torch.ops.auto_round.spinquant_mxfp4_linear(
            x, layer.weight_packed, layer.weight_scale, self.quant_config.group_size
        )
        if bias is not None:
            output = output + bias
        return output


class SpinQuantMXFP4QuarkLikeDenseLinearMethod(SpinQuantMXFP4LinearMethod):
    """Dense runtime: load-time weight restore + activation qdq + F.linear."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._prepare_activations(layer, x)
        return self._apply_dense_linear(layer.weight_dense_qdq, x, bias)


class SpinQuantMXFP4PreunpackFP8LinearMethod(SpinQuantMXFP4LinearMethod):
    """Pre-unpack runtime: load-time FP8 unpack + per-forward weight restore + F.linear."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._prepare_activations(layer, x)
        weight_dense = _dequant_preunpacked_mxfp4_weight(
            layer.weight_unpacked_fp8,
            layer.weight_scale_bf16,
            target_dtype=getattr(layer, "params_dtype", x.dtype),
        )
        return self._apply_dense_linear(weight_dense, x, bias)


# =============================================================================
# Hadamard utilities
# =============================================================================


try:
    from auto_round.algorithms.transforms.spinquant.rotation_utils import (
        get_hadamard_K as _auto_round_get_hadamard_K,
        matmul_hadU as _auto_round_matmul_hadU,
    )
    _HAS_AUTO_ROUND_HAD = True
except ImportError:
    _HAS_AUTO_ROUND_HAD = False
    _auto_round_get_hadamard_K = None
    _auto_round_matmul_hadU = None


def get_hadamard_K(n: int) -> tuple[torch.Tensor, int]:
    """Get Hadamard matrix H_K and block size K for dimension n."""
    if _HAS_AUTO_ROUND_HAD:
        return _auto_round_get_hadamard_K(n)
    if n & (n - 1) == 0:
        return torch.tensor([[1.0]]), 1
    raise ValueError(
        f"Dimension {n} is not a power of 2. Install auto-round for full Hadamard support."
    )


def matmul_hadU(X: torch.Tensor, hadamard_K: torch.Tensor | None = None, K: int | None = None) -> torch.Tensor:
    """Apply the normalized Hadamard transform to the last dimension of X."""
    if _auto_round_matmul_hadU is not None:
        return _auto_round_matmul_hadU(X, hadamard_K=hadamard_K, K=K)
    H = _build_full_hadamard(X.shape[-1], X.device).to(dtype=X.dtype)
    return X @ H


def _describe_hadamard(rot_type: str, size: int) -> str:
    """Build a human-readable description of a rotation configuration.

    Examples:
        "hadamard, size=1024, power-of-2"
        "hadamard, size=3072, K=12 (non-power-of-2)"
        "random, size=1024"
    """
    if rot_type != "hadamard":
        return f"{rot_type}, size={size}"
    is_po2 = (size & (size - 1) == 0) and size > 0
    if is_po2:
        return f"hadamard, size={size}, power-of-2"
    try:
        _, K = get_hadamard_K(size)
        return f"hadamard, size={size}, K={K} (non-power-of-2)"
    except (ValueError, ImportError):
        return f"hadamard, size={size}, non-power-of-2"


def _build_full_hadamard(n: int, device: torch.device) -> torch.Tensor:
    """Build full [n, n] normalized Hadamard matrix on device.

    Called once at model load time, NOT in the forward path.

    Uses matmul_hadU applied to an identity matrix to guarantee the explicit
    matrix is byte-identical to the butterfly algorithm used during save-time
    offline fuse and online hooks. This is critical for non-power-of-2 dimensions
    (e.g. 3072 with K=12) where the butterfly interleaving pattern differs from
    a naive first-half/second-half split.
    """
    try:
        from auto_round.algorithms.transforms.spinquant.rotation_utils import matmul_hadU
        # matmul_hadU(X) computes X @ H (normalized) via butterfly algorithm.
        # Applying it to the identity gives us the explicit H matrix.
        I = torch.eye(n, device=device, dtype=torch.float32)
        H = matmul_hadU(I)
        return H
    except ImportError:
        pass

    # Fallback for pure power-of-2 (K=1) when auto_round is not installed
    if n & (n - 1) != 0:
        raise ValueError(
            f"Dimension {n} is not a power of 2. "
            f"Install auto-round for full Hadamard support (non-power-of-2 dims)."
        )
    # Standard Walsh-Hadamard for power-of-2
    H = torch.eye(n, device=device, dtype=torch.float32)
    size = n
    while size > 1:
        half = size // 2
        B = torch.zeros(n, n, device=device, dtype=torch.float32)
        for start in range(0, n, size):
            for i in range(half):
                B[start + i, start + i] = 1.0
                B[start + i, start + half + i] = 1.0
                B[start + half + i, start + i] = 1.0
                B[start + half + i, start + half + i] = -1.0
        H = B @ H
        size = half
    H = H / math.sqrt(n)
    return H


def _build_block_hadamard(
    rotation_size: int,
    hadamard_K: torch.Tensor,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a normalized explicit Hadamard block for block-wise rotation."""
    rot_mat = hadamard_K.to(device=device, dtype=torch.float32)
    if rot_mat.shape[0] != rotation_size:
        had_1, _ = get_hadamard_K(rotation_size // K)
        rot_mat = torch.kron(
            rot_mat.to(device="cpu", dtype=torch.float32),
            had_1.to(device="cpu", dtype=torch.float32),
        ).to(device=device)
    return rot_mat / math.sqrt(rotation_size)


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


def _resolve_runtime_backend(config: dict[str, Any]) -> str:
    """Resolve runtime backend from env or config."""
    sq_config = config.get("spinquant_config", {})
    backend = (
        os.getenv("AUTO_ROUND_SPINQUANT_RUNTIME_BACKEND")
        or sq_config.get("runtime_backend")
        or config.get("runtime_backend")
        or RUNTIME_BACKEND_PACKED_FUSED
    )
    if backend not in VALID_RUNTIME_BACKENDS:
        raise ValueError(
            f"Unsupported SpinQuant runtime backend {backend!r}. "
            f"Expected one of {sorted(VALID_RUNTIME_BACKENDS)}."
        )
    return backend


def _clear_packed_weight_storage(layer: torch.nn.Module) -> None:
    """Drop packed-weight storage once an alternate runtime backend is prepared."""
    if hasattr(layer, "weight_packed"):
        del layer.weight_packed
        layer.register_parameter("weight_packed", None)
    if hasattr(layer, "weight_scale"):
        del layer.weight_scale
        layer.register_parameter("weight_scale", None)


def _unpack_mxfp4_values(weight_packed: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Unpack packed E2M1 MXFP4 values to unscaled floating-point values."""
    N, K_half = weight_packed.shape
    K = K_half * 2

    low = (weight_packed & 0x0F).to(torch.int32)
    high = ((weight_packed >> 4) & 0x0F).to(torch.int32)
    unpacked = torch.stack([low, high], dim=-1).reshape(N, K)

    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=weight_packed.device,
    )
    sign = torch.where(unpacked >= 8, -1.0, 1.0)
    mag_idx = unpacked & 0x07
    abs_val = e2m1_lut[mag_idx]
    return (abs_val * sign).to(target_dtype)


def _e8m0_to_scale(weight_scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Convert e8m0 exponents to floating-point scale values."""
    return torch.pow(2.0, weight_scale.to(torch.int32).float() - 127.0).to(target_dtype)


def _dequant_packed_mxfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Fully dequantize packed MXFP4 weights to a dense floating-point tensor."""
    N, K_half = weight_packed.shape
    K = K_half * 2
    fp_values = _unpack_mxfp4_values(weight_packed, target_dtype=target_dtype)
    scale_float = _e8m0_to_scale(weight_scale, target_dtype=target_dtype)
    fp_values = fp_values.reshape(N, -1, group_size)
    return (fp_values * scale_float.unsqueeze(-1)).reshape(N, K)


def _preunpack_mxfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed MXFP4 weight to an unpacked FP8+scale representation."""
    fp_values = _unpack_mxfp4_values(weight_packed, target_dtype=torch.float32)
    weight_fp8 = fp_values.to(torch.float8_e4m3fn)
    scale_bf16 = _e8m0_to_scale(weight_scale, target_dtype=torch.bfloat16).reshape(-1, 1)
    if group_size != MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"preunpack_fp8 backend currently expects group_size={MXFP4_BLOCK_SIZE}, got {group_size}"
        )
    return weight_fp8, scale_bf16


def _dequant_preunpacked_mxfp4_weight(
    weight_fp8: torch.Tensor,
    scale_bf16: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize unpacked FP8+scale weights back to a dense tensor for GEMM."""
    origin_shape = weight_fp8.shape
    weight_fp8 = weight_fp8.reshape(-1, MXFP4_BLOCK_SIZE)
    scale = scale_bf16.reshape(-1, 1).to(target_dtype)
    return (weight_fp8.to(target_dtype) * scale).reshape(origin_shape)


# =============================================================================
# Custom Ops: MXFP4 activation QDQ + MXFP4 Linear
# =============================================================================

try:
    from auto_round_extension.vllm_ext.mxfp4_qdq_utils import qdq_mxfp4 as _vllm_ext_qdq_mxfp4
except ImportError:
    _vllm_ext_qdq_mxfp4 = None

try:
    from quark.torch.kernel import mx as _quark_mx
except ImportError:
    _quark_mx = None

try:
    from auto_round.triton_kernels.mxfp4_gemm import triton_mxfp4_gemm as _triton_mxfp4_gemm
    _HAS_TRITON_MXFP4 = True
except (ImportError, RuntimeError):
    _HAS_TRITON_MXFP4 = False
    _triton_mxfp4_gemm = None


def _spinquant_mxfp4_act_qdq_impl(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """QDQ the rotated activation to MXFP4 semantics before GEMM."""
    if group_size <= 0 or x.shape[-1] % group_size != 0:
        raise ValueError(
            f"MXFP4 activation qdq requires the last dim to be divisible by group_size, "
            f"got shape={tuple(x.shape)}, group_size={group_size}"
        )
    if group_size == MXFP4_BLOCK_SIZE and _quark_mx is not None and x.dtype in (torch.float16, torch.bfloat16):
        return _quark_mx.qdq_mxfp4(x, scale_calculation_mode="even")
    if group_size == MXFP4_BLOCK_SIZE and _vllm_ext_qdq_mxfp4 is not None:
        return _vllm_ext_qdq_mxfp4(x)
    return _mxfp4_act_qdq_fallback(x, group_size)


def _spinquant_mxfp4_act_qdq_fake(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(x)


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
    w_dequant = _dequant_packed_mxfp4_weight(
        weight_packed,
        weight_scale,
        group_size,
        target_dtype=x.dtype,
    )
    return torch.nn.functional.linear(x, w_dequant, None)


def _mxfp4_act_qdq_fallback(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pure PyTorch MXFP4 QDQ fallback aligned with Quark/vllm-ext even mode."""
    if group_size <= 0 or x.shape[-1] % group_size != 0:
        raise ValueError(
            f"MXFP4 activation qdq requires the last dim to be divisible by group_size, "
            f"got shape={tuple(x.shape)}, group_size={group_size}"
        )

    original_dtype = x.dtype
    original_shape = x.shape
    x_fp32 = x.to(torch.float32).reshape(-1, group_size)

    sign = x_fp32.sign()
    x_abs = x_fp32.abs()
    amax = x_abs.amax(dim=-1, keepdim=True)

    # Match Quark's "even" scale mode: round the max value before extracting the exponent.
    rounded_bits = (amax.contiguous().view(torch.int32) + 0x200000) & 0x7F800000
    rounded_max = rounded_bits.view(torch.float32)
    safe_max = torch.where(rounded_max > 0, rounded_max, torch.full_like(rounded_max, F32_MIN_NORMAL))

    scale_exp = torch.floor(torch.log2(safe_max)) - 2.0
    scale_exp = torch.clamp(scale_exp, min=-127, max=127)
    scale = torch.pow(2.0, scale_exp)
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))

    x_scaled = x_abs / scale
    x_fp4 = _fp4_121_positive(x_scaled).clamp(max=FP4_E2M1_MAX)
    x_qdq = (sign * x_fp4 * scale).reshape(original_shape)
    return x_qdq.to(original_dtype)


def _fp4_121_positive(x: torch.Tensor) -> torch.Tensor:
    """Round positive values to the E2M1 FP4 grid."""
    half_step = torch.round(2.0 * x) / 2.0
    unit_step = torch.round(x)
    two_step = 2.0 * torch.round(x / 2.0)

    below_two = x < 2.0
    below_four = x < 4.0
    return (
        half_step * below_two
        + unit_step * (~below_two) * below_four
        + two_step * (~below_two) * (~below_four)
    )


# Register custom op with vLLM's library system
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.platforms import current_platform

# Create our own library for the custom op
_auto_round_lib = torch.library.Library("auto_round", "DEF")

direct_register_custom_op(
    op_name="spinquant_mxfp4_act_qdq",
    op_func=_spinquant_mxfp4_act_qdq_impl,
    mutates_args=[],
    fake_impl=_spinquant_mxfp4_act_qdq_fake,
    target_lib=_auto_round_lib,
    dispatch_key=current_platform.dispatch_key,
)

direct_register_custom_op(
    op_name="spinquant_mxfp4_linear",
    op_func=_spinquant_mxfp4_linear_impl,
    mutates_args=[],
    fake_impl=_spinquant_mxfp4_linear_fake,
    target_lib=_auto_round_lib,
    dispatch_key=current_platform.dispatch_key,
)
