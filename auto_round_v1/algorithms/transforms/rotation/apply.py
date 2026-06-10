# Copyright (c) 2026 Intel Corporation
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
"""Hadamard rotation – concrete ``BaseRotation`` implementation.

Public entry points
-------------------
* :class:`HadamardRotation` – the stateful algorithm object.
* :func:`apply_rotation_transform` – convenience one-shot function.
"""

from __future__ import annotations

from typing import Any

import torch
import tqdm

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.rotation.config import RotationConfig, normalize_rotation_config
from auto_round.algorithms.transforms.rotation.transforms import build_hadamard_transform
from auto_round.compressors.utils import is_nv_fp
from auto_round.experimental.qmodules.base import QModuleBase

__all__ = ["HadamardRotation", "apply_rotation_transform"]


def _triton_available(data_type: str = "mx_fp") -> bool:
    """Best-effort check for whether Triton kernel path can be used."""
    if is_nv_fp(data_type):
        return False
    try:
        import triton  # noqa: F401  # pylint: disable=E0401

        if not torch.cuda.is_available():
            return False
        from auto_round.algorithms.transforms.rotation.utils.triton.mxfp4 import (  # noqa: F401
            mxfp4_forward_kernel_wrapper,
        )

        return True
    except Exception:
        return False


@BaseRotation.register("hadamard")
class HadamardRotation(BaseRotation):
    """Hadamard rotation algorithm.

    Registered under ``"hadamard"`` in the
    :class:`~auto_round.algorithms.transforms.base.BaseRotation` registry.

    Typical usage (via the top-level helper)::

        from auto_round.algorithms.transforms import apply_rotation
        model = apply_rotation(model, config={"hadamard_type": "random_hadamard"})

    Or directly::

        from auto_round.algorithms.transforms.rotation import apply_rotation_transform
        model = apply_rotation_transform(model, config=RotationConfig(), need_calibration=True)
    """

    def __init__(self, config: RotationConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_config(cls, config: dict | RotationConfig) -> "HadamardRotation":
        """Build a :class:`HadamardRotation` from a raw dict or :class:`RotationConfig`."""
        if isinstance(config, dict):
            config = RotationConfig.model_validate(config)
        return cls(config)

    def apply_to_model(
        self,
        model: torch.nn.Module,
        location: str = "weight",
        use_tqdm: bool = True,
        desc: str | None = None,
        data_type: str = "mx_fp",
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Apply the Hadamard rotation to *model*.

        Args:
            model:           Target model; modified in-place.
            location:        ``"weight"`` (eager, fused into weights) or
                             ``"input"`` (activation-side, via forward hook).
            use_tqdm:        Show a progress bar while iterating modules.
            desc:            Custom progress-bar description.
            data_type:       Quantization data type (e.g. ``"mx_fp"``).
            **kwargs:        Reserved for future use.

        Returns:
            The mutated *model* with ``model.rotation_config`` set to the
            normalised :class:`RotationConfig` dict.
        """
        cfg = self.config

        # Dispatch by backend.  The transform backend (triton-fused per-Linear)
        # is implemented below; the inplace (QuaRot) backend is delegated to
        # :mod:`auto_round.algorithms.transforms.rotation.inplace`.
        from auto_round.algorithms.transforms.rotation.dispatcher import resolve_hadamard_backend

        backend = resolve_hadamard_backend(cfg, data_type)
        if backend == "inplace":
            import auto_round.envs as envs
            from auto_round.algorithms.transforms.rotation.inplace import apply_rotation_transform as _inplace_apply

            # Resolve fuse flag: explicit > env var > default(False).
            fuse_online_to_weight = cfg.fuse_online_to_weight
            if cfg.fuse_online_to_weight is not None:
                fuse_online_to_weight = bool(cfg.fuse_online_to_weight)
            elif envs.AR_FUSE_ONLINE_ROTATION:
                fuse_online_to_weight = bool(envs.AR_FUSE_ONLINE_ROTATION)

            bs = cfg.block_size
            group_size = bs if (bs is not None and bs > 0) else None

            compute_device = kwargs.get("compute_device")
            model, _hooks = _inplace_apply(
                model,
                group_size=group_size,
                allow_online_rotation=cfg.allow_online_rotation,
                rotation_matrix=cfg.hadamard_type,
                fuse_online_to_weight=fuse_online_to_weight,
                compute_device=compute_device,
            )
            setattr(model, "rotation_config", cfg.model_dump())
            return model

        # backend == "transform": original per-Linear triton-fused path.
        # Collect target modules.
        target_types = (torch.nn.Linear, QModuleBase)

        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, target_types)]

        # ---- Selective rotation: decide per-layer ----
        from auto_round.algorithms.transforms.rotation.selective import LayerSelector

        selector = LayerSelector.from_config(cfg)

        # For "auto" mode, run activation profiling if a dataloader is provided.
        if cfg.layer_selection == "auto":
            calibration_dataloader = kwargs.get("calibration_dataloader")
            if calibration_dataloader is not None:
                compute_device = kwargs.get("compute_device")
                selector.profile(model, calibration_dataloader, num_samples=32, device=compute_device)
            else:
                from auto_round.utils import logger as ar_logger

                ar_logger.warning(
                    "layer_selection='auto' but no calibration_dataloader provided. "
                    "Falling back to structural priors only (no activation profiling)."
                )

        _desc = desc or f"Applying {cfg.hadamard_type} transforms"
        applied_count = 0
        skipped_count = 0
        rotated_layers = []
        for name, module in tqdm.tqdm(modules, desc=_desc, disable=not use_tqdm):
            if "lm_head" in name:
                skipped_count += 1
                continue
            if not selector.should_rotate(name, module):
                skipped_count += 1
                continue
            # Mark this module so the class-level WrapperLinear patch only
            # rotates weight/activation for selected layers (not all layers).
            module._hadamard_rotate_enabled = True
            _apply_to_module(model, module, cfg, location, data_type)
            applied_count += 1
            rotated_layers.append(name)

        # Log selection summary.
        if cfg.layer_selection != "all":
            from auto_round.utils import logger as ar_logger

            ar_logger.info(
                "Selective rotation result: applied=%d, skipped=%d / %d total (%.1f%% coverage)",
                applied_count, skipped_count, len(modules),
                100 * applied_count / max(len(modules), 1),
            )

            # Store the decisions for serialization — the export/save path
            # uses this to know which layers to inject rotation buffers on.
            decisions = selector.get_decisions()
            setattr(model, "rotation_decisions", decisions)

        # Store the list of rotated layers on model for the serialize path.
        setattr(model, "_rotated_layers", set(rotated_layers))

        # Store config on model for serialisation / downstream inspection.
        setattr(model, "rotation_config", cfg.model_dump())

        # Bridge to SpinQuant serializer: set model._rotation_config so that
        # the export path (inject_rotation_buffers_bulk / inject_rotation_buffers_on_layer)
        # will inject per-QuantLinear rotation buffers for vLLM inference.
        _setup_serialize_bridge(model, cfg)

        return model


# ---------------------------------------------------------------------------
# Serialize bridge: connect Hadamard rotation to SpinQuant serializer for vLLM
# ---------------------------------------------------------------------------


def _setup_serialize_bridge(model: torch.nn.Module, cfg: RotationConfig) -> None:
    """Set model._rotation_config so the export path injects vLLM-compatible buffers.

    The SpinQuant serializer recognizes SpinQuantConfig and injects per-QuantLinear
    buffers (spinquant_r1_type/size/matrix). We create a synthetic config with
    r1=True, online_r1_rotation=True so that the same serialization path handles
    the Hadamard per-linear rotation case.

    For selective rotation, `model._rotated_layers` is checked by the modified
    inject logic to skip layers that were not rotated.
    """
    try:
        from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig
    except ImportError:
        return

    is_random = cfg.hadamard_type == "random_hadamard"
    spinquant_cfg = SpinQuantConfig(
        algorithm="spinquant",
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        random_r1=is_random,
        trainable_rotation=False,
        trainable_smooth=False,
        rotation_size=cfg.block_size,
    )
    # Mark as selective so serializer can check _rotated_layers
    spinquant_cfg._selective_mode = cfg.layer_selection != "all"
    model._rotation_config = spinquant_cfg


# ---------------------------------------------------------------------------
# Module-level application helper
# ---------------------------------------------------------------------------


def _apply_to_module(
    model: torch.nn.Module,
    module: torch.nn.Module,
    config: RotationConfig,
    location: str,
    data_type: str = "mx_fp",
) -> None:
    """Apply the configured Hadamard transform to a single *module*."""
    if location == "input":
        _apply_input_transform(module, config, data_type)

    elif location == "weight":
        _apply_weight_transform(module, config)

    else:
        raise NotImplementedError(f"Unsupported transform location: {location!r}")


def _apply_input_transform(module: torch.nn.Module, config: RotationConfig, data_type: str = "mx_fp") -> None:
    """Register a forward pre-hook that applies the Hadamard to the input activation."""
    from auto_round.algorithms.transforms.rotation.utils.matrix import multihead_matmul

    inp_transform = build_hadamard_transform(
        **config.model_dump(),
        location="input",
        inverse=True,
        device="cpu",
        precision=module.dtype if hasattr(module, "dtype") else None,
    )

    if config.hadamard_type != "random_hadamard":
        hadamard_weight = inp_transform.weight
    else:
        hadamard_weight = None

    if _triton_available(data_type):
        from auto_round.algorithms.transforms.rotation.utils.triton.mxfp4 import mxfp4_forward_kernel_wrapper

        def _input_hook(self, args):
            x = args[0]
            orig_shape = x.shape
            orig_dtype = x.dtype
            x_flat = x.contiguous().flatten(end_dim=-2)
            w = hadamard_weight.to(orig_dtype) if hadamard_weight is not None else self.hadamard_matrix.T.to(orig_dtype)
            qdq_input, _ = mxfp4_forward_kernel_wrapper(x_flat, w)
            return qdq_input.reshape(orig_shape).to(orig_dtype)

        module.pre_dequantized_input = True
        module.register_forward_pre_hook(_input_hook, prepend=True)
    else:

        def _input_hook(self, args):
            x = args[0]
            ori_shape = x.shape
            orig_dtype = x.dtype
            if hadamard_weight is not None:
                x = x.view(-1, hadamard_weight.shape[0])
                return multihead_matmul(x, hadamard_weight.to(x.device).to(orig_dtype)).view(ori_shape).to(orig_dtype)
            else:
                x = x.view(-1, self.hadamard_matrix.shape[0])
                return multihead_matmul(x, self.hadamard_matrix.T.to(orig_dtype)).view(ori_shape).to(orig_dtype)

        module.pre_dequantized_input = False
        module.register_forward_pre_hook(_input_hook, prepend=True)


def _apply_weight_transform(
    module: torch.nn.Module,
    config: RotationConfig,
) -> None:
    """Fuse or patch the Hadamard rotation into the weight of *module*."""
    from auto_round.algorithms.transforms.rotation.patch import (
        patch_quantlinear,
        patch_wrapperlinear_to_apply_transform,
        patch_wrapperwalayer_forward_to_apply_transform,
    )

    assert hasattr(module, "weight"), "Weight transform requires module to have a 'weight' attribute"

    w_transform = build_hadamard_transform(
        **config.model_dump(),
        location="weight",
        device=module.weight.device,
    )

    # For random Hadamard, save the matrix as a submodule for serialisation.
    if config.hadamard_type == "random_hadamard":
        from auto_round.algorithms.transforms.rotation.patch import patch_quantlinear as _patch_ql

        _patch_ql(w_transform)

    # Patch WrapperLinear and WrapperWALayer so the transform is applied
    # during calibration tuning.
    inp_transform = build_hadamard_transform(
        **config.model_dump(),
        location="input",
        inverse=True,
        device=module.weight.device,
        precision=module.weight.dtype,
    )

    patch_wrapperlinear_to_apply_transform(w_transform, inp_transform)
    patch_wrapperwalayer_forward_to_apply_transform(inp_transform)


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------


def apply_rotation_transform(
    model: torch.nn.Module,
    config: str | dict | RotationConfig | None,
    location: str = "weight",
    use_tqdm: bool = True,
    desc: str | None = None,
    data_type: str = "mx_fp",
    **kwargs,
) -> torch.nn.Module:
    """Apply a Hadamard rotation to *model*.

    This is the main public entry point when you only want Hadamard (rather
    than the polymorphic :func:`~auto_round.algorithms.transforms.apply_rotation`).

    Args:
        model:            Target model.
        config:           One of: :class:`RotationConfig`, ``dict``, ``str``
                          shorthand, or ``None`` (no-op).
        location:         ``"weight"`` or ``"input"``.
        use_tqdm:         Show progress bar.
        desc:             Custom progress-bar label.
        data_type:        Quantization data type (e.g. ``"mx_fp"``).
        **kwargs:         Forwarded to apply_to_model (e.g. calibration_dataloader,
                          compute_device for selective rotation).

    Returns:
        The transformed model.
    """
    normalised = normalize_rotation_config(config)
    if not normalised:
        return model
    rotation = HadamardRotation.from_config(normalised)
    return rotation.apply_to_model(
        model,
        location=location,
        use_tqdm=use_tqdm,
        desc=desc,
        data_type=data_type,
        **kwargs,
    )
