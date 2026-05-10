"""SpinQuant / QuaRot mixin for AutoRound compressors.

.. deprecated::
    This mixin is **deprecated** in favour of the unified rotation pipeline.
    Pass a :class:`SpinQuantConfig` (or the string alias ``"quarot"`` /
    ``"spinquant"``) via the ``rotation_configs`` parameter of
    :class:`AutoRound` instead.  The unified pipeline routes through
    :func:`auto_round.algorithms.transforms.apply_rotation` →
    :class:`SpinQuantRotation` and fully integrates with
    :meth:`BaseCompressor._apply_rotations`.

**New recommended usage**::

    from auto_round.compressors_new.entry import AutoRound
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    # QuaRot (deterministic, no training):
    AutoRound(
        model=model, scheme="W4A16",
        rotation_configs=["quarot"],          # shorthand
    )

    # Equivalent explicit config:
    AutoRound(
        model=model, scheme="W4A16",
        rotation_configs=[
            SpinQuantConfig(r1=True, r2=True, r3=True, r4=True,
                            trainable_rotation=False)
        ],
    )

    # SpinQuant training (⚠️ experimental — requires dataloader):
    # Pre-apply rotation before passing to AutoRound:
    from auto_round.algorithms.transforms import apply_rotation
    model = apply_rotation(model, SpinQuantConfig(trainable_rotation=True),
                           dataloader=my_loader)

This mixin is preserved **only** for backward-compatibility with code that
already calls :meth:`preprocess_with_spinquant`.  It will be removed in a
future release.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch.nn as nn

from auto_round.algorithms.transforms import apply_rotation
from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig,
    SpinQuantState,
)


class SpinQuantMixin:
    """Mixin that adds :meth:`preprocess_with_spinquant` to a compressor.

    .. deprecated::
        Use ``rotation_configs=["quarot"]`` or a :class:`SpinQuantConfig`
        directly.  See module-level docstring for migration examples.
    """

    def __init__(
        self,
        *args: Any,
        enable_spinquant: bool = False,
        spinquant_iters: int = 200,
        spinquant_lr: float = 1e-4,
        spinquant_smooth_lr: float = 1e-3,
        spinquant_loss_type: str = "kl_top",
        spinquant_r1: bool = True,
        spinquant_r2: bool = True,
        spinquant_trainable: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.enable_spinquant = enable_spinquant
        self.spinquant_state = SpinQuantState()

        self.spinquant_config = SpinQuantConfig(
            r1=spinquant_r1,
            r2=spinquant_r2,
            trainable_rotation=spinquant_trainable,
            iters=spinquant_iters,
            lr=spinquant_lr,
            smooth_lr=spinquant_smooth_lr,
            loss_type=spinquant_loss_type,
            fuse_rmsnorm=True,
            untie_embeddings=True,
        )

        if enable_spinquant:
            warnings.warn(
                "SpinQuantMixin is deprecated.  Use "
                "rotation_configs=['quarot'] or a SpinQuantConfig instead.  "
                "See auto_round.compressors_new.spinquant_mixin docstring "
                "for migration details.",
                DeprecationWarning,
                stacklevel=2,
            )

    def preprocess_with_spinquant(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
    ) -> nn.Module:
        """Apply SpinQuant/QuaRot preprocessing to *model*.

        .. deprecated:: Use :func:`apply_rotation` with a
            :class:`SpinQuantConfig` instead.

        Args:
            model: Model to preprocess.
            dataloader: Calibration data (required for SpinQuant training).

        Returns:
            The preprocessed model.
        """
        if not self.enable_spinquant:
            return model

        warnings.warn(
            "preprocess_with_spinquant() is deprecated.  Use "
            "apply_rotation(model, SpinQuantConfig(...)) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.spinquant_state.enabled = True
        self.spinquant_state.max_iterations = self.spinquant_config.iters

        # If dataloader not provided, try to get from compressor state
        if dataloader is None and hasattr(self, "dataloader"):
            dataloader = self.dataloader

        if dataloader is None and self.spinquant_config.trainable_rotation:
            raise ValueError(
                "dataloader required for SpinQuant training.  "
                "Either pass it to preprocess_with_spinquant() or ensure "
                "the compressor has a dataloader attribute."
            )

        # Delegate to the unified rotation pipeline
        result = apply_rotation(
            model,
            self.spinquant_config,
            dataloader=dataloader,
        )

        self.spinquant_state.rotation_names = [
            n for n, _ in model.named_parameters() if "spinquant" in n
        ]
        return result

    def get_spinquant_summary(self) -> dict[str, Any]:
        """Return a summary of SpinQuant preprocessing."""
        return self.spinquant_state.summary()


def patch_compressor_for_spinquant(compressor_class: type) -> type:
    """Dynamically patch a compressor class to support SpinQuant.

    .. deprecated::
        Use ``rotation_configs=["quarot"]`` in :class:`AutoRound` instead.
    """
    warnings.warn(
        "patch_compressor_for_spinquant() is deprecated.  Use "
        "rotation_configs=['quarot'] in AutoRound instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    original_compress = compressor_class.compress

    def compress_with_spinquant(self: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self, "enable_spinquant") and self.enable_spinquant:
            if hasattr(self, "model") and hasattr(self, "dataloader"):
                self.preprocess_with_spinquant(self.model, self.dataloader)
        return original_compress(self, *args, **kwargs)

    compressor_class.compress = compress_with_spinquant  # type: ignore[attr-defined]

    if not hasattr(compressor_class, "preprocess_with_spinquant"):
        compressor_class.preprocess_with_spinquant = SpinQuantMixin.preprocess_with_spinquant  # type: ignore[method-assign]
    if not hasattr(compressor_class, "get_spinquant_summary"):
        compressor_class.get_spinquant_summary = SpinQuantMixin.get_spinquant_summary  # type: ignore[method-assign]

    return compressor_class

