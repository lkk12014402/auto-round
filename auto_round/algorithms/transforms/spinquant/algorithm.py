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
"""SpinQuant / QuaRot rotation — ``BaseRotation`` subclass.

Registers ``"spinquant"`` in the :class:`BaseRotation` registry so that
the unified ``apply_rotation(model, config)`` entry point can dispatch to
the SpinQuant preprocessing pipeline automatically.

Usage via the unified entry point::

    from auto_round.algorithms.transforms import apply_rotation
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    config = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False,
                             trainable_rotation=False)
    model = apply_rotation(model, config)

Usage via AutoRound pipeline::

    from auto_round.compressors_new.entry import AutoRound
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    AutoRound(
        alg_configs=[SignRoundConfig(iters=200), SpinQuantConfig(r1=True, r2=True)],
        model=model,
        scheme="W4A16",
    )
"""

from __future__ import annotations

from typing import Any

import torch

from auto_round.algorithms.transforms.base import BaseRotation


@BaseRotation.register("spinquant")
class SpinQuantRotation(BaseRotation):
    """QuaRot / SpinQuant rotation registered as ``"spinquant"`` in
    :class:`BaseRotation`.

    Delegates to :class:`SpinQuantPreprocessor` for the actual rotation
    pipeline (RMSNorm fusion → rotation matrix init → optional training →
    weight fusion → online hook registration).

    For QuaRot mode (``trainable_rotation=False``), no ``dataloader`` is
    needed. For SpinQuant mode (``trainable_rotation=True``, ⚠️ experimental),
    pass ``dataloader=`` via ``**kwargs``.
    """

    def apply_to_model(
        self,
        model: torch.nn.Module,
        data_type: str = "mx_fp",
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Apply SpinQuant / QuaRot rotation to *model*.

        Args:
            model: The model to transform (modified in-place).
            data_type: Quantization data type (informational; SpinQuant does
                not change behavior based on this).
            **kwargs: Extra arguments.  ``dataloader`` is forwarded to
                :meth:`SpinQuantPreprocessor.preprocess` when the config
                has ``trainable_rotation=True``.

        Returns:
            The transformed model.
        """
        # Lazy import to avoid circular dependencies — SpinQuantPreprocessor
        # imports from this package's __init__ which imports this module.
        from auto_round.algorithms.transforms.spinquant.preprocessor import (
            SpinQuantPreprocessor,
        )

        dataloader = kwargs.get("dataloader")
        preprocessor = SpinQuantPreprocessor(model, self.config)
        return preprocessor.preprocess(dataloader)
