"""
SpinQuant rotation for Intel AutoRound.

This package provides SpinQuant-style trainable orthogonal rotation
transforms that can be applied before AutoRound quantization.

Main API::

    RotationTrainer              – HuggingFace-Trainer-style SpinQuant trainer
    RotationTrainerConfig        – Trainer hyperparameters
    SpinQuantPreprocessor       – Direct preprocessing (8-step pipeline)
    SpinQuantConfig              – Configuration dataclass
    TrainableRMSNorm             – RMSNorm + SmoothQuant wrapper
    apply_spinquant_in_place     – In-place application (AutoRound style)
    register_spinquant_hooks     – Online hook registration

Example (Trainer style, recommended)::

    from auto_round.algorithms.transforms.spinquant import (
        RotationTrainer, RotationTrainerConfig
    )

    trainer = RotationTrainer(
        model,
        config=RotationTrainerConfig(iters=200, lr=1e-4),
    )
    metrics = trainer.train(dataloader)
    model = trainer.fuse()

    # Now model is ready for AutoRound
    autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
    autoround.quantize()

Example (Preprocessor style, simplest)::

    from auto_round.algorithms.transforms.spinquant import SpinQuantPreprocessor, SpinQuantConfig

    SpinQuantPreprocessor(model, SpinQuantConfig()).preprocess(dataloader)
    AutoRound(model, tokenizer, bits=4).quantize()
"""

from auto_round.algorithms.transforms.spinquant.cayley_optimizer import (
    AdamAndSGDG,
    SGDG,
)
from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    apply_spinquant_in_place,
    register_spinquant_hooks,
    remove_spinquant_hooks,
)
from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
    TrainableRMSNorm,
)
from auto_round.algorithms.transforms.spinquant.trainer import (
    LossLogger,
    OrthogonalityMonitor,
    RotationTrainer,
    RotationTrainerCallback,
    RotationTrainerConfig,
)
from auto_round.algorithms.transforms.spinquant.training import (
    SpinQuantState,
    SpinQuantTrainingHook,
    create_spinquant_optimizer,
    spinquant_loss_fn,
)

__all__ = [
    # -- Trainer (HF-style, recommended) --
    "RotationTrainer",
    "RotationTrainerConfig",
    "RotationTrainerCallback",
    "LossLogger",
    "OrthogonalityMonitor",
    # -- Optimiser core (Cayley) --
    "AdamAndSGDG",
    "SGDG",
    # -- Preprocessor (direct) --
    "SpinQuantConfig",
    "SpinQuantPreprocessor",
    "TrainableRMSNorm",
    # -- In-place (AutoRound style) --
    "apply_spinquant_in_place",
    "register_spinquant_hooks",
    "remove_spinquant_hooks",
    # -- Legacy helpers --
    "SpinQuantTrainingHook",
    "SpinQuantState",
    "create_spinquant_optimizer",
    "spinquant_loss_fn",
]
