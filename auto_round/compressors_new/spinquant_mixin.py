"""
SpinQuant mixin for AutoRound compressors.

This module provides a mixin class that can be added to AutoRound's
compressor classes (LLMCompressor, AdamCompressor, etc.) to enable
SpinQuant preprocessing before quantization.

Usage:
    In compressors_new/base.py or entry.py, add SpinQuantMixin to the
    compressor's base classes.

Example:
    class LLMCompressor(SpinQuantMixin, BaseCompressor):
        ...
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
    SpinQuantState,
)


class SpinQuantMixin:
    """
    Mixin class that adds SpinQuant preprocessing to an AutoRound compressor.
    
    This mixin intercepts the quantization pipeline to run SpinQuant rotation
    learning before the main block-wise quantization calibration.
    
    Attributes:
        enable_spinquant: Whether to enable SpinQuant preprocessing
        spinquant_config: Configuration for SpinQuant
        spinquant_state: Training state tracker
    """

    def __init__(
        self,
        *args,
        enable_spinquant: bool = False,
        spinquant_iters: int = 200,
        spinquant_lr: float = 1e-4,
        spinquant_smooth_lr: float = 1e-3,
        spinquant_loss_type: str = "kl_top",
        spinquant_r1: bool = True,
        spinquant_r2: bool = True,
        spinquant_trainable: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.enable_spinquant = enable_spinquant
        self.spinquant_state = SpinQuantState()
        
        # Build SpinQuant config from AutoRound parameters
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
    
    def preprocess_with_spinquant(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
    ) -> nn.Module:
        """
        Apply SpinQuant preprocessing to the model.
        
        This method should be called before the main quantization calibration.
        
        Args:
            model: The model to preprocess
            dataloader: Calibration data for training rotations
        
        Returns:
            The preprocessed model
        """
        if not self.enable_spinquant:
            return model
        
        print("[AutoRound] SpinQuant preprocessing enabled")
        self.spinquant_state.enabled = True
        self.spinquant_state.max_iterations = self.spinquant_config.iters
        
        preprocessor = SpinQuantPreprocessor(model, self.spinquant_config)
        
        # If dataloader not provided, try to get from compressor state
        if dataloader is None and hasattr(self, 'dataloader'):
            dataloader = self.dataloader
        
        if dataloader is None and self.spinquant_config.trainable_rotation:
            raise ValueError(
                "dataloader required for SpinQuant training. "
                "Either pass it to preprocess_with_spinquant() or ensure "
                "the compressor has a dataloader attribute."
            )
        
        result = preprocessor.preprocess(dataloader)
        
        # Update state
        self.spinquant_state.rotation_names = [
            n for n, _ in model.named_parameters()
            if "spinquant" in n
        ]
        
        return result
    
    def get_spinquant_summary(self) -> dict[str, Any]:
        """Return a summary of SpinQuant preprocessing."""
        return self.spinquant_state.summary()


def patch_compressor_for_spinquant(compressor_class: type) -> type:
    """
    Dynamically patch an AutoRound compressor class to support SpinQuant.
    
    This is an alternative to using the mixin - it monkey-patches the
    compress method to add SpinQuant preprocessing.
    
    Args:
        compressor_class: The compressor class to patch (e.g., LLMCompressor)
    
    Returns:
        The patched class
    """
    original_compress = compressor_class.compress
    
    def compress_with_spinquant(self, *args, **kwargs):
        # Run SpinQuant preprocessing first
        if hasattr(self, 'enable_spinquant') and self.enable_spinquant:
            if hasattr(self, 'model') and hasattr(self, 'dataloader'):
                self.preprocess_with_spinquant(self.model, self.dataloader)
        
        # Then run original compression
        return original_compress(self, *args, **kwargs)
    
    compressor_class.compress = compress_with_spinquant  # type: ignore[attr-defined]
    
    # Add mixin methods if not already present
    if not hasattr(compressor_class, 'preprocess_with_spinquant'):
        compressor_class.preprocess_with_spinquant = SpinQuantMixin.preprocess_with_spinquant  # type: ignore[method-assign]
    if not hasattr(compressor_class, 'get_spinquant_summary'):
        compressor_class.get_spinquant_summary = SpinQuantMixin.get_spinquant_summary  # type: ignore[method-assign]
    
    return compressor_class

