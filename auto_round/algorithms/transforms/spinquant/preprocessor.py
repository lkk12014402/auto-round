"""
SpinQuant preprocessor for Intel AutoRound.

Main class that orchestrates the SpinQuant / QuaRot rotation pipeline:

    1. Fuse RMSNorm parameters into linear layers
    2. Replace RMSNorm with TrainableRMSNorm (if ``trainable_smooth``)
    3. Initialise rotation matrices (R1, R2, R3, R4)
    4. (If trainable) train rotations & smooth values via KL divergence
    5. Fuse offline rotations (R1, R2) into model weights
    6. Register online hooks (R3, R4) and clean up

The module delegates in-place hook registration to
``spinquant.inplace.apply`` so that the online-rotation logic follows the
same pattern as AutoRound's ``rotation.inplace`` package.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.transforms.spinquant.cayley_optimizer import AdamAndSGDG
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    InputRotationWrapperHadamard,
    apply_hadamard_to_linear,
    create_block_diag_from_head_matrix,
    deterministic_hadamard_matrix,
    fuse_rmsnorm_in_model,
    get_hadamard_K,
    get_model_arch_info,
    is_pow2,
    matmul_hadU,
    random_hadamard_matrix,
    rotate_in_channels_,
    rotate_out_channels_,
    untie_word_embeddings_if_needed,
)

# ---------------------------------------------------------------------------
# Delegate online-hook registration to the inplace sub-package.
# ---------------------------------------------------------------------------
from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    register_spinquant_hooks,
    remove_spinquant_hooks,
)

logger = logging.getLogger("auto_round.spinquant")


# NOTE: The old _make_online_r1_hook() function has been removed.
# Online R1 now uses InputRotationWrapperHadamard (nn.Module) instead of
# register_forward_pre_hook, enabling proper model save/load.


@dataclass
class SpinQuantConfig:
    """Configuration for SpinQuant preprocessing."""

    # Rotation dimensions
    r1: bool = True                    # R1: hidden_size rotation (offline fused)
    r2: bool = True                    # R2: head_dim rotation (offline fused)
    r3: bool = True                    # R3: Q/K online rotation
    r4: bool = True                    # R4: MLP activation online rotation

    # Rotation size override (None = use full dimension from model config)
    # When set, R1 uses rotation_size instead of hidden_size,
    # and R4 uses rotation_size instead of intermediate_size.
    # R2 always uses head_dim, R3 does not support custom size.
    # This follows the same convention as Quark's rotation_size.
    rotation_size: Optional[int] = None

    # Training control
    trainable_rotation: bool = True    # Learn R via Cayley SGD (False = QuaRot fixed Hadamard)
    trainable_smooth: bool = True      # Learn smooth_values via Adam (joint SmoothQuant)
    online_r1_rotation: bool = True    # Online R1: rotate target weights + hook (Quark default)

    # Training hyperparameters
    iters: int = 200                   # Training iterations
    lr: float = 1e-4                   # SGDG learning rate (rotation matrices)
    smooth_lr: float = 1e-3            # Adam learning rate (smooth values)
    batch_size: int = 1

    # Loss
    loss_type: str = "kl_top"          # "kl_top" | "kl_full" | "mse"
    kl_top_k: int = 1000

    # Pipeline steps
    fuse_rmsnorm: bool = True
    untie_embeddings: bool = True

    # Numerics
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.rotation_size is not None:
            if self.rotation_size <= 0:
                raise ValueError(f"rotation_size must be positive, got {self.rotation_size}")
            if not is_pow2(self.rotation_size):
                raise ValueError(
                    f"rotation_size must be a power of 2, got {self.rotation_size}. "
                    f"Valid values: 16, 32, 64, 128, 256, 512, 1024, ..."
                )


class TrainableRMSNorm(nn.Module):
    """
    RMSNorm wrapper with trainable ``smooth_values`` for joint
    SpinQuant + SmoothQuant.

    Original RMSNorm::

        output = x / RMS(x) * gamma

    Trainable version::

        output = x / RMS(x) * gamma * smooth_values

    The ``smooth_values`` (diagonal scaling ``D``) are learned jointly
    with rotation matrices to minimise quantisation error.
    """

    def __init__(self, original_norm: nn.Module, trainable: bool = True):
        super().__init__()
        self.original_norm = original_norm
        self.trainable = trainable

        if hasattr(original_norm, "weight"):
            shape = original_norm.weight.shape
            self.smooth_values = nn.Parameter(
                torch.ones(shape, device=original_norm.weight.device, dtype=original_norm.weight.dtype),
                requires_grad=trainable,
            )
        else:
            self.smooth_values = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.original_norm(hidden_states)
        if self.smooth_values is not None:
            out = out * self.smooth_values
        return out


class SpinQuantPreprocessor:
    """
    SpinQuant preprocessor following AutoRound's transform conventions.

    After preprocessing the model is mathematically equivalent to the
    original but with weight distributions better suited for quantisation.
    """

    def __init__(self, model: nn.Module, config: Optional[SpinQuantConfig] = None):
        self.model = model
        self.config = config or SpinQuantConfig()

        # Architecture metadata
        info = get_model_arch_info(model)
        self.hidden_size = info.get("hidden_size", 0)
        self.head_dim = info.get("head_dim", 0)
        self.num_q_heads = info.get("num_q_heads", 0)
        self.num_kv_heads = info.get("num_kv_heads", 0)
        self.intermediate_size = info.get("intermediate_size", 0)

        # Resolve effective rotation sizes based on config.rotation_size
        # R1: rotation_size or hidden_size
        self.r1_rotation_size = self.config.rotation_size or self.hidden_size
        # R2: always head_dim (not affected by rotation_size)
        self.r2_rotation_size = self.head_dim
        # R3: always head_dim (not affected by rotation_size, same as Quark)
        self.r3_rotation_size = self.head_dim
        # R4: rotation_size or intermediate_size
        self.r4_rotation_size = self.config.rotation_size or self.intermediate_size

        # Training state
        self.rotation_params: list[nn.Parameter] = []
        self.smooth_params: list[nn.Parameter] = []

        # Fusion deduplication
        self._rotated_modules: set[nn.Module] = set()

        # Hook handles (managed by inplace sub-package)
        self._hook_handles: list[Any] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def preprocess(self, dataloader: Optional[Any] = None) -> nn.Module:
        logger.info("[SpinQuant] Starting preprocessing...")
        logger.info(
            f"[SpinQuant] Model architecture info: hidden_size={self.hidden_size}, "
            f"head_dim={self.head_dim}, num_q_heads={self.num_q_heads}, "
            f"num_kv_heads={self.num_kv_heads}, intermediate_size={self.intermediate_size}"
        )
        if self.config.rotation_size is not None:
            logger.info(
                f"[SpinQuant] Custom rotation_size={self.config.rotation_size} → "
                f"R1 size={self.r1_rotation_size}, R4 size={self.r4_rotation_size} "
                f"(R2/R3 always use head_dim={self.head_dim})"
            )
        logger.info(
            f"[SpinQuant] Rotation config: R1={self.config.r1}, R2={self.config.r2}, "
            f"R3={self.config.r3}, R4={self.config.r4}, "
            f"online_r1={self.config.online_r1_rotation}, "
            f"trainable_rotation={self.config.trainable_rotation}, "
            f"trainable_smooth={self.config.trainable_smooth}"
        )

        # Validate dimensions for enabled rotations
        self._validate_dimensions()

        # Step 1: untie embeddings (only needed for offline R1 — it rotates embed_tokens)
        if self.config.untie_embeddings and not self.config.online_r1_rotation:
            if untie_word_embeddings_if_needed(self.model):
                logger.info("[SpinQuant] Untied input/output embeddings")
        elif self.config.online_r1_rotation:
            logger.info("[SpinQuant] Online R1: skipping untie embeddings (embed_tokens unchanged)")

        # Step 2: fuse RMSNorm gamma into linear weights
        # Online R1 does NOT fuse RMSNorm (matching Quark's behavior):
        # the rotation is local per-module, so gamma doesn't need to commute.
        if self.config.fuse_rmsnorm and not self.config.online_r1_rotation:
            logger.info("[SpinQuant] Fusing RMSNorm parameters into linear weights...")
            fuse_rmsnorm_in_model(self.model)
        elif self.config.online_r1_rotation:
            logger.info("[SpinQuant] Online R1: skipping RMSNorm fusion (not needed)")

        # Step 3: replace RMSNorm with TrainableRMSNorm (SmoothQuant)
        if self.config.trainable_smooth:
            logger.info("[SpinQuant] Adding trainable smooth values...")
            self._replace_norms_with_trainable()

        # Step 4: initialise rotation matrices
        logger.info("[SpinQuant] Initialising rotation matrices...")
        self._init_rotation_matrices()

        # Step 5: train if requested
        if self.config.trainable_rotation or self.config.trainable_smooth:
            if dataloader is None:
                raise ValueError("dataloader required when trainable=True")
            logger.info(f"[SpinQuant] Training for {self.config.iters} iterations...")
            self._train_rotations(dataloader)

        # Step 6: apply R1 rotation
        if self.config.r1 and self.config.online_r1_rotation:
            logger.info("[SpinQuant] Applying online R1 rotation (weight + wrapper)...")
            self._apply_online_r1()
            # Still fuse R2 and R4 offline
            self._fuse_r2_rotation()
            self._fuse_r4_rotation()
        else:
            logger.info("[SpinQuant] Fusing offline rotations into weights...")
            self._fuse_offline_rotations()

        # Step 7: register online hooks (R3 / R4)
        if self.config.r3 or self.config.r4:
            logger.info("[SpinQuant] Registering online rotation hooks...")
            self._hook_handles = register_spinquant_hooks(
                self.model, self.config,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
                r4_rotation_size=self.r4_rotation_size,
            )

        # Step 8: cleanup training artefacts
        self._cleanup()

        # Print per-layer transformation summary table
        self._print_transformation_summary()

        logger.info("[SpinQuant] Preprocessing complete!")
        return self.model

    def _validate_dimensions(self) -> None:
        """Validate dimension requirements and disable rotations that can't work."""
        # R1: check r1_rotation_size divides hidden_size and is power of 2
        if self.config.r1 and self.r1_rotation_size > 0:
            if not is_pow2(self.r1_rotation_size):
                logger.warning(
                    f"[SpinQuant] R1 rotation_size must be a power of 2, "
                    f"but got {self.r1_rotation_size}. Disabling R1."
                )
                self.config.r1 = False
            elif self.hidden_size % self.r1_rotation_size != 0:
                logger.warning(
                    f"[SpinQuant] R1 rotation_size={self.r1_rotation_size} must divide "
                    f"hidden_size={self.hidden_size}. Disabling R1."
                )
                self.config.r1 = False

        if self.config.r2 and self.head_dim > 0 and not is_pow2(self.head_dim):
            logger.warning(
                f"[SpinQuant] R2 requires head_dim to be a power of 2, "
                f"but got head_dim={self.head_dim}. Disabling R2."
            )
            self.config.r2 = False

        if self.config.r3 and self.head_dim > 0 and not is_pow2(self.head_dim):
            logger.warning(
                f"[SpinQuant] R3 requires head_dim to be a power of 2, "
                f"but got head_dim={self.head_dim}. Disabling R3."
            )
            self.config.r3 = False

        if self.config.r3 and self.config.rotation_size is not None:
            logger.warning(
                f"[SpinQuant] R3 does not support custom rotation_size "
                f"(always uses head_dim={self.head_dim}). Ignoring rotation_size for R3."
            )

        if self.config.r4 and self.r4_rotation_size > 0:
            # For R4: find K = largest pow2 factor of r4_rotation_size
            inter = self.r4_rotation_size
            K = 1
            while K * 2 <= inter and inter % (K * 2) == 0:
                K *= 2
            if K <= 1:
                logger.warning(
                    f"[SpinQuant] R4 requires rotation size divisible by a power of 2 > 1, "
                    f"but r4_rotation_size={inter} has no such factor. Disabling R4."
                )
                self.config.r4 = False
            elif self.intermediate_size % self.r4_rotation_size != 0:
                logger.warning(
                    f"[SpinQuant] R4 rotation_size={self.r4_rotation_size} must divide "
                    f"intermediate_size={self.intermediate_size}. Disabling R4."
                )
                self.config.r4 = False
            else:
                logger.info(
                    f"[SpinQuant] R4 block Hadamard: K={K}, "
                    f"blocks={inter // K} "
                    f"(r4_rotation_size={inter}, intermediate_size={self.intermediate_size})"
                )

    # ------------------------------------------------------------------
    # Step 3: Trainable RMSNorm
    # ------------------------------------------------------------------
    def _replace_norms_with_trainable(self) -> None:
        """Replace RMSNorm modules with TrainableRMSNorm wrappers."""
        layers = list(self._get_layers())
        if not layers:
            return

        for layer in layers:
            if hasattr(layer, "input_layernorm"):
                layer.input_layernorm = TrainableRMSNorm(
                    layer.input_layernorm, trainable=self.config.trainable_smooth
                )
                if self.config.trainable_smooth:
                    self.smooth_params.append(layer.input_layernorm.smooth_values)

            if hasattr(layer, "post_attention_layernorm"):
                layer.post_attention_layernorm = TrainableRMSNorm(
                    layer.post_attention_layernorm, trainable=self.config.trainable_smooth
                )
                if self.config.trainable_smooth:
                    self.smooth_params.append(layer.post_attention_layernorm.smooth_values)

        # Final norm
        final_norm = None
        for path in ("model.norm", "norm"):
            parts = path.split(".")
            obj = self.model
            for p in parts:
                if not hasattr(obj, p):
                    break
                obj = getattr(obj, p)
            else:
                final_norm = obj
                break

        if final_norm is not None:
            wrapped = TrainableRMSNorm(final_norm, trainable=self.config.trainable_smooth)
            # Replace on model
            for path in ("model.norm", "norm"):
                parts = path.split(".")
                if len(parts) == 1:
                    if hasattr(self.model, parts[0]):
                        setattr(self.model, parts[0], wrapped)
                        break
                elif len(parts) == 2:
                    parent = getattr(self.model, parts[0], None)
                    if parent is not None and hasattr(parent, parts[1]):
                        setattr(parent, parts[1], wrapped)
                        break
            if self.config.trainable_smooth:
                self.smooth_params.append(wrapped.smooth_values)

    # ------------------------------------------------------------------
    # Step 4: Initialise rotation matrices
    # ------------------------------------------------------------------
    def _init_rotation_matrices(self) -> None:
        # Use the model's actual device, not config.device, to avoid mismatch
        model_device = next(self.model.parameters()).device
        dtype = self.config.dtype

        # R1: r1_rotation_size x r1_rotation_size
        if self.config.r1:
            r1_size = self.r1_rotation_size
            if self.config.trainable_rotation and not self.config.online_r1_rotation:
                R1 = nn.Parameter(torch.eye(r1_size, device=model_device, dtype=dtype))
                logger.info(f"[SpinQuant] R1: Trainable rotation matrix [{r1_size}×{r1_size}] (identity init)")
            else:
                R1 = nn.Parameter(
                    random_hadamard_matrix(r1_size, dtype=dtype, device=model_device),
                    requires_grad=False,
                )
                if r1_size < self.hidden_size:
                    logger.info(
                        f"[SpinQuant] R1: Random Hadamard [{r1_size}×{r1_size}] block rotation "
                        f"({self.hidden_size // r1_size} blocks, fixed, offline fuse)"
                    )
                else:
                    logger.info(f"[SpinQuant] R1: Random Hadamard [{r1_size}×{r1_size}] (fixed, offline fuse)")
            self._register_rotation("spinquant_R1", R1)

        # R2_head: head_dim x head_dim
        if self.config.r2 and self.head_dim > 0:
            if self.config.trainable_rotation:
                R2_head = nn.Parameter(torch.eye(self.head_dim, device=model_device, dtype=dtype))
                logger.info(f"[SpinQuant] R2: Trainable per-head rotation [{self.head_dim}×{self.head_dim}] (identity init)")
            else:
                R2_head = nn.Parameter(
                    random_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device),
                    requires_grad=False,
                )
                logger.info(f"[SpinQuant] R2: Random Hadamard [{self.head_dim}×{self.head_dim}] per head (fixed, offline fuse)")
            self._register_rotation("spinquant_R2_head", R2_head)

        # R3_head: fixed Hadamard (online, not trainable)
        if self.config.r3 and self.head_dim > 0:
            R3 = random_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device)
            self.model.register_buffer("spinquant_R3_head", R3)
            logger.info(f"[SpinQuant] R3: Online Hadamard [{self.head_dim}×{self.head_dim}] after RoPE (monkeypatch)")

        # R4: r4_rotation_size Hadamard (online buffer)
        if self.config.r4 and self.r4_rotation_size > 0:
            # Find the largest K that divides r4_rotation_size
            r4_size = self.r4_rotation_size
            K = 1
            while K * 2 <= r4_size and r4_size % (K * 2) == 0:
                K *= 2
            if K > 1:
                had_K = deterministic_hadamard_matrix(K, dtype=dtype, device=model_device)
                self.model.register_buffer("spinquant_R4_had_K", had_K)
                self.model.register_buffer("spinquant_R4_K", torch.tensor(K, device=model_device))
                if r4_size < self.intermediate_size:
                    logger.info(
                        f"[SpinQuant] R4: Block Hadamard K={K}, "
                        f"{r4_size // K} blocks within rotation_size={r4_size}, "
                        f"{self.intermediate_size // r4_size} rotation blocks on down_proj"
                    )
                else:
                    logger.info(
                        f"[SpinQuant] R4: Block Hadamard K={K}, "
                        f"{r4_size // K} blocks on down_proj (offline fuse + online hook)"
                    )

    def _register_rotation(self, name: str, param: nn.Parameter) -> None:
        self.model.register_parameter(name, param)
        if param.requires_grad:
            self.rotation_params.append(param)

    # ------------------------------------------------------------------
    # Step 5: Training
    # ------------------------------------------------------------------
    def _train_rotations(self, dataloader: Any) -> None:
        """Simple embedded training loop (no callbacks/checkpointing).

        For advanced training features (callbacks, evaluation, checkpointing,
        custom loss), use ``RotationTrainer`` from ``trainer.py`` directly.
        This method provides a minimal training path for the one-call
        ``preprocess()`` API.
        """
        if not (self.rotation_params or self.smooth_params):
            logger.info("[SpinQuant] No trainable parameters, skipping training")
            return

        n_rot = len(self.rotation_params)
        n_smooth = len(self.smooth_params)
        total_params = sum(p.numel() for p in self.rotation_params) + sum(p.numel() for p in self.smooth_params)
        logger.info(
            f"[SpinQuant] Training: {n_rot} rotation params + {n_smooth} smooth params "
            f"= {total_params:,} trainable parameters"
        )
        for p in self.rotation_params:
            logger.debug(f"  Trainable rotation: {p.shape} on {p.device}")

        # Use model's current device for training
        model_device = next(self.model.parameters()).device

        # Dual optimiser: Adam for smooth values, SGDG (Cayley) for rotations
        optimizer = AdamAndSGDG(
            adam_params=self.smooth_params,
            sgdg_params=self.rotation_params,
            learning_rate=self.config.lr,
            smooth_learning_rate=self.config.smooth_lr,
        )

        # Clone original model for KL reference
        logger.info("[SpinQuant] Cloning original model for KL reference...")
        original_model = copy.deepcopy(self.model)
        original_model.eval()
        for p in original_model.parameters():
            p.requires_grad = False
        # Remove hooks from the clone so it runs as the un-rotated baseline
        remove_spinquant_hooks_from_model(original_model)

        self.model.train()
        step = 0
        loss_history: list[float] = []

        for batch in dataloader:
            if step >= self.config.iters:
                break

            # Move batch to model device
            if hasattr(batch, "to"):
                batch = batch.to(model_device)
            elif isinstance(batch, dict):
                batch = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in batch.items()}

            # Forward: rotated model
            out_rot = self.model(**batch)
            logits = out_rot.logits if hasattr(out_rot, "logits") else out_rot

            # Forward: original model (no grad)
            with torch.no_grad():
                out_ori = original_model(**batch)
                ori_logits = out_ori.logits if hasattr(out_ori, "logits") else out_ori

            # Loss
            loss = self._compute_loss(logits, ori_logits)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_history.append(loss.item())
            if step % 50 == 0:
                avg = sum(loss_history[-50:]) / len(loss_history[-50:]) if loss_history else 0.0
                logger.info(f"[SpinQuant] Step {step}/{self.config.iters}, loss={loss.item():.6f} (avg={avg:.6f})")

            step += 1

        # Final orthogonality check
        self._check_orthogonality()

        del original_model
        torch.cuda.empty_cache()
        self.model.eval()

    def _compute_loss(self, logits: torch.Tensor, ori_logits: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "kl_top":
            k = min(self.config.kl_top_k, logits.size(-1))
            top_ori, indices = ori_logits.topk(k, dim=-1, sorted=False)
            top_logits = logits.gather(-1, indices)
            return F.kl_div(
                F.log_softmax(top_logits.flatten(0, -2), dim=-1),
                F.softmax(top_ori.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        if self.config.loss_type == "kl_full":
            return F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
        if self.config.loss_type == "mse":
            return F.mse_loss(logits, ori_logits)
        raise ValueError(f"Unknown loss_type={self.config.loss_type!r}")

    def _check_orthogonality(self) -> None:
        max_dev = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or "spinquant_R" not in name:
                continue
            R = param.data
            if R.dim() != 2 or R.shape[0] != R.shape[1] or R.numel() == 0:
                continue
            I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
            dev = (torch.matmul(R, R.t()) - I).abs().max().item()
            max_dev = max(max_dev, dev)
            if dev > 1e-4:
                logger.warning(f"  {name} orthogonality deviation={dev:.2e}")
        if max_dev > 0:
            logger.info(f"[SpinQuant] Max orthogonality deviation={max_dev:.2e}")
            if max_dev > 1e-3:
                logger.warning("  Orthogonality constraint significantly violated!")

    def _get_embed_tokens(self) -> Optional[nn.Module]:
        """Get embedding module, supporting both model.embed_tokens and model.model.embed_tokens."""
        for attr_path in ("embed_tokens", "model.embed_tokens"):
            parts = attr_path.split(".")
            obj = self.model
            for p in parts:
                if not hasattr(obj, p):
                    break
                obj = getattr(obj, p)
            else:
                return obj
        return None

    def _get_layers(self):
        """Yield transformer layers, supporting multiple nesting patterns."""
        for attr_path in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
            parts = attr_path.split(".")
            obj = self.model
            for p in parts:
                if not hasattr(obj, p):
                    break
                obj = getattr(obj, p)
            else:
                if hasattr(obj, "__iter__"):
                    for layer in obj:
                        yield layer
                    return
        # Fallback: search recursively
        for name, module in self.model.named_modules():
            if name.endswith(".layers") or name == "layers":
                if hasattr(module, "__iter__"):
                    for layer in module:
                        yield layer
                    return

    def _get_lm_head(self) -> Optional[nn.Module]:
        """Get LM head module."""
        return getattr(self.model, "lm_head", None)

    # ------------------------------------------------------------------
    # Step 6a: Online R1 rotation (matching Quark's default behavior)
    # ------------------------------------------------------------------
    def _apply_online_r1(self) -> None:
        """Apply online R1 rotation: rotate target module weights and wrap
        them with ``InputRotationWrapperHadamard`` so the matching activation
        rotation is applied at runtime.

        This matches Quark's ``apply_online_r1()`` behavior:
        - Target modules (q/k/v_proj, gate/up_proj) get ``W_new = matmul_hadU(W)``
          on their input channels (last dim of weight matrix)
        - The module is replaced in the model tree with an
          ``InputRotationWrapperHadamard`` that applies the same Hadamard to
          activations at runtime
        - The two transforms cancel out: ``H(x) @ (W @ H).T = x @ W.T``
        - prev_modules (embed_tokens, o_proj, down_proj) are NOT modified
        - RMSNorm gamma is NOT fused
        - lm_head is NOT modified (last_layer skipped, matching Quark)

        Unlike the old hook-based approach, the wrapper is a proper nn.Module
        that is serialised by ``save_pretrained`` / ``state_dict``, enabling
        model save/load with online R1 rotation.
        """
        r1_size = self.r1_rotation_size

        # Pre-compute the Hadamard matrix for the rotation
        hadamard_K, K = get_hadamard_K(r1_size)
        model_device = next(self.model.parameters()).device
        hadamard_K = hadamard_K.to(model_device)

        n_rotated = 0
        n_wrapped = 0

        for layer in self._get_layers():
            if not (hasattr(layer, "self_attn") and hasattr(layer, "mlp")):
                continue

            layer_device = next(layer.parameters()).device
            had_K_local = hadamard_K.to(layer_device)

            attn = layer.self_attn
            mlp = layer.mlp

            # Target modules: (parent_module, attr_name)
            target_specs = []
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                if hasattr(attn, proj_name):
                    target_specs.append((attn, proj_name))
            for proj_name in ("gate_proj", "up_proj"):
                if hasattr(mlp, proj_name):
                    target_specs.append((mlp, proj_name))

            for parent, attr_name in target_specs:
                module = getattr(parent, attr_name)
                dtype = module.weight.data.dtype
                in_features = module.weight.shape[-1]

                if r1_size == in_features:
                    # Full rotation via matmul_hadU (butterfly algorithm)
                    module.weight.data = matmul_hadU(
                        module.weight.data, hadamard_K=had_K_local, K=K
                    ).to(dtype)
                elif in_features % r1_size == 0:
                    # Block rotation: get_hadamard_K already returns normalized
                    # (orthogonal) matrix — no extra division needed.
                    R_block = had_K_local.to(torch.float64)
                    if R_block.shape[0] != r1_size:
                        had_1, _ = get_hadamard_K(r1_size // K)
                        R_block = torch.kron(had_K_local.to(torch.float64), had_1.to(layer_device, torch.float64))
                    rotate_in_channels_(module, R_in=R_block)
                else:
                    raise ValueError(
                        f"Online R1: in_features={in_features} not compatible "
                        f"with r1_rotation_size={r1_size}"
                    )
                n_rotated += 1

                # Replace module with InputRotationWrapperHadamard
                wrapper = InputRotationWrapperHadamard(
                    original_module=module,
                    rotation_size=r1_size,
                    hadamard_K=had_K_local,
                    K=K,
                )
                setattr(parent, attr_name, wrapper)
                n_wrapped += 1

        logger.info(
            f"[SpinQuant] Online R1: rotated {n_rotated} target modules, "
            f"wrapped {n_wrapped} with InputRotationWrapperHadamard "
            f"(rotation_size={r1_size}, lm_head/embed_tokens/o_proj/down_proj unchanged)"
        )

    # ------------------------------------------------------------------
    # Step 6b: Fuse offline rotations
    # ------------------------------------------------------------------
    def _fuse_offline_rotations(self) -> None:
        self._rotated_modules.clear()

        if not self.config.r1:
            # Even without R1, we may still need R2 and R4
            self._fuse_r2_rotation()
            self._fuse_r4_rotation()
            return

        R1 = self._get_rotation_tensor("spinquant_R1")
        if R1 is None or R1.numel() == 0:
            return
        R1_inv = R1.t()
        r1_size = R1.shape[0]

        # Embed tokens (output rotation: W_embed @ R1)
        embed = self._get_embed_tokens()
        if embed is not None:
            with torch.no_grad():
                W_f64 = embed.weight.data.to(torch.float64)
                R_f64 = R1.to(embed.weight.device).to(torch.float64)
                if W_f64.shape[-1] == r1_size:
                    new_w = torch.matmul(W_f64, R_f64)
                elif W_f64.shape[-1] % r1_size == 0:
                    # Block rotation for embedding
                    w_reshaped = W_f64.reshape(*W_f64.shape[:-1], -1, r1_size)
                    new_w = (w_reshaped @ R_f64).reshape(W_f64.shape)
                else:
                    raise ValueError(
                        f"embed_tokens dim={W_f64.shape[-1]} not divisible by R1 size={r1_size}"
                    )
                embed.weight.data = new_w.to(embed.weight.dtype)

        # Transformer layers
        n_layers = 0
        for layer in self._get_layers():
            if not (hasattr(layer, "self_attn") and hasattr(layer, "mlp")):
                continue
            attn = layer.self_attn
            mlp = layer.mlp

            # Ensure R1_inv is on the same device as layer weights
            layer_device = next(layer.parameters()).device
            R1_inv_local = R1_inv.to(layer_device)
            R1_local = R1.to(layer_device)

            # Attention: in-channel uses R1_inv (→ W @ R1), out-channel uses R1 (→ R1_inv @ W)
            if hasattr(attn, "q_proj"):
                rotate_in_channels_(attn.q_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(attn, "k_proj"):
                rotate_in_channels_(attn.k_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(attn, "v_proj"):
                rotate_in_channels_(attn.v_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(attn, "o_proj"):
                rotate_out_channels_(attn.o_proj, R_out=R1_local, rotated_modules=self._rotated_modules)

            # MLP: same convention
            if hasattr(mlp, "gate_proj"):
                rotate_in_channels_(mlp.gate_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(mlp, "up_proj"):
                rotate_in_channels_(mlp.up_proj, R_in=R1_inv_local, rotated_modules=self._rotated_modules)
            if hasattr(mlp, "down_proj"):
                rotate_out_channels_(mlp.down_proj, R_out=R1_local, rotated_modules=self._rotated_modules)
            n_layers += 1

        logger.info(f"[SpinQuant] R1 fused into {n_layers} layers (embed_tokens, q/k/v/o_proj, gate/up/down_proj, lm_head)")

        # LM head: in-channel uses R1_inv (→ W @ R1)
        lm_head = self._get_lm_head()
        if lm_head is not None:
            lm_device = lm_head.weight.device
            rotate_in_channels_(lm_head, R_in=R1_inv.to(lm_device), rotated_modules=self._rotated_modules)

        # R2 head-dim rotation (per-head Hadamard on v_proj output + o_proj input)
        self._fuse_r2_rotation()

        # R4 offline fusion (Hadamard on down_proj input side)
        self._fuse_r4_rotation()

    def _fuse_r2_rotation(self) -> None:
        """Fuse R2 per-head rotation into v_proj and o_proj.

        Following the reference implementation, R2 applies a per-head Hadamard
        to v_proj's output channels and o_proj's input channels. This is done
        using block-diagonal Hadamard application (had_dim=head_dim).
        """
        if not self.config.r2 or self.head_dim <= 0:
            return

        R2_head = self._get_rotation_tensor("spinquant_R2_head")
        if R2_head is None:
            return

        n_fused = 0
        for layer in self._get_layers():
            if not hasattr(layer, "self_attn"):
                continue
            attn = layer.self_attn

            # v_proj: apply per-head rotation on output (each head_dim chunk of rows)
            if hasattr(attn, "v_proj"):
                apply_hadamard_to_linear(attn.v_proj, had_dim=self.head_dim, output=True)

            # o_proj: apply per-head rotation on input (each head_dim chunk of columns)
            if hasattr(attn, "o_proj"):
                apply_hadamard_to_linear(attn.o_proj, had_dim=self.head_dim, output=False)
            n_fused += 1

        logger.info(f"[SpinQuant] R2 fused into {n_fused} layers (v_proj out + o_proj in, head_dim={self.head_dim})")

    def _fuse_r4_rotation(self) -> None:
        """Fuse R4 Hadamard into down_proj's input side.

        Applies the same block Hadamard that the online hook uses. The hook
        computes x.view(-1, M, K) @ H_K.T where K is the largest power-of-2
        dividing r4_rotation_size. The offline fusion applies the matching
        transform to the weight so they cancel: (x@H_block) @ (W@H_block).T = x@W.T.
        """
        if not self.config.r4 or self.r4_rotation_size <= 0:
            return

        # Compute K: same logic as the hook in inplace/apply.py
        r4_size = self.r4_rotation_size
        K = 1
        while K * 2 <= r4_size and r4_size % (K * 2) == 0:
            K *= 2

        if K <= 1:
            return

        n_fused = 0
        for layer in self._get_layers():
            if not hasattr(layer, "mlp"):
                continue
            mlp = layer.mlp
            if hasattr(mlp, "down_proj"):
                apply_hadamard_to_linear(mlp.down_proj, had_dim=K, output=False)
                n_fused += 1

        logger.info(
            f"[SpinQuant] R4 offline fused into {n_fused} down_proj layers "
            f"(block Hadamard K={K}, r4_rotation_size={r4_size})"
        )

    # ------------------------------------------------------------------
    # Step 8: Cleanup
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        self.model.eval()

        # Remove fused offline rotation *Parameters* (not buffers)
        for name in list(self.model._parameters.keys()):
            if name.startswith("spinquant_R") and name in self.model._parameters:
                if name not in self.model._buffers:
                    delattr(self.model, name)

        # NOTE: Do NOT remove online hooks/monkeypatches here. R3/R4 hooks
        # must persist for correct inference. Only remove training-related state.

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Clear internal tracking
        self.rotation_params.clear()
        self.smooth_params.clear()
        self._rotated_modules.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_rotation_tensor(self, name: str) -> Optional[torch.Tensor]:
        if hasattr(self.model, name):
            tensor = getattr(self.model, name)
            if isinstance(tensor, (nn.Parameter, torch.Tensor)):
                return tensor.data if isinstance(tensor, nn.Parameter) else tensor
        return None

    def _print_transformation_summary(self) -> None:
        """Print a per-layer table summarizing all applied transformations and hooks."""
        lines = []
        lines.append("")
        lines.append("=" * 100)
        lines.append("SpinQuant Transformation Summary")
        lines.append("=" * 100)

        # --- Global transforms ---
        lines.append("")
        lines.append("Global Transforms:")
        lines.append(f"  {'Component':<30} {'Transform':<50} {'Status'}")
        lines.append(f"  {'-'*30} {'-'*50} {'-'*10}")

        # Untie embeddings
        tied = getattr(self.model.config, "tie_word_embeddings", False) if hasattr(self.model, "config") else "unknown"
        if self.config.online_r1_rotation:
            lines.append(f"  {'embed_tokens / lm_head':<30} {'Untie word embeddings (skipped: online R1)':<50} {'- n/a'}")
        else:
            lines.append(f"  {'embed_tokens / lm_head':<30} {'Untie word embeddings':<50} {'✓ untied' if not tied else '✗ still tied'}")

        # RMSNorm fusion
        if self.config.online_r1_rotation:
            lines.append(f"  {'All RMSNorm layers':<30} {'Fuse gamma (skipped: online R1)':<50} {'- n/a'}")
        else:
            lines.append(f"  {'All RMSNorm layers':<30} {'Fuse gamma into linear weights':<50} {'✓ fused' if self.config.fuse_rmsnorm else '✗ skipped'}")

        # R1
        if self.config.r1:
            r1_size = self.r1_rotation_size
            mode = "online" if self.config.online_r1_rotation else "offline"
            if r1_size < self.hidden_size:
                r1_desc = f"Block Hadamard {r1_size}×{r1_size}, {self.hidden_size // r1_size} blocks ({mode})"
            else:
                r1_desc = f"Hadamard {r1_size}×{r1_size} ({mode})"
        else:
            r1_desc = "Disabled"
        lines.append(f"  {'Residual stream (R1)':<30} {r1_desc:<50} {'✓' if self.config.r1 else '✗'}")

        # --- Per-layer table ---
        lines.append("")
        lines.append("Per-Layer Transforms:")

        # Determine R4 K
        r4_K = 0
        if self.config.r4 and self.r4_rotation_size > 0:
            K = 1
            r4_size = self.r4_rotation_size
            while K * 2 <= r4_size and r4_size % (K * 2) == 0:
                K *= 2
            if K > 1:
                r4_K = K

        # Table header
        col_layer = "Layer"
        col_r1 = "R1 (online)" if self.config.online_r1_rotation else "R1 (weight fuse)"
        col_r2 = "R2 (weight fuse)"
        col_r3 = "R3 (online hook)"
        col_r4 = "R4 (online hook)"
        header = f"  {col_layer:<35} {col_r1:<20} {col_r2:<20} {col_r3:<22} {col_r4:<22}"
        lines.append(header)
        lines.append(f"  {'-'*35} {'-'*20} {'-'*20} {'-'*22} {'-'*22}")

        online_r1 = self.config.online_r1_rotation

        # embed_tokens
        if online_r1:
            r1_embed = "-" if self.config.r1 else "-"  # online R1 doesn't touch embed
        else:
            r1_embed = f"W@R ({self.r1_rotation_size})" if self.config.r1 else "-"
        lines.append(f"  {'model.embed_tokens':<35} {r1_embed:<20} {'-':<20} {'-':<22} {'-':<22}")

        # Transformer layers
        layers = list(self._get_layers())
        n_layers = sum(1 for l in layers if hasattr(l, "self_attn") and hasattr(l, "mlp"))

        for i, layer in enumerate(layers):
            if not (hasattr(layer, "self_attn") and hasattr(layer, "mlp")):
                continue

            # Only show first 2 and last layer to keep output concise
            if n_layers > 5 and 2 <= i < n_layers - 1:
                if i == 2:
                    lines.append(f"  {'  ... (same pattern for all layers)':<35}")
                continue

            layer_name = f"layers.{i}"

            if online_r1 and self.config.r1:
                r1_attn = "q/k/v:W@H+hook"
                r1_mlp = "g/u:W@H+hook"
            elif self.config.r1:
                r1_attn = "q/k/v:R⁻¹@W o:W@R"
                r1_mlp = "g/u:R⁻¹@W d:W@R"
            else:
                r1_attn = "-"
                r1_mlp = "-"

            # R2 for this layer: affects v_proj out, o_proj in
            r2_status = f"H({self.head_dim}) v↔o" if self.config.r2 else "-"

            # R3: check if monkeypatch applied (look for wrapper in self_attn)
            r3_status = "-"
            if self.config.r3 and self.head_dim > 0 and is_pow2(self.head_dim):
                r3_status = f"H({self.head_dim}) Q,K post-RoPE"

            # R4: check if hook on down_proj
            r4_status = "-"
            if self.config.r4 and r4_K > 0:
                n_blocks = self.r4_rotation_size // r4_K
                r4_status = f"blockH(K={r4_K},b={n_blocks})"

            # Print attention row
            lines.append(f"  {layer_name + '.self_attn':<35} {r1_attn:<20} {r2_status:<20} {r3_status:<22} {'-':<22}")
            # Print MLP row
            lines.append(f"  {layer_name + '.mlp':<35} {r1_mlp:<20} {'-':<20} {'-':<22} {r4_status:<22}")

        # lm_head
        if online_r1:
            r1_lm = "-" if self.config.r1 else "-"  # online R1 doesn't touch lm_head
        else:
            r1_lm = f"R⁻¹@W ({self.r1_rotation_size})" if self.config.r1 else "-"
        lines.append(f"  {'model.lm_head':<35} {r1_lm:<20} {'-':<20} {'-':<22} {'-':<22}")

        # --- Hook summary ---
        lines.append("")
        lines.append("Registered Hooks:")
        r3_hooks = sum(1 for h in self._hook_handles if isinstance(h, tuple) and h[0] == "r3_monkeypatch")
        # For online R1, hooks are torch.utils.hooks.RemovableHook objects, not tuples
        r1_hooks = 0
        r4_hooks = 0
        for h in self._hook_handles:
            if isinstance(h, tuple) and h[0] == "r3_monkeypatch":
                continue
            elif isinstance(h, tuple):
                r4_hooks += 1
            else:
                # RemovableHook from register_forward_pre_hook (R1 online or R4)
                if online_r1 and self.config.r1:
                    r1_hooks += 1
                else:
                    r4_hooks += 1
        if online_r1 and self.config.r1:
            # Count more precisely: R1 hooks = 5 per layer (q/k/v + gate/up)
            r1_hooks = sum(1 for h in self._hook_handles
                          if not (isinstance(h, tuple) and h[0] == "r3_monkeypatch"))
            # R4 hooks registered separately via register_spinquant_hooks
            r4_hooks = 0
        lines.append(f"  R1 online hooks (Hadamard on target module input):          {r1_hooks} modules")
        lines.append(f"  R3 monkeypatch (apply_rotary_pos_emb → QKRotationWrapper):  {r3_hooks} attention layers")
        lines.append(f"  R4 forward_pre_hook (block Hadamard on down_proj input):    {r4_hooks} MLP layers")

        # --- Summary totals ---
        lines.append("")
        lines.append("Totals:")
        lines.append(f"  Transformer layers:   {n_layers}")
        if online_r1 and self.config.r1:
            lines.append(f"  Online R1 targets:    {n_layers}×(q/k/v_proj, gate/up_proj) = {r1_hooks} modules")
            fused_parts = []
        else:
            fused_parts = ["embed_tokens"]
            if self.config.r1:
                fused_parts.append(f"{n_layers}×(q/k/v/o_proj, gate/up/down_proj), lm_head")
        if self.config.r2:
            fused_parts.append(f"{n_layers}×(v_proj↔o_proj)")
        if self.config.r4 and r4_K > 0:
            fused_parts.append(f"{n_layers}×(down_proj)")
        lines.append(f"  Offline-fused params: {', '.join(fused_parts) if fused_parts else 'none'}")
        total_hooks = r1_hooks + r3_hooks + r4_hooks
        lines.append(f"  Online hooks:         {total_hooks} total ({r1_hooks} R1 + {r3_hooks} R3 + {r4_hooks} R4)")
        lines.append(f"  Inference overhead:   R1={'O(seq×hidden×log₂H) per module' if r1_hooks > 0 else 'none'}")
        lines.append(f"                        R3={'O(seq×heads×d_head×log₂d_head) per layer' if r3_hooks > 0 else 'none'}")
        lines.append(f"                        R4={'O(seq×inter×log₂K) per layer' if r4_hooks > 0 else 'none'}")
        lines.append("=" * 100)

        # Log as a single multi-line message
        logger.info("\n".join(lines))


def remove_spinquant_hooks_from_model(model: nn.Module) -> None:
    """Remove all SpinQuant forward hooks / pre-hooks from a model."""
    for module in model.modules():
        if hasattr(module, "_forward_hooks"):
            for hook_id in list(module._forward_hooks.keys()):
                try:
                    del module._forward_hooks[hook_id]
                except Exception:
                    pass
        if hasattr(module, "_forward_pre_hooks"):
            for hook_id in list(module._forward_pre_hooks.keys()):
                try:
                    del module._forward_pre_hooks[hook_id]
                except Exception:
                    pass

