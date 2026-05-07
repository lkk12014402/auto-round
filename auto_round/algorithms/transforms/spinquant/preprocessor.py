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
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.transforms.spinquant.cayley_optimizer import AdamAndSGDG
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    apply_hadamard_to_linear,
    create_block_diag_from_head_matrix,
    deterministic_hadamard_matrix,
    fuse_rmsnorm_in_model,
    get_model_arch_info,
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


@dataclass
class SpinQuantConfig:
    """Configuration for SpinQuant preprocessing."""

    # Rotation dimensions
    r1: bool = True                    # R1: hidden_size rotation (offline fused)
    r2: bool = True                    # R2: head_dim rotation (offline fused)
    r3: bool = True                    # R3: Q/K online rotation
    r4: bool = True                    # R4: MLP activation online rotation

    # Training control
    trainable_rotation: bool = True    # Learn R via Cayley SGD (False = QuaRot fixed Hadamard)
    trainable_smooth: bool = True      # Learn smooth_values via Adam (joint SmoothQuant)
    online_r1_rotation: bool = False   # Keep R1 online (rarely used)

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
        print("[SpinQuant] Starting preprocessing...")

        # Step 1: untie embeddings
        if self.config.untie_embeddings:
            if untie_word_embeddings_if_needed(self.model):
                print("[SpinQuant] Untied input/output embeddings")

        # Step 2: fuse RMSNorm gamma into linear weights
        if self.config.fuse_rmsnorm:
            print("[SpinQuant] Fusing RMSNorm parameters...")
            fuse_rmsnorm_in_model(self.model)

        # Step 3: replace RMSNorm with TrainableRMSNorm (SmoothQuant)
        if self.config.trainable_smooth:
            print("[SpinQuant] Adding trainable smooth values...")
            self._replace_norms_with_trainable()

        # Step 4: initialise rotation matrices
        print("[SpinQuant] Initialising rotation matrices...")
        self._init_rotation_matrices()

        # Step 5: train if requested
        if self.config.trainable_rotation or self.config.trainable_smooth:
            if dataloader is None:
                raise ValueError("dataloader required when trainable=True")
            print(f"[SpinQuant] Training for {self.config.iters} iterations...")
            self._train_rotations(dataloader)

        # Step 6: fuse offline rotations into weights
        print("[SpinQuant] Fusing offline rotations into weights...")
        self._fuse_offline_rotations()

        # Step 7: register online hooks (R3 / R4)
        if self.config.r3 or self.config.r4:
            print("[SpinQuant] Registering online rotation hooks...")
            self._hook_handles = register_spinquant_hooks(
                self.model, self.config,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
            )

        # Step 8: cleanup training artefacts
        self._cleanup()

        print("[SpinQuant] Preprocessing complete!")
        return self.model

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

        # R1: hidden_size x hidden_size
        if self.config.r1:
            if self.config.trainable_rotation and not self.config.online_r1_rotation:
                R1 = nn.Parameter(torch.eye(self.hidden_size, device=model_device, dtype=dtype))
            else:
                R1 = nn.Parameter(
                    random_hadamard_matrix(self.hidden_size, dtype=dtype, device=model_device),
                    requires_grad=False,
                )
            self._register_rotation("spinquant_R1", R1)

        # R2_head: head_dim x head_dim
        if self.config.r2 and self.head_dim > 0:
            if self.config.trainable_rotation:
                R2_head = nn.Parameter(torch.eye(self.head_dim, device=model_device, dtype=dtype))
            else:
                R2_head = nn.Parameter(
                    random_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device),
                    requires_grad=False,
                )
            self._register_rotation("spinquant_R2_head", R2_head)

        # R3_head: fixed Hadamard (online, not trainable)
        if self.config.r3 and self.head_dim > 0:
            R3 = random_hadamard_matrix(self.head_dim, dtype=dtype, device=model_device)
            self.model.register_buffer("spinquant_R3_head", R3)

        # R4: intermediate_size Hadamard (online buffer)
        if self.config.r4 and self.intermediate_size > 0:
            # Find the largest K that divides intermediate_size
            K = 1
            while K * 2 <= self.intermediate_size and self.intermediate_size % (K * 2) == 0:
                K *= 2
            if K > 1:
                had_K = deterministic_hadamard_matrix(K, dtype=dtype, device=model_device)
                self.model.register_buffer("spinquant_R4_had_K", had_K)
                self.model.register_buffer("spinquant_R4_K", torch.tensor(K, device=model_device))

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
            print("[SpinQuant] No trainable parameters, skipping training")
            return

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
        print("[SpinQuant] Cloning original model for KL reference...")
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
                print(f"[SpinQuant] Step {step}/{self.config.iters}, loss={loss.item():.6f} (avg={avg:.6f})")

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
                print(f"  [WARN] {name} orthogonality deviation={dev:.2e}")
        if max_dev > 0:
            print(f"[SpinQuant] Max orthogonality deviation={max_dev:.2e}")
            if max_dev > 1e-3:
                print("  WARNING: orthogonality constraint significantly violated!")

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
    # Step 6: Fuse offline rotations
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

        # Embed tokens (output rotation: W_embed @ R1)
        embed = self._get_embed_tokens()
        if embed is not None:
            with torch.no_grad():
                W_f64 = embed.weight.data.to(torch.float64)
                R_f64 = R1.to(embed.weight.device).to(torch.float64)
                new_w = torch.matmul(W_f64, R_f64).to(embed.weight.dtype)
                embed.weight.data = new_w

        # Transformer layers
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

    def _fuse_r4_rotation(self) -> None:
        """Fuse R4 Hadamard into down_proj's input side.

        Following the reference: apply_exact_had_to_linear(mlp.down_proj, had_dim=-1, output=False)
        This means full Hadamard on the intermediate_size (input) dimension.
        The matching online hook applies Hadamard to the activation before down_proj.
        """
        if not self.config.r4 or self.intermediate_size <= 0:
            return

        for layer in self._get_layers():
            if not hasattr(layer, "mlp"):
                continue
            mlp = layer.mlp
            if hasattr(mlp, "down_proj"):
                apply_hadamard_to_linear(mlp.down_proj, had_dim=-1, output=False)

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

        # NOTE: Do NOT remove online hooks here. R3/R4 hooks must persist
        # for correct inference. Only remove training-related state.

        # Restore monkey-patched attention forward methods
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer in self.model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "_spinquant_original_forward"):
                    layer.self_attn.forward = layer.self_attn._spinquant_original_forward
                    delattr(layer.self_attn, "_spinquant_original_forward")

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

