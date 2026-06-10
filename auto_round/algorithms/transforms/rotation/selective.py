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
"""Selective Hadamard Transform — per-layer decision logic.

This module implements the core innovation of **Selective Hadamard Transform**:
instead of applying rotation to ALL Linear layers uniformly, it uses structural
priors and/or activation statistics to decide per-layer whether rotation is
beneficial.

Key insight: Hadamard rotation helps heavy-tailed, expand, nonlinearity-buffered
layers but HARMS residual-sensitive, compress, and semantically-aligned layers
(e.g., down_proj, o_proj, lm_head).

Two modes:
    - "structural": Zero-cost decision based on layer naming/structure patterns.
    - "auto": Statistics-based (kurtosis + energy concentration) with structural
      priors as guardrails. Requires a calibration forward pass.

Usage::

    from auto_round.algorithms.transforms.rotation.selective import LayerSelector

    selector = LayerSelector(config, model)
    # For "auto" mode:
    selector.profile(calibration_dataloader)
    # Query:
    should_rotate = selector.should_rotate("model.layers.0.mlp.up_proj")
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("autoround.selective_rotation")

__all__ = ["LayerSelector", "LayerDecision", "structural_decision", "compute_layer_score"]

# ═══════════════════════════════════════════════════════════════════════════════
# Logging verbosity control — set via environment variable:
#   SELECTIVE_ROTATION_LOG_LEVEL=DEBUG   → per-layer decision + stats
#   SELECTIVE_ROTATION_LOG_LEVEL=INFO    → summary only (default)
# ═══════════════════════════════════════════════════════════════════════════════
import os as _os

_LOG_LEVEL_STR = _os.environ.get("SELECTIVE_ROTATION_LOG_LEVEL", "INFO").upper()
if _LOG_LEVEL_STR == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif _LOG_LEVEL_STR == "WARNING":
    logger.setLevel(logging.WARNING)
else:
    logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
# Structural prior patterns
# ═══════════════════════════════════════════════════════════════════════════════

# Layers that should ALWAYS be skipped (structural harm is well-established).
STRUCTURAL_SKIP_PATTERNS: list[str] = [
    "*lm_head*",
    "*down_proj*",
    "*o_proj*",
    # MoE-specific: router/gate layers (tiny params, precision-sensitive, not quantized)
    "*mlp.gate",              # MoE router (e.g., Qwen3-MoE, DeepSeek)
    "*block_sparse_moe.gate", # MoE router (Mixtral style)
    "*shared_expert_gate*",   # shared expert gating weight
    # MoE expert layers: vLLM uses FusedMoE kernel for these, which does NOT
    # support online rotation. Rotating weights without online activation rotation
    # at inference would produce incorrect results.
    "*experts*gate_proj*",    # MoE expert gate_proj
    "*experts*up_proj*",      # MoE expert up_proj
    "*experts*down_proj*",    # MoE expert down_proj
    # Embeddings (not quantized, semantic alignment)
    "*embed_tokens*",
]

# Layers that are BEST CANDIDATES for Hadamard (strong benefit expected).
STRUCTURAL_PREFER_PATTERNS: list[str] = [
    "*up_proj*",
    "*gate_proj*",
    "*q_proj*",
    "*k_proj*",
]

# Layers with conditional benefit (may help or hurt depending on distribution).
# NOTE: v_proj is conditional because vLLM merges q/k/v into one qkv_proj.
# The vLLM plugin handles this via load-time weight compensation for the
# non-rotated partitions, so selective rotation of q/k without v is valid.
STRUCTURAL_CONDITIONAL_PATTERNS: list[str] = [
    "*v_proj*",
]


@dataclass
class LayerDecision:
    """Decision record for a single layer."""

    name: str
    enabled: bool
    reason: str
    score: Optional[float] = None
    kurtosis: Optional[float] = None
    ecr: Optional[float] = None
    is_expand: Optional[bool] = None
    is_compress: Optional[bool] = None
    is_residual_consumer: Optional[bool] = None


def structural_decision(name: str) -> tuple[Optional[bool], str]:
    """Return (decision, reason) based on structural naming patterns.

    Returns:
        (True, reason) — should definitely rotate.
        (False, reason) — should definitely skip.
        (None, reason) — no strong structural signal, defer to stats.
    """
    name_lower = name.lower()

    # Hard skip patterns
    for pattern in STRUCTURAL_SKIP_PATTERNS:
        if fnmatch.fnmatch(name_lower, pattern):
            return False, f"structural_skip({pattern})"

    # Strong prefer patterns
    for pattern in STRUCTURAL_PREFER_PATTERNS:
        if fnmatch.fnmatch(name_lower, pattern):
            return True, f"structural_prefer({pattern})"

    # Conditional patterns — no definitive answer
    for pattern in STRUCTURAL_CONDITIONAL_PATTERNS:
        if fnmatch.fnmatch(name_lower, pattern):
            return None, f"structural_conditional({pattern})"

    # Unknown layer — no structural signal
    return None, "no_structural_match"


def compute_layer_score(
    name: str,
    module: nn.Module,
    kurtosis: Optional[float] = None,
    ecr: Optional[float] = None,
    kurtosis_threshold: float = 5.0,
    ecr_threshold: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """Compute a Hadamard-benefit score for a layer.

    Higher score → more likely to benefit from Hadamard rotation.

    The score combines:
        1. Distribution quality (kurtosis, ECR) — heavy-tailed → benefit
        2. Structural features (expand/compress, residual sensitivity)

    Args:
        name: Full module name (e.g., "model.layers.0.mlp.up_proj").
        module: The nn.Module (Linear or QuantLinear).
        kurtosis: Per-channel kurtosis of input activations (mean across batch).
        ecr: Energy concentration ratio of input activations.
        kurtosis_threshold: Threshold above which kurtosis indicates heavy-tail.
        ecr_threshold: Threshold above which ECR indicates outlier dominance.

    Returns:
        (score, details) where details is a dict with scoring breakdown.
    """
    score = 0.0
    details: dict[str, Any] = {}

    # Infer dimensions
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        in_f = module.in_features
        out_f = module.out_features
    elif hasattr(module, "weight"):
        out_f, in_f = module.weight.shape[0], module.weight.shape[1]
    else:
        in_f = out_f = 0

    is_expand = out_f > in_f
    is_compress = out_f < in_f
    details["in_features"] = in_f
    details["out_features"] = out_f
    details["is_expand"] = is_expand
    details["is_compress"] = is_compress

    # Structural scoring
    if is_expand:
        score += 1.0
        details["expand_bonus"] = 1.0
    if is_compress:
        score -= 1.0
        details["compress_penalty"] = -1.0

    # Residual sensitivity heuristic: down_proj and o_proj output directly
    # into residual stream without nonlinearity buffer.
    name_lower = name.lower()
    is_residual_consumer = any(p in name_lower for p in ["down_proj", "o_proj", "out_proj"])
    details["is_residual_consumer"] = is_residual_consumer
    if is_residual_consumer:
        score -= 1.5
        details["residual_penalty"] = -1.5

    # Nonlinearity buffer: up_proj/gate_proj feed into activation functions.
    has_nonlinearity_after = any(p in name_lower for p in ["up_proj", "gate_proj"])
    details["has_nonlinearity_after"] = has_nonlinearity_after
    if has_nonlinearity_after:
        score += 0.5
        details["nonlinearity_bonus"] = 0.5

    # Statistics-based scoring (if available)
    if kurtosis is not None:
        details["kurtosis"] = kurtosis
        if kurtosis > kurtosis_threshold:
            score += 1.0
            details["kurtosis_bonus"] = 1.0
        elif kurtosis < 3.0:
            score -= 0.5
            details["kurtosis_penalty"] = -0.5

    if ecr is not None:
        details["ecr"] = ecr
        if ecr > ecr_threshold:
            score += 1.0
            details["ecr_bonus"] = 1.0

    details["total_score"] = score
    return score, details


class LayerSelector:
    """Decides per-layer whether to apply Hadamard rotation.

    Supports three modes (set via ``config.layer_selection``):
        - ``"all"``: Apply to all layers (legacy behavior).
        - ``"structural"``: Use naming-based structural priors only.
        - ``"auto"``: Profile activations + structural priors.

    Example::

        selector = LayerSelector.from_config(config, model)
        selector.profile(dataloader)  # only needed for "auto"

        for name, module in model.named_modules():
            if selector.should_rotate(name):
                apply_hadamard(module)
    """

    def __init__(
        self,
        mode: str = "all",
        include_layers: Optional[list[str]] = None,
        exclude_layers: Optional[list[str]] = None,
        kurtosis_threshold: float = 5.0,
        ecr_threshold: float = 0.05,
        score_threshold: float = 1.5,
    ):
        self.mode = mode
        self.include_layers = include_layers or []
        self.exclude_layers = exclude_layers or []
        self.kurtosis_threshold = kurtosis_threshold
        self.ecr_threshold = ecr_threshold
        self.score_threshold = score_threshold

        # Populated by profile() for "auto" mode.
        self._layer_stats: dict[str, dict[str, float]] = {}
        # Decision cache.
        self._decisions: dict[str, LayerDecision] = {}

    @classmethod
    def from_config(cls, config: Any, model: Optional[nn.Module] = None) -> "LayerSelector":
        """Construct a LayerSelector from a RotationConfig."""
        return cls(
            mode=getattr(config, "layer_selection", "all"),
            include_layers=getattr(config, "include_layers", None),
            exclude_layers=getattr(config, "exclude_layers", None),
            kurtosis_threshold=getattr(config, "kurtosis_threshold", 5.0),
            ecr_threshold=getattr(config, "ecr_threshold", 0.05),
            score_threshold=getattr(config, "score_threshold", 1.5),
        )

    def should_rotate(self, name: str, module: Optional[nn.Module] = None) -> bool:
        """Determine whether a named layer should have Hadamard rotation applied.

        Args:
            name: Full dotted module name.
            module: The module instance (used for dimension inference in auto mode).

        Returns:
            True if the layer should be rotated.
        """
        # Check explicit overrides first.
        for pattern in self.exclude_layers:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pattern.lower()):
                self._decisions[name] = LayerDecision(name=name, enabled=False, reason=f"explicit_exclude({pattern})")
                logger.debug(
                    "  ❌ SKIP  %-50s | reason: explicit exclude pattern '%s'", name, pattern
                )
                return False
        for pattern in self.include_layers:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pattern.lower()):
                self._decisions[name] = LayerDecision(name=name, enabled=True, reason=f"explicit_include({pattern})")
                logger.debug(
                    "  ✅ APPLY %-50s | reason: explicit include pattern '%s'", name, pattern
                )
                return True

        # Mode-specific logic.
        if self.mode == "all":
            # Legacy: rotate everything except lm_head (handled externally).
            self._decisions[name] = LayerDecision(name=name, enabled=True, reason="mode_all")
            return True

        if self.mode == "structural":
            decision, reason = structural_decision(name)
            if decision is None:
                # No strong signal — default to skip (conservative).
                decision = False
                reason += " → conservative_skip"
            self._decisions[name] = LayerDecision(name=name, enabled=decision, reason=reason)
            if decision:
                logger.debug("  ✅ APPLY %-50s | reason: %s", name, reason)
            else:
                logger.debug("  ❌ SKIP  %-50s | reason: %s", name, reason)
            return decision

        if self.mode == "auto":
            return self._auto_decision(name, module)

        # Fallback (shouldn't reach here with validation).
        return True

    def _auto_decision(self, name: str, module: Optional[nn.Module] = None) -> bool:
        """Statistics + structural decision for 'auto' mode."""
        # Structural hard-skip (always respect).
        struct_decision, struct_reason = structural_decision(name)
        if struct_decision is False:
            self._decisions[name] = LayerDecision(name=name, enabled=False, reason=f"auto:{struct_reason}")
            logger.debug(
                "  ❌ SKIP  %-50s | structural hard-skip: %s", name, struct_reason
            )
            return False

        # Get profiled stats (if available).
        stats = self._layer_stats.get(name, {})
        kurtosis = stats.get("kurtosis")
        ecr = stats.get("ecr")

        # Compute score.
        if module is not None:
            score, details = compute_layer_score(
                name, module, kurtosis=kurtosis, ecr=ecr,
                kurtosis_threshold=self.kurtosis_threshold,
                ecr_threshold=self.ecr_threshold,
            )
        else:
            # Without module, use only stats + structural.
            score = 0.0
            details = {}
            if kurtosis is not None and kurtosis > self.kurtosis_threshold:
                score += 1.0
            if ecr is not None and ecr > self.ecr_threshold:
                score += 1.0

        # Structural-prefer bonus: if structural analysis says this layer type
        # benefits from rotation (gate_proj, up_proj, q_proj, k_proj), add a
        # baseline bonus. This ensures that e.g. MoE expert gate_proj layers
        # (which are dimensionally "compress" but structurally "prefer") still
        # get rotated even without profiling data.
        # The bonus is set high enough (2.0) to overcome compress_penalty (-1.0)
        # and still meet the default score_threshold (1.5) without profiling.
        if struct_decision is True:
            score += 2.0
            details["structural_prefer_bonus"] = 2.0

        enabled = score >= self.score_threshold
        reason = f"auto:score={score:.2f}({'≥' if enabled else '<'}{self.score_threshold})"

        # Detailed logging for auto decisions
        decision_icon = "✅ APPLY" if enabled else "❌ SKIP "
        stats_str = ""
        if kurtosis is not None:
            kurt_flag = "⚠️ heavy-tail" if kurtosis > self.kurtosis_threshold else "normal"
            stats_str += f"kurtosis={kurtosis:.2f}({kurt_flag})"
        if ecr is not None:
            ecr_flag = "⚠️ concentrated" if ecr > self.ecr_threshold else "uniform"
            stats_str += f" ecr={ecr:.4f}({ecr_flag})"
        if not stats_str:
            stats_str = "no profiling data"

        dim_str = ""
        if details.get("in_features") and details.get("out_features"):
            dim_str = f" dim={details['in_features']}→{details['out_features']}"
            if details.get("is_expand"):
                dim_str += "(expand)"
            elif details.get("is_compress"):
                dim_str += "(compress)"

        score_breakdown = []
        for key in ["structural_prefer_bonus", "expand_bonus", "compress_penalty",
                    "residual_penalty", "nonlinearity_bonus", "kurtosis_bonus",
                    "kurtosis_penalty", "ecr_bonus"]:
            v = details.get(key)
            if v is not None:
                score_breakdown.append(f"{key}={v:+.1f}")
        breakdown_str = " | breakdown: " + ", ".join(score_breakdown) if score_breakdown else ""

        logger.debug(
            "  %s %-50s | score=%.2f (threshold=%.1f) | %s%s%s",
            decision_icon, name, score, self.score_threshold, stats_str, dim_str, breakdown_str
        )

        self._decisions[name] = LayerDecision(
            name=name, enabled=enabled, reason=reason, score=score,
            kurtosis=kurtosis, ecr=ecr,
            is_expand=details.get("is_expand"),
            is_compress=details.get("is_compress"),
            is_residual_consumer=details.get("is_residual_consumer"),
        )
        return enabled

    def profile(
        self,
        model: nn.Module,
        dataloader: Any,
        num_samples: int = 32,
        device: Optional[str] = None,
    ) -> dict[str, dict[str, float]]:
        """Profile activation statistics for each Linear layer.

        Runs ``num_samples`` from ``dataloader`` through the model and collects
        per-layer kurtosis and energy concentration ratio (ECR).

        This is only needed for ``mode="auto"``. For ``mode="structural"``,
        calling this is a no-op.

        Args:
            model: The model to profile.
            dataloader: Calibration data (yields input_ids tensors or dicts).
            num_samples: Number of calibration samples to use.
            device: Compute device (defaults to model device).

        Returns:
            Dict mapping layer_name → {"kurtosis": float, "ecr": float}.
        """
        if self.mode != "auto":
            logger.info("LayerSelector.profile() called but mode=%s; skipping.", self.mode)
            return {}

        logger.info(
            "╔══════════════════════════════════════════════════════════════════════╗\n"
            "║  Selective Rotation: Activation Profiling                           ║\n"
            "║  mode=%s | num_samples=%d | device=%s                     \n"
            "╚══════════════════════════════════════════════════════════════════════╝",
            self.mode, num_samples, device or "auto"
        )

        from auto_round.algorithms.transforms.rotation.selective import _profile_activations
        self._layer_stats = _profile_activations(model, dataloader, num_samples, device)

        # Log outlier analysis summary
        if self._layer_stats:
            self._log_profiling_summary()

        return self._layer_stats

    def _log_profiling_summary(self):
        """Log a detailed summary of profiling results with outlier analysis."""
        stats = self._layer_stats
        if not stats:
            return

        # Classify layers by outlier severity
        heavy_tail_layers = []
        concentrated_layers = []
        normal_layers = []

        for name, s in sorted(stats.items()):
            kurt = s.get("kurtosis", 0)
            ecr = s.get("ecr", 0)
            is_heavy = kurt > self.kurtosis_threshold
            is_concentrated = ecr > self.ecr_threshold
            if is_heavy or is_concentrated:
                if is_heavy:
                    heavy_tail_layers.append((name, kurt, ecr))
                if is_concentrated:
                    concentrated_layers.append((name, kurt, ecr))
            else:
                normal_layers.append((name, kurt, ecr))

        logger.info(
            "┌─────────────────────────────────────────────────────────────────────┐\n"
            "│  Activation Profiling Results: %d layers analyzed                     \n"
            "│  ⚠️  Heavy-tail (kurtosis > %.1f): %d layers                          \n"
            "│  ⚠️  Concentrated (ECR > %.3f): %d layers                             \n"
            "│  ✓  Normal distribution: %d layers                                   \n"
            "└─────────────────────────────────────────────────────────────────────┘",
            len(stats), self.kurtosis_threshold, len(heavy_tail_layers),
            self.ecr_threshold, len(concentrated_layers), len(normal_layers),
        )

        # Top-N outlier layers (these benefit MOST from rotation)
        by_kurtosis = sorted(stats.items(), key=lambda x: x[1].get("kurtosis", 0), reverse=True)
        logger.info("  Top-5 layers with HIGHEST kurtosis (most outlier-prone → rotation helps):")
        for name, s in by_kurtosis[:5]:
            kurt = s["kurtosis"]
            ecr = s["ecr"]
            benefit = "🔥 STRONG" if kurt > self.kurtosis_threshold * 2 else "⚠️  moderate" if kurt > self.kurtosis_threshold else "○ low"
            logger.info("    %s %-50s | kurtosis=%6.2f  ecr=%.4f", benefit, name, kurt, ecr)

        # Bottom-N (these are LEAST likely to benefit → rotation may harm)
        logger.info("  Bottom-5 layers with LOWEST kurtosis (near-Gaussian → rotation may harm):")
        for name, s in by_kurtosis[-5:]:
            kurt = s["kurtosis"]
            ecr = s["ecr"]
            logger.info("    ○ %-50s | kurtosis=%6.2f  ecr=%.4f", name, kurt, ecr)

        # Per-layer-type aggregation
        type_stats: dict[str, list[tuple[float, float]]] = {}
        for name, s in stats.items():
            # Extract layer type (last component of name, e.g., "up_proj")
            parts = name.split(".")
            layer_type = parts[-1] if parts else name
            if layer_type not in type_stats:
                type_stats[layer_type] = []
            type_stats[layer_type].append((s["kurtosis"], s["ecr"]))

        logger.info("  Per-layer-type outlier statistics (averaged across all blocks):")
        logger.info("    %-15s  %8s  %8s  %8s  %s", "layer_type", "avg_kurt", "max_kurt", "avg_ecr", "verdict")
        logger.info("    %s", "-" * 75)
        for ltype, vals in sorted(type_stats.items()):
            avg_kurt = sum(v[0] for v in vals) / len(vals)
            max_kurt = max(v[0] for v in vals)
            avg_ecr = sum(v[1] for v in vals) / len(vals)
            if avg_kurt > self.kurtosis_threshold:
                verdict = "⚠️  outlier-prone → rotation BENEFICIAL"
            elif avg_kurt < 3.5:
                verdict = "✓  near-Gaussian → rotation RISKY"
            else:
                verdict = "○  moderate → depends on structure"
            logger.info("    %-15s  %8.2f  %8.2f  %8.4f  %s", ltype, avg_kurt, max_kurt, avg_ecr, verdict)

    def get_decisions(self) -> dict[str, LayerDecision]:
        """Return all cached decisions (populated after should_rotate calls)."""
        return dict(self._decisions)

    def summary(self, verbose: bool = True) -> str:
        """Return a human-readable summary of decisions with outlier analysis.

        Args:
            verbose: If True, include full per-layer decision table.
                     If False, only show summary statistics and per-type breakdown.
        """
        if not self._decisions:
            return "No decisions yet. Call should_rotate() first."

        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║  Selective Hadamard Rotation — Decision Summary                                                                     ║")
        lines.append(f"║  mode={self.mode:<12} score_threshold={self.score_threshold:<5}  kurtosis_threshold={self.kurtosis_threshold:<5}  ecr_threshold={self.ecr_threshold:<6}    ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

        enabled_count = 0
        skipped_layers = []
        applied_layers = []

        for name, d in sorted(self._decisions.items()):
            if d.enabled:
                enabled_count += 1
                applied_layers.append(d)
            else:
                skipped_layers.append(d)

        total = len(self._decisions)

        # Full per-layer table (only when verbose)
        if verbose:
            lines.append("")
            lines.append(f"  {'Layer':<55} {'Decision':<10} {'Score':<8} {'Kurtosis':<10} {'ECR':<10} {'Reason'}")
            lines.append(f"  {'─' * 120}")
            for name, d in sorted(self._decisions.items()):
                status = "✅ ON " if d.enabled else "❌ OFF"
                score_str = f"{d.score:.2f}" if d.score is not None else "—"
                kurt_str = f"{d.kurtosis:.2f}" if d.kurtosis is not None else "—"
                ecr_str = f"{d.ecr:.4f}" if d.ecr is not None else "—"
                lines.append(f"  {name:<55} {status:<10} {score_str:<8} {kurt_str:<10} {ecr_str:<10} {d.reason}")
            lines.append(f"  {'─' * 120}")

        # Summary statistics (always shown)
        lines.append("")
        lines.append("  ┌─── Summary ──────────────────────────────────────────────────────┐")
        lines.append(f"  │  Total layers: {total}                                            ")
        lines.append(f"  │  ✅ Rotated:   {enabled_count} ({100*enabled_count/max(total,1):.1f}%)                                  ")
        lines.append(f"  │  ❌ Skipped:   {total - enabled_count} ({100*(total-enabled_count)/max(total,1):.1f}%)                                  ")
        lines.append(f"  └─────────────────────────────────────────────────────────────────┘")

        # Outlier compression analysis (if stats available)
        if self._layer_stats:
            lines.append("")
            lines.append("  ┌─── Outlier Compression Analysis ─────────────────────────────────┐")
            lines.append("  │  Why selective rotation works:                                     ")
            lines.append("  │  • Layers WITH outliers (high kurtosis) → rotation DISPERSES them ")
            lines.append("  │    across channels, reducing quantization error                    ")
            lines.append("  │  • Layers WITHOUT outliers (low kurtosis) → rotation ADDS noise   ")
            lines.append("  │    to an already-uniform distribution (harmful)                    ")
            lines.append("  │                                                                   ")

            # Compute expected benefit
            rotated_kurt = [d.kurtosis for d in applied_layers if d.kurtosis is not None]
            skipped_kurt = [d.kurtosis for d in skipped_layers if d.kurtosis is not None]
            rotated_ecr = [d.ecr for d in applied_layers if d.ecr is not None]
            skipped_ecr = [d.ecr for d in skipped_layers if d.ecr is not None]

            if rotated_kurt:
                avg_rotated_kurt = sum(rotated_kurt) / len(rotated_kurt)
                lines.append(f"  │  Rotated layers avg kurtosis:  {avg_rotated_kurt:.2f} (outlier-prone → rotation beneficial) ")
            if skipped_kurt:
                avg_skipped_kurt = sum(skipped_kurt) / len(skipped_kurt)
                lines.append(f"  │  Skipped layers avg kurtosis:  {avg_skipped_kurt:.2f} (near-normal → rotation would harm)  ")
            if rotated_kurt and skipped_kurt:
                separation = sum(rotated_kurt) / len(rotated_kurt) - sum(skipped_kurt) / len(skipped_kurt)
                quality = "GOOD" if separation > 2.0 else "MODERATE" if separation > 0.5 else "WEAK"
                lines.append(f"  │  Kurtosis separation:          {separation:+.2f} ({quality} selective targeting) ")

            if rotated_ecr:
                avg_rotated_ecr = sum(rotated_ecr) / len(rotated_ecr)
                lines.append(f"  │  Rotated layers avg ECR:       {avg_rotated_ecr:.4f} (energy concentrated → needs dispersion)")
            if skipped_ecr:
                avg_skipped_ecr = sum(skipped_ecr) / len(skipped_ecr)
                lines.append(f"  │  Skipped layers avg ECR:       {avg_skipped_ecr:.4f} (energy uniform → already balanced)    ")

            lines.append("  │                                                                   ")
            lines.append("  │  Interpretation:                                                  ")
            lines.append("  │  • Large kurtosis separation → selective is well-targeted         ")
            lines.append("  │  • If rotated layers have LOW kurtosis → consider relaxing skip   ")
            lines.append("  └─────────────────────────────────────────────────────────────────┘")

        # Per-type breakdown (always shown)
        type_counts: dict[str, tuple[int, int]] = {}  # type → (applied, skipped)
        for d in self._decisions.values():
            parts = d.name.split(".")
            ltype = parts[-1] if parts else d.name
            if ltype not in type_counts:
                type_counts[ltype] = (0, 0)
            applied, skipped = type_counts[ltype]
            if d.enabled:
                type_counts[ltype] = (applied + 1, skipped)
            else:
                type_counts[ltype] = (applied, skipped + 1)

        lines.append("")
        lines.append("  ┌─── Per-Layer-Type Breakdown ─────────────────────────────────────┐")
        lines.append(f"  │  {'Type':<15} {'Applied':<10} {'Skipped':<10} {'Rate':<8} {'Assessment'}")
        lines.append(f"  │  {'─' * 65}")
        for ltype, (app, skip) in sorted(type_counts.items()):
            rate = f"{100*app/max(app+skip,1):.0f}%"
            if app == 0:
                assessment = "← all skipped (residual-sensitive / compress)"
            elif skip == 0:
                assessment = "← all rotated (outlier-prone / expand)"
            else:
                assessment = "← mixed (per-block differences detected)"
            lines.append(f"  │  {ltype:<15} {app:<10} {skip:<10} {rate:<8} {assessment}")
        lines.append("  └─────────────────────────────────────────────────────────────────┘")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Activation profiler (internal)
# ═══════════════════════════════════════════════════════════════════════════════


def _profile_activations(
    model: nn.Module,
    dataloader: Any,
    num_samples: int = 32,
    device: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """Run calibration data through model, collect per-layer activation stats.

    For each Linear layer, computes:
        - kurtosis: Excess kurtosis of input activations (channel-wise mean).
        - ecr: Energy Concentration Ratio = max_channel_energy / total_energy.
        - max_abs: Maximum absolute activation value (outlier magnitude).
        - channel_std_ratio: max_channel_std / min_channel_std (uniformity measure).

    Returns:
        Dict[layer_name, {"kurtosis": float, "ecr": float, "max_abs": float, "channel_std_ratio": float}]
    """
    from auto_round.experimental.qmodules.base import QModuleBase

    target_types = (nn.Linear, QModuleBase)
    stats_accum: dict[str, dict[str, list]] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def _hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is None or not isinstance(x, torch.Tensor):
                return
            # Flatten to (N, features) for stats.
            x_flat = x.detach().float().reshape(-1, x.shape[-1])
            if x_flat.shape[0] == 0:
                return

            # Kurtosis (per-channel, then mean).
            mean = x_flat.mean(dim=0)
            var = x_flat.var(dim=0) + 1e-8
            centered = x_flat - mean
            kurt = (centered.pow(4).mean(dim=0) / var.pow(2)) - 3.0  # excess kurtosis
            kurt_mean = kurt.mean().item()

            # Energy Concentration Ratio.
            channel_energy = (x_flat ** 2).sum(dim=0)
            total_energy = channel_energy.sum().item()
            max_energy = channel_energy.max().item()
            ecr = max_energy / (total_energy + 1e-12)

            # Max absolute value (outlier magnitude indicator).
            max_abs = x_flat.abs().max().item()

            # Channel std ratio (how uneven the channels are).
            channel_std = x_flat.std(dim=0)
            std_max = channel_std.max().item()
            std_min = channel_std.min().item() + 1e-8
            channel_std_ratio = std_max / std_min

            if layer_name not in stats_accum:
                stats_accum[layer_name] = {"kurtosis": [], "ecr": [], "max_abs": [], "channel_std_ratio": []}
            stats_accum[layer_name]["kurtosis"].append(kurt_mean)
            stats_accum[layer_name]["ecr"].append(ecr)
            stats_accum[layer_name]["max_abs"].append(max_abs)
            stats_accum[layer_name]["channel_std_ratio"].append(channel_std_ratio)
        return _hook

    # Register hooks on all Linear layers.
    hook_count = 0
    for name, module in model.named_modules():
        if isinstance(module, target_types) and "lm_head" not in name:
            h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)
            hook_count += 1

    logger.info("  Profiling: registered hooks on %d layers, running %d calibration samples...", hook_count, num_samples)

    # Resolve the device for input tensors and model placement.
    # Key scenarios:
    #   1. Model dispatched across multi-GPU (device_map="auto") — DON'T move, let accelerate handle it
    #   2. Model on CPU, CUDA available — move to GPU temporarily for profiling
    #   3. Model already on single GPU — use as-is
    is_dispatched = hasattr(model, "hf_device_map") and model.hf_device_map is not None
    model_was_on_cpu = False

    if is_dispatched:
        # Model is split across multiple devices by accelerate — never call model.to().
        # Input should go to the device of the embedding layer (first module in forward pass).
        try:
            # model.device returns device of first parameter for dispatched models
            input_device = model.device
        except Exception:
            input_device = torch.device("cuda:0")
        logger.info("  Profiling: model is accelerate-dispatched (device_map=%s), input_device=%s",
                    getattr(model, "hf_device_map", "unknown"), input_device)
    else:
        # Single-device model — may need to move to GPU.
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        # Determine target device.
        if device and device not in (None, "auto"):
            target_device = torch.device(device)
        elif torch.cuda.is_available():
            target_device = torch.device("cuda:0")
        else:
            target_device = model_device

        # Move model if needed.
        if str(model_device) != str(target_device):
            try:
                model.to(target_device)
                model_was_on_cpu = (str(model_device) == "cpu")
                logger.info("  Profiling: moved model from %s to %s for activation profiling.", model_device, target_device)
            except Exception as e:
                logger.warning("  Profiling: failed to move model to %s (%s), staying on %s", target_device, e, model_device)
                target_device = model_device

        input_device = target_device

    # Run calibration.
    model.eval()
    samples_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            if isinstance(batch, dict):
                if input_device:
                    batch = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model(**batch)
            elif isinstance(batch, (list, tuple)):
                inp = batch[0]
                if input_device and isinstance(inp, torch.Tensor):
                    inp = inp.to(input_device)
                model(inp)
            else:
                if input_device and isinstance(batch, torch.Tensor):
                    batch = batch.to(input_device)
                model(batch)
            samples_seen += 1
            if samples_seen % 8 == 0:
                logger.debug("  Profiling progress: %d/%d samples", samples_seen, num_samples)

    logger.info("  Profiling complete: processed %d samples across %d layers.", samples_seen, len(stats_accum))

    # Remove hooks.
    for h in hooks:
        h.remove()

    # Move model back to CPU if we moved it (pipeline will move it to GPU later in _hardware_setup)
    if model_was_on_cpu:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("  Profiling: moved model back to CPU (pipeline will handle final placement).")

    # Aggregate stats.
    result: dict[str, dict[str, float]] = {}
    for name, accum in stats_accum.items():
        if accum["kurtosis"]:
            result[name] = {
                "kurtosis": sum(accum["kurtosis"]) / len(accum["kurtosis"]),
                "ecr": sum(accum["ecr"]) / len(accum["ecr"]),
                "max_abs": max(accum["max_abs"]),
                "channel_std_ratio": sum(accum["channel_std_ratio"]) / len(accum["channel_std_ratio"]),
            }
    return result
