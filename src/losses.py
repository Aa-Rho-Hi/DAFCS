"""
Loss functions for EMBER2024 training.

  SupConLoss              — Supervised Contrastive Loss (Khosla et al. 2020)
                            + optional hard-negative up-weighting
  FocalLoss               — Binary focal loss for detection head
  ClassBalancedFocalLoss  — Class-frequency-weighted focal loss (Cui et al. 2019)
  LogitAdjustedLoss       — Logit-adjusted CE for long-tail families
                            (Menon et al. ICLR 2021)
  MultiTaskLoss           — Weighted sum of detection + family losses
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ──────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020) with optional hard-negative
    up-weighting.

    Standard SupCon
    ───────────────
    For each anchor i:
      L_i = -1/|P(i)| · Σ_{j∈P(i)} log(
                exp(z_i·z_j / τ) / Σ_{k≠i} exp(z_i·z_k / τ)
            )

    Hard-negative up-weighting (hard_neg_weight > 0)
    ─────────────────────────────────────────────────
    Negative pairs (different class, high similarity) are the most
    informative — they represent families that are easy to confuse.
    We up-weight their contribution in the denominator by:

        effective_exp_k = exp(z_i·z_k / τ) · (1 + α · softmax_k)

    where softmax_k = softmax over negatives of (z_i·z_k / τ_hard).
    This concentrates the denominator on hard negatives while leaving
    the numerator (positive terms) unchanged.

    Args:
        temperature:      τ (default 0.07).
        hard_neg_weight:  α — up-weighting strength for hard negatives
                          (0 = standard SupCon, 0.5 = moderate, 1.0 = strong).
        hard_neg_temp:    Temperature for the negative-softmax used in weighting
                          (should be ≥ τ to avoid collapse).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_neg_weight: float = 0.5,
        hard_neg_temp: float = 0.2,
    ):
        super().__init__()
        self.temperature     = temperature
        self.hard_neg_weight = hard_neg_weight
        self.hard_neg_temp   = hard_neg_temp

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) — L2-normalised projection embeddings.
            labels:   (B,)   — Integer class labels.
        Returns:
            Scalar loss.
        """
        B      = features.shape[0]
        device = features.device

        features = F.normalize(features, p=2, dim=1)

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Masks
        labels_row = labels.unsqueeze(1)
        labels_col = labels.unsqueeze(0)
        pos_mask = (labels_row == labels_col).float()                   # same class
        eye      = torch.eye(B, device=device, dtype=torch.float32)
        pos_mask = pos_mask * (1.0 - eye)                               # exclude self
        neg_mask = (1.0 - pos_mask) * (1.0 - eye)                      # different class, not self

        # Base denominator: all k ≠ i
        exp_sim = torch.exp(sim)
        base_denom = exp_sim * (1.0 - eye)      # (B, B), diagonal zeroed

        if self.hard_neg_weight > 0:
            # Compute attention weights over negatives only
            # (higher weight = harder negative = more confusable)
            neg_raw_sim = torch.matmul(features, features.T) / self.hard_neg_temp
            neg_raw_sim = neg_raw_sim - neg_raw_sim.max(dim=1, keepdim=True).values.detach()
            # Mask positives + self before softmax
            neg_raw_sim = neg_raw_sim - 1e9 * (pos_mask + eye)
            neg_attn = torch.softmax(neg_raw_sim, dim=1)                # (B, B)

            # Up-weight hard negatives in the denominator
            hard_weight = 1.0 + self.hard_neg_weight * neg_attn * neg_mask
            denom = (base_denom * hard_weight).sum(dim=1, keepdim=True) + 1e-8
        else:
            denom = base_denom.sum(dim=1, keepdim=True) + 1e-8

        log_denom = torch.log(denom)
        log_prob  = sim - log_denom             # (B, B)

        n_pos = pos_mask.sum(dim=1)             # (B,)
        valid = n_pos > 0

        if not valid.any():
            return features.new_zeros(1).squeeze()

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        return -mean_log_prob_pos[valid].mean()


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss (binary)
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for the detection head.

    FL(p_t) = -(1 - p_t)^γ * log(p_t)

    where p_t = sigmoid(logit) when target=1, else 1 - sigmoid(logit).

    Args:
        gamma:      Focusing parameter (default 2.0).
        pos_weight: Optional scalar weight for the positive class,
                    passed to F.binary_cross_entropy_with_logits.
                    Useful when benign >> malicious in the training set.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = (
            torch.tensor([pos_weight]) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) — raw output from DetectionHead (no sigmoid applied).
            targets: (B,) — float tensor of 0.0 / 1.0.
        """
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )   # (B,)

        # p_t = exp(-BCE) because BCE = -log(p_t)
        pt  = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = alpha_t * focal

        return focal.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Class-Balanced Focal Loss (multi-class)
# ──────────────────────────────────────────────────────────────────────────────

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-balanced focal loss for the family head.

    Combines:
      • Class-balanced weighting (Cui et al. 2019):
            weight_i = (1 - β) / (1 - β^{n_i})
        where n_i = number of training samples in class i, β ∈ [0, 1).
      • Focal modulation: (1 - p_t)^γ

    Args:
        samples_per_class: Array of shape (num_classes,) with sample counts.
        beta:              Class-balance hyper-parameter (default 0.999).
        gamma:             Focal focusing parameter (default 2.0).
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        beta: float = 0.999,
        gamma: float = 2.0,
        weight_mode: str = "effective_num",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        counts = samples_per_class.astype(float)
        if weight_mode == "inverse_sqrt":
            weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        else:
            # Effective number of samples (Cui et al. 2019)
            eff_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / (eff_num + 1e-8)
        # Normalise so the mean weight is 1 (preserves overall loss scale)
        weights = weights / weights.mean()
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) — raw output from FamilyHead.
            targets: (B,)   — integer class indices. Entries == -1 are ignored.
        """
        # Mask out samples without a family label
        valid = targets >= 0
        if not valid.any():
            return logits.new_zeros(1).squeeze()

        logits  = logits[valid]
        targets = targets[valid]

        # Per-sample class weight
        sample_weights = self.class_weights[targets]   # (B_valid,)

        # Standard cross-entropy per sample
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )   # (B_valid,)

        # Focal modulation
        pt    = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        # Apply class-balanced weights
        return (focal * sample_weights).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Logit-Adjusted Loss (long-tail family classification)
# ──────────────────────────────────────────────────────────────────────────────

class LogitAdjustedLoss(nn.Module):
    """
    Logit-adjusted cross-entropy for long-tail recognition.

    Reference: Menon et al., "Long-tail learning via logit adjustment",
               ICLR 2021. https://arxiv.org/abs/2007.07314

    How it works
    ────────────
    Standard cross-entropy trains the model to predict frequent classes even
    when the ground-truth is rare, because frequent classes dominate the
    gradient signal. Logit adjustment corrects this by adding τ·log(π_y) to
    the logit for class y during *training*:

        adjusted_logit_y = logit_y + τ · log(π_y)

    where π_y = n_y / N is the empirical class frequency. Equivalently, at
    inference the model's Bayes-optimal decision boundary shifts to account
    for the true (balanced) prior — without any test-time manipulation.

    Combining with focal modulation (γ > 0) further de-emphasises easy
    majority-class samples, concentrating gradient on the rare classes and
    hard examples simultaneously.

    Args:
        samples_per_class: (C,) array of training sample counts per class.
        tau:               Adjustment strength (default 1.0; paper default).
                           Larger τ = stronger correction toward rare classes.
        gamma:             Focal modulation parameter (0 = plain adjusted CE).
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        tau: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        freq = samples_per_class.astype(float) / samples_per_class.sum()
        adj  = tau * np.log(freq + 1e-12)     # (C,) — negative for rare classes
        self.register_buffer("adjustments", torch.tensor(adj, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) — raw logits from family head (before softmax).
            targets: (B,)   — integer class indices; -1 entries are skipped.
        Returns:
            Scalar loss.
        """
        valid = targets >= 0
        if not valid.any():
            return logits.new_zeros(1).squeeze()

        logits  = logits[valid]
        targets = targets[valid]

        # Shift logits: rare-class logits decrease during training, forcing the
        # classifier to develop a stronger feature-based signal for them.
        adj_logits = logits + self.adjustments.unsqueeze(0)   # broadcast (1, C)

        ce = F.cross_entropy(adj_logits, targets, reduction="none")

        if self.gamma > 0:
            pt   = torch.exp(-ce)
            loss = ((1.0 - pt) ** self.gamma) * ce
        else:
            loss = ce

        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-task combined loss
# ──────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted sum of detection focal loss + family classification loss.

    Family loss options (controlled by `use_logit_adj`):
      • True  (default): LogitAdjustedLoss — best for long-tail families;
                         learns balanced decision boundaries without test-time hacks.
      • False:           ClassBalancedFocalLoss — effective number reweighting
                         (Cui et al. 2019); kept for ablations.

    Args:
        samples_per_class: (C,) array of per-class training sample counts.
        focal_gamma:       γ for focal modulation (both losses).
        cb_beta:           β for ClassBalancedFocalLoss (unused when logit_adj=True).
        logit_adj_tau:     τ for LogitAdjustedLoss (default 1.0).
        use_logit_adj:     If True, use LogitAdjustedLoss; else ClassBalancedFocalLoss.
        detection_weight:  Loss weight for detection head.
        family_weight:     Loss weight for family head.
        pos_weight:        Optional positive-class scalar for detection focal loss.
    """

    def __init__(
        self,
        samples_per_class: np.ndarray,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        cb_beta: float = 0.999,
        logit_adj_tau: float = 1.0,
        use_logit_adj: bool = True,
        family_weight_mode: str = "effective_num",
        family_label_smoothing: float = 0.0,
        detection_weight: float = 1.0,
        family_weight: float = 1.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.det_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            pos_weight=pos_weight,
        )

        if use_logit_adj:
            self.family_loss = LogitAdjustedLoss(
                samples_per_class=samples_per_class,
                tau=logit_adj_tau,
                gamma=focal_gamma,
            )
        else:
            self.family_loss = ClassBalancedFocalLoss(
                samples_per_class=samples_per_class,
                beta=cb_beta,
                gamma=focal_gamma,
                weight_mode=family_weight_mode,
                label_smoothing=family_label_smoothing,
            )

        self.w_det    = detection_weight
        self.w_family = family_weight

    def forward(
        self,
        det_logits: torch.Tensor,
        family_logits: torch.Tensor,
        det_targets: torch.Tensor,
        family_targets: torch.Tensor,
    ):
        """Returns: (total_loss, det_loss, family_loss)."""
        l_det    = self.det_loss(det_logits, det_targets)
        l_family = self.family_loss(family_logits, family_targets)
        total    = self.w_det * l_det + self.w_family * l_family
        return total, l_det, l_family
