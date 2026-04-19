"""
Model architecture: ResidualMLP encoder + task heads.

Design
------
Input (2568)
  → LayerNorm → Linear(2568→1024) → GELU → Dropout(0.3)   [stem]
  → ResBlock(1024) → ResBlock(1024) → ResBlock(512) → ResBlock(256)  [backbone]
  ┌→ ProjectionHead: Linear(256→128) → L2-norm              [contrastive]
  ├→ DetectionHead: Linear(256→1)                           [binary detection]
  └→ FamilyHead:   Linear(256→num_families)                 [family classification]

Each ResBlock(dim_in, dim_out):
  LayerNorm → Linear(dim_in→dim_out) → GELU → Dropout → Linear(dim_out→dim_out)
  + skip connection (with a Linear projection if dim_in ≠ dim_out)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Pre-activation residual block with optional dimension change.

    Architecture (pre-activation style):
        LayerNorm → Linear(in→out) → GELU → Dropout → Linear(out→out) → +skip

    The skip connection uses a linear projection when in_features ≠ out_features.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.norm   = nn.LayerNorm(in_features)
        self.lin1   = nn.Linear(in_features, out_features)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)
        self.lin2   = nn.Linear(out_features, out_features)

        # Projection for skip when dimensions differ
        if in_features != out_features:
            self.skip_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)
        out = self.norm(x)
        out = self.lin1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.lin2(out)
        return out + residual


class ResidualMLPEncoder(nn.Module):
    """
    Main encoder backbone: stem + stack of ResBlocks.

    Args:
        input_dim:   Dimension of input feature vector (default 2568).
        hidden_dims: Sizes of successive ResBlocks (default [1024, 1024, 512, 256]).
        dropout:     Dropout rate throughout the network.

    Output: embedding of dimension hidden_dims[-1].
    """

    def __init__(
        self,
        input_dim: int = 2568,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024, 512, 256]

        # Stem: LayerNorm → Linear → GELU → Dropout
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        blocks = []
        dims = hidden_dims  # e.g. [1024, 1024, 512, 256]
        for i in range(len(dims) - 1):
            blocks.append(ResBlock(dims[i], dims[i + 1], dropout=dropout))
        # Final block stays at dims[-1]
        blocks.append(ResBlock(dims[-1], dims[-1], dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder embedding of shape (B, output_dim)."""
        x = self.stem(x)
        x = self.blocks(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Task heads
# ──────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP contrastive projection head with BatchNorm.

    SimCLR (Chen et al. 2020) shows that a non-linear projection head
    substantially outperforms a single linear layer for contrastive learning.
    BN before the final layer prevents feature collapse.

    Architecture: Linear(in→hidden) → BN → ReLU → Linear(hidden→proj_dim) → L2-norm
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        hidden = max(proj_dim * 2, in_dim)   # at least as wide as input
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class DetectionHead(nn.Module):
    """Binary malware-detection head: Linear → raw logit (no sigmoid here)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)   # (B,) raw logits


class FamilyHead(nn.Module):
    """Multi-class family-classification head: Linear → raw logits."""

    def __init__(self, in_dim: int, num_families: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_families)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # (B, num_families) raw logits


class CosineClassifier(nn.Module):
    """
    Cosine similarity classifier for long-tail family classification.

    Unlike a standard linear layer (w·x + b), this L2-normalises both the
    feature vector x and each class weight vector before taking their dot
    product, then scales by a learnable temperature s.

    Why this helps for long-tail families
    ──────────────────────────────────────
    A standard linear head conflates the *direction* and *magnitude* of the
    weight vector. Head-class weights grow large from many gradient updates,
    making them systematically preferred at inference even when the true class
    is rare. Normalising both sides decouples direction from magnitude so
    every family competes on equal geometric footing.

    References:
        • Liu et al., "Large-Margin Softmax Loss", NeurIPS 2016
        • Gidaris & Komodakis, "Dynamic Few-Shot Visual Learning without
          Forgetting", CVPR 2018
        • Kang et al., "Decoupling Representation and Classifier for
          Long-Tailed Recognition", ICLR 2020
    """

    def __init__(self, in_dim: int, num_classes: int, init_scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Log-parameterised so scale stays positive; clamped to [1, 100]
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)            # (B, D)
        w_norm = F.normalize(self.weight, p=2, dim=1)  # (C, D)
        return self.scale * (x_norm @ w_norm.T)        # (B, C)


class ArcFaceClassifier(nn.Module):
    """
    ArcFace: Additive Angular Margin loss classifier for long-tail family
    classification with many classes (Deng et al., CVPR 2019).

    Standard softmax with a cosine similarity classifier trains the model to
    produce correct predictions, but doesn't enforce a *margin* between
    classes. ArcFace adds a fixed angular margin m to the angle between the
    feature vector and the ground-truth class weight:

        logit_y_train = s · cos(θ_y + m)       ← target class
        logit_k_train = s · cos(θ_k)           ← all other classes

    This forces the encoder to produce embeddings that are at least m radians
    away from the decision boundary — significantly more discriminative than
    cosine softmax, especially for rare classes with few training samples.

    At inference (labels=None or training=False): behaves identically to
    CosineClassifier — returns s·cos(θ), so no test-time changes needed.

    Args:
        in_dim:   Feature dimension.
        num_classes: Number of family classes.
        scale:    s — logit scale (default 64, common in face recognition).
        margin:   m — angular margin in radians (default 0.5 ≈ 28.6°).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute stable trig constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold angle beyond which we use a linear approximation
        # (avoids the non-monotone region of cos(θ + m) when θ > π - m)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, in_dim) — encoder embeddings (need not be normalised).
            labels: (B,)        — integer class labels, or None for inference.

        Returns:
            (B, num_classes) — logits (with margin during training, plain cosine at inference).
        """
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = x_norm @ w_norm.T      # (B, C)

        if labels is None or not self.training:
            # Pure cosine logits for inference / evaluation
            return self.scale * cosine

        # ── Angular margin injection ───────────────────────────────────────
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))
        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Monotonicity fix: when θ > π − m, use linear approximation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to the target class
        valid_labels = labels.clamp(min=0)                         # guard against -1
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, valid_labels.unsqueeze(1), 1.0)
        # Samples with label -1 should not receive margin
        known = (labels >= 0).float().unsqueeze(1)
        one_hot = one_hot * known

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return self.scale * output


def feature_augment(
    x: torch.Tensor,
    mask_prob: float = 0.3,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """
    Lightweight feature augmentation for contrastive pre-training.

    Two independent random transformations of x can serve as "two views"
    in the contrastive batch, analogous to random crops/flips for images.

    Operations (applied independently per sample):
        1. Feature masking: zero-out each feature with probability mask_prob.
        2. Gaussian noise: add N(0, noise_std) to non-masked features.

    Both operations are applied in-place during training (no gradient through
    mask generation).
    """
    # Bernoulli keep-mask: 1 = keep, 0 = zero-out
    keep = torch.bernoulli(
        torch.full_like(x, 1.0 - mask_prob)
    )
    noise = noise_std * torch.randn_like(x)
    return x * keep + noise * keep


# ──────────────────────────────────────────────────────────────────────────────
# Full model wrappers
# ──────────────────────────────────────────────────────────────────────────────

class ContrastiveModel(nn.Module):
    """
    Stage 1 model: encoder + projection head only.

    Used during supervised contrastive pre-training.
    """

    def __init__(
        self,
        input_dim: int = 2568,
        hidden_dims: Optional[List[int]] = None,
        proj_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ResidualMLPEncoder(input_dim, hidden_dims, dropout)
        self.proj_head = ProjectionHead(self.encoder.output_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projection embeddings (B, proj_dim)."""
        emb = self.encoder(x)
        return self.proj_head(emb)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw encoder embeddings (B, encoder.output_dim), no projection."""
        return self.encoder(x)


class MultiTaskModel(nn.Module):
    """
    Stage 2 model: shared encoder + detection head + family head.

    The projection head is kept so the encoder weights from Stage 1 can be
    loaded without key mismatches (proj_head keys are simply ignored).

    Args:
        input_dim:    Input feature dimension.
        hidden_dims:  ResBlock sizes.
        proj_dim:     Projection head output dim (kept for weight compatibility).
        num_families: Number of family classes.
        dropout:      Dropout rate.
        freeze_encoder: If True, freeze encoder params (useful for warm-start fine-tuning).
    """

    def __init__(
        self,
        input_dim: int = 2568,
        hidden_dims: Optional[List[int]] = None,
        proj_dim: int = 128,
        num_families: int = 2358,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        use_cosine_classifier: bool = True,
        use_arcface: bool = True,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
    ):
        super().__init__()
        self.encoder   = ResidualMLPEncoder(input_dim, hidden_dims, dropout)
        self.proj_head = ProjectionHead(self.encoder.output_dim, proj_dim)  # kept for compat
        self.det_head  = DetectionHead(self.encoder.output_dim)

        # Family head priority: ArcFace > CosineClassifier > plain Linear
        if use_arcface:
            self.family_head = ArcFaceClassifier(
                self.encoder.output_dim, num_families,
                scale=arcface_scale, margin=arcface_margin,
            )
        elif use_cosine_classifier:
            self.family_head = CosineClassifier(self.encoder.output_dim, num_families)
        else:
            self.family_head = FamilyHead(self.encoder.output_dim, num_families)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        family_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:             (B, input_dim) input features.
            family_labels: (B,) integer labels — passed to ArcFaceClassifier
                           during training so it can inject the angular margin.
                           Pass None (or omit) for inference.
        Returns:
            det_logits:    (B,) raw detection logits.
            family_logits: (B, num_families) logits (with ArcFace margin if training).
        """
        emb = self.encoder(x)
        det = self.det_head(emb)
        # ArcFace needs labels; other heads ignore the argument
        if isinstance(self.family_head, ArcFaceClassifier):
            fam = self.family_head(emb, family_labels)
        else:
            fam = self.family_head(emb)
        return det, fam

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw encoder embeddings for prototype computation."""
        return self.encoder(x)

    def get_backbone_params(self):
        """Generator of encoder + proj_head parameters."""
        yield from self.encoder.parameters()
        yield from self.proj_head.parameters()

    def get_head_params(self):
        """Generator of task head parameters only."""
        yield from self.det_head.parameters()
        yield from self.family_head.parameters()


# ──────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_weights(module: nn.Module) -> None:
    """Kaiming uniform for linear layers, constant for norm layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_contrastive_model(cfg: dict) -> ContrastiveModel:
    m = cfg["model"]
    model = ContrastiveModel(
        input_dim=m["input_dim"],
        hidden_dims=m["hidden_dims"],
        proj_dim=m["projection_dim"],
        dropout=m["dropout"],
    )
    model.apply(init_weights)
    return model


def build_multitask_model(cfg: dict, num_families: int) -> MultiTaskModel:
    m  = cfg["model"]
    s2 = cfg.get("stage2", {})
    model = MultiTaskModel(
        input_dim=m["input_dim"],
        hidden_dims=m["hidden_dims"],
        proj_dim=m["projection_dim"],
        num_families=num_families,
        dropout=m["dropout"],
        use_cosine_classifier=s2.get("use_cosine_classifier", True),
        use_arcface=s2.get("use_arcface", True),
        arcface_scale=s2.get("arcface_scale", 64.0),
        arcface_margin=s2.get("arcface_margin", 0.5),
    )
    model.apply(init_weights)
    return model
