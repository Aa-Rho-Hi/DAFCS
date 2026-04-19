"""
Stage 2: Multi-Task Fine-tuning.

Loads the encoder pre-trained in Stage 1, adds detection and family heads,
and trains both simultaneously on the full training set.

Usage:
    python -m src.multitask --config configs/default.yaml \
        --pretrained checkpoints/stage1/best_model.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_ember_arrays,
    load_family_metadata,
    week_split_indices,
    make_multitask_loaders,
)
from src.model import build_multitask_model
from src.losses import MultiTaskLoss
from src.utils import (
    AverageMeter,
    WandbLogger,
    get_cosine_schedule_with_warmup,
    get_device,
    load_checkpoint,
    load_config,
    load_encoder_weights,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("ember2024")


# ──────────────────────────────────────────────────────────────────────────────
# Mix-up augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def mixup_batch(
    features: torch.Tensor,
    det_labels: torch.Tensor,
    fam_labels: torch.Tensor,
    alpha: float = 0.4,
):
    """
    Feature Mix-up (Zhang et al., ICLR 2018) for tabular malware features.

    Interpolates between randomly paired samples in the batch:
        x̃  = λ·x_i + (1−λ)·x_j
    Both label pairs (a and b) are returned so the caller can compute a
    convex combination of the two loss terms.

    Returns:
        mixed_features, det_a, fam_a, det_b, fam_b, lam
    """
    if alpha <= 0.0:
        # No mix-up: return originals with lam=1
        return features, det_labels, fam_labels, det_labels, fam_labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(features.size(0), device=features.device)
    mixed = lam * features + (1.0 - lam) * features[idx]
    return mixed, det_labels, fam_labels, det_labels[idx], fam_labels[idx], lam


def apply_mixup_loss(criterion, det_logits, fam_logits,
                     det_a, fam_a, det_b, fam_b, lam):
    """Compute lam·loss(a) + (1−lam)·loss(b) for mixed samples."""
    total_a, l_det_a, l_fam_a = criterion(det_logits, fam_logits, det_a, fam_a)
    total_b, l_det_b, l_fam_b = criterion(det_logits, fam_logits, det_b, fam_b)
    total  = lam * total_a  + (1.0 - lam) * total_b
    l_det  = lam * l_det_a  + (1.0 - lam) * l_det_b
    l_fam  = lam * l_fam_a  + (1.0 - lam) * l_fam_b
    return total, l_det, l_fam


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, lr: float, backbone_lr_mult: float, weight_decay: float):
    """
    Create AdamW with two parameter groups:
      - Backbone (encoder + proj_head): lr * backbone_lr_mult
      - Task heads: lr
    """
    backbone_params = list(model.get_backbone_params())
    head_params     = list(model.get_head_params())
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": head_params,     "lr": lr},
        ],
        weight_decay=weight_decay,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    log_interval: int = 100,
    epoch: int = 0,
    wandb_logger=None,
    mixup_alpha: float = 0.4,
) -> dict:
    model.train()
    meters = {k: AverageMeter(k) for k in ["loss", "det_loss", "fam_loss"]}
    optimizer.zero_grad()

    for step, (features, det_labels, fam_labels) in enumerate(
        tqdm(loader, desc=f"[Stage 2] Epoch {epoch}", leave=False)
    ):
        features   = features.to(device, non_blocking=True)
        det_labels = det_labels.to(device, non_blocking=True)
        fam_labels = fam_labels.to(device, non_blocking=True)

        # Mix-up: interpolate between paired samples in the batch
        features, det_a, fam_a, det_b, fam_b, lam = mixup_batch(
            features, det_labels, fam_labels, alpha=mixup_alpha
        )

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Pass family labels for ArcFace margin injection during training.
            # Use the 'a' labels (higher weight); 'b' labels are used in the loss.
            det_logits, fam_logits = model(features, family_labels=fam_a)

            if lam < 1.0:
                loss, l_det, l_fam = apply_mixup_loss(
                    criterion, det_logits, fam_logits,
                    det_a, fam_a, det_b, fam_b, lam,
                )
            else:
                loss, l_det, l_fam = criterion(
                    det_logits, fam_logits, det_a, fam_a
                )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        B = features.size(0)
        meters["loss"].update(loss.item() * grad_accum_steps, B)
        meters["det_loss"].update(l_det.item(), B)
        meters["fam_loss"].update(l_fam.item(), B)

        if (step + 1) % log_interval == 0:
            global_step = epoch * len(loader) + step
            logger.info(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"loss={meters['loss'].avg:.4f} | "
                f"det={meters['det_loss'].avg:.4f} | "
                f"fam={meters['fam_loss'].avg:.4f} | "
                f"LR_head={optimizer.param_groups[1]['lr']:.2e}"
            )
            if wandb_logger:
                wandb_logger.log(
                    {
                        "train/loss":     meters["loss"].avg,
                        "train/det_loss": meters["det_loss"].avg,
                        "train/fam_loss": meters["fam_loss"].avg,
                    },
                    step=global_step,
                )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> dict:
    """
    Compute validation losses and detection accuracy.
    Returns dict with loss/det_loss/fam_loss/det_acc.
    """
    from sklearn.metrics import average_precision_score

    model.eval()
    meters = {k: AverageMeter(k) for k in ["loss", "det_loss", "fam_loss"]}

    all_det_preds   = []
    all_det_targets = []
    all_fam_preds   = []
    all_fam_targets = []

    for features, det_labels, fam_labels in tqdm(loader, desc="[Stage 2] Val", leave=False):
        features   = features.to(device, non_blocking=True)
        det_labels = det_labels.to(device, non_blocking=True)
        fam_labels = fam_labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # family_labels=None → ArcFace runs in plain-cosine inference mode
            det_logits, fam_logits = model(features, family_labels=None)
            loss, l_det, l_fam = criterion(
                det_logits, fam_logits, det_labels, fam_labels
            )

        B = features.size(0)
        meters["loss"].update(loss.item(), B)
        meters["det_loss"].update(l_det.item(), B)
        meters["fam_loss"].update(l_fam.item(), B)

        all_det_preds.extend(torch.sigmoid(det_logits).cpu().numpy().tolist())
        all_det_targets.extend(det_labels.cpu().numpy().tolist())

        # Family: only valid (non -1) labels
        fam_valid = fam_labels >= 0
        if fam_valid.any():
            all_fam_preds.extend(fam_logits[fam_valid].argmax(-1).cpu().numpy().tolist())
            all_fam_targets.extend(fam_labels[fam_valid].cpu().numpy().tolist())

    metrics = {k: v.avg for k, v in meters.items()}

    # Detection PR AUC
    try:
        metrics["det_pr_auc"] = float(
            average_precision_score(all_det_targets, all_det_preds)
        )
    except Exception:
        metrics["det_pr_auc"] = float("nan")

    # Family accuracy
    if all_fam_targets:
        correct = sum(p == t for p, t in zip(all_fam_preds, all_fam_targets))
        metrics["fam_acc"] = correct / len(all_fam_targets)
    else:
        metrics["fam_acc"] = float("nan")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    set_seed(cfg["hardware"]["seed"])
    setup_logging(cfg["logging"]["level"])
    device = get_device(cfg["hardware"]["device"])
    logger.info(f"Using device: {device}")

    wandb_logger = WandbLogger(cfg, enabled=cfg["logging"]["use_wandb"])

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = cfg["data"]["data_dir"]
    logger.info(f"Loading EMBER2024 training data from {data_dir} …")

    X_train, y_train = load_ember_arrays(
        data_dir, "train",
        use_memmap=cfg["data"]["use_memmap"],
        feature_dim=cfg["data"]["input_dim"],
    )
    meta = load_family_metadata(
        data_dir, "train",
        min_confidence=cfg["data"]["family_confidence_threshold"],
        min_samples=cfg["data"]["min_family_samples"],
    )
    train_idx, val_idx = week_split_indices(meta, val_weeks=cfg["data"]["val_weeks"])

    s2 = cfg["stage2"]
    train_loader, val_loader = make_multitask_loaders(
        X_train, y_train, meta, train_idx, val_idx,
        batch_size=s2["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # ── Positive-class weight for detection (benign >> malicious) ───────────
    n_mal = int((y_train[train_idx] == 1).sum())
    n_ben = int((y_train[train_idx] == 0).sum())
    pos_weight = n_ben / max(n_mal, 1)
    logger.info(f"  Train: {n_mal:,} malicious, {n_ben:,} benign  (pos_weight={pos_weight:.2f})")

    # ── Model ───────────────────────────────────────────────────────────────
    cfg["model"]["num_families"] = meta.num_families
    model = build_multitask_model(cfg, num_families=meta.num_families).to(device)

    # Load pretrained encoder if provided
    pretrained = s2.get("pretrained_backbone") or cfg.get("stage2", {}).get("pretrained_backbone")
    if pretrained:
        logger.info(f"Loading pretrained encoder from {pretrained} …")
        load_encoder_weights(pretrained, model, device)

    logger.info(f"MultiTaskModel: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Optimiser & Scheduler ───────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        lr=s2["lr"],
        backbone_lr_mult=s2["backbone_lr_multiplier"],
        weight_decay=s2["weight_decay"],
    )
    total_steps  = s2["epochs"] * len(train_loader) // s2["gradient_accumulation_steps"]
    warmup_steps = s2["warmup_epochs"] * len(train_loader) // s2["gradient_accumulation_steps"]
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and s2["mixed_precision"])
    )

    criterion = MultiTaskLoss(
        samples_per_class=meta.samples_per_class,
        focal_gamma=s2["focal_gamma"],
        focal_alpha=s2.get("focal_alpha"),
        cb_beta=s2["class_balanced_beta"],
        logit_adj_tau=s2.get("logit_adj_tau", 1.0),
        use_logit_adj=s2.get("use_logit_adj", True),
        family_weight_mode=s2.get("family_weight_mode", "effective_num"),
        family_label_smoothing=s2.get("family_label_smoothing", 0.0),
        detection_weight=s2["detection_weight"],
        family_weight=s2["family_weight"],
        pos_weight=pos_weight,
    ).to(device)

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = 0.0     # maximise det_pr_auc
    if s2.get("resume"):
        ckpt = load_checkpoint(s2["resume"], model, optimizer, device)
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_metric  = ckpt.get("best_det_pr_auc", 0.0)

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, s2["epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, s2["gradient_accumulation_steps"],
            log_interval=cfg["logging"]["log_interval"],
            epoch=epoch, wandb_logger=wandb_logger,
            mixup_alpha=s2.get("mixup_alpha", 0.4),
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        current_metric = val_metrics.get("det_pr_auc", 0.0)
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric

        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_det_pr_auc={val_metrics['det_pr_auc']:.4f} | "
            f"val_fam_acc={val_metrics.get('fam_acc', float('nan')):.4f} | "
            f"best_pr_auc={best_metric:.4f}"
            + (" ← best" if is_best else "")
        )
        if wandb_logger:
            wandb_logger.log({"epoch": epoch, **{f"val/{k}": v for k, v in val_metrics.items()}})

        if (epoch + 1) % s2.get("save_every", 5) == 0 or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_det_pr_auc": best_metric,
                    "num_families": meta.num_families,
                    "family_to_label": meta.family_to_label,
                    "config": cfg,
                },
                checkpoint_dir=s2["checkpoint_dir"],
                filename=f"epoch_{epoch:03d}.pt",
                is_best=is_best,
            )

    wandb_logger.finish()
    logger.info("Stage 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Multi-Task Fine-tuning")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--pretrained", default=None, help="Stage 1 checkpoint")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.pretrained:
        cfg["stage2"]["pretrained_backbone"] = args.pretrained
    if args.resume:
        cfg["stage2"]["resume"] = args.resume
    if args.wandb:
        cfg["logging"]["use_wandb"] = True

    env_dir = os.environ.get("EMBER2024_DIR")
    if env_dir and cfg["data"]["data_dir"] == "/path/to/ember2024":
        cfg["data"]["data_dir"] = env_dir

    main(cfg)
