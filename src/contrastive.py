"""
Stage 1: Supervised Contrastive Pre-training.

Usage:
    python -m src.contrastive --config configs/default.yaml [--data_dir /path/to/ember2024]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Allow running as a module from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_ember_arrays,
    load_family_metadata,
    week_split_indices,
    make_contrastive_loader,
    EMBER2024ContrastiveDataset,
    PrototypicalBatchSampler,
)
from src.model import build_contrastive_model, feature_augment
from src.losses import SupConLoss
from src.utils import (
    AverageMeter,
    WandbLogger,
    get_cosine_schedule_with_warmup,
    get_device,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("ember2024")


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: SupConLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    log_interval: int = 100,
    epoch: int = 0,
    wandb_logger: "WandbLogger" = None,
    aug_mask_prob: float = 0.3,
    aug_noise_std: float = 0.05,
) -> float:
    """
    One epoch of contrastive training with two-view feature augmentation.

    Two-view strategy
    ─────────────────
    For each original batch of B samples, we create two independently augmented
    copies (view_1, view_2) and concatenate them → 2B samples with labels
    repeated [y, y]. This doubles the number of positive pairs per batch without
    changing the batch-sampler, improving the quality of the contrastive signal.

    Returns average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter("SupConLoss")
    optimizer.zero_grad()

    for step, (features, labels) in enumerate(tqdm(loader, desc=f"[Stage 1] Epoch {epoch}", leave=False)):
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        # Two independently augmented views — gives every sample a natural positive pair
        view1 = feature_augment(features, mask_prob=aug_mask_prob, noise_std=aug_noise_std)
        view2 = feature_augment(features, mask_prob=aug_mask_prob, noise_std=aug_noise_std)
        # Concatenate: (2B, D) with repeated labels (2B,)
        aug_features = torch.cat([view1, view2], dim=0)
        aug_labels   = torch.cat([labels, labels], dim=0)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings = model(aug_features)       # (2B, proj_dim) L2-normalised
            loss = criterion(embeddings, aug_labels)
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

        loss_meter.update(loss.item() * grad_accum_steps, n=features.size(0))

        if (step + 1) % log_interval == 0:
            global_step = epoch * len(loader) + step
            logger.info(
                f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss {loss_meter.avg:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )
            if wandb_logger:
                wandb_logger.log(
                    {"train/loss": loss_meter.avg, "train/lr": optimizer.param_groups[0]["lr"]},
                    step=global_step,
                )

    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: SupConLoss,
    device: torch.device,
) -> float:
    """Compute contrastive loss on a validation set."""
    model.eval()
    loss_meter = AverageMeter("ValSupConLoss")

    for features, labels in tqdm(loader, desc="[Stage 1] Val", leave=False):
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings = model(features)
            loss = criterion(embeddings, labels)

        loss_meter.update(loss.item(), n=features.size(0))

    return loss_meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
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
        data_dir, subset="train",
        use_memmap=cfg["data"]["use_memmap"],
        feature_dim=cfg["data"]["input_dim"],
    )
    logger.info("Loading family metadata …")
    meta = load_family_metadata(
        data_dir,
        subset="train",
        min_confidence=cfg["data"]["family_confidence_threshold"],
        min_samples=cfg["data"]["min_family_samples"],
    )

    train_idx, val_idx = week_split_indices(meta, val_weeks=cfg["data"]["val_weeks"])
    logger.info(f"Families for contrastive learning: {meta.num_families}")

    s1 = cfg["stage1"]

    train_loader = make_contrastive_loader(
        X_train, meta, train_idx,
        n_classes_per_batch=s1["n_classes_per_batch"],
        n_samples_per_class=s1["n_samples_per_class"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    # Validation: use family dataset with a standard shuffled loader
    from torch.utils.data import DataLoader
    val_ds = EMBER2024ContrastiveDataset(
        X_train, meta.idx_to_family_label, indices=val_idx
    )
    val_sampler = PrototypicalBatchSampler(
        labels=val_ds.local_family_labels,
        n_classes_per_batch=min(s1["n_classes_per_batch"], 32),
        n_samples_per_class=s1["n_samples_per_class"],
        n_iterations=50,   # fixed number of val batches
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_contrastive_model(cfg).to(device)
    logger.info(
        f"ContrastiveModel: {sum(p.numel() for p in model.parameters()):,} params"
    )

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=s1["lr"],
        weight_decay=s1["weight_decay"],
    )
    total_steps   = s1["epochs"] * len(train_loader) // s1["gradient_accumulation_steps"]
    warmup_steps  = s1["warmup_epochs"] * len(train_loader) // s1["gradient_accumulation_steps"]
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and s1["mixed_precision"])
    )
    criterion = SupConLoss(
        temperature=s1["temperature"],
        hard_neg_weight=s1.get("hard_neg_weight", 0.5),
        hard_neg_temp=s1.get("hard_neg_temp", 0.2),
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    if s1.get("resume"):
        from src.utils import load_checkpoint
        ckpt = load_checkpoint(s1["resume"], model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val_loss", float("inf"))

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, s1["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, s1["gradient_accumulation_steps"],
            log_interval=cfg["logging"]["log_interval"],
            epoch=epoch, wandb_logger=wandb_logger,
            aug_mask_prob=s1.get("aug_mask_prob", 0.3),
            aug_noise_std=s1.get("aug_noise_std", 0.05),
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        logger.info(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | best={best_val:.4f}"
            + (" ← best" if is_best else "")
        )
        if wandb_logger:
            wandb_logger.log({
                "epoch": epoch, "val/loss": val_loss, "val/best_loss": best_val,
            })

        # Save checkpoint
        if (epoch + 1) % s1.get("save_every", 5) == 0 or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "best_val_loss": best_val,
                    "config": cfg,
                },
                checkpoint_dir=s1["checkpoint_dir"],
                filename=f"epoch_{epoch:03d}.pt",
                is_best=is_best,
            )

    wandb_logger.finish()
    logger.info("Stage 1 complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Supervised Contrastive Pre-training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None, help="Override data.data_dir in config")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.resume:
        cfg["stage1"]["resume"] = args.resume
    if args.wandb:
        cfg["logging"]["use_wandb"] = True

    # Allow overriding data_dir via environment variable
    env_dir = os.environ.get("EMBER2024_DIR")
    if env_dir and cfg["data"]["data_dir"] == "/path/to/ember2024":
        cfg["data"]["data_dir"] = env_dir

    main(cfg)
