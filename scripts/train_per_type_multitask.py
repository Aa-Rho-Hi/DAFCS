#!/usr/bin/env python3
"""
Train MultiTask neural models per file type.

This script reuses functions from `src.multitask` to train a separate
MultiTaskModel for each file type. Checkpoints are written to
`checkpoints/stage2_{file_type}`.

Usage:
    python scripts/train_per_type_multitask.py --config configs/default.yaml
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

# Ensure repo `DAFCS` package is importable when running script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_ember_arrays, load_family_metadata, week_split_indices, make_multitask_loaders, FILE_TYPES
from src.model import build_multitask_model
from src.multitask import train_one_epoch, evaluate, build_optimizer
from src.losses import MultiTaskLoss
from src.utils import load_config, setup_logging, set_seed, save_checkpoint

logger = logging.getLogger("ember2024")


def main(cfg: dict):
    setup_logging(cfg["logging"]["level"])
    set_seed(cfg["hardware"]["seed"])
    device = torch.device(cfg["hardware"].get("device", "cpu"))

    data_dir = cfg["data"]["data_dir"]
    X_train, y_train = load_ember_arrays(data_dir, "train", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])
    meta = load_family_metadata(data_dir, "train", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=cfg["data"]["min_family_samples"])
    train_idx, val_idx = week_split_indices(meta, val_weeks=cfg["data"]["val_weeks"])

    s2 = cfg["stage2"]

    ftypes = np.array(meta.idx_to_filetype)

    for ft in FILE_TYPES:
        # restrict indices to this file type
        tr_mask = (ftypes[train_idx] == ft)
        va_mask = (ftypes[val_idx] == ft)
        tr_idx = train_idx[tr_mask]
        va_idx = val_idx[va_mask]

        if len(tr_idx) < 256:
            logger.warning(f"Skipping {ft}: too few train samples ({len(tr_idx)})")
            continue

        logger.info(f"Training MultiTaskModel for {ft}: train={len(tr_idx)}, val={len(va_idx)}")

        # loaders
        train_loader, val_loader = make_multitask_loaders(
            X_train, y_train, meta, tr_idx, va_idx,
            batch_size=s2["batch_size"], num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
        )

        cfg["model"]["num_families"] = meta.num_families
        model = build_multitask_model(cfg, num_families=meta.num_families).to(device)

        optimizer = build_optimizer(model, lr=s2["lr"], backbone_lr_mult=s2["backbone_lr_multiplier"], weight_decay=s2["weight_decay"])
        total_steps = max(1, s2["epochs"] * len(train_loader) // s2["gradient_accumulation_steps"])
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and s2.get("mixed_precision", False)))

        # positive class weight
        n_mal = int((y_train[tr_idx] == 1).sum())
        n_ben = int((y_train[tr_idx] == 0).sum())
        pos_weight = n_ben / max(n_mal, 1)

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

        best_metric = 0.0
        out_dir = Path(s2.get("checkpoint_dir", "checkpoints/stage2")) / ft
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(s2["epochs"]):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, None, scaler, device,
                grad_accum_steps=s2["gradient_accumulation_steps"],
                log_interval=cfg["logging"]["log_interval"], epoch=epoch,
                wandb_logger=None, mixup_alpha=s2.get("mixup_alpha", 0.4),
            )
            val_metrics = evaluate(model, val_loader, criterion, device)

            current_metric = val_metrics.get("det_pr_auc", 0.0)
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric

            logger.info(f"{ft} Epoch {epoch} | train_loss={train_metrics['loss']:.4f} | val_pr={val_metrics['det_pr_auc']:.4f} | best_pr={best_metric:.4f}")

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
                    checkpoint_dir=str(out_dir),
                    filename=f"epoch_{epoch:03d}.pt",
                    is_best=is_best,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
