#!/usr/bin/env python3
"""
Train LightGBM detection models per file type.

Usage:
    python scripts/train_per_type_lgbm.py --config configs/default.yaml

Saves models to `checkpoints/lgbm_per_type/{file_type}/lgbm_detection.txt`.
"""
import sys
import argparse
import logging
from pathlib import Path
import numpy as np

# Ensure repo `DAFCS` package is importable when running script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_ember_arrays, load_family_metadata, week_split_indices, FILE_TYPES
from src.utils import load_config, setup_logging, set_seed

logger = logging.getLogger("ember2024")


def main(cfg: dict):
    setup_logging(cfg["logging"]["level"])
    set_seed(cfg["hardware"]["seed"])

    data_dir = cfg["data"]["data_dir"]
    X_train, y_train = load_ember_arrays(data_dir, "train", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])
    meta = load_family_metadata(data_dir, "train", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=cfg["data"]["min_family_samples"])
    train_idx, val_idx = week_split_indices(meta, val_weeks=cfg["data"]["val_weeks"])

    # Use baseline training helper
    from src.baseline_lgbm import train_detection_model

    out_root = Path("checkpoints/lgbm_per_type")
    out_root.mkdir(parents=True, exist_ok=True)

    ftypes = np.array(meta.idx_to_filetype)

    for ft in FILE_TYPES:
        mask_train = (ftypes[train_idx] == ft)
        mask_val = (ftypes[val_idx] == ft)
        tr_idx = train_idx[mask_train]
        va_idx = val_idx[mask_val]

        if len(tr_idx) < 100:
            logger.warning(f"Skipping {ft}: too few train samples ({len(tr_idx)})")
            continue

        out_dir = out_root / ft
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training LGBM detection for {ft}: train={len(tr_idx)}, val={len(va_idx)}")

        model, model_path = train_detection_model(X_train, y_train, tr_idx, va_idx, str(out_dir))
        logger.info(f"Saved {ft} model → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
