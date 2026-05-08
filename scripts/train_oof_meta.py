#!/usr/bin/env python3
"""
Train a meta-learner using out-of-fold (OOF) stacking features.

Features produced per sample:
 - per-type LGBM calibrated score (if model exists)
 - per-type NN calibrated score (if model exists)
 - global LGBM score (if model exists)
 - flags: has_per_type_lgbm, has_per_type_nn
 - file-type one-hot (sparse)

Procedure:
 - Use StratifiedKFold on labeled training samples to produce OOF predictions
 - Fit per-file-type Platt-scaler (logistic) on each fold's train split for calibration
 - Train a LogisticRegression meta-learner on OOF features
 - Save meta-learner and per-sample OOF predictions to `checkpoints/ensemble` and `results/raw_stack_preds.npz`

This script is intended to run on the full EMBER2024 training set (not smoke).
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Ensure repo `DAFCS` package is importable when running script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_ember_arrays, load_family_metadata, week_split_indices, FILE_TYPES
from src.baseline_lgbm import _to_numpy
from src.utils import load_config, setup_logging
from src.utils import load_checkpoint
from src.model import build_multitask_model
from src.evaluate import run_multitask_inference
import lightgbm as lgb

logger = logging.getLogger("ember2024")


def load_lgbm_per_type(root: Path) -> Dict[str, lgb.Booster]:
    boosters = {}
    for ft in FILE_TYPES:
        path = root / ft / "lgbm_detection.txt"
        if path.exists():
            boosters[ft] = lgb.Booster(model_file=str(path))
            logger.info(f"Loaded LGBM for {ft} from {path}")
    return boosters


def load_nn_per_type(checkpoint_dir: str, cfg, meta) -> Dict[str, object]:
    models = {}
    try:
        import torch
        device = torch.device(cfg["hardware"].get("device", "cpu"))
    except Exception:
        device = None

    for ft in FILE_TYPES:
        ckpt = Path(checkpoint_dir) / ft / "best_model.pt"
        if ckpt.exists():
            model = build_multitask_model(cfg, num_families=meta.num_families)
            try:
                load_checkpoint(str(ckpt), model, device=device)
                if device is not None:
                    model = model.to(device)
                models[ft] = model
                logger.info(f"Loaded NN model for {ft}")
            except Exception as e:
                logger.warning(f"Failed loading NN {ft}: {e}")
    return models


def calibrate_platt(scores_train, y_train):
    """Fit a Platt-scaler (logistic) for calibration. Returns (coeffs) callable."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y_train)) < 2:
        # Degenerate: return identity
        return lambda x: x
    lr = LogisticRegression(solver="lbfgs", max_iter=200)
    lr.fit(scores_train.reshape(-1, 1), y_train)
    return lambda x: lr.predict_proba(np.array(x).reshape(-1, 1))[:, 1]


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    setup_logging(cfg["logging"]["level"])

    data_dir = cfg["data"]["data_dir"]
    X_train, y_train = load_ember_arrays(data_dir, "train", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])
    meta = load_family_metadata(data_dir, "train", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=cfg["data"]["min_family_samples"])

    # Prepare holders
    N = len(X_train)
    labeled_mask = _to_numpy(y_train) != -1
    labeled_idx = np.where(labeled_mask)[0]
    y_labeled = _to_numpy(y_train[labeled_idx])

    # Load models
    lgbm_boosters = load_lgbm_per_type(Path("checkpoints/lgbm_per_type"))
    global_lgbm_path = Path("checkpoints/lgbm/lgbm_detection.txt")
    global_booster = None
    if global_lgbm_path.exists():
        global_booster = lgb.Booster(model_file=str(global_lgbm_path))

    nn_models = load_nn_per_type(cfg.get("stage2", {}).get("checkpoint_dir", "checkpoints/stage2"), cfg, meta)

    # OOF feature arrays for labeled samples
    # We'll produce features: per-type lgbm, per-type nn, global lgbm, flags, and file-type one-hot
    num_ft = len(FILE_TYPES)
    feats_oof = []

    # For memory, we'll create a feature matrix of shape (n_labeled, 2 + 1 + num_ft + 2)
    # But per-type lgbm/nn -> we will store only the matching file-type score and zeros elsewhere
    X_oof = np.zeros((len(labeled_idx), 2 + 1 + num_ft + 2), dtype=np.float32)
    # Columns mapping:
    # 0: per-type lgbm score (for sample's type)
    # 1: per-type nn score
    # 2: global lgbm score
    # 3..3+num_ft-1: one-hot file-type
    # last two: has_per_type_lgbm, has_per_type_nn

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["hardware"]["seed"])

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_labeled)), y_labeled)):
        logger.info(f"Fold {fold}: tr={len(tr_idx)} va={len(va_idx)}")

        # map to global indices
        tr_global = labeled_idx[tr_idx]
        va_global = labeled_idx[va_idx]

        # calibrators per file-type for LGBM and NN
        lgbm_calib = {}
        nn_calib = {}

        # Fit per-file-type Platt on tr_global
        ftypes = np.array(meta.idx_to_filetype)

        for ft in FILE_TYPES:
            tr_mask = np.isin(tr_global, np.where(ftypes == ft)[0])
            if tr_mask.sum() == 0:
                continue
            tr_rows = tr_global[tr_mask]
            y_tr = _to_numpy(y_train[tr_rows])

            # LGBM raw scores
            if ft in lgbm_boosters:
                preds_tr = lgbm_boosters[ft].predict(_to_numpy(X_train[tr_rows]))
                try:
                    lgbm_calib[ft] = calibrate_platt(preds_tr, y_tr)
                except Exception:
                    lgbm_calib[ft] = lambda x: x

            # NN raw scores
            if ft in nn_models:
                res = run_multitask_inference(nn_models[ft], X_train[tr_rows], _to_numpy(y_train[tr_rows]), device=None)
                preds_tr = res["det_scores"]
                try:
                    nn_calib[ft] = calibrate_platt(preds_tr, y_tr)
                except Exception:
                    nn_calib[ft] = lambda x: x

        # Global LGBM calibrator
        if global_booster is not None:
            preds_global_tr = global_booster.predict(_to_numpy(X_train[tr_global]))
            try:
                global_calib = calibrate_platt(preds_global_tr, _to_numpy(y_train[tr_global]))
            except Exception:
                global_calib = lambda x: x
        else:
            global_calib = lambda x: x

        # Now fill OOF predictions for va_global
        for j, gi in enumerate(va_global):
            ft = ftypes[gi]
            ft_idx = FILE_TYPES.index(ft) if ft in FILE_TYPES else -1

            # per-type lgbm
            if ft in lgbm_boosters and ft in lgbm_calib:
                raw = lgbm_boosters[ft].predict(_to_numpy(X_train[gi:gi+1]))[0]
                score_lgbm = lgbm_calib[ft]([raw])[0]
                has_lgbm = 1
            else:
                score_lgbm = 0.0
                has_lgbm = 0

            # per-type nn
            if ft in nn_models and ft in nn_calib:
                res = run_multitask_inference(nn_models[ft], X_train[gi:gi+1], np.array([0]), device=None)
                raw = res["det_scores"][0]
                score_nn = nn_calib[ft]([raw])[0]
                has_nn = 1
            else:
                score_nn = 0.0
                has_nn = 0

            # global
            if global_booster is not None:
                rawg = global_booster.predict(_to_numpy(X_train[gi:gi+1]))[0]
                score_global = global_calib([rawg])[0]
            else:
                score_global = 0.0

            # build feature row
            row = np.zeros(X_oof.shape[1], dtype=np.float32)
            row[0] = score_lgbm
            row[1] = score_nn
            row[2] = score_global
            if ft_idx >= 0:
                row[3 + ft_idx] = 1.0
            row[-2] = has_lgbm
            row[-1] = has_nn

            # compute position in labeled_idx
            pos = np.where(labeled_idx == gi)[0][0]
            X_oof[pos] = row

    # Train meta learner on OOF features
    meta_clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    meta_clf.fit(X_oof, y_labeled)

    outdir = Path("checkpoints/ensemble")
    outdir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(meta_clf, outdir / "meta_oof_lr.pkl")
    logger.info(f"Saved meta learner to {outdir / 'meta_oof_lr.pkl'}")

    # Save OOF features and labels for debugging
    out = Path("results/raw_stack_oof.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X_oof=X_oof, y=y_labeled, labeled_idx=labeled_idx)
    logger.info(f"Saved OOF features → {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
