#!/usr/bin/env python3
"""
Stacked ensemble using per-file-type LightGBM and per-file-type neural models.

Steps:
 1. Load per-type LGBM models from `checkpoints/lgbm_per_type/{ft}/lgbm_detection.txt` (if present)
 2. Load per-type neural best checkpoints from `checkpoints/stage2_{ft}/best_model.pt` (if present)
 3. Use the week-based validation split to gather validation predictions and train a logistic regression meta-learner
 4. Apply the meta-learner on test and challenge sets and print metrics

Usage:
    python scripts/ensemble_stacked.py --config configs/default.yaml
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
import joblib

# Ensure repo `DAFCS` package is importable when running script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.data_loader import load_ember_arrays, load_family_metadata, week_split_indices, FILE_TYPES
from src.baseline_lgbm import _to_numpy
from src.evaluate import run_multitask_inference, run_lgbm_inference, evaluate_challenge_detection
from src.model import build_multitask_model
from src.utils import load_checkpoint

import lightgbm as lgb

logger = logging.getLogger("ember2024")


def load_lgbm_per_type(root: Path):
    boosters = {}
    for ft in FILE_TYPES:
        path = root / ft / "lgbm_detection.txt"
        if path.exists():
            boosters[ft] = lgb.Booster(model_file=str(path))
            logger.info(f"Loaded LGBM for {ft} → {path}")
    return boosters


def load_nn_per_type(cfg, device):
    models = {}
    for ft in FILE_TYPES:
        ckpt = Path(cfg.get("stage2", {}).get("checkpoint_dir", "checkpoints/stage2")) / ft / "best_model.pt"
        if ckpt.exists():
            cfg_model = cfg.copy()
            cfg_model["model"]["num_families"] = None  # will be set after metadata load
            model = build_multitask_model(cfg, num_families=cfg_model["model"].get("num_families", 0))
            try:
                load_checkpoint(str(ckpt), model)
                models[ft] = model
                logger.info(f"Loaded NN checkpoint for {ft} → {ckpt}")
            except Exception as e:
                logger.warning(f"Failed loading NN for {ft}: {e}")
    return models


def main(cfg: dict):
    setup_logging(cfg["logging"]["level"])

    data_dir = cfg["data"]["data_dir"]
    X_train, y_train = load_ember_arrays(data_dir, "train", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])
    X_test, y_test = load_ember_arrays(data_dir, "test", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])
    X_ch, y_ch = load_ember_arrays(data_dir, "challenge", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])

    # Load metadata for train/test/challenge to get per-sample file types
    meta = load_family_metadata(data_dir, "train", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=cfg["data"]["min_family_samples"])
    train_idx, val_idx = week_split_indices(meta, val_weeks=cfg["data"]["val_weeks"])

    test_meta = load_family_metadata(data_dir, "test", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=1, reference_family_to_label=meta.family_to_label)
    try:
        challenge_meta = load_family_metadata(data_dir, "challenge", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=1, reference_family_to_label=meta.family_to_label)
    except FileNotFoundError:
        challenge_meta = None

    ftypes_train = np.array(meta.idx_to_filetype)
    ftypes_test = np.array(test_meta.idx_to_filetype) if test_meta is not None else ["unknown"] * len(X_test)
    ftypes_ch = np.array(challenge_meta.idx_to_filetype) if challenge_meta is not None else ["unknown"] * len(X_ch)

    # load per-type LGBMs
    lgbm_root = Path("checkpoints/lgbm_per_type")
    lgbm_boosters = load_lgbm_per_type(lgbm_root)

    # load per-type NNs (note: model building needs num_families; build after reading meta)
    device = None
    try:
        import torch
        device = torch.device(cfg["hardware"].get("device", "cpu"))
    except Exception:
        device = None

    nn_models = {}
    for ft in FILE_TYPES:
        ckpt = Path(cfg.get("stage2", {}).get("checkpoint_dir", "checkpoints/stage2")) / ft / "best_model.pt"
        if ckpt.exists():
            model = build_multitask_model(cfg, num_families=meta.num_families)
            try:
                load_checkpoint(str(ckpt), model, device=device)
                if device is not None:
                    model = model.to(device)
                nn_models[ft] = model
                logger.info(f"Loaded NN model for {ft}")
            except Exception as e:
                logger.warning(f"Failed to load NN for {ft}: {e}")

    # Helper: produce per-sample pair of scores [lgbm_score, nn_score] for a given set of indices and file types
    def gather_scores(X, indices, idx_to_ft_list):
        n = len(indices)
        l_scores = np.zeros(n, dtype=np.float32)
        nn_scores = np.zeros(n, dtype=np.float32)

        # LGBM per-type
        for ft, booster in lgbm_boosters.items():
            # idx_to_ft_list maps full-array indices -> file type; we select rows by passed indices
            mask = np.array([idx_to_ft_list[i] == ft for i in indices])
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            # map to local rows
            rows = [indices[i] for i in idxs]
            X_sub = _to_numpy(X[rows])
            preds = booster.predict(X_sub)
            l_scores[idxs] = preds

        # NN per-type: run model inference per loaded model and pick scores where file-type matches
        for ft, model in nn_models.items():
            # run inference on the subset of indices with that file type
            mask = np.array([idx_to_ft_list[i] == ft for i in indices])
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            rows = [indices[i] for i in idxs]
            X_sub = X[rows]
            res = run_multitask_inference(model, X_sub, np.zeros(len(X_sub), dtype=int), device=device)
            nn_scores[idxs] = res["det_scores"]

        return np.stack([l_scores, nn_scores], axis=1)

    # Build validation stacking dataset (use train metadata file types)
    val_indices = val_idx
    val_feats = gather_scores(X_train, val_indices, ftypes_train)
    val_labels = _to_numpy(y_train[val_indices])

    # Train meta-learner (logistic regression)
    # If both scores are zero (missing models), fallback to baseline average
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    # handle degenerate case
    if np.unique(val_labels).size < 2:
        logger.warning("Validation labels are degenerate; skipping meta-learner training")
    else:
        clf.fit(val_feats, val_labels)
    outdir = Path("checkpoints/ensemble")
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, outdir / "meta_lr.pkl")
    logger.info(f"Trained meta-learner saved to {outdir / 'meta_lr.pkl'}")

    # Apply to test and challenge using their proper metadata file-type lists
    test_indices = np.arange(len(X_test))
    test_feats = gather_scores(X_test, test_indices, ftypes_test)
    if np.unique(val_labels).size < 2:
        # fallback: average of available scores
        test_score = np.nanmean(test_feats, axis=1)
    else:
        test_score = clf.predict_proba(test_feats)[:, 1]

    # Challenge
    ch_indices = np.arange(len(X_ch))
    ch_feats = gather_scores(X_ch, ch_indices, ftypes_ch)
    if np.unique(val_labels).size < 2:
        ch_score = np.nanmean(ch_feats, axis=1)
    else:
        ch_score = clf.predict_proba(ch_feats)[:, 1]

    # Metrics
    from sklearn.metrics import roc_auc_score, average_precision_score

    try:
        test_roc = float(roc_auc_score(_to_numpy(y_test), test_score))
        test_pr = float(average_precision_score(_to_numpy(y_test), test_score))
    except Exception:
        test_roc = test_pr = float('nan')

    # For challenge metrics compute using test benign subset + challenge malware (paper methodology)
    try:
        yt = _to_numpy(y_test)
        ben_mask = yt == 0
        ben_scores = test_score[ben_mask]
        all_scores = np.concatenate([ben_scores, ch_score])
        all_labels = np.concatenate([np.zeros(len(ben_scores)), np.ones(len(ch_score))])
        ch_res = dict(
            roc_auc=float(roc_auc_score(all_labels, all_scores)),
            pr_auc=float(average_precision_score(all_labels, all_scores)),
            det_rate=float((ch_score >= 0.5).mean()),
        )
    except Exception:
        ch_res = {"roc_auc": float('nan'), "pr_auc": float('nan'), "det_rate": float('nan')}

    results = {
        "stacked_test": {"roc_auc": test_roc, "pr_auc": test_pr},
        "stacked_challenge": {**ch_res, "n_samples": int(len(X_ch))},
    }

    out = Path("results/stacked_ensemble_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved stacked results → {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
