#!/usr/bin/env python3
"""
Compute stacked ensemble metrics on the training set and update results JSON.

Usage:
  python scripts/compute_stacked_train_metrics.py --config configs/smoke.yaml

This loads `checkpoints/ensemble/meta_lr.pkl`, per-type LGBMs in
`checkpoints/lgbm_per_type`, and per-type NN checkpoints in
`checkpoints/stage2/{FT}/best_model.pt` (if present), then scores the
training set and appends `stacked_train` metrics to
`results/stacked_ensemble_results.json`.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure repo `DAFCS` package is importable when running script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.data_loader import load_ember_arrays, load_family_metadata, week_split_indices, FILE_TYPES
from src.baseline_lgbm import _to_numpy
from src.evaluate import run_multitask_inference
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
    return boosters


def main(cfg: dict):
    setup_logging(cfg["logging"]["level"])

    data_dir = cfg["data"]["data_dir"]
    X_train, y_train = load_ember_arrays(data_dir, "train", use_memmap=cfg["data"]["use_memmap"], feature_dim=cfg["data"]["input_dim"])

    meta = load_family_metadata(data_dir, "train", min_confidence=cfg["data"]["family_confidence_threshold"], min_samples=cfg["data"]["min_family_samples"])

    # Load models
    lgbm_root = Path("checkpoints/lgbm_per_type")
    lgbm_boosters = load_lgbm_per_type(lgbm_root)

    # load meta-learner
    meta_path = Path("checkpoints/ensemble/meta_lr.pkl")
    clf = None
    if meta_path.exists():
        clf = joblib.load(meta_path)
        logger.info(f"Loaded meta-learner {meta_path}")
    else:
        logger.warning("Meta-learner not found; falling back to average of available scores")

    # load NN models per type
    nn_models = {}
    try:
        import torch
        device = torch.device(cfg["hardware"].get("device", "cpu"))
    except Exception:
        device = None

    for ft in FILE_TYPES:
        ckpt = Path(cfg.get("stage2", {}).get("checkpoint_dir", "checkpoints/stage2")) / ft / "best_model.pt"
        if ckpt.exists():
            model = build_multitask_model(cfg, num_families=meta.num_families)
            try:
                load_checkpoint(str(ckpt), model, device=device)
                if device is not None:
                    model = model.to(device)
                nn_models[ft] = model
            except Exception as e:
                logger.warning(f"Failed to load NN for {ft}: {e}")

    # Helper to gather per-sample features [lgbm_score, nn_score]
    ftypes = np.array(meta.idx_to_filetype)
    N = len(X_train)
    l_scores = np.zeros(N, dtype=np.float32)
    nn_scores = np.zeros(N, dtype=np.float32)

    for ft, booster in lgbm_boosters.items():
        mask = (ftypes == ft)
        if not mask.any():
            continue
        rows = np.where(mask)[0]
        preds = booster.predict(_to_numpy(X_train[rows]))
        l_scores[rows] = preds

    for ft, model in nn_models.items():
        mask = (ftypes == ft)
        if not mask.any():
            continue
        rows = np.where(mask)[0]
        X_sub = X_train[rows]
        res = run_multitask_inference(model, X_sub, np.zeros(len(X_sub), dtype=int), device=device)
        nn_scores[rows] = res["det_scores"]

    feats = np.stack([l_scores, nn_scores], axis=1)

    if clf is None:
        # average available nonzero entries per sample
        score = np.nanmean(feats, axis=1)
    else:
        score = clf.predict_proba(feats)[:, 1]

    # compute metrics on labeled training samples
    labeled = np.where(_to_numpy(y_train) != -1)[0]
    if labeled.size > 0 and np.unique(_to_numpy(y_train[labeled])).size > 1:
        tr_roc = float(roc_auc_score(_to_numpy(y_train[labeled]), score[labeled]))
        tr_pr = float(average_precision_score(_to_numpy(y_train[labeled]), score[labeled]))
    else:
        tr_roc = float('nan')
        tr_pr = float('nan')

    # Load existing results and update
    res_path = Path("results/stacked_ensemble_results.json")
    if res_path.exists():
        data = json.loads(res_path.read_text())
    else:
        data = {}

    data["stacked_train"] = {"roc_auc": tr_roc, "pr_auc": tr_pr}
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Updated results with stacked_train → {res_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
