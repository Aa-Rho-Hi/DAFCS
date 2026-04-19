"""
LightGBM baselines — reproduces the paper's detection and family-classification
baselines for both the test set and the challenge set.

Usage:
    python -m src.baseline_lgbm --config configs/default.yaml --task detection
    python -m src.baseline_lgbm --config configs/default.yaml --task family
    python -m src.baseline_lgbm --config configs/default.yaml --task all
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    FILE_TYPES,
    load_ember_arrays,
    load_family_metadata,
    week_split_indices,
)
from src.utils import (
    load_config,
    print_results_table,
    save_json,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("ember2024")

# ──────────────────────────────────────────────────────────────────────────────
# LightGBM hyper-parameters matching the EMBER2024 paper baseline
# ──────────────────────────────────────────────────────────────────────────────

LGBM_DETECTION_PARAMS = {
    "objective":           "binary",
    "metric":              ["binary_logloss", "auc"],
    "num_leaves":          1024,
    "max_depth":           -1,
    "min_child_samples":   20,
    "learning_rate":       0.05,
    "n_estimators":        3000,
    "subsample":           0.5,
    "colsample_bytree":    0.5,
    "reg_alpha":           0.1,
    "reg_lambda":          0.1,
    "n_jobs":              -1,
    "verbose":             -1,
    "early_stopping_rounds": 100,
}

LGBM_FAMILY_PARAMS = {
    "objective":           "multiclass",
    "metric":              ["multi_logloss"],
    "num_leaves":          1024,
    "max_depth":           -1,
    "min_child_samples":   5,
    "learning_rate":       0.05,
    "n_estimators":        2000,
    "subsample":           0.5,
    "colsample_bytree":    0.5,
    "class_weight":        "balanced",
    "n_jobs":              -1,
    "verbose":             -1,
    "early_stopping_rounds": 50,
}


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(arr: np.ndarray) -> np.ndarray:
    """Force-load a memory-mapped array into a real numpy array."""
    if isinstance(arr, np.memmap):
        return np.array(arr)
    return arr


def train_detection_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: str,
    params: Optional[Dict] = None,
):
    """Train and save a LightGBM detection model."""
    import lightgbm as lgb

    if params is None:
        params = LGBM_DETECTION_PARAMS.copy()

    # Filter labeled samples
    tr_mask = y_train[train_idx] != -1
    va_mask = y_train[val_idx] != -1
    tr_idx  = train_idx[tr_mask]
    va_idx  = val_idx[va_mask]

    logger.info(
        f"Detection: loading {len(tr_idx):,} train + {len(va_idx):,} val samples …"
    )
    # Load into RAM for LightGBM (required — LGB cannot use memmap)
    X_tr = _to_numpy(X_train[tr_idx])
    y_tr = y_train[tr_idx].astype(np.int32)
    X_va = _to_numpy(X_train[va_idx])
    y_va = y_train[val_idx[va_mask]].astype(np.int32)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 100)),
                   lgb.log_evaluation(100)],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / "lgbm_detection.txt")
    model.booster_.save_model(model_path)
    logger.info(f"LightGBM detection model saved → {model_path}")
    return model, model_path


def train_family_model(
    X_train: np.ndarray,
    meta,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: str,
    params: Optional[Dict] = None,
):
    """Train and save a LightGBM family-classification model."""
    import lightgbm as lgb

    if params is None:
        params = LGBM_FAMILY_PARAMS.copy()
    params = dict(params)
    params["num_class"] = meta.num_families

    # Filter to samples with known family labels
    fam_labels = meta.idx_to_family_label
    tr_mask = fam_labels[train_idx] >= 0
    va_mask = fam_labels[val_idx] >= 0
    tr_idx  = train_idx[tr_mask]
    va_idx  = val_idx[va_mask]

    logger.info(
        f"Family: loading {len(tr_idx):,} train + {len(va_idx):,} val samples, "
        f"{meta.num_families} classes …"
    )
    X_tr = _to_numpy(X_train[tr_idx])
    y_tr = fam_labels[tr_idx].astype(np.int32)
    X_va = _to_numpy(X_train[va_idx])
    y_va = fam_labels[va_idx].astype(np.int32)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 50)),
                   lgb.log_evaluation(50)],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(output_dir / "lgbm_family.txt")
    model.booster_.save_model(model_path)
    logger.info(f"LightGBM family model saved → {model_path}")
    return model, model_path


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_detection(model, X: np.ndarray, y: np.ndarray, file_type_list, label: str) -> Dict:
    """Evaluate detection model on a dataset, broken down by file type."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    labeled = np.where(y != -1)[0]
    X_eval = _to_numpy(X[labeled])
    y_eval = y[labeled].astype(int)

    scores = model.predict_proba(X_eval)[:, 1]
    preds  = (scores >= 0.5).astype(int)

    results = {}

    def _metrics(yt, ys):
        if len(np.unique(yt)) < 2:
            return {"pr_auc": float("nan"), "roc_auc": float("nan"), "f1": float("nan")}
        return {
            "pr_auc":  float(average_precision_score(yt, ys)),
            "roc_auc": float(roc_auc_score(yt, ys)),
            "f1":      float(f1_score(yt, (ys >= 0.5).astype(int), zero_division=0)),
        }

    results["Overall"] = _metrics(y_eval, scores)

    ft_arr = np.array([file_type_list[i] if i < len(file_type_list) else "unknown" for i in labeled])
    for ft in FILE_TYPES + ["unknown"]:
        mask = ft_arr == ft
        if mask.sum() > 0:
            results[ft] = _metrics(y_eval[mask], scores[mask])

    logger.info(f"\n[{label}] LightGBM detection:")
    for ft, m in results.items():
        logger.info(f"  {ft:<20}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")
    return results


def evaluate_family(model, X: np.ndarray, meta, label: str) -> Dict:
    """Evaluate family-classification model."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    fam_labels = meta.idx_to_family_label
    valid_idx = np.where(fam_labels >= 0)[0]
    if len(valid_idx) == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
        }

    X_eval = _to_numpy(X[valid_idx])
    y_true = fam_labels[valid_idx]

    y_pred = model.predict(X_eval)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    logger.info(
        f"[{label}] LightGBM family: accuracy={acc:.4f}  macro_F1={macro_f1:.4f}  "
        f"weighted_F1={weighted_f1:.4f}"
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
    }


def evaluate_challenge(det_model, data_dir: str) -> Dict:
    """Evaluate detection on challenge set."""
    try:
        X_ch, y_ch = load_ember_arrays(data_dir, subset="challenge", use_memmap=False)
        scores = det_model.predict_proba(_to_numpy(X_ch))[:, 1]
        det_rate  = float((scores >= 0.5).mean())
        avg_score = float(scores.mean())
        logger.info(f"Challenge: detection_rate={det_rate:.4f}  avg_score={avg_score:.4f}")
        return {"n_samples": len(X_ch), "detection_rate": det_rate, "avg_det_score": avg_score}
    except FileNotFoundError:
        logger.warning("Challenge set not found — skipping.")
        return {"error": "challenge set not found"}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: dict, task: str = "all") -> None:
    set_seed(cfg["hardware"]["seed"])
    setup_logging(cfg["logging"]["level"])

    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "LightGBM is not installed. Install requirements.txt or skip the baseline stage."
        ) from exc

    data_dir = cfg["data"]["data_dir"]
    output_dir = Path("checkpoints/lgbm")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading training data from {data_dir} …")
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

    # Test set
    X_test, y_test = load_ember_arrays(data_dir, "test", use_memmap=True, feature_dim=cfg["data"]["input_dim"])
    test_meta = load_family_metadata(
        data_dir,
        "test",
        min_confidence=cfg["data"]["family_confidence_threshold"],
        min_samples=1,
        reference_family_to_label=meta.family_to_label,
    )

    all_results = {}

    # ── Detection baseline ────────────────────────────────────────────────────
    if task in ("detection", "all"):
        det_model, _ = train_detection_model(
            X_train, y_train, train_idx, val_idx, str(output_dir)
        )
        train_det = evaluate_detection(
            det_model, X_train, y_train, meta.idx_to_filetype, "Train"
        )
        test_det  = evaluate_detection(
            det_model, X_test, y_test, test_meta.idx_to_filetype, "Test"
        )
        challenge = evaluate_challenge(det_model, data_dir)

        all_results["lgbm_detection_train"] = train_det
        all_results["lgbm_detection_test"]  = test_det
        all_results["lgbm_challenge"]        = challenge

    # ── Family baseline ───────────────────────────────────────────────────────
    if task in ("family", "all"):
        fam_model, _ = train_family_model(
            X_train, meta, train_idx, val_idx, str(output_dir)
        )
        train_fam = evaluate_family(fam_model, X_train, meta, "Train")
        test_fam  = evaluate_family(fam_model, X_test, test_meta, "Test")

        all_results["lgbm_family_train"] = train_fam
        all_results["lgbm_family_test"]  = test_fam

    print_results_table(all_results, "LightGBM Baseline Results")
    save_json(all_results, str(results_dir / "lgbm_baseline_results.json"))
    logger.info(f"Results saved to {results_dir / 'lgbm_baseline_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM baseline training + evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--task", choices=["detection", "family", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    env_dir = os.environ.get("EMBER2024_DIR")
    if env_dir and cfg["data"]["data_dir"] == "/path/to/ember2024":
        cfg["data"]["data_dir"] = env_dir

    main(cfg, task=args.task)
