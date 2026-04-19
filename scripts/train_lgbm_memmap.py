#!/usr/bin/env python3
"""
Train a LightGBM binary malware detector directly from EMBER2024 memmaps.

Data root defaults to:
    /Users/roheeeee/Documents/DACS/EMBER2024-corrected-full

Expected memmap shapes / dtypes:
    X_train.dat      float32  (2626000, 2568)
    y_train.dat      int32    (2626000,)
    X_test.dat       float32  (605929, 2568)
    y_test.dat       int32    (605929,)
    X_challenge.dat  float32  (6315, 2568)
    y_challenge.dat  int32    (6315,)

Notes
-----
- This script keeps the on-disk arrays as numpy memmaps.
- LightGBM may still build its own internal binned representation in RAM.
- Challenge evaluation uses the official-style setup:
      benign rows from the test split + malicious challenge rows
  because the challenge split alone contains only positives, so ROC-AUC would
  otherwise be undefined.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


TRAIN_SHAPE = (2_626_000, 2568)
TEST_SHAPE = (605_929, 2568)
CHALLENGE_SHAPE = (6_315, 2568)
PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_split(root: Path, subset: str, shape: tuple[int, int]) -> tuple[np.memmap, np.memmap]:
    """Load one split as memmaps with the exact expected shape/dtype."""
    x_path = root / f"X_{subset}.dat"
    y_path = root / f"y_{subset}.dat"
    X = np.memmap(x_path, dtype=np.float32, mode="r", shape=shape)
    y = np.memmap(y_path, dtype=np.int32, mode="r", shape=(shape[0],))
    return X, y


def resolve_data_root(explicit_path: str | None) -> Path:
    """Resolve the data root from CLI, env, or repo-local defaults."""
    if explicit_path:
        return Path(explicit_path)

    candidates = []
    env_dir = os.environ.get("EMBER2024_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            PROJECT_DIR / "data" / "local" / "EMBER2024-corrected-full",
            PROJECT_DIR / ".." / "EMBER2024-corrected-full",
            PROJECT_DIR / "data" / "local" / "EMBER2024-corrected-canonical",
            PROJECT_DIR / ".." / "EMBER2024-corrected-canonical",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "No EMBER2024 data directory found. Set --data-root, export EMBER2024_DIR, "
        "or run bash scripts/link_local_ember2024.sh."
    )


def split_summary(name: str, X: np.memmap, y: np.memmap) -> None:
    """Print shape, label counts, and finite checks for first/last rows."""
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    label_counts = dict(zip(labels.tolist(), counts.tolist()))
    first_ok = bool(np.isfinite(np.asarray(X[0])).all())
    last_ok = bool(np.isfinite(np.asarray(X[-1])).all())
    print(f"\n[{name}]")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y dtype: {y.dtype}")
    print(f"label counts: {label_counts}")
    print(f"first row finite: {first_ok}")
    print(f"last row finite:  {last_ok}")


def binary_label_stats(y: np.memmap) -> tuple[int, int, float]:
    """Return (n_negative, n_positive, scale_pos_weight) from a binary label memmap."""
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    count_map = dict(zip(labels.tolist(), counts.tolist()))
    n_negative = int(count_map.get(0, 0))
    n_positive = int(count_map.get(1, 0))
    if n_positive == 0:
        raise ValueError("Training split has zero positive samples; cannot compute scale_pos_weight.")
    return n_negative, n_positive, n_negative / n_positive


def batched_predict_proba(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    batch_size: int,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Predict probabilities in batches to avoid materialising large arrays at once."""
    if indices is None:
        indices = np.arange(X.shape[0], dtype=np.int64)
    probs: list[np.ndarray] = []
    for start in range(0, indices.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        probs.append(model.predict_proba(X[batch_idx])[:, 1])
    return np.concatenate(probs, axis=0) if probs else np.empty(0, dtype=np.float64)


def fpr_at_target_fnr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fnr: float = 0.01,
) -> tuple[float, float, float]:
    """
    Return (fpr, threshold, fnr) at the smallest empirical FPR whose FNR is <= target.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    target_tpr = 1.0 - target_fnr
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.flatnonzero(tpr >= target_tpr)
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")

    best_local = valid[np.argmin(fpr[valid])]
    return float(fpr[best_local]), float(thresholds[best_local]), float(1.0 - tpr[best_local])


def evaluate_binary(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fnr: float,
) -> None:
    """Print ROC-AUC and FPR@target FNR for one binary evaluation set."""
    roc_auc = float(roc_auc_score(y_true, y_score))
    fpr_1pct, threshold, actual_fnr = fpr_at_target_fnr(y_true, y_score, target_fnr=target_fnr)
    print(f"\n[{name}]")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"FPR @ {target_fnr:.1%} FNR: {fpr_1pct:.6f}")
    print(f"threshold @ target: {threshold:.6f}")
    print(f"actual FNR at threshold: {actual_fnr:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM on EMBER2024 memmaps.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing X_*.dat / y_*.dat files. Defaults to EMBER2024_DIR or repo-local data links.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Prediction batch size for test/challenge evaluation.",
    )
    parser.add_argument(
        "--target-fnr",
        type=float,
        default=0.01,
        help="False negative rate target used for FPR reporting.",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Optional path to save the trained LightGBM model.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=3000,
        help="Number of boosting rounds / trees (default: 3000).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=2048,
        help="LightGBM num_leaves (default: 2048).",
    )
    parser.add_argument(
        "--use-scale-pos-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LightGBM scale_pos_weight=n_negative/n_positive (default: True).",
    )
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)

    print(f"Data root: {root}")
    X_train, y_train = load_split(root, "train", TRAIN_SHAPE)
    X_test, y_test = load_split(root, "test", TEST_SHAPE)
    X_challenge, y_challenge = load_split(root, "challenge", CHALLENGE_SHAPE)

    split_summary("train", X_train, y_train)
    split_summary("test", X_test, y_test)
    split_summary("challenge", X_challenge, y_challenge)

    n_negative, n_positive, scale_pos_weight = binary_label_stats(y_train)
    print("\n[train_class_balance]")
    print(f"train negatives: {n_negative}")
    print(f"train positives: {n_positive}")
    print(f"negative/positive ratio: {scale_pos_weight:.6f}")
    print(f"use_scale_pos_weight: {args.use_scale_pos_weight}")

    print("\n[train_lgbm]")
    spw_text = f"{scale_pos_weight:.6f}" if args.use_scale_pos_weight else "disabled"
    print(
        "params: "
        f"n_estimators={args.n_estimators} learning_rate={args.learning_rate} "
        f"num_leaves={args.num_leaves} n_jobs=-1 "
        f"scale_pos_weight={spw_text}"
    )
    lgbm_kwargs = dict(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_jobs=-1,
        verbose=-1,
    )
    if args.use_scale_pos_weight:
        lgbm_kwargs["scale_pos_weight"] = scale_pos_weight
    model = lgb.LGBMClassifier(
        **lgbm_kwargs,
    )
    model.fit(X_train, np.asarray(y_train))

    if args.model_out:
        out_path = Path(args.model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(out_path))
        print(f"saved model: {out_path}")

    test_scores = batched_predict_proba(model, X_test, batch_size=args.batch_size)
    evaluate_binary("test", np.asarray(y_test), test_scores, target_fnr=args.target_fnr)

    test_benign_idx = np.flatnonzero(np.asarray(y_test) == 0)
    challenge_scores = batched_predict_proba(model, X_challenge, batch_size=args.batch_size)
    benign_scores = batched_predict_proba(
        model, X_test, batch_size=args.batch_size, indices=test_benign_idx
    )
    y_challenge_eval = np.concatenate(
        [np.zeros(test_benign_idx.shape[0], dtype=np.int32), np.asarray(y_challenge)],
        axis=0,
    )
    challenge_eval_scores = np.concatenate([benign_scores, challenge_scores], axis=0)
    evaluate_binary(
        "challenge_with_test_benign",
        y_challenge_eval,
        challenge_eval_scores,
        target_fnr=args.target_fnr,
    )


if __name__ == "__main__":
    main()
