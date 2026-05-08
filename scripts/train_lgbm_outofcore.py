#!/usr/bin/env python3
"""Out-of-core LightGBM training using memmap-backed .dat files.

This script streams the EMBER2024 `X_train.dat` / `y_train.dat` and writes a
filtered memmap containing only labeled samples (y != -1). It then trains a
LightGBM model using memmap slices (no large in-RAM copies).

Usage:
  python DAFCS/scripts/train_lgbm_outofcore.py --data-root DAFCS/data/local/EMBER2024-corrected-full

Notes:
 - Requires sufficient disk space for the filtered memmaps (roughly proportional
   to number of labeled samples * feature_dim * 4 bytes).
 - Uses contiguous slicing for training/validation to avoid copies.
"""

import argparse
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def build_filtered_memmap(root: Path, out_root: Path, feature_dim: int = 2568, chunk: int = 100_000):
    """Create X_labeled.dat and y_labeled.dat under out_root by streaming shards.
    Returns: n_labeled (int)
    """
    X_path = root / "X_train.dat"
    y_path = root / "y_train.dat"
    out_root.mkdir(parents=True, exist_ok=True)
    X_out = out_root / "X_train_labeled.dat"
    y_out = out_root / "y_train_labeled.dat"

    # open memmaps
    n_samples = os.path.getsize(X_path) // (feature_dim * 4)
    X_mm = np.memmap(X_path, dtype=np.float32, mode="r", shape=(n_samples, feature_dim))
    # y may be int32 or float32
    try:
        y_mm = np.memmap(y_path, dtype=np.int32, mode="r", shape=(n_samples,))
    except Exception:
        y_mm = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_samples,)).astype(np.int32)

    labeled_mask = (y_mm != -1)
    n_labeled = int(labeled_mask.sum())
    print(f"Found {n_labeled:,} labeled samples out of {n_samples:,}")

    if n_labeled == 0:
        raise SystemExit("No labeled samples found in y_train.dat")

    # create output memmaps
    if X_out.exists():
        print(f"Using existing {X_out} (will be overwritten)")
        X_out.unlink()
    if y_out.exists():
        y_out.unlink()

    X_filtered = np.memmap(X_out, dtype=np.float32, mode="w+", shape=(n_labeled, feature_dim))
    y_filtered = np.memmap(y_out, dtype=np.int32, mode="w+", shape=(n_labeled,))

    write_idx = 0
    for start in range(0, n_samples, chunk):
        stop = min(n_samples, start + chunk)
        mask = labeled_mask[start:stop]
        if not mask.any():
            continue
        rows = np.nonzero(mask)[0] + start
        count = rows.size
        X_filtered[write_idx:write_idx + count] = X_mm[rows]
        y_filtered[write_idx:write_idx + count] = y_mm[rows]
        write_idx += count
        print(f"Copied {write_idx:,}/{n_labeled:,} labeled rows", end="\r")

    # flush
    del X_filtered
    del y_filtered
    print(f"\nWrote labeled memmaps to: {X_out} ({n_labeled:,} x {feature_dim})")
    return n_labeled


def train_from_memmap(out_root: Path, n_labeled: int, feature_dim: int = 2568):
    X_labeled = np.memmap(out_root / "X_train_labeled.dat", dtype=np.float32, mode="r", shape=(n_labeled, feature_dim))
    y_labeled = np.memmap(out_root / "y_train_labeled.dat", dtype=np.int32, mode="r", shape=(n_labeled,))

    # simple temporal-style val split: last 7% for val
    n_val = max(1, int(n_labeled * 0.07))
    tr_slice = slice(0, n_labeled - n_val)
    va_slice = slice(n_labeled - n_val, n_labeled)

    # build datasets; free_raw_data=False to keep memmaps in place
    dtrain = lgb.Dataset(X_labeled[tr_slice], label=y_labeled[tr_slice], free_raw_data=False)
    dval = lgb.Dataset(X_labeled[va_slice], label=y_labeled[va_slice], reference=dtrain, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "num_leaves": 64,
        "learning_rate": 0.05,
        "verbose": -1,
        "num_threads": 8,
    }

    outdir = Path("checkpoints/lgbm_outofcore")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Starting LightGBM training...")
    t0 = time.time()
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        early_stopping_rounds=100,
        verbose_eval=50,
    )
    elapsed = (time.time() - t0) / 60.0
    print(f"Training finished in {elapsed:.1f} minutes; best_iter={bst.best_iteration}")

    model_path = outdir / "lgbm_64leaf_outofcore.txt"
    bst.save_model(str(model_path))
    print(f"Saved model → {model_path}")

    return bst


def evaluate_model(bst, root: Path, feature_dim: int = 2568):
    # load test/challenge memmaps if present
    X_test_path = root / "X_test.dat"
    y_test_path = root / "y_test.dat"
    X_ch_path = root / "X_challenge.dat"
    y_ch_path = root / "y_challenge.dat"

    if not X_test_path.exists() or not y_test_path.exists():
        print("Test set not found; skipping evaluation")
        return

    n_test = os.path.getsize(X_test_path) // (feature_dim * 4)
    X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(n_test, feature_dim))
    y_test = np.memmap(y_test_path, dtype=np.int32, mode="r", shape=(n_test,))

    # predict in batches to limit memory spikes
    batch = 50_000
    preds = []
    for i in range(0, n_test, batch):
        j = min(n_test, i + batch)
        preds.append(bst.predict(np.array(X_test[i:j])))
    preds = np.concatenate(preds)

    print(f"Test ROC: {roc_auc_score(np.array(y_test), preds):.4f} PR: {average_precision_score(np.array(y_test), preds):.4f}")

    if X_ch_path.exists() and y_ch_path.exists():
        n_ch = os.path.getsize(X_ch_path) // (feature_dim * 4)
        X_ch = np.memmap(X_ch_path, dtype=np.float32, mode="r", shape=(n_ch, feature_dim))
        y_ch = np.memmap(y_ch_path, dtype=np.int32, mode="r", shape=(n_ch,))
        ch_preds = bst.predict(np.array(X_ch[:min(n_ch, 100000)])) if n_ch > 0 else np.array([])
        print(f"Challenge sample predictions computed ({len(ch_preds)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="DAFCS/data/local/EMBER2024-corrected-full", help="Path to corrected-full root")
    parser.add_argument("--out-root", default="DAFCS/data/local/EMBER2024-corrected-full", help="Where to write labeled memmaps")
    parser.add_argument("--no-write", action="store_true", help="Do not create filtered memmap files; train with sample weights instead")
    parser.add_argument("--feature-dim", type=int, default=2568)
    args = parser.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out_root)

    if args.no_write:
        # Train using sample weights to ignore unlabeled examples, without creating new files
        print("No-write mode: training without creating filtered memmaps. Using sample weights.")
        # load memmaps
        X_path = root / "X_train.dat"
        y_path = root / "y_train.dat"
        feature_dim = args.feature_dim
        n_samples = os.path.getsize(X_path) // (feature_dim * 4)
        X_mm = np.memmap(X_path, dtype=np.float32, mode="r", shape=(n_samples, feature_dim))
        try:
            y_mm = np.memmap(y_path, dtype=np.int32, mode="r", shape=(n_samples,))
        except Exception:
            y_mm = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_samples,)).astype(np.int32)

        labeled_idx = np.nonzero(y_mm != -1)[0]
        n_labeled = len(labeled_idx)
        if n_labeled == 0:
            raise SystemExit("No labeled samples found in y_train.dat")

        # build train/val index split based on labeled ordering
        n_val = max(1, int(n_labeled * 0.07))
        tr_idx = labeled_idx[: n_labeled - n_val]
        va_idx = labeled_idx[n_labeled - n_val :]

        # prepare label array where unlabeled set to 0 (won't matter because weight=0)
        y_copy = np.array(y_mm, copy=True)
        y_copy[y_copy == -1] = 0

        # build weight arrays: train weights =1 for train indices else 0; val weights vice-versa
        w_train = np.zeros(n_samples, dtype=np.float32)
        w_val = np.zeros(n_samples, dtype=np.float32)
        w_train[tr_idx] = 1.0
        w_val[va_idx] = 1.0

        dtrain = lgb.Dataset(X_mm, label=y_copy, weight=w_train, free_raw_data=False)
        dval = lgb.Dataset(X_mm, label=y_copy, weight=w_val, reference=dtrain, free_raw_data=False)

        params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "num_leaves": 64,
            "learning_rate": 0.05,
            "verbose": -1,
            "num_threads": 8,
        }

        outdir = Path("checkpoints/lgbm_outofcore")
        outdir.mkdir(parents=True, exist_ok=True)

        print("Starting LightGBM training (no-write mode)...")
        t0 = time.time()
        # Use callback API for compatibility with different LightGBM versions
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=50),
            ],
        )
        elapsed = (time.time() - t0) / 60.0
        print(f"Training finished in {elapsed:.1f} minutes; best_iter={bst.best_iteration}")

        model_path = outdir / "lgbm_64leaf_no_write.txt"
        bst.save_model(str(model_path))
        print(f"Saved model → {model_path}")

        evaluate_model(bst, root, feature_dim=args.feature_dim)
    else:
        print(f"Building filtered memmap from {root} → {out_root}")
        n_labeled = build_filtered_memmap(root, out_root, feature_dim=args.feature_dim)

        bst = train_from_memmap(out_root, n_labeled, feature_dim=args.feature_dim)

        evaluate_model(bst, root, feature_dim=args.feature_dim)


if __name__ == "__main__":
    main()
