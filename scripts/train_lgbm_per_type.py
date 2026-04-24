#!/usr/bin/env python3
"""
Train one 64-leaf LightGBM per file type (Win32, Win64, Dot_Net, APK, ELF, PDF).

These per-type models add diversity to the ensemble: same complexity as the
paper's per-type models but trained with temporal early stopping.

Outputs: checkpoints/lgbm/lgbm_64leaf_{ft}.txt  for each file type.

Estimated runtime: ~30-40 min for Win32, 5-15 min each for smaller types.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT   = Path("/Users/roheeeee/Documents/DACS/EMBER2024-corrected-full")
OUT    = Path("checkpoints/lgbm")
BATCH  = 50_000
FILE_TYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]

OUT.mkdir(parents=True, exist_ok=True)


def predict_batched(booster, X):
    return np.concatenate(
        [booster.predict(np.array(X[i : i + BATCH])) for i in range(0, len(X), BATCH)]
    )


def load_filetype_indices(jsonl_path: Path) -> dict:
    """Return {file_type: sorted array of row indices}."""
    print(f"Reading {jsonl_path.name} for file_type labels …", flush=True)
    ft_indices: dict[str, list[int]] = {ft: [] for ft in FILE_TYPES}
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            ft = json.loads(line).get("file_type", "")
            if ft in ft_indices:
                ft_indices[ft].append(i)
            if i % 500_000 == 0 and i > 0:
                print(f"  … {i:,} rows read", flush=True)
    return {ft: np.array(idxs) for ft, idxs in ft_indices.items()}


def train_one(ft: str, idxs: np.ndarray, X_all, y_all, X_test, y_test, X_ch, y_ch_label):
    n_val  = max(int(len(idxs) * 0.08), 1000)
    tr_idx = idxs[: len(idxs) - n_val]
    va_idx = idxs[len(idxs) - n_val :]

    y_tr_raw = np.array(y_all[tr_idx])
    y_va_raw = np.array(y_all[va_idx])
    labeled_tr = tr_idx[y_tr_raw != -1]
    labeled_va = va_idx[y_va_raw != -1]

    y_tr = np.array(y_all[labeled_tr]).astype(np.int32)
    y_va = np.array(y_all[labeled_va]).astype(np.int32)

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    if n_pos == 0:
        print(f"  [{ft}] No positive samples — skipping.", flush=True)
        return None
    spw = n_neg / n_pos

    print(f"  [{ft}] Train={len(labeled_tr):,}  Val={len(labeled_va):,}  "
          f"spw={spw:.2f}", flush=True)

    print(f"  [{ft}] Loading into RAM …", flush=True)
    X_tr = np.array(X_all[labeled_tr])
    X_va = np.array(X_all[labeled_va])

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        num_leaves=64,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=spw,
        n_jobs=-1,
        verbose=-1,
    )

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    elapsed = (time.time() - t0) / 60
    print(f"  [{ft}] Done in {elapsed:.1f} min  best_iter={model.best_iteration_}",
          flush=True)

    booster = model.booster_
    out_path = OUT / f"lgbm_64leaf_{ft}.txt"
    booster.save_model(str(out_path))
    print(f"  [{ft}] Saved → {out_path}", flush=True)

    # Quick eval on matching test samples
    yt = np.array(y_test)
    import json as _json
    # Score full test set (model trained per-type but we score all — consistent with ensemble usage)
    test_scores = predict_batched(booster, X_test)
    ch_scores   = booster.predict(np.array(X_ch))

    ben_idx = np.flatnonzero(yt == 0)
    yc = np.concatenate([np.zeros(len(ben_idx)), np.ones(len(y_ch_label))])
    sc = np.concatenate([test_scores[ben_idx], ch_scores])

    print(f"  [{ft}] Test ROC-AUC: {roc_auc_score(yt, test_scores):.4f}  "
          f"Chal ROC-AUC: {roc_auc_score(yc, sc):.4f}  "
          f"DetRate: {(ch_scores >= 0.5).mean():.4f}", flush=True)

    return booster


def main():
    print("Loading full train memmaps …", flush=True)
    X_all = np.memmap(ROOT / "X_train.dat", dtype=np.float32, mode="r",
                      shape=(2_626_000, 2568))
    y_all = np.memmap(ROOT / "y_train.dat", dtype=np.int32, mode="r",
                      shape=(2_626_000,))

    print("Loading test + challenge memmaps …", flush=True)
    X_test = np.memmap(ROOT / "X_test.dat",      dtype=np.float32, mode="r", shape=(605929, 2568))
    y_test = np.memmap(ROOT / "y_test.dat",       dtype=np.int32,   mode="r", shape=(605929,))
    X_ch   = np.memmap(ROOT / "X_challenge.dat",  dtype=np.float32, mode="r", shape=(6315, 2568))
    y_ch   = np.memmap(ROOT / "y_challenge.dat",  dtype=np.int32,   mode="r", shape=(6315,))

    ft_indices = load_filetype_indices(ROOT / "2023-09-24_2024-09-21_train.jsonl")

    for ft, idxs in ft_indices.items():
        out_path = OUT / f"lgbm_64leaf_{ft}.txt"
        if out_path.exists():
            print(f"[{ft}] Already trained — skipping (delete to retrain).", flush=True)
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"Training {ft}  ({len(idxs):,} samples)", flush=True)
        train_one(ft, idxs, X_all, y_all, X_test, y_test, X_ch, y_ch)

    print("\nAll per-type models done.", flush=True)


if __name__ == "__main__":
    main()
