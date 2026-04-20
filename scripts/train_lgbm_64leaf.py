#!/usr/bin/env python3
"""
Retrain LightGBM with 64 leaves (matching paper complexity) + early stopping.
Targets better generalization to evasive malware on the challenge set.
"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/Users/roheeeee/Documents/DACS/EMBER2024-corrected-full")
OUT  = Path("checkpoints/lgbm")
BATCH = 50_000

def predict_batched(booster, X):
    return np.concatenate(
        [booster.predict(np.array(X[i:i+BATCH])) for i in range(0, len(X), BATCH)]
    )

print("Loading training data...", flush=True)
X = np.memmap(ROOT / "X_train.dat", dtype=np.float32, mode="r", shape=(2_626_000, 2568))
y = np.memmap(ROOT / "y_train.dat", dtype=np.int32,   mode="r", shape=(2_626_000,))

# Temporal validation split: last ~7% as val
n_val  = int(len(y) * 0.07)
tr_idx = np.arange(len(y) - n_val)
va_idx = np.arange(len(y) - n_val, len(y))

tr_mask = np.array(y[tr_idx]) != -1
va_mask = np.array(y[va_idx]) != -1
tr_idx, va_idx = tr_idx[tr_mask], va_idx[va_mask]

n_neg = int((np.array(y[tr_idx]) == 0).sum())
n_pos = int((np.array(y[tr_idx]) == 1).sum())
spw   = n_neg / n_pos
print(f"Train: {len(tr_idx):,}  Val: {len(va_idx):,}  scale_pos_weight={spw:.3f}", flush=True)

print("Loading into RAM...", flush=True)
X_tr = np.array(X[tr_idx]); y_tr = np.array(y[tr_idx]).astype(np.int32)
X_va = np.array(X[va_idx]); y_va = np.array(y[va_idx]).astype(np.int32)

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
print("Training (64 leaves, up to 2000 trees, early_stopping=100)...", flush=True)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
)
elapsed = (time.time() - t0) / 60
print(f"Training done in {elapsed:.1f} min  best_iter={model.best_iteration_}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
model_path = OUT / "lgbm_64leaf.txt"
model.booster_.save_model(str(model_path))
print(f"Model saved → {model_path}", flush=True)

# Evaluate on test + challenge
print("\nEvaluating...", flush=True)
X_test = np.memmap(ROOT/"X_test.dat", dtype=np.float32, mode="r", shape=(605929, 2568))
y_test = np.memmap(ROOT/"y_test.dat", dtype=np.int32,   mode="r", shape=(605929,))
X_ch   = np.memmap(ROOT/"X_challenge.dat", dtype=np.float32, mode="r", shape=(6315, 2568))
y_ch   = np.memmap(ROOT/"y_challenge.dat", dtype=np.int32,   mode="r", shape=(6315,))

booster    = model.booster_
test_scores = predict_batched(booster, X_test)
ch_scores   = booster.predict(np.array(X_ch))

yt = np.array(y_test)
print(f"Test  ROC-AUC: {roc_auc_score(yt, test_scores):.4f}  "
      f"PR-AUC: {average_precision_score(yt, test_scores):.4f}")

ben_idx = np.flatnonzero(yt == 0)
yc      = np.concatenate([np.zeros(len(ben_idx)), np.ones(len(y_ch))])
sc      = np.concatenate([test_scores[ben_idx], ch_scores])
print(f"Chal  ROC-AUC: {roc_auc_score(yc, sc):.4f}  "
      f"PR-AUC: {average_precision_score(yc, sc):.4f}  "
      f"DetRate: {(ch_scores >= 0.5).mean():.4f}")
