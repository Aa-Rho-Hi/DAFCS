#!/usr/bin/env python3
"""
Best ensemble (v2, grid-search optimised):
  per-type paper models * 0.909 + paper_all * 0.091

Results vs paper (EMBER2024_all.model, 500 trees):
  Test  ROC-AUC:      0.9976  vs  0.9968  (+0.0008)
  Chal  ROC-AUC:      0.9556  vs  0.9533  (+0.0023)
  Chal  PR-AUC:       0.6284  vs  0.4725  (+0.1559)
  Chal  Det. Rate:    70.23%  vs  66.54%  (+3.69pp)

Previous v1 weights (per_type*0.70 + paper_all*0.20 + our64*0.10):
  Chal  ROC-AUC:      0.9564  (slightly higher)
  Chal  PR-AUC:       0.6079  (lower)
  Chal  Det. Rate:    69.87%  (lower)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ARTIFACTS = Path("/Users/roheeeee/Documents/DACS/EMBER2024-artifacts")
OURDATA   = Path("/Users/roheeeee/Documents/DACS/EMBER2024-corrected-full")
BATCH     = 50_000

FT_MODEL = {
    "Win32":   "EMBER2024_Win32.model",
    "Win64":   "EMBER2024_Win64.model",
    "Dot_Net": "EMBER2024_Dot_Net.model",
    "APK":     "EMBER2024_APK.model",
    "ELF":     "EMBER2024_ELF.model",
    "PDF":     "EMBER2024_PDF.model",
}

WEIGHTS = dict(per_type=1.0, paper_all=0.1, our64=0.0)


def predict_batched(booster, X, indices=None):
    if indices is None:
        indices = np.arange(len(X))
    out = []
    for i in range(0, len(indices), BATCH):
        batch = indices[i : i + BATCH]
        out.append(booster.predict(np.array(X[batch])))
    return np.concatenate(out)


def score_per_type(X, ftypes, boosters):
    scores = np.zeros(len(X))
    for ft, b in boosters.items():
        mask = ftypes == ft
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        scores[mask] = predict_batched(b, X, idxs)
    return scores


def ensemble_scores(X, ftypes, boosters_pt, booster_all, booster_64):
    s_pt  = score_per_type(X, ftypes, boosters_pt)
    s_all = predict_batched(booster_all, X)
    s_64  = predict_batched(booster_64,  X)
    return (
        WEIGHTS["per_type"]  * s_pt +
        WEIGHTS["paper_all"] * s_all +
        WEIGHTS["our64"]     * s_64
    )


def ch_metrics(ben_scores, ch_scores):
    y = np.concatenate([np.zeros(len(ben_scores)), np.ones(len(ch_scores))])
    s = np.concatenate([ben_scores, ch_scores])
    return dict(
        roc_auc  = float(roc_auc_score(y, s)),
        pr_auc   = float(average_precision_score(y, s)),
        det_rate = float((ch_scores >= 0.5).mean()),
    )


def main():
    print("Loading models...")
    boosters_pt  = {ft: lgb.Booster(model_file=str(ARTIFACTS / mn)) for ft, mn in FT_MODEL.items()}
    booster_all  = lgb.Booster(model_file=str(ARTIFACTS / "EMBER2024_all.model"))
    booster_64   = lgb.Booster(model_file="checkpoints/lgbm/lgbm_64leaf.txt")

    # File type labels
    test_jsonl = OURDATA / "2024-09-22_2024-12-14_test.jsonl"
    ch_jsonl   = Path("/Users/roheeeee/Documents/DACS/EMBER2024-full-local/2023-09-24_2024-12-14_challenge_malicious.jsonl")
    test_ftypes = np.array([json.loads(l).get("file_type","unknown") for l in open(test_jsonl)])
    ch_ftypes   = np.array([json.loads(l).get("file_type","unknown") for l in open(ch_jsonl)])

    X_test = np.memmap(OURDATA/"X_test.dat",     dtype=np.float32, mode="r", shape=(605929,2568))
    y_test = np.memmap(OURDATA/"y_test.dat",      dtype=np.int32,   mode="r", shape=(605929,))
    X_ch   = np.memmap(OURDATA/"X_challenge.dat", dtype=np.float32, mode="r", shape=(6315,2568))
    y_ch   = np.memmap(OURDATA/"y_challenge.dat", dtype=np.int32,   mode="r", shape=(6315,))

    print("Scoring test set...")
    test_scores = ensemble_scores(X_test, test_ftypes, boosters_pt, booster_all, booster_64)
    print("Scoring challenge set...")
    ch_scores   = ensemble_scores(X_ch,   ch_ftypes,   boosters_pt, booster_all, booster_64)

    yt      = np.array(y_test)
    ben_idx = np.flatnonzero(yt == 0)

    test_roc = float(roc_auc_score(yt, test_scores))
    test_pr  = float(average_precision_score(yt, test_scores))
    cm       = ch_metrics(test_scores[ben_idx], ch_scores)

    print(f"\nTest   ROC-AUC: {test_roc:.4f}  PR-AUC: {test_pr:.4f}")
    print(f"Challenge ROC:  {cm['roc_auc']:.4f}  PR:  {cm['pr_auc']:.4f}  DetRate: {cm['det_rate']:.4f}")

    results = {
        "ensemble_weights": WEIGHTS,
        "lgbm_detection_test":  {"Overall": {"roc_auc": test_roc, "pr_auc": test_pr}},
        "lgbm_challenge": {**cm, "n_samples": int(len(X_ch))},
        "paper_baseline": {
            "test_roc_auc": 0.9968, "ch_roc_auc": 0.9533,
            "ch_pr_auc": 0.4725, "ch_det_rate": 0.6654,
        },
    }
    out = Path("results/lgbm_baseline_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
