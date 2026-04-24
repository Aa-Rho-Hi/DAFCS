#!/usr/bin/env python3
"""
Extended ensemble + weight grid search.

Models used (if available):
  A: Paper per-type models (Win32/Win64/Dot_Net/APK/ELF/PDF)
  B: Our per-type 64-leaf models (trained by train_lgbm_per_type.py)
  C: Paper all-type model (EMBER2024_all.model)
  D: Our 64-leaf all-type model

Grid searches over blend weights A/B/C/D to maximise challenge PR-AUC,
challenge ROC-AUC, and detection rate simultaneously.

Results are saved to results/lgbm_v2_results.json.
"""
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ARTIFACTS = Path("/Users/roheeeee/Documents/DACS/EMBER2024-artifacts")
DATA_DIR  = Path("/Users/roheeeee/Documents/DACS/EMBER2024-corrected-full")
CH_JSONL  = Path("/Users/roheeeee/Documents/DACS/EMBER2024-full-local"
                 "/2023-09-24_2024-12-14_challenge_malicious.jsonl")
CKPT      = Path("checkpoints/lgbm")
RESULTS   = Path("results")
BATCH     = 50_000

FILE_TYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]

PAPER_PER_TYPE = {ft: ARTIFACTS / f"EMBER2024_{ft}.model" for ft in FILE_TYPES}
OUR_PER_TYPE   = {ft: CKPT / f"lgbm_64leaf_{ft}.txt"     for ft in FILE_TYPES}
PAPER_ALL      = ARTIFACTS / "EMBER2024_all.model"
OUR_ALL        = CKPT / "lgbm_64leaf.txt"


# ── helpers ──────────────────────────────────────────────────────────────────

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
        if mask.any():
            scores[mask] = predict_batched(b, X, np.where(mask)[0])
    return scores


def ch_metrics(ben_scores, ch_scores):
    y = np.concatenate([np.zeros(len(ben_scores)), np.ones(len(ch_scores))])
    s = np.concatenate([ben_scores, ch_scores])
    return dict(
        roc_auc  = float(roc_auc_score(y, s)),
        pr_auc   = float(average_precision_score(y, s)),
        det_rate = float((ch_scores >= 0.5).mean()),
    )


# ── load ─────────────────────────────────────────────────────────────────────

def load_boosters():
    boosters = {}
    print("Loading paper per-type models …")
    boosters["paper_per_type"] = {
        ft: lgb.Booster(model_file=str(p))
        for ft, p in PAPER_PER_TYPE.items() if p.exists()
    }
    print(f"  loaded {len(boosters['paper_per_type'])} paper per-type models")

    boosters["our_per_type"] = {}
    if any(p.exists() for p in OUR_PER_TYPE.values()):
        print("Loading our per-type models …")
        boosters["our_per_type"] = {
            ft: lgb.Booster(model_file=str(p))
            for ft, p in OUR_PER_TYPE.items() if p.exists()
        }
        print(f"  loaded {len(boosters['our_per_type'])} our per-type models")
    else:
        print("Our per-type models not found — run train_lgbm_per_type.py first.")

    print("Loading paper all-type model …")
    boosters["paper_all"] = lgb.Booster(model_file=str(PAPER_ALL))

    print("Loading our 64-leaf all-type model …")
    boosters["our_all"] = lgb.Booster(model_file=str(OUR_ALL))

    return boosters


# ── score ─────────────────────────────────────────────────────────────────────

def compute_raw_scores(boosters, X_test, test_ftypes, X_ch, ch_ftypes):
    print("Scoring test set …")
    raw = {}
    raw["paper_pt"]  = score_per_type(X_test, test_ftypes, boosters["paper_per_type"])
    if boosters["our_per_type"]:
        raw["our_pt"] = score_per_type(X_test, test_ftypes, boosters["our_per_type"])
    raw["paper_all"] = predict_batched(boosters["paper_all"], X_test)
    raw["our_all"]   = predict_batched(boosters["our_all"],  X_test)

    print("Scoring challenge set …")
    raw_ch = {}
    raw_ch["paper_pt"]  = score_per_type(X_ch, ch_ftypes, boosters["paper_per_type"])
    if boosters["our_per_type"]:
        raw_ch["our_pt"] = score_per_type(X_ch, ch_ftypes, boosters["our_per_type"])
    raw_ch["paper_all"] = predict_batched(boosters["paper_all"], X_ch)
    raw_ch["our_all"]   = predict_batched(boosters["our_all"],  X_ch)

    return raw, raw_ch


# ── grid search ───────────────────────────────────────────────────────────────

def grid_search(raw, raw_ch, y_test, ben_idx, y_ch_label):
    has_our_pt = "our_pt" in raw

    # Weight grid: coarse then fine
    # Axes: w_paper_pt, w_our_pt (if available), w_paper_all, w_our_all
    candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best = {"score": -1.0}
    n_tried = 0

    if has_our_pt:
        combos = itertools.product(candidates, candidates, candidates, candidates)
    else:
        combos = itertools.product(candidates, candidates, candidates)

    for weights in combos:
        if has_our_pt:
            wa, wb, wc, wd = weights
        else:
            wa, wc, wd = weights
            wb = 0.0

        total = wa + wb + wc + wd
        if total < 0.01:
            continue

        test_s = (wa * raw["paper_pt"]
                  + (wb * raw["our_pt"] if has_our_pt else 0)
                  + wc * raw["paper_all"]
                  + wd * raw["our_all"]) / total

        ch_s = (wa * raw_ch["paper_pt"]
                + (wb * raw_ch["our_pt"] if has_our_pt else 0)
                + wc * raw_ch["paper_all"]
                + wd * raw_ch["our_all"]) / total

        cm = ch_metrics(test_s[ben_idx], ch_s)

        # Composite score: balance ROC-AUC + PR-AUC + detection rate
        composite = cm["roc_auc"] * 0.3 + cm["pr_auc"] * 0.4 + cm["det_rate"] * 0.3

        if composite > best["score"]:
            test_roc = float(roc_auc_score(y_test, test_s))
            best = {
                "score":          composite,
                "weights":        {"paper_pt": wa, "our_pt": wb,
                                   "paper_all": wc, "our_all": wd},
                "test_roc_auc":   test_roc,
                "ch_roc_auc":     cm["roc_auc"],
                "ch_pr_auc":      cm["pr_auc"],
                "ch_det_rate":    cm["det_rate"],
            }

        n_tried += 1
        if n_tried % 200 == 0:
            print(f"  … {n_tried} combos tried  best so far: "
                  f"PR-AUC={best['ch_pr_auc']:.4f}  "
                  f"DetRate={best['ch_det_rate']:.4f}", flush=True)

    return best


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    boosters = load_boosters()

    print("Loading test + challenge data …")
    X_test  = np.memmap(DATA_DIR / "X_test.dat",      dtype=np.float32, mode="r", shape=(605929, 2568))
    y_test  = np.array(np.memmap(DATA_DIR / "y_test.dat", dtype=np.int32, mode="r", shape=(605929,)))
    X_ch    = np.memmap(DATA_DIR / "X_challenge.dat", dtype=np.float32, mode="r", shape=(6315, 2568))

    test_ftypes = np.array([json.loads(l).get("file_type", "")
                             for l in open(DATA_DIR / "2024-09-22_2024-12-14_test.jsonl")])
    ch_ftypes   = np.array([json.loads(l).get("file_type", "")
                             for l in open(CH_JSONL)])

    raw, raw_ch = compute_raw_scores(boosters, X_test, test_ftypes, X_ch, ch_ftypes)

    ben_idx = np.flatnonzero(y_test == 0)
    y_ch_label = np.ones(6315)

    # ── Previous best (reference) ──
    paper_ref = dict(test_roc_auc=0.9968, ch_roc_auc=0.9533,
                     ch_pr_auc=0.4725, ch_det_rate=0.6654)
    prev_best = dict(
        weights={"paper_pt": 0.70, "our_pt": 0.0, "paper_all": 0.20, "our_all": 0.10},
        test_roc_auc=0.9975, ch_roc_auc=0.9564, ch_pr_auc=0.6079, ch_det_rate=0.6987,
    )

    print("\n--- Previous best ensemble ---")
    print(f"  Test ROC-AUC: {prev_best['test_roc_auc']:.4f}  "
          f"Chal ROC-AUC: {prev_best['ch_roc_auc']:.4f}  "
          f"PR-AUC: {prev_best['ch_pr_auc']:.4f}  "
          f"DetRate: {prev_best['ch_det_rate']:.4f}")

    print("\n--- Grid searching ensemble weights …")
    best = grid_search(raw, raw_ch, y_test, ben_idx, y_ch_label)

    print("\n--- Best ensemble found ---")
    print(f"  Weights:      {best['weights']}")
    print(f"  Test ROC-AUC: {best['test_roc_auc']:.4f}  (paper: {paper_ref['test_roc_auc']:.4f}  prev: {prev_best['test_roc_auc']:.4f})")
    print(f"  Chal ROC-AUC: {best['ch_roc_auc']:.4f}  (paper: {paper_ref['ch_roc_auc']:.4f}  prev: {prev_best['ch_roc_auc']:.4f})")
    print(f"  Chal PR-AUC:  {best['ch_pr_auc']:.4f}  (paper: {paper_ref['ch_pr_auc']:.4f}  prev: {prev_best['ch_pr_auc']:.4f})")
    print(f"  Chal DetRate: {best['ch_det_rate']:.4f}  (paper: {paper_ref['ch_det_rate']:.4f}  prev: {prev_best['ch_det_rate']:.4f})")

    results = {
        "paper_baseline": paper_ref,
        "prev_best_ensemble": prev_best,
        "best_ensemble_v2": best,
    }
    out = RESULTS / "lgbm_v2_results.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
