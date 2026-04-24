#!/usr/bin/env python3
"""
Calibrated meta-stacking — beyond the paper baseline.

What the paper does:
  - Score each file with per-type LightGBM models
  - Weighted average of raw scores (fixed weights, no calibration)

What we do extra:
  1. Isotonic regression calibration — maps raw LightGBM scores to proper
     probabilities. Per-type models have different score distributions
     (PDF model's 0.7 ≠ Win32 model's 0.7). Calibration aligns them.
  2. Logistic regression meta-stacker — learns the optimal per-feature-type
     weighting from held-out data instead of hand-tuning.
  3. File-type indicator features — the stacker knows which type it's
     scoring, allowing type-aware weighting.

Calibration/meta set: last 15% of training data (held out).

Outputs:
  checkpoints/lgbm/calibrators.pkl   — isotonic regressors per model
  checkpoints/lgbm/meta_stacker.pkl  — logistic regression meta model
  results/lgbm_calibrated_results.json
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

ARTIFACTS = Path("/Users/roheeeee/Documents/DACS/EMBER2024-artifacts")
DATA_DIR  = Path("/Users/roheeeee/Documents/DACS/EMBER2024-corrected-full")
CH_JSONL  = Path("/Users/roheeeee/Documents/DACS/EMBER2024-full-local"
                 "/2023-09-24_2024-12-14_challenge_malicious.jsonl")
CKPT      = Path("checkpoints/lgbm")
RESULTS   = Path("results")
BATCH     = 50_000

FILE_TYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]
FT_IDX     = {ft: i for i, ft in enumerate(FILE_TYPES)}

PAPER_PER_TYPE = {ft: ARTIFACTS / f"EMBER2024_{ft}.model" for ft in FILE_TYPES}
PAPER_ALL      = ARTIFACTS / "EMBER2024_all.model"
OUR_ALL        = CKPT / "lgbm_64leaf.txt"
OUR_PER_TYPE   = {ft: CKPT / f"lgbm_64leaf_{ft}.txt" for ft in FILE_TYPES}


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


def build_stacking_features(raw_scores: dict, ftypes: np.ndarray) -> np.ndarray:
    """
    Columns: [paper_pt, paper_all, our_all, ft_onehot×6]
    If our_per_type models exist, also include our_pt column.
    """
    cols = [
        raw_scores["paper_pt"],
        raw_scores["paper_all"],
        raw_scores["our_all"],
    ]
    if "our_pt" in raw_scores:
        cols.append(raw_scores["our_pt"])

    # File-type one-hot (lets the meta-model learn per-type offsets)
    ft_oh = np.zeros((len(ftypes), len(FILE_TYPES)), dtype=np.float32)
    for ft, idx in FT_IDX.items():
        ft_oh[ftypes == ft, idx] = 1.0
    cols.append(ft_oh)

    return np.hstack([c.reshape(-1, 1) if c.ndim == 1 else c for c in cols])


def ch_metrics(ben_scores, ch_scores, threshold=0.5):
    y = np.concatenate([np.zeros(len(ben_scores)), np.ones(len(ch_scores))])
    s = np.concatenate([ben_scores, ch_scores])
    return dict(
        roc_auc  = float(roc_auc_score(y, s)),
        pr_auc   = float(average_precision_score(y, s)),
        det_rate = float((ch_scores >= threshold).mean()),
    )


# ── load models ──────────────────────────────────────────────────────────────

def load_boosters():
    print("Loading paper per-type models …")
    pt = {ft: lgb.Booster(model_file=str(p))
          for ft, p in PAPER_PER_TYPE.items() if p.exists()}
    print(f"  {len(pt)} paper per-type models loaded")

    our_pt = {ft: lgb.Booster(model_file=str(p))
              for ft, p in OUR_PER_TYPE.items() if p.exists()}
    if our_pt:
        print(f"  {len(our_pt)} our per-type models loaded")

    print("Loading paper_all …")
    paper_all = lgb.Booster(model_file=str(PAPER_ALL))
    print("Loading our 64-leaf …")
    our_all = lgb.Booster(model_file=str(OUR_ALL))
    return pt, our_pt, paper_all, our_all


# ── score a split ─────────────────────────────────────────────────────────────

def raw_score_split(X, ftypes, pt, our_pt, paper_all, our_all):
    out = {}
    out["paper_pt"]  = score_per_type(X, ftypes, pt)
    out["paper_all"] = predict_batched(paper_all, X)
    out["our_all"]   = predict_batched(our_all,   X)
    if our_pt:
        out["our_pt"] = score_per_type(X, ftypes, our_pt)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    pt, our_pt, paper_all, our_all = load_boosters()

    # ── Load full training data ───────────────────────────────────────────
    print("\nLoading training data …")
    X_train = np.memmap(DATA_DIR / "X_train.dat", dtype=np.float32, mode="r",
                        shape=(2_626_000, 2568))
    y_train = np.array(np.memmap(DATA_DIR / "y_train.dat", dtype=np.int32, mode="r",
                                 shape=(2_626_000,)))

    # Stratified random 15% split — temporal split fails because malicious samples
    # are concentrated in the earlier portion of the temporally-ordered dataset.
    rng = np.random.default_rng(42)
    all_idx = np.arange(len(y_train))
    ben_idx_all = all_idx[y_train == 0]
    mal_idx_all = all_idx[y_train == 1]

    n_ben_meta = int(len(ben_idx_all) * 0.15)
    n_mal_meta = int(len(mal_idx_all) * 0.15)

    meta_ben = rng.choice(ben_idx_all, n_ben_meta, replace=False)
    meta_mal = rng.choice(mal_idx_all, n_mal_meta, replace=False)
    meta_idx = np.sort(np.concatenate([meta_ben, meta_mal]))
    y_meta   = y_train[meta_idx]
    print(f"  Meta/calibration set: {len(meta_idx):,} samples  "
          f"({(y_meta==1).sum():,} malicious, {(y_meta==0).sum():,} benign)")

    # File types for ALL training samples (read once, then index into it)
    print("  Reading file-type labels for all training samples …")
    train_jsonl = DATA_DIR / "2023-09-24_2024-09-21_train.jsonl"
    all_ftypes_list = []
    with open(train_jsonl) as f:
        for i, line in enumerate(f):
            all_ftypes_list.append(json.loads(line).get("file_type", "Win32"))
            if i % 500_000 == 0 and i > 0:
                print(f"    … {i:,} rows", flush=True)
    all_ftypes  = np.array(all_ftypes_list)
    meta_ftypes = all_ftypes[meta_idx]
    print(f"  File types for meta set: {dict(zip(*np.unique(meta_ftypes, return_counts=True)))}")

    # ── Score meta set with all base models ───────────────────────────────
    print("\nScoring meta set …")
    X_meta = np.array(X_train[meta_idx])
    raw_meta = raw_score_split(X_meta, meta_ftypes, pt, our_pt, paper_all, our_all)

    # ── Fit isotonic calibrators ──────────────────────────────────────────
    print("\nFitting isotonic calibrators …")
    calibrators = {}
    for key, scores in raw_meta.items():
        if key == "our_pt" or key.endswith("_pt"):
            # Per-type models: calibrate per file type for more accuracy
            cal_dict = {}
            for ft in FILE_TYPES:
                mask = meta_ftypes == ft
                if mask.sum() < 50:
                    continue
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(scores[mask], y_meta[mask])
                cal_dict[ft] = ir
            calibrators[key] = cal_dict
        else:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(scores, y_meta)
            calibrators[key] = ir

    def apply_calibration(raw_scores, ftypes, calibrators):
        cal = {}
        for key, scores in raw_scores.items():
            c = calibrators[key]
            if isinstance(c, dict):
                out = np.zeros(len(scores))
                for ft, ir in c.items():
                    mask = ftypes == ft
                    if mask.any():
                        out[mask] = ir.predict(scores[mask])
                cal[key] = out
            else:
                cal[key] = c.predict(scores)
        return cal

    cal_meta = apply_calibration(raw_meta, meta_ftypes, calibrators)

    # ── Fit logistic regression meta-stacker ─────────────────────────────
    print("\nFitting logistic regression meta-stacker …")
    F_meta = build_stacking_features(cal_meta, meta_ftypes)
    meta_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                  class_weight="balanced", random_state=42)
    meta_clf.fit(F_meta, y_meta)

    meta_train_auc = roc_auc_score(y_meta, meta_clf.predict_proba(F_meta)[:, 1])
    print(f"  Meta-train ROC-AUC (in-sample): {meta_train_auc:.4f}")
    print(f"  Coefficients: {dict(zip(['paper_pt','paper_all','our_all'] + (['our_pt'] if our_pt else []) + FILE_TYPES, meta_clf.coef_[0].round(3).tolist()))}")

    # ── Save calibrators + stacker ────────────────────────────────────────
    CKPT.mkdir(parents=True, exist_ok=True)
    with open(CKPT / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)
    with open(CKPT / "meta_stacker.pkl", "wb") as f:
        pickle.dump(meta_clf, f)
    print("\nSaved calibrators.pkl and meta_stacker.pkl")

    # ── Evaluate on test set ──────────────────────────────────────────────
    print("\nLoading test + challenge data …")
    X_test  = np.memmap(DATA_DIR / "X_test.dat",      dtype=np.float32, mode="r", shape=(605929, 2568))
    y_test  = np.array(np.memmap(DATA_DIR / "y_test.dat", dtype=np.int32, mode="r", shape=(605929,)))
    X_ch    = np.memmap(DATA_DIR / "X_challenge.dat", dtype=np.float32, mode="r", shape=(6315, 2568))

    test_ftypes = np.array([json.loads(l).get("file_type", "Win32")
                             for l in open(DATA_DIR / "2024-09-22_2024-12-14_test.jsonl")])
    ch_ftypes   = np.array([json.loads(l).get("file_type", "Win32")
                             for l in open(CH_JSONL)])

    print("Scoring test set …")
    raw_test = raw_score_split(X_test, test_ftypes, pt, our_pt, paper_all, our_all)
    cal_test = apply_calibration(raw_test, test_ftypes, calibrators)
    F_test   = build_stacking_features(cal_test, test_ftypes)
    test_stacked = meta_clf.predict_proba(F_test)[:, 1]

    print("Scoring challenge set …")
    raw_ch = raw_score_split(X_ch, ch_ftypes, pt, our_pt, paper_all, our_all)
    cal_ch = apply_calibration(raw_ch, ch_ftypes, calibrators)
    F_ch   = build_stacking_features(cal_ch, ch_ftypes)
    ch_stacked = meta_clf.predict_proba(F_ch)[:, 1]

    ben_idx = np.flatnonzero(y_test == 0)
    cm = ch_metrics(test_stacked[ben_idx], ch_stacked)
    test_roc = float(roc_auc_score(y_test, test_stacked))
    test_pr  = float(average_precision_score(y_test, test_stacked))

    # ── Reference: raw v2 ensemble (no calibration) ───────────────────────
    raw_ens_test = (raw_test["paper_pt"] + 0.1 * raw_test["paper_all"]) / 1.1
    raw_ens_ch   = (raw_ch["paper_pt"]   + 0.1 * raw_ch["paper_all"])   / 1.1
    cm_raw = ch_metrics(raw_ens_test[ben_idx], raw_ens_ch)
    test_roc_raw = float(roc_auc_score(y_test, raw_ens_test))

    paper_ref = dict(test_roc_auc=0.9968, ch_roc_auc=0.9533,
                     ch_pr_auc=0.4725, ch_det_rate=0.6654)

    print("\n" + "="*65)
    print(f"{'Metric':<30} {'Paper':>8}  {'V2 (raw)':>10}  {'Calibrated':>12}")
    print("-"*65)
    print(f"{'Test ROC-AUC':<30} {paper_ref['test_roc_auc']:>8.4f}  {test_roc_raw:>10.4f}  {test_roc:>12.4f}")
    print(f"{'Challenge ROC-AUC':<30} {paper_ref['ch_roc_auc']:>8.4f}  {cm_raw['roc_auc']:>10.4f}  {cm['roc_auc']:>12.4f}")
    print(f"{'Challenge PR-AUC':<30} {paper_ref['ch_pr_auc']:>8.4f}  {cm_raw['pr_auc']:>10.4f}  {cm['pr_auc']:>12.4f}")
    print(f"{'Challenge Det. Rate':<30} {paper_ref['ch_det_rate']:>8.4f}  {cm_raw['det_rate']:>10.4f}  {cm['det_rate']:>12.4f}")
    print("="*65)

    results = {
        "paper_baseline": paper_ref,
        "v2_raw_ensemble": {
            "test_roc_auc": test_roc_raw,
            **{f"ch_{k}": v for k, v in cm_raw.items()},
        },
        "calibrated_stacked": {
            "test_roc_auc":  test_roc,
            "test_pr_auc":   test_pr,
            "ch_roc_auc":    cm["roc_auc"],
            "ch_pr_auc":     cm["pr_auc"],
            "ch_det_rate":   cm["det_rate"],
            "n_meta_samples": int(len(meta_idx)),
        },
    }
    out = RESULTS / "lgbm_calibrated_results.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
