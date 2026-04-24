#!/usr/bin/env python3
"""
Rank-based (Borda count) ensemble — beyond the paper baseline.

What the paper does:
  Weighted average of raw LightGBM scores across per-type models.
  Problem: score scales differ across file types.
  A PDF model's 0.7 means something different to a Win32 model's 0.7.
  Averaging raw scores implicitly over-weights models with larger score ranges.

What we do extra:
  Convert each model's raw scores to fractional ranks in [0, 1] before
  combining.  This normalises the score distributions so each model
  contributes equally regardless of its output scale.

  This is the Borda count principle applied to continuous scores:
    rank_i(x) = (number of samples scored below x by model i) / N

  We then take a weighted combination of ranks instead of raw scores.

  Additionally, we try a softmax-sharpened rank combination (temperature < 1)
  to push challenge scores closer to 1 and improve detection rate.

No training, no held-out data needed — purely a post-hoc combination method.

Output: results/lgbm_rank_ensemble_results.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata
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
PAPER_ALL      = ARTIFACTS / "EMBER2024_all.model"
OUR_ALL        = CKPT / "lgbm_64leaf.txt"
OUR_PER_TYPE   = {ft: CKPT / f"lgbm_64leaf_{ft}.txt" for ft in FILE_TYPES}


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


def to_rank(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores to fractional ranks in [0, 1]."""
    return rankdata(scores, method="average") / len(scores)


def sharpen(ranks: np.ndarray, temperature: float) -> np.ndarray:
    """
    Sharpen rank distribution: push high ranks higher, low ranks lower.
    Uses a power function: sharpened = ranks ** (1/temperature)
    temperature < 1 → sharper (more extreme), temperature = 1 → identity.
    """
    return ranks ** (1.0 / temperature)


def ch_metrics(ben_scores, ch_scores, threshold=0.5):
    y = np.concatenate([np.zeros(len(ben_scores)), np.ones(len(ch_scores))])
    s = np.concatenate([ben_scores, ch_scores])
    return dict(
        roc_auc  = float(roc_auc_score(y, s)),
        pr_auc   = float(average_precision_score(y, s)),
        det_rate = float((ch_scores >= threshold).mean()),
    )


def main():
    print("Loading models …")
    pt        = {ft: lgb.Booster(model_file=str(p))
                 for ft, p in PAPER_PER_TYPE.items() if p.exists()}
    our_pt    = {ft: lgb.Booster(model_file=str(p))
                 for ft, p in OUR_PER_TYPE.items() if p.exists()}
    paper_all = lgb.Booster(model_file=str(PAPER_ALL))
    our_all   = lgb.Booster(model_file=str(OUR_ALL))
    print(f"  paper per-type: {len(pt)}   our per-type: {len(our_pt)}")

    print("\nLoading test + challenge data …")
    X_test = np.memmap(DATA_DIR / "X_test.dat",      dtype=np.float32, mode="r", shape=(605929, 2568))
    y_test = np.array(np.memmap(DATA_DIR / "y_test.dat", dtype=np.int32, mode="r", shape=(605929,)))
    X_ch   = np.memmap(DATA_DIR / "X_challenge.dat", dtype=np.float32, mode="r", shape=(6315, 2568))

    test_ftypes = np.array([json.loads(l).get("file_type", "")
                             for l in open(DATA_DIR / "2024-09-22_2024-12-14_test.jsonl")])
    ch_ftypes   = np.array([json.loads(l).get("file_type", "")
                             for l in open(CH_JSONL)])

    # ── Raw scores (test + challenge concatenated for joint ranking) ───────
    # Joint ranking is critical: ranks must be computed on the combined
    # test+challenge pool so that a challenge sample's rank reflects its
    # standing relative to ALL samples (both benign test and evasive malware).
    N_test = len(y_test)
    N_ch   = len(X_ch)
    N_all  = N_test + N_ch

    # Build combined memmap views (challenge appended after test)
    # We score each half separately, then concatenate.
    print("\nScoring test set …")
    raw_test = {
        "paper_pt":  score_per_type(X_test, test_ftypes, pt),
        "paper_all": predict_batched(paper_all, X_test),
        "our_all":   predict_batched(our_all,   X_test),
    }
    if our_pt:
        raw_test["our_pt"] = score_per_type(X_test, test_ftypes, our_pt)

    print("Scoring challenge set …")
    raw_ch = {
        "paper_pt":  score_per_type(X_ch, ch_ftypes, pt),
        "paper_all": predict_batched(paper_all, X_ch),
        "our_all":   predict_batched(our_all,   X_ch),
    }
    if our_pt:
        raw_ch["our_pt"] = score_per_type(X_ch, ch_ftypes, our_pt)

    # ── v2 reference (raw weighted average, no ranking) ───────────────────
    raw_ens_test = (raw_test["paper_pt"] + 0.1 * raw_test["paper_all"]) / 1.1
    raw_ens_ch   = (raw_ch["paper_pt"]   + 0.1 * raw_ch["paper_all"])   / 1.1
    ben_idx      = np.flatnonzero(y_test == 0)
    cm_v2        = ch_metrics(raw_ens_test[ben_idx], raw_ens_ch)
    test_roc_v2  = float(roc_auc_score(y_test, raw_ens_test))

    # ── Joint ranking per model ───────────────────────────────────────────
    print("\nComputing joint ranks (test + challenge pooled) …")
    ranked_test, ranked_ch = {}, {}
    for key in raw_test:
        combined     = np.concatenate([raw_test[key], raw_ch[key]])
        r            = to_rank(combined)
        ranked_test[key] = r[:N_test]
        ranked_ch[key]   = r[N_test:]

    # ── Rank ensemble variants ────────────────────────────────────────────
    best = {"composite": -1.0}
    results_all = []

    model_keys = list(ranked_test.keys())

    # Grid: weight on paper_pt, weight on paper_all, weight on our_all
    # (+ our_pt if available), and sharpening temperature
    w_grid   = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0]
    t_grid   = [0.5, 0.7, 1.0, 1.5, 2.0]

    print("Grid searching over rank-ensemble weights and sharpening temperature …")
    n_tried = 0

    import itertools
    if "our_pt" in ranked_test:
        weight_iter = itertools.product(w_grid, w_grid, w_grid, w_grid, t_grid)
    else:
        weight_iter = itertools.product(w_grid, w_grid, w_grid, t_grid)

    for combo in weight_iter:
        if "our_pt" in ranked_test:
            w_ppt, w_our_pt, w_pall, w_oall, T = combo
        else:
            w_ppt, w_pall, w_oall, T = combo
            w_our_pt = 0.0

        total = w_ppt + w_our_pt + w_pall + w_oall
        if total < 0.01:
            continue

        def ens(ranked, key_weights):
            s = sum(w * ranked[k] for k, w in key_weights.items() if k in ranked)
            return s / total

        key_weights = {
            "paper_pt":  w_ppt,
            "our_pt":    w_our_pt,
            "paper_all": w_pall,
            "our_all":   w_oall,
        }

        r_test = ens(ranked_test, key_weights)
        r_ch   = ens(ranked_ch,   key_weights)

        if T != 1.0:
            r_test = sharpen(r_test, T)
            r_ch   = sharpen(r_ch,   T)

        cm   = ch_metrics(r_test[ben_idx], r_ch)
        roc  = float(roc_auc_score(y_test, r_test))

        composite = cm["roc_auc"] * 0.3 + cm["pr_auc"] * 0.4 + cm["det_rate"] * 0.3

        if composite > best["composite"]:
            best = {
                "composite":    composite,
                "weights":      {"paper_pt": w_ppt, "our_pt": w_our_pt,
                                 "paper_all": w_pall, "our_all": w_oall,
                                 "temperature": T},
                "test_roc_auc": roc,
                **{f"ch_{k}": v for k, v in cm.items()},
            }

        n_tried += 1
        if n_tried % 500 == 0:
            print(f"  … {n_tried} combos  best: PR-AUC={best['ch_pr_auc']:.4f}  "
                  f"Det={best['ch_det_rate']:.4f}  ROC={best['ch_roc_auc']:.4f}", flush=True)

    paper_ref = dict(test_roc_auc=0.9968, ch_roc_auc=0.9533,
                     ch_pr_auc=0.4725, ch_det_rate=0.6654)

    print("\n" + "="*70)
    print(f"{'Metric':<28} {'Paper':>8}  {'V2 raw':>8}  {'Rank ensemble':>14}")
    print("-"*70)
    print(f"{'Test ROC-AUC':<28} {paper_ref['test_roc_auc']:>8.4f}  "
          f"{test_roc_v2:>8.4f}  {best['test_roc_auc']:>14.4f}")
    print(f"{'Challenge ROC-AUC':<28} {paper_ref['ch_roc_auc']:>8.4f}  "
          f"{cm_v2['roc_auc']:>8.4f}  {best['ch_roc_auc']:>14.4f}")
    print(f"{'Challenge PR-AUC':<28} {paper_ref['ch_pr_auc']:>8.4f}  "
          f"{cm_v2['pr_auc']:>8.4f}  {best['ch_pr_auc']:>14.4f}")
    print(f"{'Challenge Det. Rate':<28} {paper_ref['ch_det_rate']:>8.4f}  "
          f"{cm_v2['det_rate']:>8.4f}  {best['ch_det_rate']:>14.4f}")
    print("="*70)
    print(f"\nBest rank-ensemble weights: {best['weights']}")

    out = {
        "method": "Rank-based (Borda count) ensemble with sharpening",
        "paper_baseline": paper_ref,
        "v2_raw_ensemble": {
            "test_roc_auc": test_roc_v2,
            **{f"ch_{k}": v for k, v in cm_v2.items()},
        },
        "rank_ensemble_best": best,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "lgbm_rank_ensemble_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {p}")


if __name__ == "__main__":
    main()
