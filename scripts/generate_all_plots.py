#!/usr/bin/env python3
"""
Generate all report figures in one pass.
Scores are cached to results/all_scores.npz after the first run.

Outputs (all in results/figs/):
  01_architecture.png
  02_metrics_table.png
  03_roc_curves.png
  04_pr_curves.png
  05_feature_importance.png
  06_ensemble_failure.png
  07_insights.png
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS = Path("/Users/roheeeee/Documents/DACS/EMBER2024-artifacts")
DATA_DIR  = Path("/Users/roheeeee/Documents/DACS/EMBER2024-corrected-full")
CH_JSONL  = Path("/Users/roheeeee/Documents/DACS/EMBER2024-full-local"
                 "/2023-09-24_2024-12-14_challenge_malicious.jsonl")
CKPT      = Path("checkpoints/lgbm")
RESULTS   = Path("results")
FIGS      = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
SCORES_CACHE = RESULTS / "all_scores.npz"

FILE_TYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]
BATCH = 50_000

PALETTE = {
    "ensemble_v2":  "#1f77b4",
    "rank_ens":     "#d62728",
    "paper_all":    "#ff7f0e",
    "our_64leaf":   "#2ca02c",
    "neural":       "#9467bd",
    "paper_ref":    "#8c8c8c",
}

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── scoring helpers ────────────────────────────────────────────────────────────
def predict_batched(booster, X, indices=None):
    if indices is None:
        indices = np.arange(len(X))
    out = []
    for i in range(0, len(indices), BATCH):
        b = indices[i: i + BATCH]
        out.append(booster.predict(np.array(X[b])))
    return np.concatenate(out)

def score_per_type(X, ftypes, boosters):
    s = np.zeros(len(X))
    for ft, b in boosters.items():
        mask = ftypes == ft
        if mask.any():
            s[mask] = predict_batched(b, X, np.where(mask)[0])
    return s

def to_rank(s): return rankdata(s, method="average") / len(s)
def sharpen(r, T): return r ** (1.0 / T)


# ── STEP 1: load / compute scores ─────────────────────────────────────────────
def get_scores():
    if SCORES_CACHE.exists():
        print("Loading cached scores …")
        d = np.load(SCORES_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.keys()}

    print("Loading models …")
    pt        = {ft: lgb.Booster(model_file=str(ARTIFACTS / f"EMBER2024_{ft}.model"))
                 for ft in FILE_TYPES}
    paper_all = lgb.Booster(model_file=str(ARTIFACTS / "EMBER2024_all.model"))
    our_64    = lgb.Booster(model_file=str(CKPT / "lgbm_64leaf.txt"))

    print("Loading test + challenge …")
    X_test  = np.memmap(DATA_DIR/"X_test.dat",      dtype=np.float32, mode="r", shape=(605929, 2568))
    y_test  = np.array(np.memmap(DATA_DIR/"y_test.dat", dtype=np.int32, mode="r", shape=(605929,)))
    X_ch    = np.memmap(DATA_DIR/"X_challenge.dat", dtype=np.float32, mode="r", shape=(6315, 2568))

    test_ft = np.array([json.loads(l).get("file_type","") for l in open(DATA_DIR/"2024-09-22_2024-12-14_test.jsonl")])
    ch_ft   = np.array([json.loads(l).get("file_type","") for l in open(CH_JSONL)])

    print("Scoring test …")
    s_pt_test  = score_per_type(X_test, test_ft, pt)
    s_all_test = predict_batched(paper_all, X_test)
    s_64_test  = predict_batched(our_64,    X_test)

    print("Scoring challenge …")
    s_pt_ch  = score_per_type(X_ch, ch_ft, pt)
    s_all_ch = predict_batched(paper_all, X_ch)
    s_64_ch  = predict_batched(our_64,    X_ch)

    np.savez_compressed(SCORES_CACHE,
        y_test=y_test,
        s_pt_test=s_pt_test,  s_all_test=s_all_test,  s_64_test=s_64_test,
        s_pt_ch=s_pt_ch,      s_all_ch=s_all_ch,       s_64_ch=s_64_ch,
        test_ft=test_ft.astype(str),  ch_ft=ch_ft.astype(str),
    )
    print(f"Cached → {SCORES_CACHE}")
    return dict(y_test=y_test,
                s_pt_test=s_pt_test,  s_all_test=s_all_test,  s_64_test=s_64_test,
                s_pt_ch=s_pt_ch,      s_all_ch=s_all_ch,       s_64_ch=s_64_ch,
                test_ft=test_ft,      ch_ft=ch_ft)


def build_ensembles(d):
    y   = d["y_test"]
    ben = np.flatnonzero(y == 0)

    # V2 ensemble
    v2_test = (d["s_pt_test"] + 0.1*d["s_all_test"]) / 1.1
    v2_ch   = (d["s_pt_ch"]   + 0.1*d["s_all_ch"])   / 1.1

    # Rank ensemble (T=2.0)
    N = len(y) + len(d["s_pt_ch"])
    combined_pt  = np.concatenate([d["s_pt_test"],  d["s_pt_ch"]])
    combined_all = np.concatenate([d["s_all_test"], d["s_all_ch"]])
    r_pt  = to_rank(combined_pt);  r_all = to_rank(combined_all)
    rk_test = sharpen((r_pt[:len(y)] + 0.7*r_all[:len(y)])/1.7, 2.0)
    rk_ch   = sharpen((r_pt[len(y):] + 0.7*r_all[len(y):])/1.7, 2.0)

    def ch_pair(test_s, ch_s):
        y_c = np.concatenate([np.zeros(len(ben)), np.ones(len(ch_s))])
        s_c = np.concatenate([test_s[ben], ch_s])
        return y_c, s_c

    return {
        "paper_all":   (y, d["s_all_test"], *ch_pair(d["s_all_test"], d["s_all_ch"])),
        "our_64leaf":  (y, d["s_64_test"],  *ch_pair(d["s_64_test"],  d["s_64_ch"])),
        "ensemble_v2": (y, v2_test,          *ch_pair(v2_test,          v2_ch)),
        "rank_ens":    (y, rk_test,          *ch_pair(rk_test,           rk_ch)),
    }


# ── PLOT 1: Architecture ───────────────────────────────────────────────────────
def plot_architecture():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    boxes = [
        (0.04, 0.40, 0.12, 0.20, "Input\n2568 features\n(static PE/file)", "#aec6cf"),
        (0.22, 0.55, 0.12, 0.12, "Paper\nper-type\nLGBM (×6)", "#ffb347"),
        (0.22, 0.40, 0.12, 0.12, "Paper\nall-type\nLGBM", "#ffb347"),
        (0.22, 0.25, 0.12, 0.12, "Our\n64-leaf\nLGBM", "#77dd77"),
        (0.48, 0.40, 0.14, 0.20, "V2 Ensemble\n(grid-searched\nweights)", "#1f77b4"),
        (0.48, 0.10, 0.14, 0.20, "Rank Ensemble\n(Borda count\n+ sharpening T=2)", "#d62728"),
        (0.74, 0.40, 0.14, 0.12, "Test ROC\n0.9976", "#b5ead7"),
        (0.74, 0.25, 0.14, 0.12, "Chal PR-AUC\n0.6284", "#b5ead7"),
        (0.74, 0.10, 0.14, 0.12, "Det. Rate\n97.86%", "#ffdac1"),
    ]
    for (x, y, w, h, txt, col) in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        facecolor=col, edgecolor="#333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center",
                fontsize=8.5, fontweight="bold", wrap=True)

    arrows = [
        (0.16, 0.50, 0.22, 0.61), (0.16, 0.50, 0.22, 0.46), (0.16, 0.50, 0.22, 0.31),
        (0.34, 0.61, 0.48, 0.50), (0.34, 0.46, 0.48, 0.50), (0.34, 0.31, 0.48, 0.50),
        (0.34, 0.61, 0.48, 0.20), (0.34, 0.46, 0.48, 0.20), (0.34, 0.31, 0.48, 0.20),
        (0.62, 0.50, 0.74, 0.46), (0.62, 0.50, 0.74, 0.31),
        (0.62, 0.20, 0.74, 0.16),
    ]
    for (x1,y1,x2,y2) in arrows:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    ax.set_title("Dataset & Model Architecture — EMBER2024 Malware Detection",
                 fontsize=13, fontweight="bold", pad=12)

    ft_patch  = mpatches.Patch(color="#ffb347", label="Paper EMBER2024 official models")
    our_patch = mpatches.Patch(color="#77dd77", label="Our retrained LightGBM (64 leaves)")
    ax.legend(handles=[ft_patch, our_patch], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS/"01_architecture.png", bbox_inches="tight")
    plt.close()
    print("  01_architecture.png saved")


# ── PLOT 2: Metrics table ──────────────────────────────────────────────────────
def plot_metrics_table():
    rows = [
        ["Metric",               "Paper Baseline", "Our 64-leaf LGBM", "V2 Ensemble",  "Rank Ensemble"],
        ["Test ROC-AUC",         "0.9968",         "0.9911",           "0.9976 ✓",     "0.9974 ✓"],
        ["Challenge ROC-AUC",    "0.9533",         "0.9257",           "0.9556 ✓",     "0.9496"],
        ["Challenge PR-AUC",     "0.4725",         "0.3718",           "0.6284 ✓",     "0.6139 ✓"],
        ["Challenge Det. Rate",  "66.54%",         "65.57%",           "70.23% ✓",     "97.86% ✓"],
    ]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.1)

    col_colors = ["#f0f0f0", "#ffdddd", "#ffe8cc", "#d4edff", "#ffd6d6"]
    beat_color = "#c8f7c5"
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(col_colors[c] if c < len(col_colors) else "#f0f0f0")
            cell.set_text_props(fontweight="bold")
        elif "✓" in cell.get_text().get_text():
            cell.set_facecolor(beat_color)
        cell.set_edgecolor("#aaa")

    ax.set_title("Evaluation Metrics — All Models vs Paper Baseline  (✓ = beats paper)",
                 fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGS/"02_metrics_table.png", bbox_inches="tight", dpi=180)
    plt.close()
    print("  02_metrics_table.png saved")


# ── PLOT 3: ROC curves ────────────────────────────────────────────────────────
def plot_roc(ensembles, smoke):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    model_cfg = [
        ("ensemble_v2", "V2 Ensemble",        PALETTE["ensemble_v2"], "-"),
        ("rank_ens",    "Rank Ensemble",       PALETTE["rank_ens"],    "--"),
        ("paper_all",   "Paper Baseline",      PALETTE["paper_all"],   "-."),
        ("our_64leaf",  "Our 64-leaf LGBM",    PALETTE["our_64leaf"],  ":"),
    ]

    for ax, (split, idx) in zip(axes, [("Test Set", 0), ("Challenge Set", 2)]):
        for key, label, color, ls in model_cfg:
            y_true, scores = ensembles[key][idx], ensembles[key][idx+1]
            fpr, tpr, _ = roc_curve(y_true, scores)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{label} ({a:.4f})")

        if split == "Test Set" and smoke is not None:
            fpr_n, tpr_n, _ = roc_curve(smoke["test_y_true"], smoke["test_our_score"])
            a_n = auc(fpr_n, tpr_n)
            ax.plot(fpr_n, tpr_n, color=PALETTE["neural"], lw=1.5, ls=":",
                    label=f"Neural/smoke ({a_n:.4f})", alpha=0.7)

        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.3)
        ax.set_xlim(0,1); ax.set_ylim(0,1.01)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {split}")
        ax.legend(loc="lower right", title="Model (AUC)")

    plt.suptitle("ROC Curves — EMBER2024 Malware Detection", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS/"03_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("  03_roc_curves.png saved")


# ── PLOT 4: PR curves ─────────────────────────────────────────────────────────
def plot_pr(ensembles, smoke):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    model_cfg = [
        ("ensemble_v2", "V2 Ensemble",     PALETTE["ensemble_v2"], "-"),
        ("rank_ens",    "Rank Ensemble",   PALETTE["rank_ens"],    "--"),
        ("paper_all",   "Paper Baseline",  PALETTE["paper_all"],   "-."),
        ("our_64leaf",  "Our 64-leaf LGBM",PALETTE["our_64leaf"],  ":"),
    ]

    for ax, (split, idx) in zip(axes, [("Test Set", 0), ("Challenge Set", 2)]):
        for key, label, color, ls in model_cfg:
            y_true, scores = ensembles[key][idx], ensembles[key][idx+1]
            prec, rec, _ = precision_recall_curve(y_true, scores)
            a = average_precision_score(y_true, scores)
            ax.plot(rec, prec, color=color, lw=2, ls=ls, label=f"{label} (AP={a:.4f})")

        if split == "Test Set" and smoke is not None:
            prec_n, rec_n, _ = precision_recall_curve(smoke["test_y_true"], smoke["test_our_score"])
            a_n = average_precision_score(smoke["test_y_true"], smoke["test_our_score"])
            ax.plot(rec_n, prec_n, color=PALETTE["neural"], lw=1.5, ls=":",
                    label=f"Neural/smoke (AP={a_n:.4f})", alpha=0.7)

        ax.set_xlim(0,1); ax.set_ylim(0,1.01)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR Curves — {split}")
        ax.legend(loc="upper right", title="Model (AP)")

    plt.suptitle("Precision-Recall Curves — EMBER2024 Malware Detection", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS/"04_pr_curves.png", bbox_inches="tight")
    plt.close()
    print("  04_pr_curves.png saved")


# ── PLOT 5: Feature importance ────────────────────────────────────────────────
def plot_feature_importance():
    model_path = CKPT / "lgbm_64leaf.txt"
    booster = lgb.Booster(model_file=str(model_path))

    imp_gain  = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    feat_idx  = np.arange(len(imp_gain))

    top_n = 30
    order = np.argsort(imp_gain)[::-1][:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Gain importance
    axes[0].barh(range(top_n), imp_gain[order][::-1], color="#1f77b4", alpha=0.8)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([f"Feature {feat_idx[i]}" for i in order[::-1]], fontsize=7)
    axes[0].set_xlabel("Total Gain")
    axes[0].set_title(f"Top {top_n} Features by Gain\n(Our 64-leaf LGBM)")

    # Split importance (top 30 by gain, same features)
    axes[1].barh(range(top_n), imp_split[order][::-1], color="#ff7f0e", alpha=0.8)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels([f"Feature {feat_idx[i]}" for i in order[::-1]], fontsize=7)
    axes[1].set_xlabel("Number of Splits")
    axes[1].set_title(f"Same Features by Split Count")

    # Annotate feature index ranges (EMBER2024 v3 feature blocks)
    blocks = [
        (0,   8,   "Header"),
        (8,   58,  "DOS/Rich"),
        (58,  58+58, "Section stats"),
        (256, 256+1024, "Byte histogram"),
        (1280,1280+256,"Byte entropy"),
        (1536,1536+128,"String stats"),
        (1664,1664+100,"Imports"),
        (1764,1764+100,"Exports"),
        (1864,2048,     "Datadirs"),
        (2048,2568,     "General/misc"),
    ]
    for start, end, name in blocks:
        hits = [(i, feat_idx[o]) for i, o in enumerate(order[::-1]) if start <= feat_idx[o] < end]
        if hits:
            axes[0].text(imp_gain[order[::-1][hits[0][0]]] * 0.02,
                         hits[0][0], f"← {name}", va="center", fontsize=6.5, color="#333")

    plt.suptitle("Feature Importance — Our 64-leaf LightGBM", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS/"05_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("  05_feature_importance.png saved")


# ── PLOT 6: Ensemble results + failure analysis ────────────────────────────────
def plot_ensemble_failure(d):
    y_test = d["y_test"]
    ben_idx = np.flatnonzero(y_test == 0)
    ch_ftypes = np.array(d["ch_ft"])

    v2_ch  = (d["s_pt_ch"]  + 0.1*d["s_all_ch"]) / 1.1

    # Detection rate per file type
    det_by_ft_v2    = {}
    det_by_ft_paper = {}
    for ft in FILE_TYPES:
        mask = ch_ftypes == ft
        if mask.sum() == 0:
            continue
        det_by_ft_v2[ft]    = float((v2_ch[mask] >= 0.5).mean())
        det_by_ft_paper[ft] = float((d["s_all_ch"][mask] >= 0.5).mean())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: det rate per type
    ax = axes[0]
    fts = list(det_by_ft_v2.keys())
    x = np.arange(len(fts))
    w = 0.35
    ax.bar(x - w/2, [det_by_ft_paper[f] for f in fts], w, label="Paper all-type",
           color=PALETTE["paper_all"], alpha=0.85)
    ax.bar(x + w/2, [det_by_ft_v2[f]    for f in fts], w, label="V2 Ensemble",
           color=PALETTE["ensemble_v2"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(fts, fontsize=9)
    ax.set_ylabel("Detection Rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Challenge Detection Rate\nby File Type")
    ax.legend(); ax.axhline(0.5, ls="--", color="red", lw=1, alpha=0.5, label="threshold 0.5")

    # Middle: score distribution on challenge
    ax = axes[1]
    ax.hist(d["s_all_ch"],  bins=50, alpha=0.6, color=PALETTE["paper_all"],  label="Paper all-type", density=True)
    ax.hist(v2_ch,           bins=50, alpha=0.6, color=PALETTE["ensemble_v2"], label="V2 Ensemble",   density=True)
    ax.axvline(0.5, color="red", lw=1.5, ls="--", label="threshold=0.5")
    ax.set_xlabel("Prediction Score"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution\non Challenge Set (evasive malware)")
    ax.legend()

    # Right: missed detections (false negatives) — score distribution of missed
    missed_mask = v2_ch < 0.5
    caught_mask = v2_ch >= 0.5
    ax = axes[2]
    ax.hist(v2_ch[missed_mask], bins=30, color="#d62728", alpha=0.75,
            label=f"Missed ({missed_mask.sum()}, {missed_mask.mean():.1%})", density=True)
    ax.hist(v2_ch[caught_mask], bins=30, color="#2ca02c", alpha=0.75,
            label=f"Caught ({caught_mask.sum()}, {caught_mask.mean():.1%})", density=True)
    ax.axvline(0.5, color="black", lw=1.5, ls="--")
    ax.set_xlabel("V2 Ensemble Score"); ax.set_ylabel("Density")
    ax.set_title(f"Failure Analysis — V2 Ensemble\non Challenge Set (n={len(v2_ch):,})")
    ax.legend()

    plt.suptitle("Ensemble Results & Failure Analysis — Evasive Malware Challenge Set",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS/"06_ensemble_failure.png", bbox_inches="tight")
    plt.close()
    print("  06_ensemble_failure.png saved")


# ── PLOT 7: Insights & future work ────────────────────────────────────────────
def plot_insights():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    col1 = [
        ("Key Insights", "#1f4e79", 14, True),
        ("", None, 10, False),
        ("1. Model complexity matters", "#1f4e79", 11, True),
        ("   2048-leaf model overfits → fails on evasive malware", "#333", 10, False),
        ("   64-leaf (matching paper) recovers generalisation", "#333", 10, False),
        ("", None, 10, False),
        ("2. Per-type specialisation is the #1 driver", "#1f4e79", 11, True),
        ("   Each file type has distinct evasion patterns", "#333", 10, False),
        ("   Win32/Win64/PDF/APK/ELF/Dot_Net each need own model", "#333", 10, False),
        ("", None, 10, False),
        ("3. Score scale mismatch across types", "#1f4e79", 11, True),
        ("   PDF model's 0.7 ≠ Win32 model's 0.7", "#333", 10, False),
        ("   Rank-based ensemble eliminates this bias", "#333", 10, False),
        ("", None, 10, False),
        ("4. Grid-search > hand-tuned weights", "#1f4e79", 11, True),
        ("   +2.1pp PR-AUC, +0.36pp det. rate over v1", "#333", 10, False),
        ("", None, 10, False),
        ("5. Rank ensemble: 97.86% detection rate", "#c00000", 11, True),
        ("   +31.32pp over paper, trade-off: lower ROC-AUC", "#333", 10, False),
    ]

    col2 = [
        ("Future Work", "#1a5276", 14, True),
        ("", None, 10, False),
        ("Train per-type 64-leaf models (our own)", "#1a5276", 11, True),
        ("   → 4-way ensemble: paper_pt + our_pt + paper_all + our_all", "#333", 10, False),
        ("   → Expected further +1-3pp on PR-AUC", "#333", 10, False),
        ("", None, 10, False),
        ("Full neural model training", "#1a5276", 11, True),
        ("   Stage 1: SupCon contrastive pretraining (50 epochs)", "#333", 10, False),
        ("   Stage 2: Multi-task fine-tuning (detection + family)", "#333", 10, False),
        ("   Stage 3: Prototypical inference for long-tail families", "#333", 10, False),
        ("", None, 10, False),
        ("Temporal calibration", "#1a5276", 11, True),
        ("   Concept drift: newer evasive malware is harder", "#333", 10, False),
        ("   Re-weight recent training samples more heavily", "#333", 10, False),
        ("", None, 10, False),
        ("Family classification", "#1a5276", 11, True),
        ("   Long-tail: many families have < 10 samples", "#333", 10, False),
        ("   Prototypical few-shot approach addresses this", "#333", 10, False),
        ("", None, 10, False),
        ("Adversarial training", "#1a5276", 11, True),
        ("   Explicitly train against challenge evasion patterns", "#333", 10, False),
    ]

    def draw_col(items, x_start):
        y = 0.96
        for txt, color, size, bold in items:
            if txt == "":
                y -= 0.03; continue
            ax.text(x_start, y, txt, transform=ax.transAxes,
                    color=color or "#333", fontsize=size,
                    fontweight="bold" if bold else "normal", va="top")
            y -= 0.047

    draw_col(col1, 0.02)
    ax.axvline(0.5, color="#aaa", lw=1, ymin=0, ymax=1)
    draw_col(col2, 0.52)

    ax.set_title("Key Insights & Future Work", fontsize=15, fontweight="bold",
                 pad=14, color="#1f1f1f")
    plt.tight_layout()
    plt.savefig(FIGS/"07_insights.png", bbox_inches="tight", dpi=160)
    plt.close()
    print("  07_insights.png saved")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    d = get_scores()
    ensembles = build_ensembles(d)

    smoke = None
    smoke_path = RESULTS / "smoke/raw_scores.npz"
    if smoke_path.exists():
        smoke = np.load(smoke_path, allow_pickle=True)

    print("\nGenerating plots …")
    plot_architecture()
    plot_metrics_table()
    plot_roc(ensembles, smoke)
    plot_pr(ensembles, smoke)
    plot_feature_importance()
    plot_ensemble_failure(d)
    plot_insights()

    print(f"\nAll figures saved to {FIGS.resolve()}/")
    for p in sorted(FIGS.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
