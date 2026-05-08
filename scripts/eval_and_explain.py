#!/usr/bin/env python3
"""Evaluate chunked-RF ensembles and MLPs and produce metric plots + feature explainability.

Outputs saved under --out-dir:
 - metrics.csv (per-model and ensemble)
 - ROC and PR curve PNGs
 - feature_importance_*.png for RF and permutation-based importance for MLP

Notes:
 - For RF ensembles we average predicted probabilities across chunk models and
   average their sklearn feature_importances_ for a global RF importance.
 - For MLPs (PyTorch state_dict) we compute permutation importance on a sample
   of the evaluation indices (controlled by --max-samples-for-importance) and
   report the top-K features.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn


def detect_meta_field(meta_path, sample_lines=200):
    keys = {}
    with open(meta_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for k in obj.keys():
                keys[k.lower()] = keys.get(k.lower(), 0) + 1
    for candidate in ("file_type", "filetype", "type", "mime", "mime_type", "file_type_guess"):
        if candidate in keys:
            return candidate
    if keys:
        return max(keys.items(), key=lambda x: x[1])[0]
    raise RuntimeError("Could not detect a metadata field for file type")


def iterate_metadata(meta_path):
    with open(meta_path, 'r', encoding='utf8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                yield {}


def collect_indices_for_type(meta_path, field_name, desired_type, y_mm):
    indices = []
    for i, obj in enumerate(iterate_metadata(meta_path)):
        val = obj.get(field_name) or obj.get(field_name.lower())
        if val is None:
            val = 'unknown'
        if val == desired_type:
            if y_mm[i] != -1:
                indices.append(i)
    return indices


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPWrapper:
    """Wrap a PyTorch MLP state_dict so it provides predict_proba(X) in numpy."""

    def __init__(self, state_path, in_dim, hidden_dim=1024, device='cpu'):
        self.device = torch.device(device)
        self.model = MLP(in_dim, hidden_dim=hidden_dim).to(self.device)
        sd = torch.load(state_path, map_location=self.device)
        # assume sd is state_dict
        if isinstance(sd, dict):
            try:
                self.model.load_state_dict(sd)
            except Exception:
                # fallback: maybe the saved object is the model itself
                try:
                    self.model = sd
                except Exception:
                    raise
        else:
            # loaded a raw model object
            self.model = sd.to(self.device)
        self.model.eval()

    def predict_proba(self, X, batch_size=1024):
        probs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = X[i : i + batch_size]
                xt = torch.from_numpy(np.array(xb, dtype=np.float32)).to(self.device)
                logits = self.model(xt).cpu().numpy()
                p = 1.0 / (1.0 + np.exp(-logits))
                probs.append(p)
        return np.concatenate(probs)


def evaluate_predictions(ys, probs):
    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float('nan')
    ap = average_precision_score(ys, probs) if len(np.unique(ys)) > 1 else float('nan')
    labels = (probs > 0.5).astype(int)
    prec = precision_score(ys, labels, zero_division=0)
    rec = recall_score(ys, labels, zero_division=0)
    f1 = f1_score(ys, labels, zero_division=0)
    fpr, tpr, roc_th = roc_curve(ys, probs) if len(np.unique(ys)) > 1 else ([], [], [])
    prec_curve, rec_curve, pr_th = precision_recall_curve(ys, probs)
    return dict(auc=auc, ap=ap, precision=prec, recall=rec, f1=f1, fpr=fpr, tpr=tpr, roc_th=roc_th, pr_prec=prec_curve, pr_rec=rec_curve, pr_th=pr_th)


def plot_roc(fpr, tpr, outpath, title='ROC'):
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_pr(precision, recall, outpath, title='Precision-Recall'):
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_feature_importance(importances, outpath, feature_dim, top_k=50, title='Feature importance'):
    # importances: 1D array length feature_dim
    idx = np.argsort(importances)[::-1][:top_k]
    vals = importances[idx]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(idx)), vals)
    plt.xticks(range(len(idx)), [str(i) for i in idx], rotation=90)
    plt.xlabel('Feature index (top features)')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='DAFCS/data/local/EMBER2024-corrected-full')
    parser.add_argument('--feature-dim', type=int, default=2568)
    parser.add_argument('--subset', choices=('train','test','challenge'), default='test')
    parser.add_argument('--file-type', default=None, help='Exact file type to evaluate; if omitted and --all used, evaluates all types')
    parser.add_argument('--all', action='store_true', help='Evaluate all file types in the chosen subset')
    parser.add_argument('--models-dir', default='checkpoints/rf_chunks', help='Directory with rf_<type>_chunk*.joblib')
    parser.add_argument('--mlp-model', default=None, help='Path to mlp_<type>.pt (state_dict) to evaluate alongside RF ensemble')
    parser.add_argument('--mlp-hidden-dim', type=int, default=1024)
    parser.add_argument('--out-dir', default='results/eval_explain')
    parser.add_argument('--test-batch-size', type=int, default=2000)
    parser.add_argument('--max-samples-for-importance', type=int, default=20000, help='Max rows to use for permutation importance to save time')
    parser.add_argument('--top-k-features', type=int, default=50, help='Number of top features to plot')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-eval-samples', type=int, default=None, help='If set, randomly subsample this many eval rows per file type for faster metrics')
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    # decide paths for subset
    if args.subset == 'train':
        Xp = root / 'X_train.dat'
        yp = root / 'y_train.dat'
        meta = root / 'train_metadata.jsonl'
    elif args.subset == 'test':
        Xp = root / 'X_test.dat'
        yp = root / 'y_test.dat'
        meta = root / 'test_metadata.jsonl'
    else:
        Xp = root / 'X_challenge.dat'
        yp = root / 'y_challenge.dat'
        # challenge metadata file may be present with challenge in name; fallback to test metadata
        meta_candidates = list(root.glob('*challenge*.jsonl'))
        meta = meta_candidates[0] if meta_candidates else root / 'test_metadata.jsonl'

    assert Xp.exists() and yp.exists() and meta.exists(), f"Missing files for subset {args.subset}: {Xp}, {yp}, {meta}"

    n = os.path.getsize(str(Xp)) // (args.feature_dim * 4)
    X_mm = np.memmap(str(Xp), dtype=np.float32, mode='r', shape=(n, args.feature_dim))
    y_mm = np.memmap(str(yp), dtype=np.float32, mode='r', shape=(n,))

    field = detect_meta_field(str(meta))

    # if evaluating all types, build type list
    types = []
    for obj in iterate_metadata(str(meta)):
        v = obj.get(field) or obj.get(field.lower())
        types.append(v if v is not None else 'unknown')

    if args.all:
        types_to_eval = sorted(set(types))
        if args.file_type:
            types_to_eval = [t for t in types_to_eval if t == args.file_type]
    else:
        if not args.file_type:
            raise SystemExit('Specify --file-type or use --all')
        types_to_eval = [args.file_type]

    metrics_lines = []

    for ftype in types_to_eval:
        print(f"Processing file type: {ftype}")
        inds = [i for i, t in enumerate(types) if t == ftype]
        if not inds:
            print(f"  No samples for {ftype} in subset {args.subset}; skipping")
            continue
        # subsample eval indices if requested (for faster metric computation)
        if args.max_eval_samples and len(inds) > args.max_eval_samples:
            rng = np.random.RandomState(0)
            eval_inds = list(rng.choice(inds, size=args.max_eval_samples, replace=False))
        else:
            eval_inds = inds

        # smaller view for importance if requested
        if args.max_samples_for_importance and len(inds) > args.max_samples_for_importance:
            rng2 = np.random.RandomState(0)
            imp_inds = list(rng2.choice(inds, size=args.max_samples_for_importance, replace=False))
        else:
            imp_inds = inds

        ys = np.array(y_mm[eval_inds], dtype=np.int32)

        # RF ensemble evaluation
        # prefer an existing top-models list if present
        top_list_path = Path(args.models_dir) / f"top_models_{ftype}.txt"
        if top_list_path.exists():
            model_paths = []
            with open(top_list_path, 'r') as fh:
                for line in fh:
                    p = line.strip().split(',')[0]
                    if p:
                        model_paths.append(p)
        else:
            pattern = os.path.join(args.models_dir, f"rf_{ftype}_chunk*.joblib")
            model_paths = sorted(glob.glob(pattern))
        rf_probs = None
        rf_importances = None
        type_out = out / ftype.replace('/', '_')
        type_out.mkdir(parents=True, exist_ok=True)

        if model_paths:
            print(f"  Loading {len(model_paths)} RF chunk models")
            for mp in model_paths:
                rf = joblib.load(mp)
                # predict in batches over eval_inds
                preds = []
                for i in range(0, len(eval_inds), args.test_batch_size):
                    batch = eval_inds[i : i + args.test_batch_size]
                    Xb = np.array(X_mm[batch], dtype=np.float32)
                    p = rf.predict_proba(Xb)[:, 1]
                    preds.append(p)
                preds = np.concatenate(preds)
                rf_probs = preds if rf_probs is None else rf_probs + preds
                # accumulate feature importances if available
                if hasattr(rf, 'feature_importances_'):
                    imp = np.array(rf.feature_importances_, dtype=np.float32)
                    rf_importances = imp if rf_importances is None else rf_importances + imp
            rf_probs = rf_probs / float(len(model_paths))
            if rf_importances is not None:
                rf_importances = rf_importances / float(len(model_paths))

            rf_metrics = evaluate_predictions(ys, rf_probs)
            print(f"  RF ensemble metrics: AUC={rf_metrics['auc']:.4f} AP={rf_metrics['ap']:.4f} F1={rf_metrics['f1']:.4f}")

            # save plots for RF
            plot_roc(rf_metrics['fpr'], rf_metrics['tpr'], type_out / f"roc_RF_{args.subset}_{ftype}.png", title=f"RF ROC {ftype} {args.subset}")
            plot_pr(rf_metrics['pr_prec'], rf_metrics['pr_rec'], type_out / f"pr_RF_{args.subset}_{ftype}.png", title=f"RF PR {ftype} {args.subset}")
            if rf_importances is not None:
                plot_feature_importance(rf_importances, type_out / f"rf_feature_importance_{ftype}.png", args.feature_dim, top_k=args.top_k_features, title=f"RF feature importance {ftype}")

            metrics_lines.append((ftype, 'RF_ensemble', len(eval_inds), rf_metrics['auc'], rf_metrics['ap'], rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1']))
            # save JSON for RF
            import json as _json
            # Save curves as npy and build JSON-serializable summary
            roc_path = type_out / f"rf_roc_{args.subset}_{ftype}.npy"
            pr_path = type_out / f"rf_pr_{args.subset}_{ftype}.npy"
            np.save(str(roc_path), np.stack([rf_metrics['fpr'], rf_metrics['tpr']], axis=0))
            np.save(str(pr_path), np.stack([rf_metrics['pr_prec'], rf_metrics['pr_rec']], axis=0))
            rf_json = {
                'file_type': ftype,
                'subset': args.subset,
                'n_samples': int(len(eval_inds)),
                'model_paths': model_paths,
                'metrics': {
                    'auc': float(rf_metrics['auc']),
                    'ap': float(rf_metrics['ap']),
                    'pr_auc': float(rf_metrics['ap']),
                    'precision': float(rf_metrics['precision']),
                    'recall': float(rf_metrics['recall']),
                    'f1': float(rf_metrics['f1']),
                },
                'roc_curve_npy': str(roc_path),
                'pr_curve_npy': str(pr_path),
                'feature_importances': rf_importances.tolist() if rf_importances is not None else None,
            }
            with open(type_out / 'rf_metrics.json', 'w') as fh:
                _json.dump(rf_json, fh, indent=2)
        else:
            print(f"  No RF chunk models found for {ftype} (pattern: {pattern})")

        # MLP evaluation (if provided)
        if args.mlp_model:
            # assume user passed a path pattern with {type} substitution or exact path
            mlp_path = args.mlp_model.format(type=ftype)
            if not Path(mlp_path).exists():
                print(f"  MLP model not found at {mlp_path}; skipping MLP for {ftype}")
            else:
                print(f"  Loading MLP from {mlp_path}")
                wrapper = TorchMLPWrapper(mlp_path, in_dim=args.feature_dim, hidden_dim=args.mlp_hidden_dim, device=args.device)
                # predict probs on the (possibly subsampled) eval indices
                X_eval_view = np.memmap(str(Xp), dtype=np.float32, mode='r', shape=(n, args.feature_dim))[eval_inds]
                probs = wrapper.predict_proba(X_eval_view, batch_size=args.test_batch_size)
                mlp_metrics = evaluate_predictions(ys, probs)
                print(f"  MLP metrics: AUC={mlp_metrics['auc']:.4f} AP={mlp_metrics['ap']:.4f} F1={mlp_metrics['f1']:.4f}")
                plot_roc(mlp_metrics['fpr'], mlp_metrics['tpr'], type_out / f"roc_MLP_{args.subset}_{ftype}.png", title=f"MLP ROC {ftype} {args.subset}")
                plot_pr(mlp_metrics['pr_prec'], mlp_metrics['pr_rec'], type_out / f"pr_MLP_{args.subset}_{ftype}.png", title=f"MLP PR {ftype} {args.subset}")

                # permutation importance on sample imp_inds
                print(f"  Computing permutation importance for MLP on up to {len(imp_inds)} samples (this may take a while)")
                X_sample = np.array(X_mm[imp_inds], dtype=np.float32)
                y_sample = np.array(y_mm[imp_inds], dtype=np.int32)
                baseline_auc = roc_auc_score(y_sample, wrapper.predict_proba(X_sample)) if len(np.unique(y_sample))>1 else float('nan')
                importances = np.zeros(args.feature_dim, dtype=np.float32)
                # To save time, only compute for top_k features by variance
                feat_var = X_sample.var(axis=0)
                feat_idx = np.argsort(feat_var)[::-1][: min(args.top_k_features*5, args.feature_dim)]
                for fi in feat_idx:
                    Xm = X_sample.copy()
                    rng = np.random.RandomState(0)
                    rng.shuffle(Xm[:, fi])
                    auc_shuf = roc_auc_score(y_sample, wrapper.predict_proba(Xm)) if len(np.unique(y_sample))>1 else float('nan')
                    importances[fi] = baseline_auc - auc_shuf if not math.isnan(baseline_auc) and not math.isnan(auc_shuf) else 0.0

                # plot top-k permutation importances
                if importances.sum() > 0:
                    plot_feature_importance(importances, type_out / f"mlp_permutation_importance_{ftype}.png", args.feature_dim, top_k=args.top_k_features, title=f"MLP permutation importance {ftype}")
                else:
                    print("  Permutation importance produced zero importances (likely single-class labels or too few samples)")

                metrics_lines.append((ftype, 'MLP', len(eval_inds), mlp_metrics['auc'], mlp_metrics['ap'], mlp_metrics['precision'], mlp_metrics['recall'], mlp_metrics['f1']))
                # Save curves as npy and JSON summary for MLP
                m_roc_path = type_out / f"mlp_roc_{args.subset}_{ftype}.npy"
                m_pr_path = type_out / f"mlp_pr_{args.subset}_{ftype}.npy"
                np.save(str(m_roc_path), np.stack([mlp_metrics['fpr'], mlp_metrics['tpr']], axis=0))
                np.save(str(m_pr_path), np.stack([mlp_metrics['pr_prec'], mlp_metrics['pr_rec']], axis=0))
                import json as _json2
                mlp_json = {
                    'file_type': ftype,
                    'subset': args.subset,
                    'n_samples': int(len(eval_inds)),
                    'mlp_path': mlp_path,
                    'metrics': {
                        'auc': float(mlp_metrics['auc']),
                        'ap': float(mlp_metrics['ap']),
                        'pr_auc': float(mlp_metrics['ap']),
                        'precision': float(mlp_metrics['precision']),
                        'recall': float(mlp_metrics['recall']),
                        'f1': float(mlp_metrics['f1']),
                    },
                    'roc_curve_npy': str(m_roc_path),
                    'pr_curve_npy': str(m_pr_path),
                    'permutation_importances': importances.tolist(),
                }
                with open(type_out / 'mlp_metrics.json', 'w') as fh:
                    _json2.dump(mlp_json, fh, indent=2)

    # write metrics CSV
    import csv

    out_csv = out / 'eval_metrics.csv'
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file_type','model','n_samples','auc','ap','precision','recall','f1'])
        for r in metrics_lines:
            w.writerow(r)
    print(f"Wrote metrics to {out_csv}")


if __name__ == '__main__':
    main()
