#!/usr/bin/env python3
"""Select the top-K chunk RF models for a given file type by evaluating each
on the test set and choosing the best by AUC or AP, then report ensemble metrics.

Usage:
  python3 DAFCS/scripts/select_top_chunks.py --file-type WIN32 --k 20
"""
import argparse
import glob
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


def detect_meta_field(meta_path, sample_lines=200):
    keys = {}
    with open(meta_path, "r", encoding="utf8") as f:
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


def read_test_types(meta_test, field):
    types = []
    with open(meta_test, "r", encoding="utf8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                val = obj.get(field) or obj.get(field.lower())
                types.append(val if val is not None else "unknown")
            except Exception:
                types.append("unknown")
    return types


def eval_model_on_indices(model_path, X_test_mm, test_inds, batch_size=2000):
    rf = joblib.load(model_path)
    preds = []
    for start in range(0, len(test_inds), batch_size):
        batch = test_inds[start : start + batch_size]
        Xb = np.array(X_test_mm[batch], dtype=np.float32)
        p = rf.predict_proba(Xb)[:, 1]
        preds.append(p)
    preds = np.concatenate(preds)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="DAFCS/data/local/EMBER2024-corrected-full")
    parser.add_argument("--feature-dim", type=int, default=2568)
    parser.add_argument("--models-dir", default="checkpoints/rf_chunks")
    parser.add_argument("--file-type", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--metric", choices=("auc","ap"), default="auc")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="If set, randomly subsample this many test rows of the given file type for faster evaluation")
    parser.add_argument("--test-seed", type=int, default=0, help="RNG seed used when subsampling test rows")
    parser.add_argument("--out-list", default="checkpoints/rf_chunks/top_models_{file_type}.txt")
    args = parser.parse_args()

    root = Path(args.data_root)
    X_test_path = root / "X_test.dat"
    y_test_path = root / "y_test.dat"
    meta_test = root / "test_metadata.jsonl"

    assert X_test_path.exists() and y_test_path.exists() and meta_test.exists(), "Missing test files"

    field = detect_meta_field(str(meta_test))
    test_types = read_test_types(str(meta_test), field)

    test_inds = [i for i,t in enumerate(test_types) if t == args.file_type]
    if not test_inds:
        raise SystemExit(f"No test samples for {args.file_type}")

    # optionally subsample the test indices for faster evaluation
    if args.max_test_samples is not None and len(test_inds) > args.max_test_samples:
        rng = np.random.RandomState(args.test_seed)
        test_inds = list(rng.choice(test_inds, size=args.max_test_samples, replace=False))

    n_test = os.path.getsize(str(X_test_path)) // (args.feature_dim * 4)
    X_test_mm = np.memmap(str(X_test_path), dtype=np.float32, mode='r', shape=(n_test, args.feature_dim))
    y_test_mm = np.memmap(str(y_test_path), dtype=np.float32, mode='r', shape=(n_test,))

    pattern = os.path.join(args.models_dir, f"rf_{args.file_type}_chunk*.joblib")
    model_paths = sorted(glob.glob(pattern))
    if not model_paths:
        raise SystemExit(f"No models found for pattern {pattern}")

    scores = []
    preds_cache = {}
    print(f"Evaluating {len(model_paths)} models for {args.file_type} on {len(test_inds)} test samples")
    for mp in model_paths:
        preds = eval_model_on_indices(mp, X_test_mm, test_inds, batch_size=args.batch_size)
        ys = np.array(y_test_mm[test_inds], dtype=np.int32)
        # compute both ROC AUC and PR AUC (average precision)
        try:
            auc_val = float(roc_auc_score(ys, preds))
        except Exception:
            auc_val = float('-inf')
        try:
            pr_auc = float(average_precision_score(ys, preds))
        except Exception:
            pr_auc = float('-inf')

        # threshold metrics at 0.5
        try:
            labels = (preds > 0.5).astype(int)
            prec = float(precision_score(ys, labels, zero_division=0))
            rec = float(recall_score(ys, labels, zero_division=0))
            f1 = float(f1_score(ys, labels, zero_division=0))
        except Exception:
            prec = rec = f1 = 0.0

        score = auc_val if args.metric == 'auc' else pr_auc
        scores.append((mp, score))
        preds_cache[mp] = preds
        print(f"  {Path(mp).name}: {args.metric}={score:.6f} PR_AUC={pr_auc:.6f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    scores.sort(key=lambda x: x[1], reverse=True)
    topk = scores[:args.k]
    print(f"Selected top {len(topk)} models")

    # ensemble of top-k
    probs_accum = None
    for mp, s in topk:
        preds = preds_cache[mp]
        probs_accum = preds if probs_accum is None else probs_accum + preds
    probs = probs_accum / float(len(topk))
    ys = np.array(y_test_mm[test_inds], dtype=np.int32)
    auc = roc_auc_score(ys, probs)
    ap = average_precision_score(ys, probs)
    labels = (probs > 0.5).astype(int)
    acc = accuracy_score(ys, labels)
    prec = precision_score(ys, labels, zero_division=0)
    rec = recall_score(ys, labels, zero_division=0)
    f1 = f1_score(ys, labels, zero_division=0)
    print(f"Ensemble of top-{len(topk)}: AUC={auc:.4f} AP={ap:.4f} PR_AUC={ap:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} ACC={acc:.4f}")

    outp = str(args.out_list).format(file_type=args.file_type)
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        for mp, s in topk:
            f.write(f"{mp},{s}\n")
    print(f"Wrote top model list to {outp}")


if __name__ == '__main__':
    main()
