#!/usr/bin/env python3
"""Evaluate a chunked RandomForest ensemble for a given file type.

This script finds chunk models saved by `train_rf_chunked.py` (pattern
`rf_<file_type>_chunk*.joblib`), loads them, averages their predicted
probabilities on the test set for the specified file type, and prints AUC / AP /
accuracy. Optionally writes per-sample test probabilities to an output file.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="DAFCS/data/local/EMBER2024-corrected-full")
    parser.add_argument("--feature-dim", type=int, default=2568)
    parser.add_argument("--file-type", default=None, help="File type to evaluate (exact match); ignored if --all is used")
    parser.add_argument("--models-dir", default="checkpoints/rf_chunks")
    parser.add_argument("--test-batch-size", type=int, default=2000)
    parser.add_argument("--save-probs", default=None, help="Optional directory to save per-type .npy probs")
    parser.add_argument("--all", action="store_true", help="Evaluate all file types found in test metadata")
    parser.add_argument("--exclude-types", default=None, help="Comma-separated list of file types to exclude when --all is used")
    parser.add_argument("--x-test", default=None, help="Path to X test memmap (overrides X_test.dat in data-root)")
    parser.add_argument("--y-test", default=None, help="Path to y test memmap (overrides y_test.dat in data-root)")
    parser.add_argument("--meta-test", default=None, help="Path to test metadata jsonl (overrides test_metadata.jsonl in data-root)")
    args = parser.parse_args()

    root = Path(args.data_root)
    X_test_path = Path(args.x_test) if args.x_test else root / "X_test.dat"
    y_test_path = Path(args.y_test) if args.y_test else root / "y_test.dat"
    meta_test = Path(args.meta_test) if args.meta_test else root / "test_metadata.jsonl"

    assert X_test_path.exists() and y_test_path.exists() and meta_test.exists(), "Missing test files (check --x-test/--y-test/--meta-test or data-root)"

    field = detect_meta_field(str(meta_test))
    test_types = read_test_types(str(meta_test), field)

    # Determine file types to evaluate
    if args.all:
        types_to_eval = sorted(set(test_types))
        if args.exclude_types:
            excl = [s.strip() for s in args.exclude_types.split(',') if s.strip()]
            types_to_eval = [t for t in types_to_eval if t not in excl]
    else:
        if not args.file_type:
            raise SystemExit("Either --file-type or --all must be specified")
        types_to_eval = [args.file_type]

    n_test = os.path.getsize(str(X_test_path)) // (args.feature_dim * 4)
    X_test_mm = np.memmap(str(X_test_path), dtype=np.float32, mode="r", shape=(n_test, args.feature_dim))
    y_test_mm = np.memmap(str(y_test_path), dtype=np.float32, mode="r", shape=(n_test,))

    results = []
    for ftype in types_to_eval:
        pattern = os.path.join(args.models_dir, f"rf_{ftype}_chunk*.joblib")
        model_paths = sorted(glob.glob(pattern))
        if not model_paths:
            print(f"Skipping {ftype}: no models found for pattern {pattern}")
            continue
        print(f"Evaluating {ftype} with {len(model_paths)} chunk models")

        test_inds = [i for i, t in enumerate(test_types) if t == ftype]
        if not test_inds:
            print(f"  no test samples for {ftype}; skipping")
            continue

        probs_accum = None
        for mp in model_paths:
            rf = joblib.load(mp)
            preds = []
            for start in range(0, len(test_inds), args.test_batch_size):
                batch = test_inds[start : start + args.test_batch_size]
                Xb = np.array(X_test_mm[batch], dtype=np.float32)
                p = rf.predict_proba(Xb)[:, 1]
                preds.append(p)
            preds = np.concatenate(preds)
            probs_accum = preds if probs_accum is None else probs_accum + preds

        probs = probs_accum / float(len(model_paths))
        ys = np.array(y_test_mm[test_inds], dtype=np.int32)

        auc = roc_auc_score(ys, probs)
        ap = average_precision_score(ys, probs)

        # precision-recall curve AUC (explicit) and F1 metrics
        from sklearn.metrics import precision_recall_curve, auc as _auc, f1_score

        precision, recall, thresh = precision_recall_curve(ys, probs)
        pr_auc = _auc(recall, precision)
        # F1 at 0.5
        f1_at_05 = f1_score(ys, (probs > 0.5).astype(int))
        # max F1 across thresholds
        f1s = []
        for t in thresh:
            f1s.append(f1_score(ys, (probs >= t).astype(int)))
        max_f1 = max(f1s) if f1s else f1_at_05

        acc = accuracy_score(ys, (probs > 0.5).astype(int))

        print(f"{ftype}: n_test={len(test_inds)} AUC={auc:.4f} AP={ap:.4f} PR_AUC={pr_auc:.4f} ACC={acc:.4f} F1@0.5={f1_at_05:.4f} maxF1={max_f1:.4f}")

        if args.save_probs:
            outp = Path(args.save_probs) / f"probs_{ftype}.npy"
            outp.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(outp), probs)
            print(f"  Saved probabilities to {outp}")

        results.append((ftype, len(test_inds), auc, ap, pr_auc, acc, f1_at_05, max_f1))

    # summary
    if results:
        print("\nSummary:")
        print("type,n,AUC,AP,PR_AUC,ACC,F1@0.5,maxF1")
        for r in results:
            print(",".join([str(x) for x in r]))


if __name__ == "__main__":
    main()
