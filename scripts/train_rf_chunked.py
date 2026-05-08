#!/usr/bin/env python3
"""Train chunked RandomForest ensemble per file type.

For each requested file type, this script splits the labeled training indices into
chunks and trains a small RandomForest on each chunk. Models are saved to
`checkpoints/rf_chunks/<file_type>_chunkNN.joblib`. After training, the script
averages chunk-model probabilities on the test set for a quick evaluation.

Designed to be memory-friendly: each chunk is loaded into RAM separately.
"""
import argparse
import json
import os
from pathlib import Path
import math

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def iterate_metadata(meta_path):
    with open(meta_path, "r", encoding="utf8") as f:
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
            val = "unknown"
        if val == desired_type:
            if y_mm[i] != -1:
                indices.append(i)
    return indices


def train_chunk_models(X_mm, y_mm, indices, args, out_dir, file_type):
    os.makedirs(out_dir, exist_ok=True)
    n = len(indices)
    nchunks = math.ceil(n / args.chunk_size)
    models = []
    for ci in range(nchunks):
        start = ci * args.chunk_size
        end = min(n, (ci + 1) * args.chunk_size)
        chunk_inds = indices[start:end]
        if not chunk_inds:
            continue
        X_chunk = np.array(X_mm[chunk_inds], dtype=np.float32)
        y_chunk = np.array(y_mm[chunk_inds], dtype=np.int32)
        # filter any remaining unlabeled
        mask = (y_chunk == 0) | (y_chunk == 1)
        X_chunk = X_chunk[mask]
        y_chunk = y_chunk[mask]
        if X_chunk.shape[0] == 0:
            print(f"Chunk {ci} empty after filtering; skipping")
            continue
        unique = np.unique(y_chunk)
        if unique.size < 2:
            print(f"Chunk {ci} for {file_type} has single-class labels {unique}; skipping")
            continue

        print(f"Training RF chunk {ci+1}/{nchunks} for {file_type}: rows={X_chunk.shape[0]}")
        rf = RandomForestClassifier(
            n_estimators=args.trees_per_chunk,
            n_jobs=args.n_jobs,
            class_weight='balanced',
            random_state=42,
        )
        rf.fit(X_chunk, y_chunk)
        model_path = Path(out_dir) / f"rf_{file_type}_chunk{ci:03d}.joblib"
        joblib.dump(rf, model_path)
        print(f"Saved chunk model -> {model_path}")
        models.append(model_path)

    return models


def eval_ensemble(models, X_test_mm, y_test_mm, test_indices, args):
    # average probabilities from chunk models
    probs_accum = None
    n_models = 0
    for model_path in models:
        rf = joblib.load(model_path)
        # predict in batches
        preds = []
        for start in range(0, len(test_indices), args.test_batch_size):
            batch = test_indices[start : start + args.test_batch_size]
            Xb = np.array(X_test_mm[batch], dtype=np.float32)
            p = rf.predict_proba(Xb)[:, 1]
            preds.append(p)
        preds = np.concatenate(preds)
        if probs_accum is None:
            probs_accum = preds
        else:
            probs_accum += preds
        n_models += 1

    if n_models == 0:
        raise SystemExit("No models to evaluate")
    probs = probs_accum / float(n_models)
    ys = np.array(y_test_mm[test_indices], dtype=np.int32)
    auc = roc_auc_score(ys, probs)
    ap = average_precision_score(ys, probs)
    acc = accuracy_score(ys, (probs > 0.5).astype(int))
    return auc, ap, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="DAFCS/data/local/EMBER2024-corrected-full")
    parser.add_argument("--feature-dim", type=int, default=2568)
    parser.add_argument("--only-types", default=None, help="Comma-separated list of file types to process")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Rows per chunk (per type)")
    parser.add_argument("--trees-per-chunk", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--out-dir", default="checkpoints/rf_chunks")
    parser.add_argument("--test-batch-size", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.data_root)
    X_path = root / "X_train.dat"
    y_path = root / "y_train.dat"
    meta_path = root / "train_metadata.jsonl"

    X_test_path = root / "X_test.dat"
    y_test_path = root / "y_test.dat"
    meta_test = root / "test_metadata.jsonl"

    assert X_path.exists() and y_path.exists() and meta_path.exists(), "Missing train files"
    assert X_test_path.exists() and y_test_path.exists() and meta_test.exists(), "Missing test files"

    n_samples = os.path.getsize(str(X_path)) // (args.feature_dim * 4)
    X_mm = np.memmap(str(X_path), dtype=np.float32, mode="r", shape=(n_samples, args.feature_dim))
    y_mm = np.memmap(str(y_path), dtype=np.float32, mode="r", shape=(n_samples,))

    # detect file type
    field = detect_meta_field(str(meta_path))
    print(f"Using metadata field '{field}' for file types")

    # read test metadata to collect test indices per type
    test_types = []
    with open(meta_test, 'r', encoding='utf8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                val = obj.get(field) or obj.get(field.lower())
                test_types.append(val if val is not None else 'unknown')
            except Exception:
                test_types.append('unknown')

    only = None
    if args.only_types:
        only = [s.strip() for s in args.only_types.split(',') if s.strip()]

    # iterate requested types
    # To limit memory, we collect indices by scanning metadata once per type
    for file_type in (only if only is not None else sorted(set(test_types))):
        print(f"\nProcessing file type: {file_type}")
        train_indices = collect_indices_for_type(str(meta_path), field, file_type, y_mm)
        if not train_indices:
            print("  no labeled training samples for this type; skipping")
            continue
        print(f"  labeled train samples = {len(train_indices)}")

        models = train_chunk_models(X_mm, y_mm, train_indices, args, args.out_dir, file_type)

        # evaluate ensemble on test indices for this type
        test_inds = [i for i,v in enumerate(test_types) if v == file_type]
        if not test_inds:
            print("  no test samples for this type; skipping eval")
            continue
        auc, ap, acc = eval_ensemble(models, np.memmap(str(X_test_path), dtype=np.float32, mode='r', shape=(os.path.getsize(str(X_test_path))//(args.feature_dim*4), args.feature_dim)), np.memmap(str(y_test_path), dtype=np.float32, mode='r', shape=(os.path.getsize(str(y_test_path))//4,)), test_inds, args)
        print(f"Evaluation for {file_type}: AUC={auc:.4f} AP={ap:.4f} ACC={acc:.4f}")


if __name__ == '__main__':
    main()
