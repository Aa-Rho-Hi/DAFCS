#!/usr/bin/env python3
"""Train a small streaming MLP per file type using memmapped features.

This script streams batches from `X_train.dat` and `y_train.dat` and uses
`train_metadata.jsonl` to group samples by file type. For each file type with
enough labeled samples it trains a lightweight PyTorch MLP using mini-batches
and saves the model and OOF predictions.

Designed to run on limited RAM by only loading batch-sized slices into memory.
"""
import argparse
import json
import os
import random
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


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
    # prefer common names
    for candidate in ("file_type", "filetype", "type", "mime", "mime_type", "file_type_guess"):
        if candidate in keys:
            return candidate
    # otherwise pick the most frequent key seen
    if keys:
        return max(keys.items(), key=lambda x: x[1])[0]
    raise RuntimeError("Could not detect a metadata field for file type")


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


def infer_n_samples(x_path, feature_dim):
    return os.path.getsize(x_path) // (feature_dim * 4)


def iterate_metadata(meta_path):
    with open(meta_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                yield {}


def list_file_types(meta_path, field_name):
    counts = {}
    for obj in iterate_metadata(meta_path):
        val = obj.get(field_name) or obj.get(field_name.lower())
        if val is None:
            val = "unknown"
        counts[val] = counts.get(val, 0) + 1
    return counts


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


def train_for_type(X_mm, y_mm, indices, args, out_dir, dtype=np.float32):
    if len(indices) < 32:
        print(f"Skipping type (too few labeled samples): {len(indices)}")
        return

    # split train/val — try to stratify when possible
    n = len(indices)
    labels = np.array([int(y_mm[i]) for i in indices], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    train_idx = None
    val_idx = None
    if unique.size >= 2 and np.all(counts >= 2):
        try:
            train_idx, val_idx = train_test_split(
                indices, test_size=args.val_frac, stratify=labels, random_state=42
            )
            print(f"Using stratified split: pos/neg counts in val -> {np.bincount(np.array([int(y_mm[i]) for i in val_idx]))}")
        except Exception as e:
            print(f"Stratified split failed ({e}), falling back to simple split")

    if train_idx is None:
        n_val = max(1, int(n * args.val_frac))
        train_idx = indices[: n - n_val]
        val_idx = indices[n - n_val :]

    device = torch.device(args.device)

    model = MLP(args.feature_dim, hidden_dim=args.hidden_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # OOF preds for indices (store in memmap sized n, we'll save aligned to indices)
    oof_preds = np.zeros(n, dtype=np.float32)

    batch_size = args.batch_size

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        model.train()
        for start in range(0, len(train_idx), batch_size):
            batch_inds = train_idx[start : start + batch_size]
            X_batch = np.array(X_mm[batch_inds], dtype=np.float32)
            y_batch = np.array(y_mm[batch_inds], dtype=np.float32)

            X_t = torch.from_numpy(X_batch).to(device)
            y_t = torch.from_numpy(y_batch).to(device)

            opt.zero_grad()
            logits = model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

        # compute val preds
        model.eval()
        all_val_logits = []
        all_val_y = []
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                batch_inds = val_idx[start : start + batch_size]
                X_batch = np.array(X_mm[batch_inds], dtype=np.float32)
                y_batch = np.array(y_mm[batch_inds], dtype=np.float32)
                X_t = torch.from_numpy(X_batch).to(device)
                logits = model(X_t).cpu().numpy()
                all_val_logits.append(logits)
                all_val_y.append(y_batch)

        preds = np.concatenate(all_val_logits) if len(all_val_logits) > 0 else np.array([])
        yval = np.concatenate(all_val_y) if len(all_val_y) > 0 else np.array([])

        # Handle degenerate validation sets (only one class present)
        if yval.size == 0:
            auc = float("nan")
            ap = float("nan")
            print(f"Epoch {epoch+1}/{args.epochs} no validation samples available")
        else:
            unique = np.unique(yval)
            pos = int((yval == 1).sum())
            neg = int((yval == 0).sum())
            if unique.size < 2:
                # can't compute AUC/AP with single-class labels; report class counts and accuracy
                auc = float("nan")
                ap = float("nan")
                # compute accuracy from logits thresholded at 0
                acc = float(((preds > 0).astype(int) == yval).mean()) if preds.size else float("nan")
                print(
                    f"Epoch {epoch+1}/{args.epochs} val single-class (pos={pos} neg={neg}) — accuracy={acc:.4f}; AUC/AP=nan"
                )
            else:
                # convert logits to probabilities for metrics
                probs = 1.0 / (1.0 + np.exp(-preds))
                try:
                    auc = roc_auc_score(yval, probs)
                except Exception:
                    auc = float("nan")
                try:
                    ap = average_precision_score(yval, probs)
                except Exception:
                    ap = float("nan")
                acc = float(((probs > 0.5).astype(int) == yval).mean())
                print(
                    f"Epoch {epoch+1}/{args.epochs} val AUC={auc:.4f} AP={ap:.4f} ACC={acc:.4f} (pos={pos} neg={neg})"
                )

        # Track best model by preferred metric: AP (if available) > AUC > accuracy
        try:
            metric_value = float(ap) if not math.isnan(ap) else (float(auc) if not math.isnan(auc) else (float(acc) if 'acc' in locals() and not math.isnan(acc) else float('nan')))
        except Exception:
            metric_value = float('nan')
        if 'best_metric' not in locals():
            best_metric = -float('inf')
            best_state = None
        if not math.isnan(metric_value) and metric_value > best_metric:
            best_metric = metric_value
            # store a CPU-copy of the state dict to avoid device issues later
            best_state = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in model.state_dict().items()}
            print(f"  New best model at epoch {epoch+1} with metric={best_metric:.6f}")

    # if we found a best state during training, load it before producing OOF preds
    if 'best_state' in locals() and best_state is not None:
        try:
            model.load_state_dict(best_state)
            print(f"Loaded best model with metric={best_metric:.6f} for OOF prediction")
        except Exception:
            # best_state might be in CPU tensors and model on device; try converting
            try:
                sd = {k: v.to(device) if hasattr(v, 'to') else v for k, v in best_state.items()}
                model.load_state_dict(sd)
                print(f"Loaded best model (converted) with metric={best_metric:.6f} for OOF prediction")
            except Exception:
                print("Warning: failed to load best_state into model; using last-epoch weights for OOF preds")

    # produce OOF preds for all indices (predict in batches)
    model.eval()
    with torch.no_grad():
        for i_start in range(0, n, batch_size):
            batch_inds = indices[i_start : i_start + batch_size]
            X_batch = np.array(X_mm[batch_inds], dtype=np.float32)
            X_t = torch.from_numpy(X_batch).to(device)
            logits = model(X_t).cpu().numpy()
            oof_preds[i_start : i_start + len(batch_inds)] = logits

    # Save model and preds
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"mlp_{args.file_type}.pt"
    # prefer saving the best state if available
    if 'best_state' in locals() and best_state is not None:
        try:
            torch.save(best_state, model_path)
        except Exception:
            # fallback to saving current model
            torch.save(model.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    preds_path = out_dir / f"oof_preds_{args.file_type}.npy"
    np.save(preds_path, oof_preds)
    print(f"Saved model -> {model_path}; preds -> {preds_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="DAFCS/data/local/EMBER2024-corrected-full")
    parser.add_argument("--feature-dim", type=int, default=2568)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out-dir", default="checkpoints/mlp_per_type")
    parser.add_argument(
        "--only-types",
        default=None,
        help="Comma-separated list of file types to train (exact match). If omitted, all types are processed.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    X_path = root / "X_train.dat"
    y_path = root / "y_train.dat"
    meta_path = root / "train_metadata.jsonl"

    if not X_path.exists() or not y_path.exists() or not meta_path.exists():
        raise SystemExit("Expected X_train.dat, y_train.dat and train_metadata.jsonl in data-root")

    n_samples = infer_n_samples(str(X_path), args.feature_dim)
    print(f"Detected n_samples={n_samples}")

    X_mm = np.memmap(str(X_path), dtype=np.float32, mode="r", shape=(n_samples, args.feature_dim))
    # Labels are stored as float32 (0.0/1.0) in this dataset — map as float32 and cast when needed
    try:
        y_mm = np.memmap(str(y_path), dtype=np.float32, mode="r", shape=(n_samples,))
    except Exception:
        # fallback to int32 if float32 memmap fails
        y_mm = np.memmap(str(y_path), dtype=np.int32, mode="r", shape=(n_samples,)).astype(np.float32)

    # detect file type field
    field = detect_meta_field(str(meta_path))
    print(f"Using metadata field '{field}' for file types")

    counts = list_file_types(str(meta_path), field)
    print("Found file types and counts (sample):")
    for k, v in list(counts.items())[:20]:
        print(f"  {k}: {v}")

    out_dir = Path(args.out_dir)

    # iterate file types and train per-type
    only = None
    if args.only_types:
        only = [s.strip() for s in args.only_types.split(",") if s.strip()]
        print(f"Restricting to only types: {only}")

    for file_type in sorted(counts.keys()):
        if only is not None and file_type not in only:
            continue
        print(f"\nTraining for file type: {file_type} (scanning metadata to collect indices)")
        indices = collect_indices_for_type(str(meta_path), field, file_type, y_mm)
        if not indices:
            print("  no labeled samples for this type; skipping")
            continue
        # convert indices to int list and sort
        indices = list(indices)
        indices.sort()

        # store file_type for the train function to know name
        args.file_type = file_type.replace("/", "_")
        train_for_type(X_mm, y_mm, indices, args, out_dir)


if __name__ == "__main__":
    main()
