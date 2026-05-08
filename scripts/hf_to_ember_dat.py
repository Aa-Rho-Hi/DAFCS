#!/usr/bin/env python3
"""Convert a Hugging Face dataset into EMBER2024-style .dat files and metadata.

Usage:
  python DAFCS/scripts/hf_to_ember_dat.py --hf_id joyce8/EMBER2024-capa --out_dir DAFCS/data/local/EMBER2024-capa-converted

The converter streams features and labels to disk to avoid loading the entire
dataset into RAM. It attempts to auto-detect the feature field and a label
field; if the dataset schema differs, pass explicit field names with
`--feat_field` and `--label_field`.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset


def detect_feature_field(example: dict) -> Optional[str]:
    # common names: 'features', 'vector', 'ember_vector', 'x', 'feat'
    candidates = [
        "features",
        "vector",
        "ember_vector",
        "x",
        "feat",
        "features_float",
    ]
    for c in candidates:
        if c in example:
            return c
    # fallback: pick any list/ndarray-like field with length 2568
    for k, v in example.items():
        if isinstance(v, (list, tuple)) and len(v) in (2568,):
            return k
        try:
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.size in (2568,):
                return k
        except Exception:
            continue
    return None


def detect_label_field(example: dict) -> Optional[str]:
    for c in ("label", "labels", "malicious", "y", "target"):
        if c in example:
            return c
    return None


def stream_split_to_disk(dset, split_name: str, out_dir: Path, feat_field: str, label_field: Optional[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f"X_{split_name}.dat"
    y_path = out_dir / f"y_{split_name}.dat"
    meta_path = out_dir / f"{split_name}_metadata.jsonl"

    # remove existing files if any
    if x_path.exists():
        x_path.unlink()
    if y_path.exists():
        y_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    # open files in append-binary mode
    with open(x_path, "ab") as fx, open(y_path, "ab") as fy, open(meta_path, "w") as fm:
        for ex in dset:
            feat = ex.get(feat_field)
            if feat is None:
                raise RuntimeError(f"Feature field '{feat_field}' missing in example: {ex.keys()}")
            arr = np.asarray(feat, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.ravel()
            if arr.size == 0:
                raise RuntimeError("Empty feature vector encountered")
            # write binary row
            fx.write(arr.tobytes())

            # label
            label = -1.0
            if label_field and label_field in ex:
                try:
                    label = float(ex[label_field])
                except Exception:
                    label = float(int(bool(ex[label_field])))
            fy.write(np.array([label], dtype=np.float32).tobytes())

            # metadata
            meta = {
                "sha256": ex.get("sha256", ""),
                "label": ex.get(label_field, -1) if label_field else -1,
                "file_type": ex.get("file_type", ""),
                "family": ex.get("family", ""),
                "family_confidence": ex.get("family_confidence", 0.0),
                "week": ex.get("week", 0),
            }
            fm.write(json.dumps(meta) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_id", required=True, help="Hugging Face dataset id, e.g. joyce8/EMBER2024-capa")
    parser.add_argument("--out_dir", default="DAFCS/data/local/EMBER2024-capa-converted", help="Output directory")
    parser.add_argument("--feat_field", default=None, help="Feature field name (auto-detected if omitted)")
    parser.add_argument("--label_field", default=None, help="Label field name (auto-detected if omitted)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Loading HF dataset: {args.hf_id}")
    ds = load_dataset(args.hf_id)

    # ds can be a DatasetDict or a Dataset
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
    else:
        splits = ["train"]
        ds = {"train": ds}

    # detect fields using the first available example from the first split
    sample = None
    for s in splits:
        if len(ds[s]) > 0:
            sample = ds[s][0]
            break
    if sample is None:
        raise RuntimeError("Dataset appears empty")

    feat_field = args.feat_field or detect_feature_field(sample)
    label_field = args.label_field or detect_label_field(sample)

    if feat_field is None:
        raise RuntimeError("Could not auto-detect feature field. Please pass --feat_field")

    print(f"Detected feature field: {feat_field}; label field: {label_field}")

    # For each split stream to disk
    for split in splits:
        print(f"Converting split: {split}")
        stream_split_to_disk(ds[split], split, out_dir, feat_field, label_field)
        print(f"Wrote: {out_dir}/X_{split}.dat, y_{split}.dat, {split}_metadata.jsonl")

    # Write a small summary
    summary = {"splits": splits}
    with open(out_dir / "mock_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Conversion complete.")


if __name__ == "__main__":
    main()
