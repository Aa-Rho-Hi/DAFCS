#!/usr/bin/env python3
"""Convert NPZ shard files in a source folder into EMBER2024 .dat memmaps and metadata.

This script looks for NPZ files whose basenames contain keywords `train`, `test`,
or `challenge` and concatenates them into `X_train.dat`, `y_train.dat`, etc under
the destination folder (e.g. `DAFCS/data/local/EMBER2024-corrected-full`).

It also writes minimal `<split>_metadata.jsonl` files so the repo loaders can run.

Usage:
  python3 DAFCS/scripts/npz_to_ember_dat.py --src DAFCS/data/local/Project --out DAFCS/data/local/EMBER2024-corrected-full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np


def find_shards(src: Path, keyword: str) -> List[Path]:
    out = []
    for p in src.glob("*.npz"):
        if keyword in p.name.lower():
            out.append(p)
    # also consider patterns like ember_*_train_*.npz
    for p in src.rglob("*.npz"):
        if keyword in p.name.lower() and p not in out:
            out.append(p)
    return sorted(out)


def write_split(shards: List[Path], out_dir: Path, split: str):
    if not shards:
        print(f"No shards found for split={split}")
        return
    # If the destination path is a symlink to a non-existent mount (common when
    # linking to external disks), remove the symlink and create a real directory
    # inside the repo so we can materialize files here.
    if out_dir.exists() and out_dir.is_symlink():
        try:
            out_dir.unlink()
            print(f"Removed stale symlink: {out_dir}")
        except Exception as e:
            print(f"Warning: failed to remove symlink {out_dir}: {e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f"X_{split}.dat"
    y_path = out_dir / f"y_{split}.dat"
    meta_path = out_dir / f"{split}_metadata.jsonl"

    # remove existing outputs (user can re-run)
    if x_path.exists():
        x_path.unlink()
    if y_path.exists():
        y_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    with open(x_path, "ab") as fx, open(y_path, "ab") as fy, open(meta_path, "w") as fm:
        for shard in shards:
            print(f"Processing shard: {shard}")
            with np.load(shard, allow_pickle=False) as d:
                if "X" not in d.files:
                    print(f"  skipping shard (no X): {shard}")
                    continue
                X = d["X"].astype(np.float32)
                y = d["y"].astype(np.float32) if "y" in d.files else np.full((X.shape[0],), -1.0, dtype=np.float32)

                # write binary rows
                fx.write(X.tobytes())
                fy.write(y.astype(np.float32).tobytes())

                # minimal metadata per row
                file_type = ""
                name = shard.name.lower()
                for ft in ("win32", "win64", "dotnet", "apk", "elf", "pdf"):
                    if ft in name:
                        file_type = ft.upper()
                        break

                for lab in y:
                    meta = {
                        "sha256": "",
                        "label": float(lab),
                        "file_type": file_type,
                        "family": "",
                        "family_confidence": 0.0,
                        "week": 0,
                    }
                    fm.write(json.dumps(meta) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder with shard .npz files")
    parser.add_argument("--out", required=True, help="Output folder to write EMBER2024-style files")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out)  # do not resolve - we want to operate on the repo path (avoid following stale symlinks)
    if not src.exists():
        print(f"Source not found: {src}")
        raise SystemExit(2)

    # find shards for train/test/challenge
    train_shards = find_shards(src, "train")
    test_shards = find_shards(src, "test")
    challenge_shards = find_shards(src, "challenge")

    # Heuristic: some files use names like ember_apk_train_train_0.npz -> include 'train_train'
    if not train_shards:
        train_shards = [p for p in src.rglob("*.npz") if "train" in p.name.lower() and "test" not in p.name.lower()]

    write_split(train_shards, out, "train")
    write_split(test_shards, out, "test")
    write_split(challenge_shards, out, "challenge")

    print("Conversion finished. Run verify_full_dataset.py to validate the layout.")


if __name__ == "__main__":
    main()
