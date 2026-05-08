#!/usr/bin/env python3
"""Create EMBER2024-* dataset folders under data/local by linking files found
in a user-provided Project folder (e.g. Google Drive 'Project').

This script is conservative: it will only create symlinks for files it can find
in the source folder. After running it, you should run
`python3 DAFCS/scripts/verify_full_dataset.py --data-root DAFCS/data/local` to
see which items are still missing.

Usage:
  python3 DAFCS/scripts/materialize_dataset_from_project.py --src DAFCS/data/local/Project --dest DAFCS/data/local
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Dict, List


DATASET_RULES: Dict[str, Dict] = {
    "EMBER2024-corrected-canonical": {
        "required_files": [
            "README.md",
            "MANIFEST.json",
            "official_detection_model_results.json",
            "2024-09-22_2024-12-14_test.jsonl",
            "2023-09-24_2024-12-14_challenge_malicious.jsonl",
            "X_test.dat",
            "y_test.dat",
            "X_challenge.dat",
            "y_challenge.dat",
        ],
    },
    "EMBER2024-corrected-full": {
        "required_files": [
            "MANIFEST_PRE_VECTORIZE.json",
            "2023-09-24_2024-09-21_train.jsonl",
            "2024-09-22_2024-12-14_test.jsonl",
            "2023-09-24_2024-12-14_challenge_malicious.jsonl",
            "X_train.dat",
            "y_train.dat",
            "X_test.dat",
            "y_test.dat",
            "X_challenge.dat",
            "y_challenge.dat",
        ],
    },
    "EMBER2024-artifacts": {
        "required_files": [
            "X_test.dat",
            "y_test.dat",
            "X_challenge.dat",
            "y_challenge.dat",
        ],
        "glob_counts": {
            "*_test.jsonl": 72,
            "*_challenge_malicious.jsonl": 64,
            "*.model": 14,
        },
    },
    "EMBER2024-evaldata": {
        "required_files": [
            "2024-09-22_2024-12-14_test.jsonl",
            "2023-09-24_2024-12-14_challenge_malicious.jsonl",
            "X_test.dat",
            "y_test.dat",
            "X_challenge.dat",
            "y_challenge.dat",
        ],
    },
    "EMBER2024-full-local": {
        "required_files": [
            "2024-09-22_2024-12-14_test.jsonl",
            "2023-09-24_2024-12-14_challenge_malicious.jsonl",
            "X_test.dat",
            "y_test.dat",
            "X_challenge.dat",
            "y_challenge.dat",
        ],
        "glob_counts": {"*_train.jsonl": 312},
    },
    "EMBER2024-train-artifacts": {
        "glob_counts": {"*_train.jsonl": 312},
        "required_files": [],
    },
}


def find_files(root: Path, name: str) -> List[Path]:
    """Find files under root whose basename equals name or matches relaxed patterns."""
    matches = []
    # exact basename first
    for p in root.rglob(name):
        if p.is_file():
            matches.append(p)
    if matches:
        return matches

    # relaxed: look for files where basename contains key parts
    key = name.split(".")[0]
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        b = p.name.lower()
        if key.lower() in b:
            matches.append(p)
    return matches


def find_glob(root: Path, pattern: str) -> List[Path]:
    # walk and fnmatch against basename
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if fnmatch.fnmatch(p.name, pattern):
            out.append(p)
    return out


def symlink_into(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(str(src), str(dest))
        print(f"linked: {dest} -> {src}")
    except Exception as e:
        print(f"failed to link {src} -> {dest}: {e}")


def materialize(src_root: Path, dest_root: Path) -> None:
    print(f"Materializing datasets from {src_root} -> {dest_root}")
    for ds_name, rules in DATASET_RULES.items():
        target_dir = dest_root / ds_name
        # if a symlink or file exists at the target path, keep it (we won't overwrite)
        if target_dir.exists() and not target_dir.is_dir():
            print(f"Note: target path exists and is not a dir (skipping mkdir): {target_dir}")
        else:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                # race or symlink edge-case: continue
                print(f"Note: could not mkdir (exists): {target_dir}")
        print(f"\nProcessing dataset: {ds_name} -> {target_dir}")

        # required files
        for rel in rules.get("required_files", []):
            found = find_files(src_root, rel)
            if found:
                # pick the first match
                src = found[0]
                symlink_into(src, target_dir / rel)
            else:
                print(f"  NOT FOUND: {rel}")

        # glob counts: bring matching files
        for pattern, expected in rules.get("glob_counts", {}).items():
            matches = find_glob(src_root, pattern)
            if not matches:
                print(f"  NO MATCHES for pattern {pattern}")
                continue
            # link up to expected number (or all if fewer)
            for i, m in enumerate(matches[: expected]):
                dest = target_dir / m.name
                symlink_into(m, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source Project folder to scan")
    parser.add_argument("--dest", default="DAFCS/data/local", help="Destination data/local root")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dest_root = Path(args.dest).resolve()
    if not src_root.exists():
        print(f"ERROR: source root not found: {src_root}")
        raise SystemExit(2)

    materialize(src_root, dest_root)

    print("\nMaterialization complete. Run verify_full_dataset.py to check remaining gaps.")


if __name__ == "__main__":
    main()
