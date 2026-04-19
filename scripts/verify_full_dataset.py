#!/usr/bin/env python3
"""Verify that the complete EMBER2024 dataset workspace is present under data/local."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASET_RULES = {
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
            "EMBER2024_APK.model",
            "EMBER2024_Dot_Net.model",
            "EMBER2024_ELF.model",
            "EMBER2024_PDF.model",
            "EMBER2024_PE.model",
            "EMBER2024_Win32.model",
            "EMBER2024_Win64.model",
            "EMBER2024_all.model",
            "EMBER2024_behavior.model",
            "EMBER2024_exploit.model",
            "EMBER2024_file_property.model",
            "EMBER2024_group.model",
            "EMBER2024_packer.model",
            "EMBER2024_family.model",
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
        "glob_counts": {
            "*_train.jsonl": 312,
        },
    },
    "EMBER2024-train-artifacts": {
        "glob_counts": {
            "*_train.jsonl": 312,
        },
        "required_files": [],
    },
}


def check_dataset(root: Path, dataset_name: str, rules: dict) -> list[str]:
    errors: list[str] = []
    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        return [f"{dataset_name}: missing directory"]

    for rel_path in rules.get("required_files", []):
        path = dataset_dir / rel_path
        if not path.is_file():
            errors.append(f"{dataset_name}: missing file {rel_path}")

    for pattern, expected_count in rules.get("glob_counts", {}).items():
        actual_count = len(list(dataset_dir.glob(pattern)))
        if actual_count != expected_count:
            errors.append(
                f"{dataset_name}: expected {expected_count} files for pattern '{pattern}', found {actual_count}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify full EMBER2024 local dataset layout.")
    parser.add_argument(
        "--data-root",
        default="data/local",
        help="Root containing EMBER2024-* directories (default: data/local).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}")
        return 1

    print(f"Verifying full dataset under: {data_root}")
    errors: list[str] = []

    for dataset_name, rules in DATASET_RULES.items():
        dataset_errors = check_dataset(data_root, dataset_name, rules)
        if dataset_errors:
            errors.extend(dataset_errors)
        else:
            print(f"OK  {dataset_name}")

    if errors:
        print("")
        print("Verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("")
    print("Full EMBER2024 dataset verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
