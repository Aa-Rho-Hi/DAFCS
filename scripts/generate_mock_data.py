"""
Generate a small synthetic EMBER2024-shaped dataset for testing the pipeline
WITHOUT needing the real ~30 GB dataset.

Creates:
  <output_dir>/
    X_train.dat          float32 (N_train, 2568)
    y_train.dat          float32 (N_train,)
    X_test.dat           float32 (N_test, 2568)
    y_test.dat           float32 (N_test,)
    X_challenge.dat      float32 (N_challenge, 2568)
    y_challenge.dat      float32 (N_challenge,)  -- all 1s (malicious)
    train_metadata.jsonl   one JSON record per line
    test_metadata.jsonl
    challenge_metadata.jsonl

Usage:
    python scripts/generate_mock_data.py --output_dir /tmp/ember2024_mock
    python scripts/generate_mock_data.py --output_dir /tmp/ember2024_mock --small
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_DIM   = 2568
N_FAMILIES    = 200          # We'll create 200 families in training data
SAMPLES_PER_FAMILY = 100     # So each family has 100 malicious samples
N_BENIGN_TRAIN  = 30_000     # Benign training samples
N_TEST          = 10_000
N_CHALLENGE     = 500        # Evasive malware
TOTAL_WEEKS     = 52
FILE_TYPES = ["Win32PE", "Win64PE", "DotNetPE", "APK", "ELF", "PDF"]

rng = np.random.default_rng(42)


def make_features(n: int, dim: int = FEATURE_DIM) -> np.ndarray:
    """Synthetic features: mostly zeros (like sparse EMBER features), some signal."""
    X = rng.random((n, dim), dtype=np.float32) * 0.1
    # Add a few large-magnitude features to simulate EMBER structure
    hot_cols = rng.integers(0, dim, size=(n, 50))
    for i in range(n):
        X[i, hot_cols[i]] = rng.random(50).astype(np.float32)
    return X


def make_malicious_features(n: int, family_id: int, dim: int = FEATURE_DIM) -> np.ndarray:
    """Features with a weak family-specific signal for testing clustering."""
    X = make_features(n, dim)
    # Add family-specific signal in a dedicated region
    family_offset = (family_id * 7) % (dim - 20)
    X[:, family_offset:family_offset + 10] += 0.5 + (family_id % 5) * 0.1
    return X


def make_record(
    sha256: str,
    label: int,
    file_type: str,
    family: str,
    family_confidence: float,
    week: int,
) -> dict:
    return {
        "sha256":             sha256,
        "label":              label,
        "file_type":          file_type,
        "family":             family,
        "family_confidence":  round(family_confidence, 3),
        "week":               week,
        "behavior":           [],
        "file_property":      [],
        "packer":             [],
        "exploit":            [],
        "group":              [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate(output_dir: str, small: bool = False) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if small:
        global N_FAMILIES, SAMPLES_PER_FAMILY, N_BENIGN_TRAIN, N_TEST, N_CHALLENGE
        N_FAMILIES = 50
        SAMPLES_PER_FAMILY = 20
        N_BENIGN_TRAIN = 5_000
        N_TEST = 2_000
        N_CHALLENGE = 100

    print(f"Generating mock EMBER2024 dataset in: {output_dir}")
    print(f"  {N_FAMILIES} families × {SAMPLES_PER_FAMILY} malicious samples")
    print(f"  {N_BENIGN_TRAIN} benign training samples")

    # ── Training set ──────────────────────────────────────────────────────────
    X_parts  = []
    y_parts  = []
    records  = []
    sha_base = 0

    # Malicious (with families)
    for fam_id in range(N_FAMILIES):
        family_name = f"family_{fam_id:04d}"
        n = SAMPLES_PER_FAMILY
        X_parts.append(make_malicious_features(n, fam_id))
        y_parts.append(np.ones(n, dtype=np.float32))

        for i in range(n):
            sha256 = f"{sha_base + i:064x}"
            week   = rng.integers(1, TOTAL_WEEKS - 4 + 1)  # keep last 4 weeks for val
            ft     = random.choice(FILE_TYPES[:3])  # mostly PE files
            conf   = float(rng.uniform(0.75, 1.0))
            records.append(make_record(sha256, 1, ft, family_name, conf, int(week)))
        sha_base += n

    # Benign
    X_parts.append(make_features(N_BENIGN_TRAIN))
    y_parts.append(np.zeros(N_BENIGN_TRAIN, dtype=np.float32))
    for i in range(N_BENIGN_TRAIN):
        sha256 = f"{sha_base + i:064x}"
        week   = rng.integers(1, TOTAL_WEEKS + 1)
        ft     = random.choice(FILE_TYPES)
        records.append(make_record(sha256, 0, ft, "", 0.0, int(week)))
    sha_base += N_BENIGN_TRAIN

    # A few families in the val period (weeks 49-52) to test val split
    for fam_id in range(min(10, N_FAMILIES)):
        family_name = f"family_{fam_id:04d}"
        n = 5
        X_parts.append(make_malicious_features(n, fam_id))
        y_parts.append(np.ones(n, dtype=np.float32))
        for i in range(n):
            sha256 = f"{sha_base + i:064x}"
            week   = rng.integers(49, TOTAL_WEEKS + 1)
            ft     = random.choice(FILE_TYPES[:3])
            conf   = float(rng.uniform(0.75, 1.0))
            records.append(make_record(sha256, 1, ft, family_name, conf, int(week)))
        sha_base += n

    # Shuffle (so train/val split is by week, not position)
    n_total = sum(len(x) for x in X_parts)
    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    # Don't shuffle — keep records aligned with X rows

    assert len(X_train) == len(records), f"{len(X_train)} != {len(records)}"

    # Write binary files
    X_train.tofile(output_dir / "X_train.dat")
    y_train.tofile(output_dir / "y_train.dat")
    print(f"  Written X_train.dat ({X_train.shape}) + y_train.dat")

    # Write JSONL metadata
    with open(output_dir / "train_metadata.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Written train_metadata.jsonl ({len(records):,} records)")

    # ── Test set ──────────────────────────────────────────────────────────────
    X_test_parts = []
    y_test_parts = []
    test_records = []
    sha_base_test = sha_base + 1_000_000

    # Test malicious (mix of known + unknown families)
    n_test_mal = N_TEST // 2
    for i in range(n_test_mal):
        fam_id = i % N_FAMILIES
        family_name = f"family_{fam_id:04d}"
        X_test_parts.append(make_malicious_features(1, fam_id))
        y_test_parts.append(np.array([1.0], dtype=np.float32))
        sha256 = f"{sha_base_test + i:064x}"
        ft = random.choice(FILE_TYPES[:3])
        conf = float(rng.uniform(0.75, 1.0))
        test_records.append(make_record(sha256, 1, ft, family_name, conf, week=53))

    # Test benign
    n_test_ben = N_TEST - n_test_mal
    X_test_parts.append(make_features(n_test_ben))
    y_test_parts.append(np.zeros(n_test_ben, dtype=np.float32))
    for i in range(n_test_ben):
        sha256 = f"{sha_base_test + n_test_mal + i:064x}"
        ft = random.choice(FILE_TYPES)
        test_records.append(make_record(sha256, 0, ft, "", 0.0, week=53))

    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    X_test.tofile(output_dir / "X_test.dat")
    y_test.tofile(output_dir / "y_test.dat")
    with open(output_dir / "test_metadata.jsonl", "w") as f:
        for rec in test_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Written X_test.dat ({X_test.shape}) + test_metadata.jsonl")

    # ── Challenge set ─────────────────────────────────────────────────────────
    # All malicious, from known families (evasive — harder detection)
    X_ch = []
    y_ch = np.ones(N_CHALLENGE, dtype=np.float32)
    challenge_records = []
    sha_base_ch = sha_base + 2_000_000

    for i in range(N_CHALLENGE):
        fam_id = i % N_FAMILIES
        family_name = f"family_{fam_id:04d}"
        # Add some noise to simulate obfuscation
        feat = make_malicious_features(1, fam_id)
        feat += rng.random(feat.shape, dtype=np.float32) * 0.3
        X_ch.append(feat)
        sha256 = f"{sha_base_ch + i:064x}"
        ft = random.choice(FILE_TYPES[:3])
        conf = float(rng.uniform(0.75, 1.0))
        challenge_records.append(make_record(sha256, 1, ft, family_name, conf, week=54))

    X_ch = np.concatenate(X_ch, axis=0)
    X_ch.tofile(output_dir / "X_challenge.dat")
    y_ch.tofile(output_dir / "y_challenge.dat")
    with open(output_dir / "challenge_metadata.jsonl", "w") as f:
        for rec in challenge_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Written X_challenge.dat ({X_ch.shape})")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "n_train":         int(len(X_train)),
        "n_test":          int(len(X_test)),
        "n_challenge":     int(len(X_ch)),
        "n_families":      N_FAMILIES,
        "feature_dim":     FEATURE_DIM,
        "note":            "Synthetic mock data — not real EMBER2024",
    }
    with open(output_dir / "mock_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Set EMBER2024_DIR={output_dir} to use this data.")
    print(f"Or pass --data_dir {output_dir} to any training script.")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic EMBER2024-shaped data")
    parser.add_argument(
        "--output_dir", default="/tmp/ember2024_mock",
        help="Directory to write mock data into"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Generate a very small dataset (fast, for unit testing)"
    )
    args = parser.parse_args()
    generate(args.output_dir, small=args.small)
