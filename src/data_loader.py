"""
EMBER2024 data loading utilities.

Supports:
  - Memory-mapped feature arrays (handles 27 GB+ training set without OOM)
  - Family metadata streaming from JSONL (line-by-line, no full-file RAM load)
  - Week-based validation split (last N weeks of training period)
  - Prototypical batch sampler (N classes × K samples per batch)
  - Challenge set loading
"""

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger("ember2024")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_DIM = 2568      # Full EMBER v3 PE feature vector
NON_PE_DIM  = 696       # Non-PE feature vector (truncate to this)

LABEL_BENIGN     = 0
LABEL_MALICIOUS  = 1
LABEL_UNLABELED  = -1

FILE_TYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level array loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_array_direct(
    data_dir: Path,
    prefix: str,          # "X_train", "y_train", etc.
    dtype: np.dtype,
    shape: Tuple,
    use_memmap: bool,
) -> np.ndarray:
    path = data_dir / f"{prefix}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run scripts/generate_mock_data.py to create synthetic data for testing."
        )
    if use_memmap:
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)
    else:
        arr = np.fromfile(path, dtype=dtype)
        return arr.reshape(shape)


def load_ember_arrays(
    data_dir: str,
    subset: str = "train",
    use_memmap: bool = True,
    feature_dim: int = FEATURE_DIM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load feature matrix X and label vector y for a given subset.

    Tries the `ember` package first; falls back to direct binary file reading.

    Args:
        data_dir:    Path to EMBER2024 data directory.
        subset:      "train", "test", or "challenge".
        use_memmap:  Memory-map the arrays (recommended; avoids OOM on large data).
        feature_dim: Expected feature dimension (used for shape inference).

    Returns:
        X: float32 array, shape (N, feature_dim)
        y: int32 array, shape (N,)
    """
    data_dir = Path(data_dir)

    # ── Try ember package (wraps the same .dat files) ──────────────────────
    if subset in ("train", "test"):
        try:
            import ember as ember_pkg
            logger.info("Loading features via ember package …")
            X_tr, y_tr, X_te, y_te = ember_pkg.read_vectorized_features(
                str(data_dir), feature_version=3
            )
            if subset == "train":
                return X_tr, y_tr
            return X_te, y_te
        except Exception as e:
            logger.warning(f"ember package unavailable ({e}); using direct binary loading.")

    # ── Direct binary loading ───────────────────────────────────────────────
    X_path = data_dir / f"X_{subset}.dat"
    y_path = data_dir / f"y_{subset}.dat"

    n_samples = os.path.getsize(X_path) // (feature_dim * 4)  # float32 = 4 bytes
    logger.info(f"  {subset}: {n_samples:,} samples × {feature_dim} features")

    X = _load_array_direct(data_dir, f"X_{subset}", np.float32, (n_samples, feature_dim), use_memmap)
    y = _load_array_direct(data_dir, f"y_{subset}", np.int32, (n_samples,), use_memmap)

    # Backward-compatible path for older local/mock datasets that wrote labels
    # as float32 instead of int32. Official EMBER2024 vectorization uses int32.
    if subset in ("train", "test", "challenge"):
        y_probe = np.asarray(y[: min(32, len(y))])
        valid_probe = set(np.unique(y_probe).tolist())
        if valid_probe and not valid_probe.issubset({-1, 0, 1}):
            logger.warning(
                "Label file %s appears to use float32 encoding; reloading compatibly.",
                y_path,
            )
            y = _load_array_direct(data_dir, f"y_{subset}", np.float32, (n_samples,), use_memmap).astype(np.int32)
    return X, y


def _metadata_files_for_subset(data_dir: Path, subset: str) -> List[Path]:
    """
    Find metadata JSONL files for a subset across both the simplified project
    layout and the official EMBER2024 naming scheme.
    """
    patterns = [
        f"{subset}_metadata*.jsonl",
        f"{subset}_features*.jsonl",
    ]

    if subset == "train":
        patterns.extend(["*_train.jsonl"])
    elif subset == "test":
        patterns.extend(["*_test.jsonl"])
    elif subset == "challenge":
        patterns.extend(["*challenge*.jsonl"])

    seen = set()
    matches: List[Path] = []
    for pattern in patterns:
        for path in sorted(data_dir.glob(pattern)):
            if path not in seen:
                matches.append(path)
                seen.add(path)
    return matches


def _build_week_lookup(jsonl_files: List[Path]) -> Dict[Path, int]:
    """
    Build a file -> week-number mapping from official EMBER2024 shard names.

    The official files are named like:
      2024-09-22_2024-09-28_Win32_test.jsonl

    We map each unique date-range token to a 1-based week number according to
    sorted order so train weeks 49-52 align naturally with the final four
    training periods.
    """
    week_keys = []
    key_for_path: Dict[Path, Optional[str]] = {}
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2})_")

    for path in jsonl_files:
        match = pattern.match(path.name)
        week_key = match.group(1) if match else None
        key_for_path[path] = week_key
        if week_key is not None:
            week_keys.append(week_key)

    unique_keys = sorted(set(week_keys))
    key_to_week = {key: idx + 1 for idx, key in enumerate(unique_keys)}
    return {
        path: (key_to_week[key] if key is not None else -1)
        for path, key in key_for_path.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metadata / family label loading
# ──────────────────────────────────────────────────────────────────────────────

class FamilyMetadata:
    """
    Holds the family / file-type / week mapping loaded from JSONL metadata.

    Attributes
    ----------
    num_families       : int  — number of qualifying families
    family_to_label    : dict — family name → integer class label
    label_to_family    : dict — integer class label → family name
    idx_to_family_label: np.ndarray (int32) — index into the feature matrix → class label (-1 if unknown)
    idx_to_filetype    : list[str]           — index → file_type string
    idx_to_week        : np.ndarray (int16)  — index → week number (-1 if unknown)
    family_to_indices  : dict                — family name → list[int] of feature-matrix row indices
    samples_per_class  : np.ndarray          — counts[i] = number of training samples in class i
    """

    def __init__(self):
        self.num_families = 0
        self.family_to_label: Dict[str, int] = {}
        self.label_to_family: Dict[int, str] = {}
        self.idx_to_family_label: np.ndarray = np.array([], dtype=np.int32)
        self.idx_to_filetype: List[str] = []
        self.idx_to_week: np.ndarray = np.array([], dtype=np.int16)
        self.family_to_indices: Dict[str, List[int]] = {}
        self.samples_per_class: np.ndarray = np.array([], dtype=np.int64)


def load_family_metadata(
    data_dir: str,
    subset: str = "train",
    min_confidence: float = 0.7,
    min_samples: int = 10,
    reference_family_to_label: Optional[Dict[str, int]] = None,
) -> FamilyMetadata:
    """
    Stream JSONL metadata files to build family labels.

    Each JSONL line is a JSON object with (at minimum):
        sha256, label, file_type, family, family_confidence, week

    Memory approach: we accumulate only the fields we need per record,
    so the full JSONL content is never held in memory at once.

    Args:
        data_dir:       Path to EMBER2024 data directory.
        subset:         "train" or "test".
        min_confidence: Only use family labels with family_confidence >= this.
        min_samples:    Exclude families with fewer training samples than this.
        reference_family_to_label:
                        Optional fixed family → label mapping. When provided,
                        this label space is reused instead of fitting a new one.
                        This is required for test/challenge evaluation so labels
                        remain aligned with the training classifier.

    Returns:
        FamilyMetadata object.
    """
    import json

    data_dir = Path(data_dir)

    # Find JSONL files across both project-local and official naming schemes.
    jsonl_files = _metadata_files_for_subset(data_dir, subset)
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL metadata found in {data_dir} for subset='{subset}'.\n"
            "Expected project-local or official EMBER2024 JSONL shard names."
        )

    logger.info(f"Streaming {len(jsonl_files)} metadata file(s) for subset='{subset}' …")
    file_to_week = _build_week_lookup(jsonl_files)

    # First pass: collect per-record info
    raw_family: List[str]  = []      # family name or ""
    raw_conf:   List[float] = []     # family_confidence
    raw_label:  List[int]  = []      # 0/1/-1
    raw_ftype:  List[str]  = []      # file_type
    raw_week:   List[int]  = []      # week index

    for jf in jsonl_files:
        default_week = file_to_week.get(jf, -1)
        with open(jf, "r", buffering=1 << 20) as fh:   # 1 MB read buffer
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Corrupt line; push placeholders so row indices stay aligned
                    raw_family.append("")
                    raw_conf.append(0.0)
                    raw_label.append(LABEL_UNLABELED)
                    raw_ftype.append("unknown")
                    raw_week.append(default_week)
                    continue

                rec_week = rec.get("week")
                if rec_week is None:
                    rec_week_id = rec.get("week_id")
                    if rec_week_id is not None:
                        rec_week = int(rec_week_id) + 1
                    else:
                        rec_week = default_week

                raw_family.append(rec.get("family", "") or "")
                raw_conf.append(float(rec.get("family_confidence", 0.0) or 0.0))
                raw_label.append(int(rec.get("label", LABEL_UNLABELED)))
                raw_ftype.append(rec.get("file_type", "unknown") or "unknown")
                raw_week.append(int(rec_week if rec_week is not None else -1))

    n_total = len(raw_family)
    logger.info(f"  Parsed {n_total:,} metadata records")

    # Count family samples (malicious + confident only)
    family_counts: Dict[str, int] = defaultdict(int)
    for i in range(n_total):
        if (
            raw_label[i] == LABEL_MALICIOUS
            and raw_family[i]
            and raw_family[i] != "unknown"
            and raw_conf[i] >= min_confidence
        ):
            family_counts[raw_family[i]] += 1

    if reference_family_to_label is not None:
        family_to_label = dict(reference_family_to_label)
        label_to_family = {i: f for f, i in family_to_label.items()}
        valid_families = set(family_to_label)
        sorted_families = [label_to_family[i] for i in sorted(label_to_family)]
        logger.info(
            f"  Reusing reference family mapping with {len(valid_families)} classes "
            f"(seen in {subset}: {len(family_counts)})"
        )
    else:
        valid_families = {f for f, c in family_counts.items() if c >= min_samples}
        logger.info(
            f"  Families with ≥{min_samples} samples: {len(valid_families)} "
            f"(total families seen: {len(family_counts)})"
        )
        sorted_families = sorted(valid_families)
        family_to_label = {f: i for i, f in enumerate(sorted_families)}
        label_to_family = {i: f for f, i in family_to_label.items()}

    # Build per-sample arrays
    idx_to_family_label = np.full(n_total, -1, dtype=np.int32)
    family_to_indices: Dict[str, List[int]] = defaultdict(list)

    for i in range(n_total):
        f = raw_family[i]
        if (
            f in valid_families
            and raw_label[i] == LABEL_MALICIOUS
            and raw_conf[i] >= min_confidence
        ):
            lbl = family_to_label[f]
            idx_to_family_label[i] = lbl
            family_to_indices[f].append(i)

    samples_per_class = np.array(
        [len(family_to_indices[sorted_families[i]]) for i in range(len(sorted_families))],
        dtype=np.int64,
    )

    meta = FamilyMetadata()
    meta.num_families = len(valid_families)
    meta.family_to_label = family_to_label
    meta.label_to_family = label_to_family
    meta.idx_to_family_label = idx_to_family_label
    meta.idx_to_filetype = raw_ftype
    meta.idx_to_week = np.array(raw_week, dtype=np.int16)
    meta.family_to_indices = dict(family_to_indices)
    meta.samples_per_class = samples_per_class

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Validation split by week
# ──────────────────────────────────────────────────────────────────────────────

def week_split_indices(
    meta: FamilyMetadata,
    val_weeks: int = 4,
    total_weeks: int = 52,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_indices, val_indices) based on week number.

    Files in the last `val_weeks` of the training period become validation.
    Falls back to a simple index-based 92/8 split if week info is absent.
    """
    weeks = meta.idx_to_week
    cutoff_week = total_weeks - val_weeks + 1   # e.g., 49 for val_weeks=4

    has_week_info = (weeks != -1).any()
    if has_week_info:
        train_idx = np.where(weeks < cutoff_week)[0]
        val_idx   = np.where((weeks >= cutoff_week) & (weeks != -1))[0]
        logger.info(
            f"Week-based split: train={len(train_idx):,}  val={len(val_idx):,} "
            f"(weeks {cutoff_week}-{total_weeks} → val)"
        )
    else:
        logger.warning("No week info in metadata; falling back to 92/8 index split.")
        n = len(weeks)
        cutoff = int(n * (1 - val_weeks / total_weeks))
        train_idx = np.arange(cutoff)
        val_idx   = np.arange(cutoff, n)

    return train_idx, val_idx


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Datasets
# ──────────────────────────────────────────────────────────────────────────────

class EMBER2024DetectionDataset(Dataset):
    """
    Binary detection dataset (malicious=1 / benign=0).

    Unlabeled samples (y == -1) are excluded.

    Args:
        X:       Feature matrix (N, D), may be a memory-mapped array.
        y:       Label vector (N,).
        indices: Optional subset of row indices to use.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if indices is None:
            labeled = np.where(y != LABEL_UNLABELED)[0]
            self.indices = labeled
        else:
            self.indices = indices[y[indices] != LABEL_UNLABELED]

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, label


class EMBER2024FamilyDataset(Dataset):
    """
    Family classification dataset (malicious files with known family labels).

    Args:
        X:                  Feature matrix.
        idx_to_family_label: Per-sample family label (-1 if unknown).
        indices:            Optional subset of row indices.
    """

    def __init__(
        self,
        X: np.ndarray,
        idx_to_family_label: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if indices is None:
            valid = np.where(idx_to_family_label >= 0)[0]
        else:
            valid = indices[idx_to_family_label[indices] >= 0]
        self.indices = valid
        self.X = X
        self.family_labels = idx_to_family_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        label = torch.tensor(self.family_labels[idx], dtype=torch.long)
        return x, label


class EMBER2024MultiTaskDataset(Dataset):
    """
    Combined dataset for Stage 2 multi-task training.

    Returns (features, detection_label, family_label) where family_label=-1
    means the family is unknown (detection loss is still applied).

    Args:
        X:                  Feature matrix.
        y:                  Binary detection labels.
        idx_to_family_label: Per-sample integer family label (-1 if unknown).
        indices:            Subset of rows (labeled samples only).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idx_to_family_label: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if indices is None:
            self.indices = np.where(y != LABEL_UNLABELED)[0]
        else:
            self.indices = indices[y[indices] != LABEL_UNLABELED]
        self.X = X
        self.y = y
        self.family_labels = idx_to_family_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x   = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        det = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        fam = torch.tensor(int(self.family_labels[idx]), dtype=torch.long)
        return x, det, fam


class EMBER2024ContrastiveDataset(Dataset):
    """
    Flat dataset used together with PrototypicalBatchSampler.

    Returns (features, family_label) for samples that have a valid family label
    and are above the confidence threshold.
    """

    def __init__(
        self,
        X: np.ndarray,
        idx_to_family_label: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if indices is None:
            valid = np.where(idx_to_family_label >= 0)[0]
        else:
            valid = indices[idx_to_family_label[indices] >= 0]
        self.indices = valid
        self.X = X
        self.family_labels = idx_to_family_label

        # Build contiguous local index → family_label (needed by sampler)
        self.local_family_labels = idx_to_family_label[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x   = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        lbl = torch.tensor(self.local_family_labels[i], dtype=torch.long)
        return x, lbl


# ──────────────────────────────────────────────────────────────────────────────
# Prototypical Batch Sampler
# ──────────────────────────────────────────────────────────────────────────────

class PrototypicalBatchSampler(Sampler):
    """
    Samples batches of the form: N classes × K samples each.

    This is the standard "episodic" sampler used for metric learning.
    Each batch contains exactly `n_classes_per_batch * n_samples_per_class`
    samples, drawn from `n_classes_per_batch` randomly chosen families.

    Args:
        labels:              Array of class labels (aligned with dataset __getitem__ indices).
        n_classes_per_batch: N — number of classes per batch.
        n_samples_per_class: K — number of samples per class.
        n_iterations:        Number of batches per epoch.
    """

    def __init__(
        self,
        labels: np.ndarray,
        n_classes_per_batch: int = 64,
        n_samples_per_class: int = 8,
        n_iterations: Optional[int] = None,
    ):
        super().__init__()
        self.n_cls  = n_classes_per_batch
        self.n_samp = n_samples_per_class

        unique_classes = np.unique(labels)
        # Build class → local indices mapping
        self.cls_to_indices = {
            c: np.where(labels == c)[0] for c in unique_classes
        }
        # Only keep classes with enough samples
        self.valid_classes = [
            c for c, idxs in self.cls_to_indices.items()
            if len(idxs) >= n_samples_per_class
        ]
        if len(self.valid_classes) < n_classes_per_batch:
            logger.warning(
                f"Only {len(self.valid_classes)} classes have ≥{n_samples_per_class} samples; "
                f"sampling with replacement across classes."
            )

        if n_iterations is None:
            # One epoch = enough iterations to see all samples ~once
            total = sum(len(self.cls_to_indices[c]) for c in self.valid_classes)
            n_iterations = max(1, total // (n_classes_per_batch * n_samples_per_class))
        self.n_iterations = n_iterations

    def __len__(self):
        return self.n_iterations

    def __iter__(self):
        for _ in range(self.n_iterations):
            # Sample N classes (with replacement if fewer valid classes than N)
            chosen_classes = np.random.choice(
                self.valid_classes,
                size=self.n_cls,
                replace=(len(self.valid_classes) < self.n_cls),
            )
            batch_indices = []
            for c in chosen_classes:
                idxs = self.cls_to_indices[c]
                chosen = np.random.choice(idxs, size=self.n_samp, replace=(len(idxs) < self.n_samp))
                batch_indices.extend(chosen.tolist())
            yield batch_indices


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ──────────────────────────────────────────────────────────────────────────────

def make_detection_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 1024,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for binary detection."""
    train_ds = EMBER2024DetectionDataset(X_train, y_train, indices=train_indices)
    val_ds   = EMBER2024DetectionDataset(X_train, y_train, indices=val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def make_multitask_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta: FamilyMetadata,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 1024,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for multi-task training."""
    train_ds = EMBER2024MultiTaskDataset(
        X_train, y_train, meta.idx_to_family_label, indices=train_indices
    )
    val_ds = EMBER2024MultiTaskDataset(
        X_train, y_train, meta.idx_to_family_label, indices=val_indices
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def make_contrastive_loader(
    X_train: np.ndarray,
    meta: FamilyMetadata,
    train_indices: np.ndarray,
    n_classes_per_batch: int = 64,
    n_samples_per_class: int = 8,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a contrastive DataLoader using PrototypicalBatchSampler."""
    ds = EMBER2024ContrastiveDataset(
        X_train, meta.idx_to_family_label, indices=train_indices
    )
    sampler = PrototypicalBatchSampler(
        labels=ds.local_family_labels,
        n_classes_per_batch=n_classes_per_batch,
        n_samples_per_class=n_samples_per_class,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def make_test_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 2048,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    """DataLoader for test or challenge set (no shuffling)."""
    ds = EMBER2024DetectionDataset(X, y)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
