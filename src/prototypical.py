"""
Stage 3: Prototypical Few-Shot Inference.

For each family: compute prototype = mean L2-normalised embedding of all
training samples. At inference time, assign the nearest prototype by cosine
similarity.  Handles families with as few as 1 sample gracefully.

Usage:
    python -m src.prototypical --config configs/default.yaml \
        --encoder_ckpt checkpoints/stage1/best_model.pt \
        --output_dir   checkpoints/stage3
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_ember_arrays,
    load_family_metadata,
    week_split_indices,
    EMBER2024FamilyDataset,
)
from src.model import build_contrastive_model, build_multitask_model
from src.utils import (
    get_device,
    load_checkpoint,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("ember2024")


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 2048,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract encoder embeddings for all samples in `dataset`.

    Returns:
        embeddings: float32 array (N, D)
        labels:     int32 array   (N,)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()

    all_embs  = []
    all_labels = []

    for features, labels in tqdm(loader, desc="Extracting embeddings", leave=False):
        features = features.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Use encode() to get raw encoder output (not the projection head)
            embs = model.encode(features)
        # L2-normalise for cosine similarity
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().float().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Prototype computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute per-class prototype = mean of L2-normalised embeddings.

    Args:
        embeddings:  (N, D) float32 — L2-normalised
        labels:      (N,)   int32   — class indices (no -1 expected here)
        num_classes: C

    Returns:
        prototypes: (C, D) float32 — L2-normalised mean embeddings
    """
    D = embeddings.shape[1]
    prototypes = np.zeros((num_classes, D), dtype=np.float32)
    counts     = np.zeros(num_classes, dtype=np.int64)

    for emb, lbl in zip(embeddings, labels):
        if 0 <= lbl < num_classes:
            prototypes[lbl] += emb
            counts[lbl] += 1

    # Families with zero samples get a zero prototype (won't be matched)
    nonzero = counts > 0
    prototypes[nonzero] /= counts[nonzero, None]

    # Re-normalise each prototype to unit length (proper Euclidean mean on sphere)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    prototypes = prototypes / norms

    logger.info(
        f"Computed {nonzero.sum()} / {num_classes} prototypes "
        f"(avg {counts[nonzero].mean():.1f} samples per class)"
    )
    return prototypes


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def prototypical_predict(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    batch_size: int = 4096,
    similarity: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each embedding to its nearest prototype.

    Args:
        embeddings:  (N, D) float32 — L2-normalised query embeddings
        prototypes:  (C, D) float32 — L2-normalised class prototypes
        batch_size:  Process this many queries at once (avoids OOM for large N)
        similarity:  "cosine" or "euclidean"

    Returns:
        pred_labels:  (N,) int32   — predicted class index
        confidence:   (N,) float32 — similarity score to nearest prototype
    """
    N = embeddings.shape[0]
    pred_labels = np.empty(N, dtype=np.int32)
    confidence  = np.empty(N, dtype=np.float32)

    proto_t = torch.from_numpy(prototypes)  # (C, D)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        emb_batch = torch.from_numpy(embeddings[start:end])  # (B, D)

        if similarity == "cosine":
            # Both are L2-normalised → dot product = cosine similarity
            sim = torch.matmul(emb_batch, proto_t.T)   # (B, C)
        else:
            # Negative squared Euclidean distance (higher = closer)
            sim = -torch.cdist(emb_batch, proto_t, p=2).pow(2)

        best_sim, best_cls = sim.max(dim=1)
        pred_labels[start:end] = best_cls.numpy()
        confidence[start:end]  = best_sim.numpy()

    return pred_labels, confidence


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline: compute prototypes + evaluate
# ──────────────────────────────────────────────────────────────────────────────

def build_and_save_prototypes(
    model: torch.nn.Module,
    X_train: np.ndarray,
    meta,
    train_idx: np.ndarray,
    output_dir: str,
    batch_size: int = 2048,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 4,
) -> str:
    """
    Extract train embeddings, compute prototypes, save to .npz.

    Returns path to saved prototypes file.
    """
    # Only training split (no leakage from validation)
    train_family_ds = EMBER2024FamilyDataset(
        X_train, meta.idx_to_family_label, indices=train_idx
    )
    logger.info(f"Extracting embeddings for {len(train_family_ds):,} training samples …")
    embeddings, labels = extract_embeddings(
        model, train_family_ds, batch_size=batch_size, device=device, num_workers=num_workers
    )

    prototypes = compute_prototypes(embeddings, labels, num_classes=meta.num_families)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "prototypes.npz"
    np.savez(
        out_path,
        prototypes=prototypes,
        label_to_family=np.array(
            [meta.label_to_family[i] for i in range(meta.num_families)], dtype=object
        ),
    )
    logger.info(f"Prototypes saved → {out_path}")
    return str(out_path)


def evaluate_prototypical(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    meta,
    prototypes: np.ndarray,
    indices: Optional[np.ndarray] = None,
    split: str = "val",
    batch_size: int = 2048,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 4,
    similarity: str = "cosine",
) -> Dict:
    """
    Run prototypical inference on eval set and compute metrics.

    Returns dict with accuracy, macro_f1, per-class f1.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    eval_ds = EMBER2024FamilyDataset(
        X_eval, meta.idx_to_family_label, indices=indices
    )
    logger.info(f"Extracting embeddings for {len(eval_ds):,} {split} samples …")
    embeddings, true_labels = extract_embeddings(
        model, eval_ds, batch_size=batch_size, device=device, num_workers=num_workers
    )

    pred_labels, _ = prototypical_predict(
        embeddings, prototypes, batch_size=4096, similarity=similarity
    )

    acc      = float(accuracy_score(true_labels, pred_labels))
    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
    report   = classification_report(
        true_labels, pred_labels, zero_division=0, output_dict=True
    )

    logger.info(f"[{split}] Prototypical | accuracy={acc:.4f} | macro_F1={macro_f1:.4f}")

    return {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "report":    report,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    set_seed(cfg["hardware"]["seed"])
    setup_logging(cfg["logging"]["level"])
    device = get_device(cfg["hardware"]["device"])

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = cfg["data"]["data_dir"]
    X_train, y_train = load_ember_arrays(
        data_dir, "train",
        use_memmap=cfg["data"]["use_memmap"],
        feature_dim=cfg["data"]["input_dim"],
    )
    meta = load_family_metadata(
        data_dir, "train",
        min_confidence=cfg["data"]["family_confidence_threshold"],
        min_samples=cfg["data"]["min_family_samples"],
    )
    train_idx, val_idx = week_split_indices(meta, val_weeks=cfg["data"]["val_weeks"])

    s3 = cfg["stage3"]

    # ── Model ───────────────────────────────────────────────────────────────
    cfg["model"]["num_families"] = meta.num_families
    # Prefer Stage 1 model (ContrastiveModel) for embeddings; fall back to Stage 2
    encoder_ckpt = s3.get("encoder_checkpoint")
    try:
        # Try loading as ContrastiveModel first
        model = build_contrastive_model(cfg)
        if encoder_ckpt:
            ckpt = load_checkpoint(encoder_ckpt, model, device=device)
        model = model.to(device)
        logger.info("Using ContrastiveModel encoder for prototypes.")
    except Exception:
        model = build_multitask_model(cfg, num_families=meta.num_families)
        if encoder_ckpt:
            ckpt = load_checkpoint(encoder_ckpt, model, device=device)
        model = model.to(device)
        logger.info("Using MultiTaskModel encoder for prototypes.")

    # ── Compute + save prototypes ────────────────────────────────────────────
    proto_path = build_and_save_prototypes(
        model, X_train, meta, train_idx,
        output_dir=s3["output_dir"],
        batch_size=s3["batch_size"],
        device=device,
        num_workers=cfg["data"]["num_workers"],
    )

    # ── Quick eval on validation split ──────────────────────────────────────
    saved = np.load(proto_path, allow_pickle=True)
    prototypes = saved["prototypes"]

    val_metrics = evaluate_prototypical(
        model, X_train, meta, prototypes,
        indices=val_idx,
        split="val",
        batch_size=s3["batch_size"],
        device=device,
        num_workers=cfg["data"]["num_workers"],
        similarity=s3.get("similarity", "cosine"),
    )
    save_json(val_metrics, Path(s3["output_dir"]) / "val_metrics_proto.json")
    logger.info("Stage 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Prototypical Inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--encoder_ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.encoder_ckpt:
        cfg["stage3"]["encoder_checkpoint"] = args.encoder_ckpt

    env_dir = os.environ.get("EMBER2024_DIR")
    if env_dir and cfg["data"]["data_dir"] == "/path/to/ember2024":
        cfg["data"]["data_dir"] = env_dir

    main(cfg)
