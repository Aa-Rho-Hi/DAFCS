"""
Full evaluation script: our model vs LightGBM baseline, broken down by
file type, for both detection and family classification tasks.

Usage:
    python -m src.evaluate --config configs/default.yaml \
        --checkpoint     checkpoints/stage2/best_model.pt \
        --proto_checkpoint checkpoints/stage3/prototypes.npz \
        [--lgbm_model    checkpoints/lgbm/lgbm_model.txt]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    EMBER2024DetectionDataset,
    EMBER2024FamilyDataset,
    FILE_TYPES,
    FamilyMetadata,
    load_ember_arrays,
    load_family_metadata,
)
from src.model import build_multitask_model, feature_augment
from src.prototypical import extract_embeddings, prototypical_predict
from src.utils import (
    get_device,
    load_checkpoint,
    load_config,
    print_results_table,
    save_json,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("ember2024")


def resolve_checkpoint_path(candidate: Optional[str], default_dir: Optional[str] = None) -> Optional[str]:
    """Resolve either an explicit file path or a checkpoint directory."""
    if candidate:
        path = Path(candidate)
        if path.is_dir():
            best = path / "best_model.pt"
            return str(best) if best.exists() else None
        return str(path)
    if default_dir:
        best = Path(default_dir) / "best_model.pt"
        return str(best) if best.exists() else None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_detection_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """Compute ROC AUC, PR AUC for binary detection."""
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        f1_score,
    )
    # Skip if only one class present
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan")}

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc  = float(average_precision_score(y_true, y_score))
    y_pred  = (y_score >= 0.5).astype(int)
    f1      = float(f1_score(y_true, y_pred, zero_division=0))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}


def compute_family_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute the family-classification metrics used in the report table."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Our model: inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_multitask_inference(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
    num_workers: int = 4,
    tta_n_views: int = 1,
    tta_mask_prob: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Run MultiTaskModel on dataset X/y, with optional Test-Time Augmentation.

    TTA (tta_n_views > 1)
    ─────────────────────
    For each batch, generate `tta_n_views` independently augmented copies,
    run inference on each, then average:
      • Detection: mean of sigmoid scores
      • Family:    mean of softmax probabilities, then argmax

    This is a free accuracy boost at inference — no retraining needed.
    Recommended: tta_n_views=5, tta_mask_prob=0.1 (mild masking).

    Returns dict:
        det_scores   : (N,) float32 — (averaged) sigmoid probabilities
        family_preds : (N,) int32   — argmax of averaged family probabilities
        det_labels   : (N,) int     — ground-truth detection labels (0/1)
    """
    import torch.nn.functional as F

    ds = EMBER2024DetectionDataset(X, y)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    model.eval()

    det_scores_list   = []
    family_preds_list = []
    det_labels_list   = []
    real_indices_list = []

    idx = 0
    for features, labels in tqdm(loader, desc="Model inference", leave=False):
        B = features.size(0)
        features = features.to(device, non_blocking=True)

        if tta_n_views <= 1:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                det_logits, fam_logits = model(features, family_labels=None)
            det_prob = torch.sigmoid(det_logits)
            fam_prob = F.softmax(fam_logits, dim=-1)
        else:
            # Accumulate probabilities over N augmented views
            det_acc = torch.zeros(B, device=device)
            fam_acc = torch.zeros(B, fam_logits.shape[-1] if False else 1, device=device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Get num_families from one clean pass
                dl, fl = model(features, family_labels=None)
                det_acc += torch.sigmoid(dl)
                fam_acc  = F.softmax(fl, dim=-1)

            for _ in range(tta_n_views - 1):
                aug = feature_augment(features, mask_prob=tta_mask_prob, noise_std=0.02)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    dl, fl = model(aug, family_labels=None)
                det_acc += torch.sigmoid(dl)
                fam_acc  = fam_acc + F.softmax(fl, dim=-1)

            det_prob = det_acc / tta_n_views
            fam_prob = fam_acc / tta_n_views

        det_scores_list.append(det_prob.cpu().float().numpy())
        family_preds_list.append(fam_prob.argmax(dim=-1).cpu().numpy())
        det_labels_list.append(labels.numpy())
        real_indices_list.extend(ds.indices[idx:idx + B].tolist())
        idx += B

    return {
        "det_scores":   np.concatenate(det_scores_list),
        "family_preds": np.concatenate(family_preds_list),
        "det_labels":   np.concatenate(det_labels_list),
        "real_indices": np.array(real_indices_list),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM inference (if model is available)
# ──────────────────────────────────────────────────────────────────────────────

def run_lgbm_inference(
    lgbm_model_path: str,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Load a saved LightGBM model and run inference."""
    import lightgbm as lgb

    model = lgb.Booster(model_file=lgbm_model_path)
    labeled = np.where(y != -1)[0]
    X_sub = np.array(X[labeled])   # force load from memmap if needed
    scores = model.predict(X_sub)
    return {
        "det_scores":   scores,
        "det_labels":   y[labeled].astype(int),
        "real_indices": labeled,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-file-type breakdown
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_by_filetype(
    det_scores: np.ndarray,
    det_labels: np.ndarray,
    real_indices: np.ndarray,
    file_type_list,          # list[str] aligned with the full feature array
    model_name: str = "Model",
) -> Dict[str, Dict]:
    """
    Break down detection metrics by file type.

    Args:
        det_scores:    (N,) detection probabilities
        det_labels:    (N,) 0/1 ground-truth
        real_indices:  (N,) indices into the full feature array
        file_type_list: list of file_type strings, one per feature-array row
    """
    results = {}

    # Overall
    results["Overall"] = compute_detection_metrics(det_labels, det_scores)

    # Per file type
    file_types = [file_type_list[i] if i < len(file_type_list) else "unknown"
                  for i in real_indices]
    file_types = np.array(file_types)

    for ft in FILE_TYPES + ["unknown"]:
        mask = file_types == ft
        if mask.sum() == 0:
            continue
        results[ft] = compute_detection_metrics(det_labels[mask], det_scores[mask])

    logger.info(f"\n{model_name} detection metrics by file type:")
    for ft, m in results.items():
        logger.info(
            f"  {ft:<20}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}"
        )
    return results


def _masked_detection_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    file_types: np.ndarray,
    wanted_types: Optional[set[str]],
) -> Dict:
    if wanted_types is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    else:
        mask = np.isin(file_types, list(wanted_types))
    if mask.sum() == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan"), "n_total": 0}
    metrics = compute_detection_metrics(labels[mask], scores[mask])
    metrics["n_total"] = int(mask.sum())
    return metrics


def evaluate_challenge_detection(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_real_indices: np.ndarray,
    test_file_type_list,
    challenge_scores: np.ndarray,
    challenge_labels: np.ndarray,
    challenge_real_indices: np.ndarray,
    challenge_file_type_list,
    model_name: str = "Model",
) -> Dict[str, Dict]:
    """
    Official challenge methodology:
      evaluate on benign test samples + challenge malicious samples.
    """
    test_types = np.array(
        [test_file_type_list[i] if i < len(test_file_type_list) else "unknown" for i in test_real_indices],
        dtype=object,
    )
    challenge_types = np.array(
        [
            challenge_file_type_list[i] if i < len(challenge_file_type_list) else "unknown"
            for i in challenge_real_indices
        ],
        dtype=object,
    )

    benign_mask = test_labels == 0
    test_scores_ben = test_scores[benign_mask]
    test_labels_ben = test_labels[benign_mask]
    test_types_ben = test_types[benign_mask]

    all_scores = np.concatenate([test_scores_ben, challenge_scores], axis=0)
    all_labels = np.concatenate([test_labels_ben, challenge_labels], axis=0)
    all_types = np.concatenate([test_types_ben, challenge_types], axis=0)

    type_map = {
        "Overall": None,
        "Win32": {"Win32"},
        "Win64": {"Win64"},
        "Dot_Net": {"Dot_Net"},
        "APK": {"APK"},
        "ELF": {"ELF"},
        "PDF": {"PDF"},
    }

    results: Dict[str, Dict] = {}
    for name, wanted in type_map.items():
        results[name] = _masked_detection_metrics(all_scores, all_labels, all_types, wanted)

    logger.info(f"\n{model_name} challenge metrics (test benign + challenge malware):")
    for ft, m in results.items():
        logger.info(
            f"  {ft:<20} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
            f"F1={m['f1']:.4f}  n={m['n_total']}"
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Challenge set evaluation
# ──────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    our_test_det: Dict,
    our_challenge_det: Dict,
    lgbm_test_det: Optional[Dict],
    lgbm_challenge_det: Optional[Dict],
    our_fam:  Dict,
    lgbm_fam: Optional[Dict],
) -> Dict:
    """Build a structured results dict for pretty-printing and JSON export."""
    table = {}

    # Detection — test set by file type
    for ft, metrics in our_test_det.items():
        entry = {"ours_pr_auc": metrics["pr_auc"], "ours_roc_auc": metrics["roc_auc"]}
        if lgbm_test_det and ft in lgbm_test_det:
            entry["lgbm_pr_auc"] = lgbm_test_det[ft]["pr_auc"]
            entry["lgbm_roc_auc"] = lgbm_test_det[ft]["roc_auc"]
            entry["delta_pr_auc"] = metrics["pr_auc"] - lgbm_test_det[ft]["pr_auc"]
        table[f"test_detection/{ft}"] = entry

    # Detection — challenge methodology by file type
    for ft, metrics in our_challenge_det.items():
        entry = {"ours_pr_auc": metrics["pr_auc"], "ours_roc_auc": metrics["roc_auc"]}
        if lgbm_challenge_det and ft in lgbm_challenge_det:
            entry["lgbm_pr_auc"] = lgbm_challenge_det[ft]["pr_auc"]
            entry["lgbm_roc_auc"] = lgbm_challenge_det[ft]["roc_auc"]
            entry["delta_pr_auc"] = metrics["pr_auc"] - lgbm_challenge_det[ft]["pr_auc"]
        table[f"challenge_detection/{ft}"] = entry

    # Family classification — overall
    table["family/overall"] = dict(our_fam)
    if lgbm_fam:
        table["family/overall"]["lgbm_accuracy"] = lgbm_fam.get("accuracy")
        table["family/overall"]["lgbm_macro_f1"] = lgbm_fam.get("macro_f1")
        table["family/overall"]["lgbm_weighted_f1"] = lgbm_fam.get("weighted_f1")
        table["family/overall"]["lgbm_macro_precision"] = lgbm_fam.get("macro_precision")
        table["family/overall"]["lgbm_macro_recall"] = lgbm_fam.get("macro_recall")
        table["family/overall"]["delta_macro_f1"] = (
            our_fam.get("macro_f1", 0) - lgbm_fam.get("macro_f1", 0)
        )

    return table


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    set_seed(cfg["hardware"]["seed"])
    setup_logging(cfg["logging"]["level"])
    device = get_device(cfg["hardware"]["device"])

    ev_cfg    = cfg.get("evaluation", {})
    output_dir = Path(ev_cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir  = cfg["data"]["data_dir"]
    batch_size = ev_cfg.get("batch_size", 2048)

    # ── Test / challenge data ────────────────────────────────────────────────
    logger.info("Loading test set …")
    X_test, y_test = load_ember_arrays(data_dir, "test", use_memmap=True, feature_dim=cfg["data"]["input_dim"])
    logger.info("Loading challenge set …")
    X_challenge, y_challenge = load_ember_arrays(
        data_dir, "challenge", use_memmap=True, feature_dim=cfg["data"]["input_dim"]
    )
    meta = load_family_metadata(
        data_dir, "train",
        min_confidence=cfg["data"]["family_confidence_threshold"],
        min_samples=cfg["data"]["min_family_samples"],
    )
    test_meta = load_family_metadata(
        data_dir,
        "test",
        min_confidence=cfg["data"]["family_confidence_threshold"],
        min_samples=1,
        reference_family_to_label=meta.family_to_label,
    )
    try:
        challenge_meta = load_family_metadata(
            data_dir,
            "challenge",
            min_confidence=cfg["data"]["family_confidence_threshold"],
            min_samples=1,
            reference_family_to_label=meta.family_to_label,
        )
    except FileNotFoundError:
        challenge_meta = None

    # ── Our model ────────────────────────────────────────────────────────────
    cfg["model"]["num_families"] = meta.num_families
    model = build_multitask_model(cfg, num_families=meta.num_families).to(device)

    checkpoint = resolve_checkpoint_path(
        ev_cfg.get("checkpoint"), cfg.get("stage2", {}).get("checkpoint_dir")
    )
    if checkpoint and Path(checkpoint).exists():
        load_checkpoint(checkpoint, model, device=device)
    else:
        logger.warning("No checkpoint provided — using random weights (for debugging only).")

    # Detection inference
    logger.info("Running our model on test set …")
    our_test_result = run_multitask_inference(
        model, X_test, y_test, device, batch_size, num_workers=cfg["data"]["num_workers"]
    )
    our_test_det = evaluate_by_filetype(
        our_test_result["det_scores"],
        our_test_result["det_labels"],
        our_test_result["real_indices"],
        test_meta.idx_to_filetype if test_meta else ["unknown"] * len(X_test),
        model_name="Ours test",
    )
    logger.info("Running our model on challenge set …")
    our_challenge_result = run_multitask_inference(
        model, X_challenge, y_challenge, device, batch_size, num_workers=cfg["data"]["num_workers"]
    )
    our_challenge_det = evaluate_challenge_detection(
        test_scores=our_test_result["det_scores"],
        test_labels=our_test_result["det_labels"],
        test_real_indices=our_test_result["real_indices"],
        test_file_type_list=test_meta.idx_to_filetype if test_meta else ["unknown"] * len(X_test),
        challenge_scores=our_challenge_result["det_scores"],
        challenge_labels=our_challenge_result["det_labels"],
        challenge_real_indices=our_challenge_result["real_indices"],
        challenge_file_type_list=challenge_meta.idx_to_filetype if challenge_meta else ["unknown"] * len(X_challenge),
        model_name="Ours challenge",
    )

    # Family classification (softmax head)
    test_fam_ds = EMBER2024FamilyDataset(X_test, test_meta.idx_to_family_label)
    our_fam = {"accuracy": float("nan"), "macro_f1": float("nan")}
    if len(test_fam_ds) > 0:
        fam_loader = DataLoader(
            test_fam_ds, batch_size=batch_size, shuffle=False,
            num_workers=cfg["data"]["num_workers"], pin_memory=(device.type == "cuda"),
        )
        all_preds, all_true = [], []
        model.eval()
        for feats, labels in tqdm(fam_loader, desc="Family eval", leave=False):
            feats = feats.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _, fam_logits = model(feats)
            all_preds.extend(fam_logits.argmax(-1).cpu().numpy())
            all_true.extend(labels.numpy())
        our_fam = compute_family_metrics(np.array(all_true), np.array(all_preds))
        logger.info(f"Our family: accuracy={our_fam['accuracy']:.4f}  macro_F1={our_fam['macro_f1']:.4f}")

    # Prototypical family metrics (if prototypes exist)
    proto_path = ev_cfg.get("prototypical_checkpoint")
    prototypes = None
    if proto_path and Path(proto_path).exists():
        saved = np.load(proto_path, allow_pickle=True)
        prototypes = saved["prototypes"]
        if len(test_fam_ds) > 0:
            embs, true_fam = extract_embeddings(
                model, test_fam_ds, batch_size=batch_size, device=device,
                num_workers=cfg["data"]["num_workers"],
            )
            pred_fam, _ = prototypical_predict(embs, prototypes)
            proto_fam = compute_family_metrics(true_fam, pred_fam)
            logger.info(
                f"Prototypical family: accuracy={proto_fam['accuracy']:.4f}  "
                f"macro_F1={proto_fam['macro_f1']:.4f}"
            )
            our_fam["proto_accuracy"] = proto_fam["accuracy"]
            our_fam["proto_macro_f1"] = proto_fam["macro_f1"]

    # ── LightGBM baseline ────────────────────────────────────────────────────
    lgbm_test_det = lgbm_challenge_det = lgbm_fam = None
    lgbm_model_path = ev_cfg.get("lgbm_model")
    if lgbm_model_path and Path(lgbm_model_path).exists():
        logger.info("Running LightGBM baseline on test set …")
        try:
            lgbm_test_result = run_lgbm_inference(lgbm_model_path, X_test, y_test)
            lgbm_challenge_result = run_lgbm_inference(lgbm_model_path, X_challenge, y_challenge)
            lgbm_test_det = evaluate_by_filetype(
                lgbm_test_result["det_scores"],
                lgbm_test_result["det_labels"],
                lgbm_test_result["real_indices"],
                test_meta.idx_to_filetype if test_meta else [],
                model_name="LightGBM test",
            )
            lgbm_challenge_det = evaluate_challenge_detection(
                test_scores=lgbm_test_result["det_scores"],
                test_labels=lgbm_test_result["det_labels"],
                test_real_indices=lgbm_test_result["real_indices"],
                test_file_type_list=test_meta.idx_to_filetype if test_meta else [],
                challenge_scores=lgbm_challenge_result["det_scores"],
                challenge_labels=lgbm_challenge_result["det_labels"],
                challenge_real_indices=lgbm_challenge_result["real_indices"],
                challenge_file_type_list=challenge_meta.idx_to_filetype if challenge_meta else [],
                model_name="LightGBM challenge",
            )
        except Exception as e:
            logger.warning(f"LightGBM inference failed: {e}")

    # ── Final table ──────────────────────────────────────────────────────────
    table = build_comparison_table(
        our_test_det=our_test_det,
        our_challenge_det=our_challenge_det,
        lgbm_test_det=lgbm_test_det,
        lgbm_challenge_det=lgbm_challenge_det,
        our_fam=our_fam,
        lgbm_fam=lgbm_fam,
    )
    print_results_table(table, title="EMBER2024 Evaluation Results")

    # Save to disk
    save_json(table, str(output_dir / "results.json"))
    raw_scores = {
        "test_y_true": our_test_result["det_labels"],
        "test_our_score": our_test_result["det_scores"],
        "test_file_types": np.array(
            [test_meta.idx_to_filetype[i] if i < len(test_meta.idx_to_filetype) else "unknown"
             for i in our_test_result["real_indices"]],
            dtype=object,
        ),
        "challenge_y_true": our_challenge_result["det_labels"],
        "challenge_our_score": our_challenge_result["det_scores"],
        "challenge_file_types": np.array(
            [
                challenge_meta.idx_to_filetype[i]
                if challenge_meta is not None and i < len(challenge_meta.idx_to_filetype)
                else "unknown"
                for i in our_challenge_result["real_indices"]
            ],
            dtype=object,
        ),
    }
    # Backward-compatible aliases for older notebook cells.
    raw_scores["y_true"] = raw_scores["test_y_true"]
    raw_scores["our_score"] = raw_scores["test_our_score"]
    raw_scores["file_types"] = raw_scores["test_file_types"]
    if lgbm_test_det is not None and lgbm_model_path and Path(lgbm_model_path).exists():
        lgbm_test_result = run_lgbm_inference(lgbm_model_path, X_test, y_test)
        lgbm_challenge_result = run_lgbm_inference(lgbm_model_path, X_challenge, y_challenge)
        raw_scores["test_lgbm_score"] = lgbm_test_result["det_scores"]
        raw_scores["challenge_lgbm_score"] = lgbm_challenge_result["det_scores"]
        raw_scores["lgbm_score"] = lgbm_test_result["det_scores"]
    np.savez(output_dir / "raw_scores.npz", **raw_scores)
    logger.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Stage 2 checkpoint")
    parser.add_argument("--proto_checkpoint", default=None, help="Stage 3 prototypes.npz")
    parser.add_argument("--lgbm_model", default=None, help="LightGBM model file")
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.checkpoint:
        cfg["evaluation"]["checkpoint"] = args.checkpoint
    if args.proto_checkpoint:
        cfg["evaluation"]["prototypical_checkpoint"] = args.proto_checkpoint
    if args.lgbm_model:
        cfg["evaluation"]["lgbm_model"] = args.lgbm_model

    env_dir = os.environ.get("EMBER2024_DIR")
    if env_dir and cfg["data"]["data_dir"] == "/path/to/ember2024":
        cfg["data"]["data_dir"] = env_dir

    main(cfg)
