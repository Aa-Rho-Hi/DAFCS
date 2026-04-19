"""
Utility functions: logging, checkpointing, config loading, metrics helpers.
"""

import os
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger with console (and optional file) handler."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger("ember2024")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Load a YAML config file and apply optional flat key=value overrides.

    Example overrides dict:
        {"stage1.lr": 5e-4, "data.data_dir": "/data/ember2024"}
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            node = cfg
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value

    return cfg


def get_nested(cfg: Dict, *keys, default=None):
    """Safe nested dict access: get_nested(cfg, 'stage1', 'lr')."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────

def get_device(preferred: str = "cuda") -> torch.device:
    """Return the best available device matching `preferred`."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    """
    Save a training checkpoint.

    Args:
        state: Dict containing model state_dict, optimizer state_dict, epoch, metrics, etc.
        checkpoint_dir: Directory to save into.
        filename: Checkpoint file name.
        is_best: If True, also copy to 'best_model.pt'.

    Returns:
        Path to saved checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)
    if is_best:
        best_path = ckpt_dir / "best_model.pt"
        shutil.copy(path, best_path)
        logging.getLogger("ember2024").info(f"New best model saved → {best_path}")
    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into a model (and optionally optimizer).

    Returns the full checkpoint dict so callers can inspect epoch/metrics.
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logging.getLogger("ember2024").info(
        f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})"
    )
    return ckpt


def load_encoder_weights(checkpoint_path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load only the encoder backbone weights from a checkpoint (ignore heads)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # Keep only encoder keys
    encoder_state = {k: v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    log = logging.getLogger("ember2024")
    log.info(f"Loaded encoder weights: {len(encoder_state)} keys")
    if missing:
        log.debug(f"  Missing keys (expected for new heads): {missing[:5]}...")
    if unexpected:
        log.warning(f"  Unexpected keys: {unexpected[:5]}...")


# ──────────────────────────────────────────────────────────────────────────────
# AverageMeter
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Track running mean of a scalar (e.g., loss, accuracy)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# W&B Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class WandbLogger:
    """
    Thin wrapper around wandb that degrades gracefully when disabled or unavailable.
    """

    def __init__(self, cfg: Dict, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=get_nested(cfg, "logging", "wandb_project", default="ember2024"),
                entity=get_nested(cfg, "logging", "wandb_entity"),
                name=get_nested(cfg, "logging", "wandb_run_name"),
                config=cfg,
            )
        except Exception as e:
            logging.getLogger("ember2024").warning(f"W&B init failed: {e}. Disabling.")
            self.enabled = False

    def log(self, metrics: Dict, step: Optional[int] = None):
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self._wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# LR Scheduler helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty table printer
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results: Dict, title: str = "Results") -> None:
    """Print a clean comparison table to stdout."""
    header = f"\n{'='*70}\n{title:^70}\n{'='*70}"
    print(header)
    for section, metrics in results.items():
        print(f"\n  [{section}]")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k:<40} {v:.4f}")
                else:
                    print(f"    {k:<40} {v}")
    print("=" * 70)
