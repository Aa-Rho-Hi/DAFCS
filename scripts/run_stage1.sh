#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Supervised Contrastive Pre-training
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="${EMBER2024_DIR:-${1:-}}"
RESUME="${RESUME:-}"
WANDB="${WANDB:-}"

if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: Set EMBER2024_DIR or pass data_dir as first argument."
    exit 1
fi

echo "==================================================================="
echo "  Stage 1: Supervised Contrastive Pre-training"
echo "  data_dir=$DATA_DIR"
echo "==================================================================="

ARGS="--config $CONFIG --data_dir $DATA_DIR"
[[ -n "$RESUME"   ]] && ARGS="$ARGS --resume $RESUME"
[[ -n "$WANDB"    ]] && ARGS="$ARGS --wandb"

python -m src.contrastive $ARGS

echo ""
echo "Stage 1 complete. Best checkpoint: checkpoints/stage1/best_model.pt"
