#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Prototypical Few-Shot Inference
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="${EMBER2024_DIR:-${1:-}}"
# Use Stage 1 encoder for prototypes (better embeddings than Stage 2 for few-shot)
ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/stage1/best_model.pt}"

if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: Set EMBER2024_DIR or pass data_dir as first argument."
    exit 1
fi

echo "==================================================================="
echo "  Stage 3: Prototypical Few-Shot Inference"
echo "  data_dir=$DATA_DIR"
echo "  encoder=$ENCODER_CKPT"
echo "==================================================================="

python -m src.prototypical \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --encoder_ckpt "$ENCODER_CKPT"

echo ""
echo "Prototypes saved: checkpoints/stage3/prototypes.npz"
