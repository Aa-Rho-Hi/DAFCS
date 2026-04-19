#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Supervised Contrastive Pre-training
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source "$SCRIPT_DIR/common_data_dir.sh"

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="$(resolve_ember2024_dir "$PROJECT_DIR" "${1:-}" || true)"
RESUME="${RESUME:-}"
WANDB="${WANDB:-}"

if [[ -z "$DATA_DIR" ]]; then
    print_ember2024_dir_help
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
