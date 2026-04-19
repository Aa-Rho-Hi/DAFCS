#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Multi-Task Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source "$SCRIPT_DIR/common_data_dir.sh"

CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="$(resolve_ember2024_dir "$PROJECT_DIR" "${1:-}" || true)"
PRETRAINED="${PRETRAINED:-checkpoints/stage1/best_model.pt}"
RESUME="${RESUME:-}"
WANDB="${WANDB:-}"

if [[ -z "$DATA_DIR" ]]; then
    print_ember2024_dir_help
    exit 1
fi

echo "==================================================================="
echo "  Stage 2: Multi-Task Fine-tuning"
echo "  data_dir=$DATA_DIR"
echo "  pretrained encoder=$PRETRAINED"
echo "==================================================================="

ARGS="--config $CONFIG --data_dir $DATA_DIR"
[[ -f "$PRETRAINED" ]] && ARGS="$ARGS --pretrained $PRETRAINED" || echo "  (no pretrained encoder found; training from scratch)"
[[ -n "$RESUME"     ]] && ARGS="$ARGS --resume $RESUME"
[[ -n "$WANDB"      ]] && ARGS="$ARGS --wandb"

python -m src.multitask $ARGS

echo ""
echo "Stage 2 complete. Best checkpoint: checkpoints/stage2/best_model.pt"
