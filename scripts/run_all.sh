#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline: baseline → Stage 1 → Stage 2 → Stage 3 → evaluate
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATA_DIR="${EMBER2024_DIR:-${1:-}}"
if [[ -z "$DATA_DIR" ]]; then
    echo "Usage: EMBER2024_DIR=/path/to/ember2024 bash scripts/run_all.sh"
    echo "   or: bash scripts/run_all.sh /path/to/ember2024"
    echo ""
    echo "To use mock data for testing:"
    echo "   python scripts/generate_mock_data.py --output_dir /tmp/ember2024_mock"
    echo "   bash scripts/run_all.sh /tmp/ember2024_mock"
    exit 1
fi

export EMBER2024_DIR="$DATA_DIR"
CONFIG="${CONFIG:-configs/default.yaml}"
HAVE_LGBM=0

if python - <<'PY' >/dev/null 2>&1
import lightgbm
PY
then
    HAVE_LGBM=1
fi

echo "==================================================================="
echo "  EMBER2024 Full Pipeline"
echo "  data_dir=$DATA_DIR"
echo "==================================================================="

# 0. LightGBM baseline
echo ""
echo "── Step 0: LightGBM Baseline ────────────────────────────────────"
if [[ "$HAVE_LGBM" -eq 1 ]]; then
    bash scripts/run_baseline.sh "$DATA_DIR"
else
    echo "Skipping baseline: lightgbm is not installed in the current environment."
fi

# 1. Contrastive pre-training
echo ""
echo "── Step 1: Supervised Contrastive Pre-training ──────────────────"
bash scripts/run_stage1.sh "$DATA_DIR"

# 2. Multi-task fine-tuning
echo ""
echo "── Step 2: Multi-Task Fine-tuning ───────────────────────────────"
bash scripts/run_stage2.sh "$DATA_DIR"

# 3. Prototypical inference
echo ""
echo "── Step 3: Prototypical Few-Shot Inference ──────────────────────"
bash scripts/run_stage3.sh "$DATA_DIR"

# 4. Full evaluation
echo ""
echo "── Step 4: Full Evaluation ──────────────────────────────────────"
EVAL_ARGS=(
    --config "$CONFIG"
    --data_dir "$DATA_DIR"
    --checkpoint checkpoints/stage2/best_model.pt
    --proto_checkpoint checkpoints/stage3/prototypes.npz
)
if [[ "$HAVE_LGBM" -eq 1 && -f checkpoints/lgbm/lgbm_detection.txt ]]; then
    EVAL_ARGS+=(--lgbm_model checkpoints/lgbm/lgbm_detection.txt)
fi
python -m src.evaluate "${EVAL_ARGS[@]}"

echo ""
echo "==================================================================="
echo "  Pipeline complete!"
echo "  Results: results/results.json"
echo "  Notebook: notebooks/analysis.ipynb"
echo "==================================================================="
