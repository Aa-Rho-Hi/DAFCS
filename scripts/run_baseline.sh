#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run LightGBM baselines (detection + family classification)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="${EMBER2024_DIR:-${1:-}}"
TASK="${TASK:-all}"   # "detection", "family", or "all"

# ── Validate data dir ─────────────────────────────────────────────────────────
if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: Set EMBER2024_DIR env var or pass data directory as first argument."
    echo "  Example: EMBER2024_DIR=/data/ember2024 bash scripts/run_baseline.sh"
    echo "  Or use mock data: bash scripts/run_baseline.sh /tmp/ember2024_mock"
    exit 1
fi

echo "==================================================================="
echo "  LightGBM Baseline  |  task=$TASK"
echo "  data_dir=$DATA_DIR"
echo "==================================================================="

python -m src.baseline_lgbm \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --task "$TASK"

echo ""
echo "Baseline results saved to results/lgbm_baseline_results.json"
