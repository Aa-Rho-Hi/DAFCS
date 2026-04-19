#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_DATA_DIR="$PROJECT_DIR/data/local"

mkdir -p "$LOCAL_DATA_DIR"

DATASETS=(
    EMBER2024-corrected-canonical
    EMBER2024-corrected-full
    EMBER2024-artifacts
    EMBER2024-evaldata
    EMBER2024-full-local
    EMBER2024-train-artifacts
)

for dataset in "${DATASETS[@]}"; do
    source_dir="$PROJECT_DIR/../$dataset"
    target_link="$LOCAL_DATA_DIR/$dataset"
    relative_target="../../../$dataset"

    if [[ ! -e "$source_dir" ]]; then
        continue
    fi

    ln -sfn "$relative_target" "$target_link"
    echo "linked $target_link -> $relative_target"
done

echo ""
echo "Local dataset links are available under $LOCAL_DATA_DIR"
