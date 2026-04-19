#!/usr/bin/env bash
# Materialize the complete EMBER2024 workspace under data/local from a shared root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_ROOT="$PROJECT_DIR/data/local"
MODE="${MODE:-rsync}"  # rsync | copy | symlink
SOURCE_ROOT="${EMBER2024_SHARED_ROOT:-${1:-$PROJECT_DIR/..}}"

DATASETS=(
    EMBER2024-corrected-canonical
    EMBER2024-corrected-full
    EMBER2024-artifacts
    EMBER2024-evaldata
    EMBER2024-full-local
    EMBER2024-train-artifacts
)

usage() {
    cat <<'EOF'
Usage:
  bash scripts/sync_full_dataset.sh /path/to/shared/root

Environment:
  EMBER2024_SHARED_ROOT   Shared storage root that contains the EMBER2024-* directories.
  MODE                    One of: rsync, copy, symlink (default: rsync)

Examples:
  bash scripts/sync_full_dataset.sh /mnt/team-share/ember2024
  MODE=symlink bash scripts/sync_full_dataset.sh /Volumes/lab-data
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "ERROR: source root not found: $SOURCE_ROOT"
    usage
    exit 1
fi

case "$MODE" in
    rsync|copy|symlink) ;;
    *)
        echo "ERROR: invalid MODE='$MODE' (expected rsync, copy, or symlink)"
        exit 1
        ;;
esac

mkdir -p "$TARGET_ROOT"

echo "==================================================================="
echo "  EMBER2024 Full Dataset Sync"
echo "  source_root=$SOURCE_ROOT"
echo "  target_root=$TARGET_ROOT"
echo "  mode=$MODE"
echo "==================================================================="

missing=0

for dataset in "${DATASETS[@]}"; do
    source_dir="$SOURCE_ROOT/$dataset"
    target_dir="$TARGET_ROOT/$dataset"

    if [[ ! -e "$source_dir" ]]; then
        echo "missing source: $source_dir"
        missing=1
        continue
    fi

    echo ""
    echo ">>> $dataset"

    case "$MODE" in
        symlink)
            rm -rf "$target_dir"
            ln -s "$source_dir" "$target_dir"
            echo "linked $target_dir -> $source_dir"
            ;;
        copy)
            rm -rf "$target_dir"
            mkdir -p "$target_dir"
            cp -a "$source_dir"/. "$target_dir"/
            echo "copied into $target_dir"
            ;;
        rsync)
            mkdir -p "$target_dir"
            rsync -a --info=progress2 "$source_dir"/ "$target_dir"/
            echo "synced into $target_dir"
            ;;
    esac
done

if [[ "$missing" -ne 0 ]]; then
    echo ""
    echo "ERROR: one or more source dataset directories were missing."
    exit 1
fi

echo ""
echo "Full dataset materialized under $TARGET_ROOT"
echo "Next step: python scripts/verify_full_dataset.py"
