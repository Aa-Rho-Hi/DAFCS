#!/usr/bin/env bash
set -euo pipefail

# download_ember2024.sh
# Helper to fetch EMBER2024 dataset into the repository's data/local folder.
# Supports three modes:
#   --http <URL>    Download a tar/zip archive from HTTP(S) and extract it
#   --rsync <SRC>   Rsync a source directory containing EMBER2024-* folders
#   --scp <SRC>     Copy via scp (user@host:/path/to/EMBER2024-corrected-full)
#
# Usage examples:
#   MODE=http ./download_ember2024.sh --http https://example.com/EMBER2024.tar.gz
#   ./download_ember2024.sh --rsync user@host:/data/ember_shared
#   ./download_ember2024.sh --scp user@host:/data/EMBER2024-corrected-full

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_DATA_DIR="$PROJECT_DIR/data/local"

usage(){
    cat <<'EOF'
usage: download_ember2024.sh [--http URL | --rsync SRC | --scp SRC] [--dest DIR]

Fetch EMBER2024 dataset into DAFCS/data/local/ by one of the supported transport
methods. After a successful download the script will try to verify the dataset
with DAFCS/scripts/verify_full_dataset.py.

Options:
  --http URL        Download an archive (tar.gz, zip) via HTTP(S) and extract it
  --rsync SRC       Rsync from a source root that contains EMBER2024-* folders
  --scp SRC         Copy via scp a single dataset directory (user@host:/path)
  --dest DIR        Destination parent folder (default: DAFCS/data/local)
  -h, --help        Show this help

Examples:
  ./download_ember2024.sh --http https://files.example.com/EMBER2024-full.tar.gz
  ./download_ember2024.sh --rsync user@host:/shared/ember_root
  ./download_ember2024.sh --scp user@host:/data/EMBER2024-corrected-full
EOF
}

if [[ ${#@} -eq 0 ]]; then
    usage
    exit 1
fi

DEST="$LOCAL_DATA_DIR"
MODE=""
SRC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --http)
            MODE="http"; SRC="$2"; shift 2 ;;
        --rsync)
            MODE="rsync"; SRC="$2"; shift 2 ;;
        --scp)
            MODE="scp"; SRC="$2"; shift 2 ;;
        --dest)
            DEST="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown arg: $1"; usage; exit 2 ;;
    esac
done

mkdir -p "$DEST"

echo "[download_ember2024] destination: $DEST"

if [[ "$MODE" == "http" ]]; then
    echo "[download_ember2024] HTTP download from: $SRC"
    TMP_ARCHIVE="/tmp/ember2024_download.$(date +%s)"
    # prefer aria2c if available for multi-connection downloads
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 16 -s 16 -o "$TMP_ARCHIVE" "$SRC"
    else
        # wget/curl fallback
        if command -v wget >/dev/null 2>&1; then
            wget -O "$TMP_ARCHIVE" "$SRC"
        else
            curl -L -o "$TMP_ARCHIVE" "$SRC"
        fi
    fi

    echo "[download_ember2024] downloaded archive → $TMP_ARCHIVE"
    # Try tar extraction first
    if tar -tzf "$TMP_ARCHIVE" >/dev/null 2>&1; then
        tar -xzf "$TMP_ARCHIVE" -C "$DEST"
    elif unzip -t "$TMP_ARCHIVE" >/dev/null 2>&1; then
        unzip -q "$TMP_ARCHIVE" -d "$DEST"
    else
        echo "[download_ember2024] Unknown archive format. Leaving file at $TMP_ARCHIVE"
        exit 3
    fi

    echo "[download_ember2024] extraction complete"

elif [[ "$MODE" == "rsync" ]]; then
    echo "[download_ember2024] rsync from: $SRC -> $DEST"
    # If the user passed a directory that contains EMBER2024-* children, rsync them
    rsync -avP --progress "$SRC"/ "$DEST"/

elif [[ "$MODE" == "scp" ]]; then
    echo "[download_ember2024] scp from: $SRC -> $DEST"
    scp -r "$SRC" "$DEST/"

else
    echo "No valid mode selected. Use --http, --rsync, or --scp."; usage; exit 2
fi

echo "[download_ember2024] making sure expected dataset structure exists under $DEST"
ls -1 "$DEST" | sed -n '1,200p'

echo "[download_ember2024] running verify script"
python3 "$SCRIPT_DIR/verify_full_dataset.py" --data-root "$DEST" || {
    echo "[download_ember2024] verify failed — try running verify manually for more details.";
    exit 4;
}

echo "[download_ember2024] done. If verify passed, you can run training scripts now."
