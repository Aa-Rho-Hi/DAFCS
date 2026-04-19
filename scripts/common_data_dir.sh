#!/usr/bin/env bash
set -euo pipefail

resolve_ember2024_dir() {
    local project_dir="$1"
    local explicit_dir="${2:-}"

    if [[ -n "${EMBER2024_DIR:-}" ]]; then
        printf '%s\n' "$EMBER2024_DIR"
        return 0
    fi

    if [[ -n "$explicit_dir" ]]; then
        printf '%s\n' "$explicit_dir"
        return 0
    fi

    local candidates=(
        "$project_dir/data/local/EMBER2024-corrected-full"
        "$project_dir/data/local/EMBER2024-corrected-canonical"
        "$project_dir/data/local/ember2024_mock"
        "$project_dir/../EMBER2024-corrected-full"
        "$project_dir/../EMBER2024-corrected-canonical"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

print_ember2024_dir_help() {
    cat <<'EOF'
ERROR: No EMBER2024 data directory found.

Use one of:
  1. export EMBER2024_DIR=/path/to/ember2024
  2. bash scripts/link_local_ember2024.sh
  3. pass the data directory as the first argument
EOF
}
