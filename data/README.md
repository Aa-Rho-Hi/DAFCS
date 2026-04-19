# EMBER2024 Data Layout

This repository keeps only the GitHub-safe EMBER2024 subset in Git.

- `reference/` contains tracked metadata and smaller official model artifacts.
- `local/` is ignored by Git and is intended for symlinks to the full local dataset.

If you need the complete EMBER2024 workspace, use shared
storage root containing the `EMBER2024-*` directories and run:

```bash
bash scripts/sync_full_dataset.sh /path/to/shared/root
python scripts/verify_full_dataset.py
```

`sync_full_dataset.sh` supports:

- `MODE=rsync` to copy/update a full local working copy under `data/local`
- `MODE=copy` for a direct copy without rsync
- `MODE=symlink` if the shared root is already mounted locally

To link the existing sibling EMBER2024 directories into this repository:

```bash
bash scripts/link_local_ember2024.sh
```

After linking, the training and evaluation scripts will auto-discover:

- `data/local/EMBER2024-corrected-full`
- `data/local/EMBER2024-corrected-canonical`

Large raw files such as multi-gigabyte memmaps and JSONL shards are intentionally
kept out of GitHub history.
