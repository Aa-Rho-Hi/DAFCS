# EMBER2024 Data Layout

This repository does not store EMBER2024 dataset files or official model
artifacts in Git.

- `local/` is ignored by Git and is intended for local dataset links or copies.
- `reference/` is ignored by Git to avoid accidentally committing large data files.

If the `EMBER2024-*` directories already exist next to the repository, run:

```bash
bash scripts/link_local_ember2024.sh
python scripts/verify_full_dataset.py
```

If the dataset lives elsewhere on your machine or on a mounted volume, run:

```bash
bash scripts/sync_full_dataset.sh /path/to/source/root
python scripts/verify_full_dataset.py
```

`sync_full_dataset.sh` supports:

- `MODE=rsync` to copy/update a full local working copy under `data/local`
- `MODE=copy` for a direct copy without rsync
- `MODE=symlink` if the source root is already mounted locally

After linking, the training and evaluation scripts will auto-discover:

- `data/local/EMBER2024-corrected-full`
- `data/local/EMBER2024-corrected-canonical`

Large raw files such as multi-gigabyte memmaps, JSONL shards, and official
model artifacts are intentionally kept out of Git history.
