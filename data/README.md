# EMBER2024 Data Layout

This repository keeps only the GitHub-safe EMBER2024 subset in Git.

- `reference/` contains tracked metadata and smaller official model artifacts.
- `local/` is ignored by Git and is intended for symlinks to the full local dataset.

To link the existing sibling EMBER2024 directories into this repository:

```bash
bash scripts/link_local_ember2024.sh
```

After linking, the training and evaluation scripts will auto-discover:

- `data/local/EMBER2024-corrected-full`
- `data/local/EMBER2024-corrected-canonical`

Large raw files such as multi-gigabyte memmaps and JSONL shards are intentionally
kept out of GitHub history.
