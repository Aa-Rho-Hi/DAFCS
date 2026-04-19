# Full EMBER2024 Inventory

This repository cannot store the full EMBER2024 workspace in normal Git history.
The complete local workspace currently consists of these top-level directories:

- `EMBER2024-corrected-canonical` about `14G`
- `EMBER2024-corrected-full` about `63G`
- `EMBER2024-artifacts` about `29G`
- `EMBER2024-evaldata` about `14G`
- `EMBER2024-full-local` about `97G`
- `EMBER2024-train-artifacts` about `83G`

Approximate total footprint: more than `300G`.

Expected repetitive file counts for completeness checks:

- `EMBER2024-train-artifacts`: `312` `*_train.jsonl` files
- `EMBER2024-full-local`: `312` `*_train.jsonl` files
- `EMBER2024-artifacts`: `72` `*_test.jsonl` files
- `EMBER2024-artifacts`: `64` `*_challenge_malicious.jsonl` files
- `EMBER2024-artifacts`: `14` `.model` files

Use `bash scripts/sync_full_dataset.sh` to materialize the full dataset from a
shared root, then run `python scripts/verify_full_dataset.py`.
