# EMBER2024 Project

## Summary

This repository benchmarks malware detection and family classification on the
EMBER2024 dataset using both official-style LightGBM baselines and a custom
PyTorch pipeline based on contrastive pre-training, multi-task fine-tuning, and
prototypical inference for long-tail families.

Dataset used:
- EMBER2024
- canonical local memmaps built from the official raw JSONL shards
- train `(2626000, 2568)`, test `(605929, 2568)`, challenge `(6315, 2568)`

## What's In This Repo

- data loaders for the cleaned canonical EMBER2024 train, test, and challenge splits
- official-style LightGBM baseline scripts for malware detection comparison
- PyTorch models for contrastive pre-training, multi-task fine-tuning, and prototype-based family inference
- evaluation utilities for ROC-AUC, PR-AUC, family metrics, and challenge-set comparisons
- a notebook for report-ready plots, comparison tables, and embedding visualisation

## Current Status

- canonical local EMBER2024 data preparation and verification are working
- the evaluation pipeline has been repaired for the corrected test and challenge methodology
- LightGBM CPU sanity runs are working on the cleaned memmaps
- the deep-learning pipeline is scaffolded and ready for longer training runs

## Quick Start

```bash
cd /Users/roheeeee/Documents/DACS/ember2024-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_lgbm_memmap.py --n-estimators 300
```

This runs a quick LightGBM sanity benchmark against the cleaned EMBER2024 local dataset.

PyTorch project for the EMBER2024 malware benchmark with three stages:

1. Supervised contrastive pre-training on malicious files with confident family labels.
2. Multi-task fine-tuning for malware detection and family classification.
3. Prototypical inference for long-tail family prediction.

The repository also includes:

- A LightGBM baseline runner for comparison.
- A mock-data generator so the full pipeline can be exercised without the real dataset.
- An analysis notebook that reads saved results and raw score files.

## Layout

```text
ember2024-project/
├── README.md
├── data/
├── requirements.txt
├── configs/
│   ├── default.yaml
│   └── smoke.yaml
├── src/
├── scripts/
└── notebooks/
```

## Dataset placement

This repository now keeps the GitHub-safe EMBER2024 subset under `data/reference`
and expects the full local dataset under `data/local` when you want to train or
rerun the full benchmark.

Link the existing sibling dataset directories into the repo with:

```bash
bash scripts/link_local_ember2024.sh
```

After that, the run scripts auto-discover `data/local/EMBER2024-corrected-full`
or `data/local/EMBER2024-corrected-canonical` without needing an explicit path.

Files that exceed practical GitHub limits, including multi-gigabyte memmaps,
large JSONL shards, and `EMBER2024_family.model`, are intentionally left out of
Git history.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `lightgbm` is only required for the baseline scripts.
- `ember` is optional if you are loading directly from `X_*.dat` and `y_*.dat`, but recommended.

## Expected data files

The loaders support direct binary arrays and JSONL metadata:

```text
<data_dir>/
├── X_train.dat
├── y_train.dat
├── X_test.dat
├── y_test.dat
├── X_challenge.dat
├── y_challenge.dat
├── train_metadata.jsonl
├── test_metadata.jsonl
└── challenge_metadata.jsonl
```

The metadata loader expects fields like:

- `sha256`
- `label`
- `file_type`
- `family`
- `family_confidence`
- `week`

## Quick smoke test without EMBER2024

Generate a small synthetic dataset:

```bash
python scripts/generate_mock_data.py --output_dir /tmp/ember2024_mock --small
```

Run the smoke pipeline on CPU:

```bash
python -m src.contrastive --config configs/smoke.yaml --data_dir /tmp/ember2024_mock
python -m src.multitask --config configs/smoke.yaml --data_dir /tmp/ember2024_mock --pretrained checkpoints/stage1_smoke/best_model.pt
python -m src.prototypical --config configs/smoke.yaml --data_dir /tmp/ember2024_mock --encoder_ckpt checkpoints/stage1_smoke/best_model.pt
python -m src.evaluate --config configs/smoke.yaml --data_dir /tmp/ember2024_mock --checkpoint checkpoints/stage2_smoke/best_model.pt --proto_checkpoint checkpoints/stage3_smoke/prototypes.npz
```

Outputs land in:

- `checkpoints/stage1_smoke`
- `checkpoints/stage2_smoke`
- `checkpoints/stage3_smoke`
- `results/smoke`

## Real-data workflow

Set the data directory:

```bash
export EMBER2024_DIR=/path/to/ember2024
```

Run the baseline:

```bash
bash scripts/run_baseline.sh
```

Run Stage 1:

```bash
bash scripts/run_stage1.sh
```

Run Stage 2:

```bash
PRETRAINED=checkpoints/stage1/best_model.pt bash scripts/run_stage2.sh
```

Run Stage 3:

```bash
ENCODER_CKPT=checkpoints/stage1/best_model.pt bash scripts/run_stage3.sh
```

Run the full pipeline:

```bash
bash scripts/run_all.sh
```

`run_all.sh` skips the LightGBM baseline automatically if `lightgbm` is not installed.

## Important implementation details

- Validation uses the last `val_weeks` of the training period.
- Family labels on test and challenge sets are projected onto the training label space.
- Stage 1 uses episodic `N classes x K samples` batches for supervised contrastive learning.
- Stage 2 applies focal loss for detection and class-balanced focal loss for family classification.
- Stage 3 computes mean encoder embeddings as class prototypes and uses cosine similarity by default.
- Evaluation writes both `results.json` and `raw_scores.npz` for the notebook plots.

## Notebook

After running evaluation, open:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook reads:

- `results/results.json`
- `results/lgbm_baseline_results.json`
- `results/raw_scores.npz`

## Common issues

If you cannot download the real dataset:

- Use `scripts/generate_mock_data.py` to validate the entire code path locally.
- Keep `configs/smoke.yaml` for CPU-only sanity checks before moving to the real dataset.

If the baseline step fails:

- Install `lightgbm`, or skip the baseline and run the PyTorch stages directly.

If the dataset is too large for RAM:

- Keep `data.use_memmap: true` in `configs/default.yaml`.
- Reduce `num_workers` if disk contention becomes the bottleneck.
