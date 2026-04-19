# EMBER2024 Project

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
├── requirements.txt
├── configs/
│   ├── default.yaml
│   └── smoke.yaml
├── src/
├── scripts/
└── notebooks/
```

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
