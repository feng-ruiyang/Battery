# Battery SoH Code Workspace

This directory contains the implementation workspace for the BNN-Transformer battery SoH project.

The current code supports Severson data ingestion, canonicalization, cycle encoder
pretraining, latent caching, and latent-window SoH predictor training.

## Setup

```bash
conda env create -f environment.yaml
conda activate battery-soh
pip install -e .[dev]
```

Install PyTorch separately if the server CUDA version requires a specific build.

## Severson Preprocess

After downloading the raw Severson data, run:

```bash
python scripts/preprocess.py \
  --dataset severson \
  --raw-dir data/raw/severson \
  --output-dir data/processed/severson
```

The parser writes:

- `cells.jsonl`: one metadata row per cell.
- `cycles.jsonl`: one row per cell-cycle pair.
- `arrays/<cell_id>/cycle_<index>.npz`: variable-length cycle arrays.
  The `.npz` files can contain fields such as `I`, `V`, `T`, `Qc`, `Qd`, `Qdlin`, `Tdlin`, `dQdV`, and `t`.

## Model Dry Runs

The model code can be shape-checked before the real latent cache exists:

```bash
python scripts/train_predictor.py --config configs/train_vbll_dynamics.yaml --dry-run
python scripts/pretrain_encoder.py --config configs/pretrain_encoder.yaml --dry-run
```

## Training Data Flow

After preprocessing, train the cycle autoencoder, cache per-cycle latents, and train
the SoH trajectory predictor:

```bash
python scripts/pretrain_encoder.py --config configs/pretrain_encoder.yaml

python scripts/cache_latents.py \
  --config configs/pretrain_encoder.yaml \
  --checkpoint outputs/severson_cycle_autoencoder/runs/<run_name>/best.pt

python scripts/train_predictor.py --config configs/train_latent_tf_vbll.yaml
```

The latent cache defaults to `data/processed/severson/latents.npz`. Predictor
training reads that cache, applies the cell-level split, builds sliding latent
windows, and saves `last.pt` / `best.pt` under the configured experiment output
directory.

By default, training runs are written to timestamped subdirectories:

```text
outputs/<experiment>/runs/YYYYMMDD_HHMMSS/
```

Set `experiment.timestamped_output: false` to restore fixed-path checkpoint
behavior.

## Logging

Training scripts write metrics to `metrics.csv` in each experiment output
directory when `logging.backend: csv`. To use Weights & Biases, install the
training extras and set:

```yaml
logging:
  backend: wandb
  project: battery-soh-bnn
  offline: true
```
