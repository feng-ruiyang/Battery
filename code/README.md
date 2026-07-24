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

## HUST Zero-Shot Preprocess

BatteryLife standardized HUST pickle files can be converted to the same
canonical representation without training on HUST:

```bash
PYTHONPATH=src python scripts/preprocess.py \
  --dataset hust \
  --raw-dir ~/data/hust/raw \
  --output-dir ~/data/hust/processed
```

The parser preserves the missing HUST temperature channel and places every cell
in `splits/all_external_test.jsonl`. Create two caches with the frozen Severson
encoder. The first retains the existing missing-channel behavior (`T=0`):

```bash
PYTHONPATH=src python scripts/cache_latents.py \
  --config configs/pretrain_encoder_server.yaml \
  --checkpoint outputs/severson_cycle_autoencoder/runs/<encoder_run>/best.pt \
  --processed-dir ~/data/hust/processed \
  --output ~/data/hust/latents_t_zero.npz
```

The second fills only the wholly missing temperature channel with the Severson
training mean reported by `audit_hust_compatibility.py`:

```bash
PYTHONPATH=src python scripts/cache_latents.py \
  --config configs/pretrain_encoder_server.yaml \
  --checkpoint outputs/severson_cycle_autoencoder/runs/<encoder_run>/best.pt \
  --processed-dir ~/data/hust/processed \
  --channel-fill T=33.698144901739695 \
  --output ~/data/hust/latents_t_severson_mean.npz
```

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

## Encoder Checkpoint Matrix

The deterministic predictor uses the same latent-state transformer and
autoregressive trajectory decoder as the VBLL predictor, but replaces the VBLL
output layer with a deterministic linear layer and trains the trajectory with
masked MSE. Its default config is:

```bash
python scripts/train_predictor.py \
  --config configs/train_latent_tf_deterministic.yaml
```

To compare encoder checkpoint selection independently from predictor type,
first cache `best.pt` and `last.pt` separately:

```bash
python scripts/cache_latents.py \
  --config configs/pretrain_encoder.yaml \
  --checkpoint outputs/severson_cycle_autoencoder/runs/<encoder_run>/best.pt \
  --output data/processed/severson/latent_cache/<encoder_run>/best/latents.npz

python scripts/cache_latents.py \
  --config configs/pretrain_encoder.yaml \
  --checkpoint outputs/severson_cycle_autoencoder/runs/<encoder_run>/last.pt \
  --output data/processed/severson/latent_cache/<encoder_run>/last/latents.npz
```

Run each predictor config against each cache with explicit run names:

```bash
python scripts/train_predictor.py \
  --config configs/train_latent_tf_vbll.yaml \
  --latent-cache data/processed/severson/latent_cache/<encoder_run>/best/latents.npz \
  --run-name <encoder_run>_best_vbll

python scripts/train_predictor.py \
  --config configs/train_latent_tf_deterministic.yaml \
  --latent-cache data/processed/severson/latent_cache/<encoder_run>/best/latents.npz \
  --run-name <encoder_run>_best_deterministic
```

Repeat the two commands with the `last` cache and `<encoder_run>_last`. Each
predictor run still saves its own `best.pt` and `last.pt`; use predictor
`best.pt` for the primary test-set comparison.

The complete four-run matrix can also be executed sequentially:

```bash
python scripts/run_predictor_matrix.py \
  --encoder-config configs/pretrain_encoder.yaml \
  --encoder-run-dir outputs/severson_cycle_autoencoder/runs/<encoder_run> \
  --processed-dir data/processed/severson \
  --split-file data/processed/severson/splits/cell_random_seed_42.jsonl \
  --cache-root data/processed/severson/latent_cache/<encoder_run>
```

Pass `--reuse-cache` when resuming after both latent cache files have already
been generated. The script stops immediately if caching or any predictor run
fails.

## Free-Running Rollout Evaluation

Evaluate a predictor checkpoint without teacher forcing:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/severson_latent_tf_vbll/runs/<run_name>/best.pt \
  --split test \
  --num-samples 32 \
  --batch-size 8
```

For VBLL, one last-layer weight sample is held fixed along each complete
autoregressive trajectory. Deterministic checkpoints produce one mean
trajectory. The evaluator writes:

- overall and macro-cell MAE/RMSE;
- MAE/RMSE at horizons 1, 5, 10, 20, 50, and 80;
- VBLL empirical CRPS, interval coverage, interval width, and predictive
  standard deviation;
- `metrics.json` and compressed rollout summaries in `predictions.npz`.

Use `--max-windows 128` for a smoke test before evaluating every test window.
Add `--save-samples` only when the full sample tensor is needed because it can
be large.

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
