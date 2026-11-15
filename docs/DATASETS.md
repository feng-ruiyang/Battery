# Dataset workflows

This directory contains the implementation workspace for the BNN-Transformer battery SoH project.

The current code supports Severson, HUST, NASA, and CALCE-CS2 data ingestion, canonicalization,
cycle encoder pretraining, latent caching, and latent-window SoH predictor training.

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

## NASA Zero-Shot Preprocess

The parser reads MAT files from `~/data/nasa/raw`. If that directory has no MAT
files, it falls back to already extracted MAT files under
`~/data/nasa/extracted`; ZIP files in that directory are only inventoried. Then run:

```bash
PYTHONPATH=src python scripts/preprocess.py \
  --config configs/preprocess_nasa.yaml
```

All charge, discharge, and impedance operations are indexed in
`operations.jsonl`, with their arrays under `operations/<cell_id>/`. Only
discharge operations populate `cycles.jsonl` and therefore enter the current
encoder and rollout experiment. For each cell, SoH is defined as:

```text
SoH_k = discharge Capacity_k / 2 Ah nominal capacity
```

Create NASA latents with the frozen Severson encoder:

```bash
PYTHONPATH=src python scripts/cache_latents.py \
  --config configs/pretrain_encoder_server.yaml \
  --checkpoint outputs/severson_cycle_autoencoder/runs/<encoder_run>/best.pt \
  --processed-dir ~/data/nasa/processed \
  --output ~/data/nasa/processed/latents.npz
```

Run a small rollout check before the complete 128-trajectory evaluation:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --checkpoint outputs/severson_latent_tf_vbll/runs/<predictor_run>/best.pt \
  --latent-cache ~/data/nasa/processed/latents.npz \
  --split-file ~/data/nasa/processed/splits/all_external_test.jsonl \
  --split test \
  --num-samples 128 \
  --batch-size 2 \
  --max-windows 32 \
  --output-dir outputs/nasa_rollout_smoke
```

Remove `--max-windows 32` and change the output directory for the complete run.
Because the experiment encodes discharge segments only while the Severson encoder
was trained on complete cycles, interpret it as a deliberately strong domain-shift
test.

To train the complete model from random initialization on NASA discharge cycles,
first create the cell-level split if preprocessing was completed by an older
parser version:

```bash
PYTHONPATH=src python scripts/create_cell_split.py \
  --processed-dir ~/data/nasa/processed \
  --seed 42
```

Then train the NASA autoencoder, cache all cell latents with its best encoder,
and train the NASA predictor:

```bash
PYTHONPATH=src python scripts/fit_channel_normalizer.py \
  --processed-dir ~/data/nasa/processed \
  --split-file ~/data/nasa/processed/splits/cell_random_seed_42.jsonl \
  --split train \
  --channels V I T \
  --sequence-length 1000 \
  --output ~/data/nasa/processed/normalization/train_vit_seed_42.json

PYTHONPATH=src python scripts/pretrain_encoder.py \
  --config configs/pretrain_encoder_nasa.yaml

PYTHONPATH=src python scripts/cache_latents.py \
  --config configs/pretrain_encoder_nasa.yaml \
  --checkpoint outputs/nasa_discharge_cycle_autoencoder/runs/<encoder_run>/best.pt

PYTHONPATH=src python scripts/train_predictor.py \
  --config configs/train_latent_tf_vbll_nasa.yaml
```

Resume a NASA autoencoder run from its best checkpoint while preserving the
existing run directory:

```bash
PYTHONPATH=src python scripts/pretrain_encoder.py \
  --config configs/pretrain_encoder_nasa.yaml \
  --resume outputs/nasa_discharge_cycle_autoencoder/runs/<encoder_run>/best.pt
```

## CALCE-CS2 Cross-Protocol Preprocess

The CALCE parser currently targets the cleaner CS2 full-cycle subset:

```text
CS2_33, CS2_34, CS2_35, CS2_36, CS2_37, CS2_38
```

Download the ZIP files under `~/data/calce/raw/cs2`:

```bash
mkdir -p ~/data/calce/raw/cs2
cd ~/data/calce/raw/cs2

for CELL in CS2_33 CS2_34 CS2_35 CS2_36 CS2_37 CS2_38; do
  wget -c --tries=5 --timeout=60 \
    "https://web.calce.umd.edu/batteries/data/${CELL}.zip" \
    -O "${CELL}.zip"
done
```

Then preprocess:

```bash
PYTHONPATH=src python -u scripts/preprocess.py \
  --config configs/preprocess_calce.yaml
```

CALCE CS2 Excel files store cumulative `Charge_Capacity(Ah)` and
`Discharge_Capacity(Ah)` within each workbook. The parser computes per-cycle
capacity as `max - min` inside each `Cycle_Index`, then defines SoH as:

```text
SoH_k = per-cycle discharge capacity_k / 1.1 Ah nominal CS2 capacity
```

The parser filters incomplete boundary cycles and local diagnostic outliers. It
writes `cycle_filter_diagnostics.jsonl` with the drop reason for every excluded
cycle. The current first-pass cross-protocol split is:

```text
train: CS2_34, CS2_35, CS2_36, CS2_37
val:   CS2_33
test:  CS2_38
```

Fit train-only normalization over the two available channels:

```bash
PYTHONPATH=src python -u scripts/fit_channel_normalizer.py \
  --processed-dir ~/data/calce/processed \
  --split-file ~/data/calce/processed/splits/cell_random_seed_42.jsonl \
  --split train \
  --channels V I \
  --sequence-length 1000 \
  --output ~/data/calce/processed/normalization/train_vi_seed_42.json
```

Train a CALCE autoencoder by copying the NASA encoder config and changing
paths, channels, and `model.input_dim` to 2:

```bash
cp configs/pretrain_encoder_nasa.yaml configs/pretrain_encoder_calce.yaml
```

Set:

```yaml
experiment:
  name: calce_cs2_cycle_autoencoder
  output_dir: outputs/calce_cs2_cycle_autoencoder
data:
  processed_dir: ~/data/calce/processed
  split_file: ~/data/calce/processed/splits/cell_random_seed_42.jsonl
  latent_cache: ~/data/calce/processed/latents.npz
  normalizer: ~/data/calce/processed/normalization/train_vi_seed_42.json
  channels: [V, I]
  sequence_length: 1000
model:
  input_dim: 2
```

Then run:

```bash
PYTHONPATH=src python -u scripts/pretrain_encoder.py \
  --config configs/pretrain_encoder_calce.yaml

PYTHONPATH=src python -u scripts/cache_latents.py \
  --config configs/pretrain_encoder_calce.yaml \
  --checkpoint outputs/calce_cs2_cycle_autoencoder/runs/<encoder_run>/best.pt \
  --output ~/data/calce/processed/latent_cache/<encoder_run>_best/latents.npz
```

The first completed CALCE-CS2 encoder run produced 5430 finite cycle latents
with shape `(5430, 128)` and best validation reconstruction MSE around `0.0305`.

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
