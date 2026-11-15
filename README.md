# Rollout-aligned probabilistic battery SoH forecasting

Research code for a frozen cycle encoder, a causal latent-history Transformer,
and an autoregressive predictor with a variational Bayesian last layer (VBLL).
One sampled output-layer weight is held fixed along each forecast trajectory.
Prediction intervals summarize variation across sampled trajectories.

## Repository and artifact layout

- `code/src/battery_soh/`: data, models, objectives, evaluation and utilities.
- `code/scripts/`: preprocessing, training, evaluation and release commands.
- `code/configs/`: dataset examples and the main experiment configuration.
- `code/tests/`: model, data and experiment-protocol tests.
- `code/docs/REPRODUCIBILITY.md`: experiment settings and evaluation details.
- `code/docs/ARTIFACTS.md`: ZIP layout, verification and manual Hub upload.
- `manuscript/`: paper sources and figures.

Datasets, checkpoints and predictions are available on
[Hugging Face](https://huggingface.co/datasets/fryabcfry/severson-2019-processed).
`release_index.json` lists the experiments, archive contents and download hashes.

## Install

Use Python 3.11 or later and install a PyTorch build compatible with your device.
From this repository:

```bash
python -m pip install -e './code[train,dev]'
cd code
PYTHONPATH=src python -m pytest -q
```

Each experiment archive includes its configuration and run environment.

## Evaluate a released checkpoint

Download the selected experiment ZIP, the matching latent-cache ZIP and the
processed metadata ZIP. Verify their hashes against `SHA256SUMS`, then extract
into new directories:

```bash
cd code
python scripts/package_release.py extract /path/to/experiment.zip \
  --destination /path/to/experiment --sha256 <zip-sha256>
python scripts/evaluate.py \
  --checkpoint /path/to/experiment/run/best.pt \
  --latent-cache /path/to/matching-cache/latents.npz \
  --split-file /path/to/metadata/splits/cell_random_seed_42.jsonl \
  --split test --num-samples 128 --batch-size 32 --num-workers 8 \
  --seed 42 --save-samples --output-dir outputs/reproduced_evaluation
```

For deterministic checkpoints, use `--use-mean --num-samples 1`.
`FILES.json` lists the archive contents. Set the checkpoint, cache and split
paths to your local directories. Archives with `config.portable.json` include
both portable and original training configurations.

## Train and inspect controlled variants

The main controlled protocol uses a fixed cell split and encoder cache,
30 historical cycles, an 80-cycle forecast, latent dimension 256 and predictor
hidden dimensions 512.

```bash
cd code
python scripts/train_predictor.py \
  --config /path/to/experiment/config.portable.json \
  --latent-cache /path/to/matching-cache/latents.npz \
  --split-file /path/to/metadata/splits/cell_random_seed_42.jsonl \
  --run-name my_reproduction
```

The AutoDL runner provides experiment preparation, GPU preflight and automated
best/last evaluation:

```bash
PYTHONPATH=src python scripts/run_revision_fair_rollout.py prepare --variant fair
PYTHONPATH=src python scripts/run_revision_fair_rollout.py preflight --run-dir <prepared-run>
PYTHONPATH=src python scripts/run_revision_fair_rollout.py run --run-dir <prepared-run>
```

Additional deletion variants are `no_ss`, `no_sro` and `neither`.
See [experiment settings](code/docs/REPRODUCIBILITY.md) for the complete protocol.
The runner loads the reference configuration and initializes training with seed 42.

## Data

The processed dataset is hosted at
[Hugging Face](https://huggingface.co/datasets/fryabcfry/severson-2019-processed).
The processed-data revision is `79f40996bf8eec2fb8a3bda391ed3072d853dfd5`.
`severson_processed_20260807.tar.gz` is 3,587,570,430 bytes;
SHA-256: `f45812e423ba81acb3a53b5859ef3e9c1290358eb959c01d0144987159d65e58`.
Download the processed data:

```bash
curl -L --retry 3 -C - -o severson_processed_20260807.tar.gz \
  'https://huggingface.co/datasets/fryabcfry/severson-2019-processed/resolve/79f40996bf8eec2fb8a3bda391ed3072d853dfd5/severson_processed_20260807.tar.gz?download=true'
```

On AutoDL, the provider's `source /etc/network_turbo` may be used in the download
shell to accelerate access.

The processed Severson-derived dataset contains 140 cells and 114,314 cycle
entries. The metadata archive includes cell splits and train-fitted normalization.
The dataset card provides source attribution and dataset terms.

The external Severson processing repository remains a Git submodule:

```bash
git submodule update --init --recursive
```

NASA, CALCE and HUST workflows have their own preprocessing scripts and configurations.
