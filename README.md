# Rollout-aligned probabilistic battery SoH forecasting

Research code for a frozen cycle encoder, a causal latent-history Transformer,
and an autoregressive predictor with a variational Bayesian last layer (VBLL).
One sampled output-layer weight is held fixed along each forecast trajectory.
The released intervals represent output-layer-posterior trajectory variation;
no additional observation noise is sampled during rollout.

## Repository and artifact layout

- `code/src/battery_soh/`: data, models, objectives, evaluation and utilities.
- `code/scripts/`: preprocessing, training, evaluation and release commands.
- `code/configs/`: dataset examples and an audited main-run reference.
- `code/tests/`: model, data and experiment-protocol tests.
- `code/docs/REPRODUCIBILITY.md`: controlled experiments and their provenance.
- `code/docs/ARTIFACTS.md`: ZIP layout, verification and manual Hub upload.
- `manuscript/`: historical writing material. Its tables are not a release index.

Large datasets, checkpoints and prediction tensors belong in the artifact
release, not Git history. Use `release_index.json` and each ZIP's `ARTIFACT.json`
to identify the experiment and its completeness. Do not select a checkpoint
using its filename or run-directory name alone.

## Install

Use Python 3.11 or later and install a PyTorch build compatible with your device.
From this repository:

```bash
python -m pip install -e './code[train,dev]'
cd code
PYTHONPATH=src python -m pytest -q
```

Reference run environments are captured inside the corresponding artifact.
GPU profiling and exact reproduction require matching the recorded software,
hardware, input-cache checksum and evaluation settings.

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
ZIPs preserve original run-relative evaluation paths; consult `FILES.json`.
A checkpoint's embedded paths may refer to the original server. Override those
paths explicitly as above. `config.portable.json`, where supplied, changes only
storage locations and logging; the original configuration is also retained.

## Train and inspect controlled variants

The main controlled protocol uses a fixed cell split and encoder cache,
30 historical cycles, an 80-cycle forecast, latent dimension 256 and predictor
hidden dimensions 512. Do not substitute the older 128-dimensional example
config when reproducing the scale256 experiments.

```bash
cd code
python scripts/train_predictor.py \
  --config /path/to/experiment/config.portable.json \
  --latent-cache /path/to/matching-cache/latents.npz \
  --split-file /path/to/metadata/splits/cell_random_seed_42.jsonl \
  --run-name my_reproduction
```

The AutoDL runner provides guarded preparation, GPU preflight and automated
best/last evaluation:

```bash
PYTHONPATH=src python scripts/run_revision_fair_rollout.py prepare --variant fair
PYTHONPATH=src python scripts/run_revision_fair_rollout.py preflight --run-dir <prepared-run>
PYTHONPATH=src python scripts/run_revision_fair_rollout.py run --run-dir <prepared-run>
```

Additional deletion variants are `no_ss`, `no_sro` and `neither`.
See [the reproducibility protocol](code/docs/REPRODUCIBILITY.md) before using
these commands. The audited reference is used for configuration verification;
new training does not initialize from its trained weights.

## Data provenance

The existing processed dataset is hosted at
[Hugging Face](https://huggingface.co/datasets/fryabcfry/severson-2019-processed).
Its audited revision is `79f40996bf8eec2fb8a3bda391ed3072d853dfd5`.
The existing `severson_processed_20260807.tar.gz` is 3,587,570,430 bytes;
SHA-256: `f45812e423ba81acb3a53b5859ef3e9c1290358eb959c01d0144987159d65e58`.
The server copy matches this hash. Download the existing file directly instead
of uploading a duplicate:

```bash
curl -L --retry 3 -C - -o severson_processed_20260807.tar.gz \
  'https://huggingface.co/datasets/fryabcfry/severson-2019-processed/resolve/79f40996bf8eec2fb8a3bda391ed3072d853dfd5/severson_processed_20260807.tar.gz?download=true'
```

On AutoDL, the provider's `source /etc/network_turbo` may be used in the download
shell when ordinary outbound access is unavailable. No credentials are embedded
in the download URL.

The main release concerns a processed Severson-derived corpus. It contains 140
processed cells and 114,314 cycle entries; this is not presented as the exact
124-cell subset of the original paper. Splits and train-fitted normalization
must travel with the processed data. Dataset cards and manifests document the
source version, paths and hashes. Preserve the upstream dataset attribution and
terms; this repository does not introduce a new license for third-party data.

The external Severson processing repository remains a Git submodule:

```bash
git submodule update --init --recursive
```

NASA, CALCE and HUST utilities are retained as separate dataset workflows.
Their configurations are not interchangeable with the controlled Severson runs.
