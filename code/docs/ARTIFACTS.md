# Artifact release

Datasets, checkpoints and predictions are hosted on
[Hugging Face](https://huggingface.co/datasets/fryabcfry/severson-2019-processed).
The release contains ZIP archives, file manifests and checksums.

## Directory layout

```text
README.md                         dataset/release card
release_index.json                archive IDs, status, sizes, SHA-256
SHA256SUMS
inventory.json                    experiment inventory
code/                             code snapshot ZIP
metadata/                         splits, normalization, processed metadata
latents/                          encoder latent caches
experiments/                      completed experiment ZIPs
legacy/                           experiments with earlier configurations
evaluation_only/                  predictions and metrics; checkpoint unavailable
```

Each experiment ZIP contains its original `run/` files, `ARTIFACT.json`,
`FILES.json`, and available training configurations. `FILES.json` records
the hash and size of each payload file. ZIP64 supports large archives.

## Download and extract

Download the selected ZIPs and `SHA256SUMS`, preserving their relative paths:

```bash
sha256sum -c SHA256SUMS
python scripts/package_release.py extract experiments/<id>.zip \
  --destination extracted/<id> --sha256 <hash-from-release-index>
```

Choose a new extraction directory. The extraction command validates archive
paths and file hashes. Source paths in manifests record the original run location.

## Build archives

```bash
python scripts/package_release.py build --plan release_plan.json \
  --output /root/autodl-fs/BatteryProject/hf_release_YYYYMMDD --watch
```

Recipes list source files and experiment completion states. The builder waits
for each run to finish, creates its ZIP and manifest, and verifies CRCs and
SHA-256 hashes. Use a new release directory when changing archive recipes.

## Upload archives

Upload the ZIPs, `*.manifest.json` sidecars and top-level metadata to the dataset
repository, preserving relative paths. The processed-data archive and experiment
ZIPs share the same repository.
