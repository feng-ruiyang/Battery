# Artifact release and manual Hugging Face upload

The code repository and binary artifact release are separate. No upload is
performed by the packaging command. The owner can download the generated ZIPs,
copy them to their cloud drive, and upload them to the intended Hugging Face repo.

## Upload directory

```text
README.md                         dataset/release card
release_index.json                archive IDs, status, sizes, SHA-256
SHA256SUMS
inventory.json                    checkpoint provenance and exclusions
code/                             exact code snapshot ZIP
metadata/                         splits, normalization, processed metadata
latents/                          one ZIP per encoder/cache identity
experiments/                      controlled complete run ZIPs
legacy/                           completed historical, nonmatched runs
evaluation_only/                  predictions without recovered checkpoints
```

Upload the archive files and top-level metadata together, preserving relative
paths. `*.manifest.json` sidecars are small and useful for verifying downloads.
Files ending in `.partial`, `.updating` or `.lock` are not release artifacts.
Keep the existing processed-data revision until its manifest has been compared
with this release. The target Hub repository is an explicit owner-supplied value;
it is never guessed or silently created.

Each experiment ZIP contains its original `run/` files, `ARTIFACT.json`,
`FILES.json`, and, where available, original/portable training configurations.
`FILES.json` records the hash and size of every other member. ZIP64 is enabled
for large datasets. Checkpoints use ZIP containers rather than tar archives.

## Package and verify

```bash
python scripts/package_release.py build --plan release_plan.json \
  --output /root/autodl-fs/BatteryProject/hf_release_YYYYMMDD --watch
```

Recipes explicitly list source paths and completion-state prerequisites.
Existing archives are immutable: a changed recipe requires a new release version.
The builder verifies ZIP CRCs, computes SHA-256 hashes and atomically publishes
completed ZIPs. Active checkpoints are not packaged until the pipeline completes.
No source files are deleted or overwritten.

```bash
sha256sum -c SHA256SUMS
python scripts/package_release.py extract experiments/<id>.zip \
  --destination extracted/<id> --sha256 <hash-from-release-index>
```

Extraction refuses unsafe paths, symlinks, unknown members, checksum mismatches
and overwriting an existing destination. Historical source paths in manifests
are provenance records, not paths that a download must recreate.
