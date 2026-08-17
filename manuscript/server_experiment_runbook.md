# Server Experiment Runbook

This note records the server-side workflow for AutoDL experiments. It is meant
to prevent path, cache, and configuration mistakes when running multiple
ablation jobs in parallel.

## 1. Path Conventions

There are two common layouts.

Old/main 6000D instance:

```bash
CODE=/root/Battery/code
PROCESSED=/root/data/severson-2019/processed
```

New AutoDL instances using the fast data disk:

```bash
WORK_ROOT=/root/autodl-tmp
CODE=/root/autodl-tmp/BatteryProject/Battery/code
PROCESSED=/root/autodl-tmp/data/severson-2019/processed
```

Shared persistent disk:

```bash
FS_ROOT=/root/autodl-fs/BatteryProject
```

When unsure, locate paths first:

```bash
find /root /root/autodl-tmp -maxdepth 6 -type d -path '*/Battery/code' 2>/dev/null
find /root /root/autodl-tmp -maxdepth 6 -type f -name cycles.jsonl 2>/dev/null | grep 'severson-2019'
find /root /root/autodl-tmp /root/autodl-fs -type f -name latents.npz 2>/dev/null
```

## 2. Environment Bootstrap

```bash
cd "$CODE"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

git pull --ff-only
git log -3 --oneline

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

nvidia-smi
```

If GitHub SSH is not configured:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "autodl" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
git remote set-url origin git@github.com:feng-ruiyang/Battery.git
```

## 3. Data Checks

Expected Severson/MIT processed data:

```text
cells.jsonl: 140 lines
cycles.jsonl: 114314 lines
```

Check:

```bash
wc -l "$PROCESSED/cells.jsonl" "$PROCESSED/cycles.jsonl"
find "$PROCESSED/arrays" -name '*.npz' | wc -l
```

If `processed/` is missing because the copied folder is one level too shallow:

```bash
if [ -f "$(dirname "$PROCESSED")/cycles.jsonl" ] && [ ! -f "$PROCESSED/cycles.jsonl" ]; then
  cd "$(dirname "$PROCESSED")"
  mkdir -p processed
  for item in arrays cells.jsonl cycles.jsonl latent_cache normalization scan_summary.jsonl splits; do
    [ -e "$item" ] && mv "$item" processed/
  done
  cd "$CODE"
fi
```

If split is missing:

```bash
SPLIT="$PROCESSED/splits/cell_random_seed_42.jsonl"
mkdir -p "$PROCESSED/splits"

PYTHONPATH=src python -u scripts/create_cell_split.py \
  --processed-dir "$PROCESSED" \
  --output "$SPLIT" \
  --seed 42 \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

Do not fit the channel normalizer just to train predictor. Predictor only needs
the latent cache and split file. The normalizer is needed for encoder/cache
generation.

## 4. Copy Processed Data Between Instances

From shared disk to a new fast disk instance:

```bash
mkdir -p /root/autodl-tmp/BatteryProject /root/autodl-tmp/data

rsync -a --info=progress2 \
  /root/autodl-fs/BatteryProject/Battery/ \
  /root/autodl-tmp/BatteryProject/Battery/

rsync -a --info=progress2 \
  /root/autodl-fs/BatteryProject/data/severson-2019 \
  /root/autodl-tmp/data/
```

From an instance back to shared disk:

```bash
mkdir -p /root/autodl-fs/BatteryProject/data

rsync -a --info=progress2 \
  "$PROCESSED/../" \
  /root/autodl-fs/BatteryProject/data/severson-2019/
```

## 5. Fast Encoder Cache

After a fast scale256 encoder finishes, cache latents with the matching
pretrain encoder config. Do not use a predictor config with `cache_latents.py`.

Find the run and config:

```bash
cd "$CODE"

find outputs/severson_cycle_autoencoder/runs -maxdepth 1 -type d -name 'severson_scale256*encoder*' -print | sort
find outputs/generated_configs configs -type f \( \
  -name 'pretrain_encoder*scale256*.yaml' -o \
  -name 'pretrain_encoder_severson*.yaml' \
\) -print | sort
```

Cache example:

```bash
ENC_RUN_DIR=outputs/severson_cycle_autoencoder/runs/severson_scale256_fast_encoder_YYYYMMDD_HHMMSS
ENC_RUN_NAME=$(basename "$ENC_RUN_DIR")
ENC_CKPT="$ENC_RUN_DIR/best.pt"
ENC_CFG=outputs/generated_configs/pretrain_encoder_severson_scale256_fast_YYYYMMDD_HHMMSS.yaml

CACHE_DIR="$PROCESSED/latent_cache/${ENC_RUN_NAME}_best"
CACHE="$CACHE_DIR/latents.npz"
mkdir -p "$CACHE_DIR"

nohup env PYTHONPATH=src python -u scripts/cache_latents.py \
  --config "$ENC_CFG" \
  --checkpoint "$ENC_CKPT" \
  --output "$CACHE" \
  > /root/autodl-tmp/cache_fast_encoder.out 2>&1 &
```

Monitor:

```bash
tail -f /root/autodl-tmp/cache_fast_encoder.out
pgrep -af "cache_latents.py" || true
```

Sync fast cache to shared disk:

```bash
mkdir -p /root/autodl-fs/BatteryProject/scale256_fast_artifacts/latent_cache

rsync -a --info=progress2 \
  "$CACHE_DIR/" \
  "/root/autodl-fs/BatteryProject/scale256_fast_artifacts/latent_cache/${ENC_RUN_NAME}_best/"
```

## 6. Predictor Experiment Definitions

All final comparison runs should share:

```text
same encoder checkpoint/cache
same cell split
same model size: latent_dim=256, history_dim=512, decoder_dim=512
same VBLL prior: prior variance = 0.5, when applicable
same eval: test split, free rollout, batch size 32
```

The four runs are:

```text
Deterministic-TF:
  model.kind = latent_tf_deterministic
  training.objective = mse
  scheduled_sampling.enabled = false
  stochastic_rollout.enabled = false
  eval uses --use-mean, --num-samples 1

VBLL-TF-NLL:
  model.kind = latent_tf_vbll
  training.objective = nll
  scheduled_sampling.enabled = false
  stochastic_rollout.enabled = false
  eval uses --num-samples 128 --save-samples

VBLL-TF-CRPS:
  model.kind = latent_tf_vbll
  scheduled_sampling.enabled = false
  stochastic_rollout.enabled = true
  stochastic_rollout.prefix_mode = teacher_forced
  stochastic_rollout.objective = crps
  stochastic_rollout.negative_weight = 0.1

VBLL-Rollout-CRPS:
  model.kind = latent_tf_vbll
  scheduled_sampling.enabled = true
  stochastic_rollout.enabled = true
  stochastic_rollout.prefix_mode = generated
  stochastic_rollout.objective = crps
  stochastic_rollout.negative_weight = 0.1
```

Use `start_probability` and `end_probability` for scheduled sampling. Do not use
`max_probability`; older code ignores it and falls back to default
`end_probability=0.5`.

Common rollout curriculum:

```yaml
stochastic_rollout:
  enabled: true
  start_epoch: 10
  num_samples: 4
  weight: 0.1
  objective: crps
  negative_weight: 0.1
  validation_num_samples: 8
  validation_horizon: 80
  validation_seed: 42
  horizon_schedule:
    - {start_epoch: 10, horizon: 20}
    - {start_epoch: 30, horizon: 40}
    - {start_epoch: 50, horizon: 80}
```

## 7. Predictor Template

Use this pattern for VBLL predictor runs. Change only `RUN_NAME`, `CFG`, and the
training block in the Python config generation.

```bash
set -euo pipefail

cd "$CODE"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

mkdir -p outputs/logs outputs/generated_configs "$PROCESSED/splits" "$PROCESSED/latent_cache"

SPLIT="$PROCESSED/splits/cell_random_seed_42.jsonl"
if [ ! -f "$SPLIT" ]; then
  PYTHONPATH=src python -u scripts/create_cell_split.py \
    --processed-dir "$PROCESSED" \
    --output "$SPLIT" \
    --seed 42 \
    --train-ratio 0.7 \
    --val-ratio 0.15
fi

FAST_CACHE=$(find "$PROCESSED/latent_cache" /root/autodl-fs/BatteryProject/scale256_fast_artifacts/latent_cache \
  -maxdepth 4 -name latents.npz 2>/dev/null | grep 'fast_encoder' | sort | tail -n 1)

test -f "$FAST_CACHE"

RUN_TAG=$(date +%Y%m%d_%H%M%S)
RUN_NAME="CHANGE_ME_${RUN_TAG}"
CFG="outputs/generated_configs/${RUN_NAME}.yaml"
LOG="outputs/logs/${RUN_NAME}.log"
```

After generating `CFG`, always dry-run before `nohup`:

```bash
PYTHONPATH=src python -u scripts/train_predictor.py \
  --config "$CFG" \
  --latent-cache "$FAST_CACHE" \
  --run-name "$RUN_NAME" \
  --dry-run
```

Launch VBLL training and stochastic eval:

```bash
nohup bash -lc "
set -euo pipefail
cd $CODE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

PYTHONPATH=src python -u scripts/train_predictor.py \
  --config $CFG \
  --latent-cache $FAST_CACHE \
  --run-name $RUN_NAME

PRED_RUN_DIR=outputs/severson_latent_tf_vbll/runs/$RUN_NAME

for CKPT in best last; do
  PYTHONPATH=src python -u scripts/evaluate.py \
    --checkpoint \$PRED_RUN_DIR/\${CKPT}.pt \
    --split test \
    --latent-cache $FAST_CACHE \
    --split-file $SPLIT \
    --num-samples 128 \
    --batch-size 32 \
    --num-workers 8 \
    --save-samples \
    --output-dir \$PRED_RUN_DIR/rollout_test_\${CKPT}_samples_128
done

echo PRED_RUN_DIR=\$PRED_RUN_DIR
echo DONE
" > "$LOG" 2>&1 &
```

Monitor:

```bash
tail -f "$LOG"
pgrep -af "train_predictor.py|evaluate.py" || true
nvidia-smi
```

## 8. Sync Runs for Aggregation

From each machine:

```bash
mkdir -p /root/autodl-fs/BatteryProject/scale256_fast_main_runs

if [ -d "$CODE/outputs/severson_latent_tf_vbll/runs" ]; then
  rsync -a --info=progress2 \
    "$CODE/outputs/severson_latent_tf_vbll/runs/" \
    /root/autodl-fs/BatteryProject/scale256_fast_main_runs/
fi

if [ -d "$CODE/outputs/severson_latent_tf_deterministic/runs" ]; then
  mkdir -p /root/autodl-fs/BatteryProject/scale256_fast_deterministic_runs
  rsync -a --info=progress2 \
    "$CODE/outputs/severson_latent_tf_deterministic/runs/" \
    /root/autodl-fs/BatteryProject/scale256_fast_deterministic_runs/
fi
```

On aggregation machine:

```bash
mkdir -p "$CODE/outputs/severson_latent_tf_vbll/runs"
rsync -a --info=progress2 \
  /root/autodl-fs/BatteryProject/scale256_fast_main_runs/ \
  "$CODE/outputs/severson_latent_tf_vbll/runs/"

mkdir -p "$CODE/outputs/severson_latent_tf_deterministic/runs"
rsync -a --info=progress2 \
  /root/autodl-fs/BatteryProject/scale256_fast_deterministic_runs/ \
  "$CODE/outputs/severson_latent_tf_deterministic/runs/" 2>/dev/null || true
```

## 9. Common Failure Modes

`cd: /root/Battery/code: No such file or directory`

Use the fast disk path:

```bash
CODE=/root/autodl-tmp/BatteryProject/Battery/code
```

`create_cell_split.py: unrecognized arguments --train-frac`

Use:

```bash
--train-ratio 0.7 --val-ratio 0.15
```

`KeyError: input_dim` in `cache_latents.py`

The config is a predictor config. Use `pretrain_encoder*.yaml`, not
`train_latent_tf_vbll*.yaml`.

Log file missing after `tail -f "$LOG"`

The `nohup` command has not been started yet, or the shell variables were only
prepared. Run the launch block first.

`nohup: ignoring input` only

This is normal at process start. If nothing else appears, check:

```bash
pgrep -af "train_predictor.py|evaluate.py|cache_latents.py" || true
tail -n 120 "$LOG"
```

Training slows down later

Generated-prefix CRPS uses horizon curriculum. Going from H=20 to H=80 makes
the rollout part roughly 4x heavier. This is expected.

## 10. Current Interpretation Notes

Mechanism ablation with the conservative scale256 cache showed:

```text
VBLL-TF-NLL:        CRPS 0.004288, width90 0.046699, cov90 0.9991
VBLL-TF-CRPS:       CRPS 0.004387, width90 0.048477, cov90 0.9984
VBLL-Rollout-CRPS:  CRPS 0.002154, width90 0.014664, cov90 0.9595
```

The conclusion is that teacher-forced CRPS alone does not explain the gain.
Generated-prefix rollout alignment is the key improvement.
