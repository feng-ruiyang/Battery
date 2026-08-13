# Severson Scale256 Seed Rerun Runbook

This runbook is for the clean seed rerun after the earlier seed experiments were found to mix different configs.

The rule is simple:

- Encoder seed runs may change only `experiment.seed` and `experiment.run_name`.
- Predictor seed runs may change only `experiment.seed`, `experiment.run_name`, and `data.latent_cache`.
- The cell split remains `cell_random_seed_42.jsonl`.
- The normalizer remains `train_vit_seed_42.json`.

## Fixed Configs

The config generator is:

```bash
code/scripts/prepare_severson_scale256_seed_configs.py
```

It writes the controlled YAML configs for:

- scale256 fast encoder
- scale256 VBLL rollout-CRPS predictor

Do not manually edit the generated YAML unless there is a real code change.

## Fixed Fast Encoder Parameters

These match the original scale256 fast encoder:

```text
latent_dim = 256
d_model = 256
num_layers = 4
num_heads = 8
batch_size = 256
lr = 2e-4
scheduler = reduce_on_plateau
scheduler patience = 8
min_lr = 2e-5
early stopping patience = 20
max_epochs = 120
```

## Fixed Predictor Parameters

These match the main `42 x 42` rollout-aligned predictor:

```text
window = 30
horizon = 80
latent_dim = 256
history_dim = 512
decoder_dim = 512
batch_size = 128
max_epochs = 120
lr = 2e-4
beta_kl = 1e-4
lambda_state = 1.0
vbll_rho_lr_multiplier = 5.0
scheduled_sampling start/end = 10/40
scheduled_sampling max probability = 0.25
stochastic_rollout start_epoch = 10
stochastic_rollout num_samples = 4
stochastic_rollout objective = crps
stochastic_rollout weight = 0.1
negative_weight = 0.1
validation_num_samples = 8
validation_horizon = 80
validation_seed = 42
horizon schedule = 20 from epoch 10, 40 from epoch 30, 80 from epoch 50
```

## Experiments To Rerun

The clean matrix is:

| Encoder seed | Predictor seed | Status |
|---:|---:|---|
| 42 | 42 | Existing main run, keep |
| 42 | 9181 | Rerun |
| 42 | 8209 | Rerun |
| 8161 | 42 | Rerun after encoder 8161 cache |
| 8161 | 9181 | Rerun after encoder 8161 cache |
| 8161 | 8209 | Rerun after encoder 8161 cache |

## Server Setup Check

Run this first on each AutoDL machine:

```bash
cd /root/autodl-tmp/BatteryProject/Battery/code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

PROCESSED=/root/autodl-tmp/data/severson-2019/processed
SPLIT=$PROCESSED/splits/cell_random_seed_42.jsonl
NORM=$PROCESSED/normalization/train_vit_seed_42.json

git log -1 --oneline
wc -l "$PROCESSED/cells.jsonl" "$PROCESSED/cycles.jsonl"
ls -lh "$SPLIT" "$NORM"
```

## Train Encoder 8161

Only this encoder needs to be rerun. Encoder 42 uses the existing main cache.

```bash
cd /root/autodl-tmp/BatteryProject/Battery/code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

mkdir -p outputs/logs outputs/generated_configs

RUN_TAG=$(date +%Y%m%d_%H%M%S)
PROCESSED=/root/autodl-tmp/data/severson-2019/processed

eval "$(
  PYTHONPATH=src python -u scripts/prepare_severson_scale256_seed_configs.py \
    --mode encoder \
    --seed 8161 \
    --processed-dir "$PROCESSED" \
    --run-tag "$RUN_TAG"
)"

LOG="outputs/logs/pretrain_${RUN_NAME}.log"

nohup env PYTHONPATH=src python -u scripts/pretrain_encoder.py \
  --config "$CONFIG" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "RUN_NAME=$RUN_NAME"
echo "CONFIG=$CONFIG"
echo "LOG=$LOG"
```

Monitor:

```bash
tail -f "$LOG"
```

## Cache Encoder 8161

After encoder training finishes:

```bash
cd /root/autodl-tmp/BatteryProject/Battery/code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

PROCESSED=/root/autodl-tmp/data/severson-2019/processed
ENC_RUN_NAME=<paste encoder RUN_NAME>
ENC_CFG=<paste encoder CONFIG>
ENC_RUN_DIR="outputs/severson_cycle_autoencoder/runs/$ENC_RUN_NAME"
CACHE_DIR="$PROCESSED/latent_cache/${ENC_RUN_NAME}_best"
CACHE="$CACHE_DIR/latents.npz"

mkdir -p "$CACHE_DIR"

nohup env PYTHONPATH=src python -u scripts/cache_latents.py \
  --config "$ENC_CFG" \
  --checkpoint "$ENC_RUN_DIR/best.pt" \
  --output "$CACHE" \
  > "outputs/logs/cache_${ENC_RUN_NAME}.log" 2>&1 &

echo "PID=$!"
echo "CACHE=$CACHE"
```

Verify:

```bash
CACHE=<paste CACHE> python - <<'PY'
import os
import numpy as np

with np.load(os.environ["CACHE"], allow_pickle=False) as data:
    z = data["z"]
    soh = data["soh"]
print("z shape:", z.shape)
print("soh shape:", soh.shape)
print("finite z:", bool(np.isfinite(z).all()))
print("finite soh:", bool(np.isfinite(soh).all()))
assert z.shape == (114314, 256)
assert np.isfinite(z).all()
assert np.isfinite(soh).all()
PY
```

## Train Predictor

Use this template for any seed pair. Change only `ENC_SEED`, `PRED_SEED`, and `CACHE`.

For existing encoder 42 cache:

```bash
CACHE=/root/autodl-tmp/data/severson-2019/processed/latent_cache/severson_scale256_fast_encoder_20260808_002315_best/latents.npz
```

For encoder 8161 cache:

```bash
CACHE=<paste encoder 8161 cache path>
```

Training command:

```bash
cd /root/autodl-tmp/BatteryProject/Battery/code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

mkdir -p outputs/logs outputs/generated_configs

RUN_TAG=$(date +%Y%m%d_%H%M%S)
PROCESSED=/root/autodl-tmp/data/severson-2019/processed

ENC_SEED=42
PRED_SEED=9181
CACHE=/root/autodl-tmp/data/severson-2019/processed/latent_cache/severson_scale256_fast_encoder_20260808_002315_best/latents.npz

eval "$(
  PYTHONPATH=src python -u scripts/prepare_severson_scale256_seed_configs.py \
    --mode predictor \
    --encoder-seed "$ENC_SEED" \
    --seed "$PRED_SEED" \
    --latent-cache "$CACHE" \
    --processed-dir "$PROCESSED" \
    --run-tag "$RUN_TAG"
)"

LOG="outputs/logs/train_${RUN_NAME}.log"

nohup env PYTHONPATH=src python -u scripts/train_predictor.py \
  --config "$CONFIG" \
  --latent-cache "$CACHE" \
  --run-name "$RUN_NAME" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "RUN_NAME=$RUN_NAME"
echo "CONFIG=$CONFIG"
echo "LOG=$LOG"
```

Recommended assignment:

```text
Machine A: ENC_SEED=42,   PRED_SEED=9181
Machine B: ENC_SEED=42,   PRED_SEED=8209
Machine C: ENC_SEED=8161, PRED_SEED=42
Machine D: ENC_SEED=8161, PRED_SEED=9181
Machine E: ENC_SEED=8161, PRED_SEED=8209
```

## Evaluate A Predictor Run

After training:

```bash
cd /root/autodl-tmp/BatteryProject/Battery/code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

PROCESSED=/root/autodl-tmp/data/severson-2019/processed
SPLIT=$PROCESSED/splits/cell_random_seed_42.jsonl
RUN_DIR="outputs/severson_latent_tf_vbll/runs/<paste RUN_NAME>"
CACHE=<paste CACHE>

LOG="outputs/logs/eval_$(basename "$RUN_DIR")_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
set -euo pipefail
cd /root/autodl-tmp/BatteryProject/Battery/code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate battery-soh

for CKPT in best last; do
  env PYTHONPATH=src python -u scripts/evaluate.py \
    --checkpoint \"$RUN_DIR/\${CKPT}.pt\" \
    --split test \
    --latent-cache \"$CACHE\" \
    --split-file \"$SPLIT\" \
    --num-samples 1 \
    --batch-size 64 \
    --num-workers 8 \
    --use-mean \
    --output-dir \"$RUN_DIR/rollout_test_\${CKPT}_mean\"

  env PYTHONPATH=src python -u scripts/evaluate.py \
    --checkpoint \"$RUN_DIR/\${CKPT}.pt\" \
    --split test \
    --latent-cache \"$CACHE\" \
    --split-file \"$SPLIT\" \
    --num-samples 128 \
    --batch-size 64 \
    --num-workers 8 \
    --save-samples \
    --output-dir \"$RUN_DIR/rollout_test_\${CKPT}_samples_128\"
done
" > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
```

## Package Results

```bash
cd /root/autodl-tmp/BatteryProject/Battery/code

RUN_DIR="outputs/severson_latent_tf_vbll/runs/<paste RUN_NAME>"
HOST_TAG=$(hostname)
OUT="/root/autodl-fs/BatteryProject/experiment_archives/${HOST_TAG}_seed_eval_$(date +%Y%m%d_%H%M%S).tar.gz"

mkdir -p /root/autodl-fs/BatteryProject/experiment_archives

tar -czf "$OUT" "$RUN_DIR" outputs/logs

ls -lh "$OUT"
tar -tzf "$OUT" | head -n 30
```

## Sanity Checks

Before trusting a run, check the checkpoint config:

```bash
RUN_DIR="outputs/severson_latent_tf_vbll/runs/<paste RUN_NAME>"

RUN_DIR="$RUN_DIR" python - <<'PY'
import os
import torch
from pathlib import Path

p = Path(os.environ["RUN_DIR"]) / "best.pt"
payload = torch.load(p, map_location="cpu", weights_only=False)
cfg = payload["config"]

print("epoch:", payload.get("epoch"))
print("metrics:", payload.get("metrics"))
print("experiment.seed:", cfg["experiment"].get("seed"))
print("latent_cache:", cfg["data"].get("latent_cache"))
print("batch_size:", cfg["training"].get("batch_size"))
print("max_epochs:", cfg["training"].get("max_epochs"))
print("scheduled_sampling:", cfg["training"].get("scheduled_sampling"))
print("stochastic_rollout:", cfg["training"].get("stochastic_rollout"))
PY
```

Expected predictor fields:

```text
batch_size: 128
max_epochs: 120
scheduled_sampling.start_epoch: 10
scheduled_sampling.end_epoch: 40
stochastic_rollout.start_epoch: 10
stochastic_rollout.validation_num_samples: 8
stochastic_rollout.validation_seed: 42
```
