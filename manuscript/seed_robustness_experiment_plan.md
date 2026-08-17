# Seed Robustness Experiment Plan

This document is for the experiment branch. It defines two different
robustness checks for the Severson scale-256 experiments:

1. **Training-seed robustness:** keep the cell split fixed and vary encoder /
   predictor initialization seeds.
2. **Split-seed robustness:** vary the cell-level train/validation/test split
   and rerun the full training and evaluation pipeline for each split.

These two checks answer different questions and should not be mixed in one
aggregate table.

## Goal: Training-Seed Robustness

The current main results are based on one pretrained encoder seed and one
predictor seed. To check whether the conclusion is stable to random
initialization, run a small crossed seed study:

```text
encoder seeds:   42, 8161
predictor seeds: 42, 8191, 8209
```

Seed `42` is the original main-experiment seed. Seed `8161` is the 1024th
prime number. The predictor seeds `8191` and `8209` are fixed additional
prime-number seeds. In the paper, describe these simply as fixed random seeds;
do not attach informal meanings to the numbers.

This is not a new data split experiment. Keep the existing cell split fixed:

```text
splits/cell_random_seed_42.jsonl
```

The purpose is to test model-training randomness, not split randomness.

## Goal: Split-Seed Robustness

The split-seed experiment tests whether the conclusion depends on one lucky or
unlucky cell-level split. For this experiment, each split seed defines a
separate closed experimental pipeline:

```text
split seed -> train/val/test cells -> train-only normalizer -> encoder
-> latent cache -> predictor -> test rollout evaluation
```

The evaluation method must stay identical across split seeds and across
methods. Only the cell split changes. Do not reuse a normalizer, encoder
checkpoint, latent cache, validation set, or test set from another split seed.

Recommended split seeds:

```text
split seeds: 42, 8161, 8191
```

Seed `42` is the main paper split. The other seeds are fixed additional
split seeds. In the paper, describe them simply as fixed split seeds.

For every split seed, compare methods under the same split:

| Method | Training prefix | Stochastic loss | Evaluation |
|---|---|---|---|
| Deterministic-TF | true SoH | none | free rollout, point mean |
| VBLL-TF-NLL | true SoH | NLL only | free stochastic rollout |
| VBLL-TF-CRPS | true SoH | CRPS + negative penalty | free stochastic rollout |
| VBLL-Rollout-CRPS | sampled SoH | CRPS + negative penalty | free stochastic rollout |

If compute is limited, prioritize `VBLL-TF-CRPS` and
`VBLL-Rollout-CRPS`, because that pair isolates whether generated-prefix
training helps when the stochastic scoring terms are matched.

## Split-Seed Evaluation Protocol

For each split seed `S`, use a separate split file:

```text
processed/splits/cell_random_seed_<S>.jsonl
```

Then run the following steps:

1. Generate or verify the cell-level split for seed `S`.
2. Fit channel normalization statistics using only the training cells of
   split `S`.
3. Train the cycle encoder using only the training/validation cells of
   split `S`; select `best.pt` only by validation reconstruction loss.
4. Cache latents for all cells using the encoder checkpoint and normalizer
   from split `S`.
5. Train each predictor method on the train/validation windows of split `S`.
6. Select predictor `best.pt` using validation metrics only. Never use test
   metrics for checkpoint selection.
7. Evaluate on the test cells of split `S` with the same free-running
   rollout protocol for all methods.

The required VBLL evaluation settings are:

```text
horizon: 80
history window: 30
num_samples: 128
rollout mode: free-running autoregressive rollout
metrics: MAE, RMSE, CRPS, PICP, MPIW, negative sample rate
```

For deterministic models, report only MAE and RMSE unless a separate
deterministic interval construction is explicitly defined. Do not fill CRPS,
PICP, or MPIW for deterministic runs by ad hoc assumptions.

## Split-Seed Aggregation

The split-seed aggregate should be paired by split seed. A valid aggregate row
has the following keys:

```text
split_seed, method, checkpoint, MAE, RMSE, CRPS, PICP90, MPIW90,
negative_rate, test_cells, num_windows
```

The recommended final summary is:

- mean and standard deviation across split seeds for each method
- paired per-split differences, especially
  `VBLL-TF-CRPS - VBLL-Rollout-CRPS` for CRPS and MPIW
- number of split seeds where the rollout-aligned method improves CRPS
- number of split seeds where the rollout-aligned method under-covers the
  nominal 90% interval

Suggested output files:

```text
outputs/analysis/severson_split_seed_runs.csv
outputs/analysis/severson_split_seed_summary.csv
```

Do not pool all rollout windows from different split seeds and report one
single window-level metric as the main split-robustness result. That would
overweight splits with more test windows and hide split-to-split variability.
Window-level pooled metrics can be reported only as a secondary diagnostic.

## Split-Seed Interpretation Rules

Use cautious wording:

```text
Across split seeds, rollout-aligned training [consistently / often / sometimes]
reduces CRPS relative to the teacher-forced CRPS control.
```

Do not claim broad dataset generalization from split-seed robustness. It only
tests robustness to different held-out cells within the same processed
Severson/MIT corpus.

If the rollout-aligned method improves CRPS but lowers PICP, report the
trade-off directly:

```text
The rollout-aligned objective improves the proper score and sharpness, but the
lower PICP indicates under-coverage on some splits.
```

## Current Status

Completed:

- Encoder seed `42`, predictor seed `42`
- Main comparison and three controls:
  - `Deterministic-TF`
  - `VBLL-TF-NLL`
  - `VBLL-TF-CRPS`
  - `VBLL-Rollout-CRPS`
- Fast scale-256 encoder cache:
  - `severson_scale256_fast_encoder_20260808_002315_best`

In progress:

- Encoder seed `8161`
  - Use the same fast scale-256 encoder configuration as the completed fast
    encoder run.
  - The inferred fast encoder settings are:
    - `latent_dim: 256`
    - `d_model: 256`
    - `num_layers: 3`
    - `num_heads: 8`
    - `batch_size: 256`
    - `lr: 2.0e-4`
    - `weight_decay: 1.0e-5`
    - `max_epochs: 100`
    - early stopping enabled with patience `12`
  - Do not use the older slow encoder settings.

## Experiment Matrix

Run the following six main-method seed combinations for
`VBLL-Rollout-CRPS`:

| Encoder seed | Predictor seed | Status |
|---:|---:|---|
| 42 | 42 | Completed main run |
| 42 | 8191 | Completed and archived |
| 42 | 8209 | Completed and archived |
| 8161 | 42 | Completed and archived |
| 8161 | 8191 | Completed and archived |
| 8161 | 8209 | Training in progress |

Current best-checkpoint partial results:

| Encoder seed | Predictor seed | MAE | RMSE | CRPS | PICP90 | MPIW90 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 42 | 0.00373 | 0.01530 | 0.00281 | 0.96342 | 0.02088 |
| 42 | 8191 | 0.00249 | 0.01488 | 0.00269 | 0.99899 | 0.02823 |
| 42 | 8209 | 0.00365 | 0.01650 | 0.00274 | 0.98066 | 0.02437 |
| 8161 | 42 | 0.00264 | 0.01505 | 0.00306 | 0.99914 | 0.03369 |
| 8161 | 8191 | 0.00292 | 0.01512 | 0.00380 | 0.99917 | 0.04440 |

Partial aggregate over the five completed best-checkpoint rows:

| Metric | n | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| MAE | 5 | 0.003086 | 0.000574 | 0.002487 | 0.003730 |
| RMSE | 5 | 0.015370 | 0.000652 | 0.014878 | 0.016505 |
| CRPS | 5 | 0.003022 | 0.000458 | 0.002694 | 0.003802 |
| PICP90 | 5 | 0.988277 | 0.016027 | 0.963420 | 0.999171 |
| MPIW90 | 5 | 0.030314 | 0.009198 | 0.020880 | 0.044400 |

Current valid archives on AutoDL shared storage:

```text
/root/autodl-fs/BatteryProject/experiment_archives/autodl-container-fa464e9524-aeb03d7c_seed_eval_20260811_185849.tar.gz
  -> encoder 42, predictor 8191

/root/autodl-fs/BatteryProject/experiment_archives/autodl-container-fxqnqh00m3-ad39220d_seed_eval_20260812_101347.tar.gz
  -> encoder 42, predictor 8209

/root/autodl-fs/BatteryProject/experiment_archives/autodl-container-fxqnqh00m3-ad39220d_seed_eval_20260811_185851.tar.gz
  -> encoder 8161, predictor 42

/root/autodl-fs/BatteryProject/experiment_archives/autodl-container-fa464e9524-aeb03d7c_seed_eval_20260812_095544.tar.gz
  -> encoder 8161, predictor 8191
```

These archives each contain `best.pt`, `last.pt`, four `metrics.json` files,
and four `predictions.npz` files for best/last mean and 128-sample rollout
evaluation.

The following partial or corrupt archives should not be used:

```text
autodl-container-fa464e9524-aeb03d7c_seed_eval_20260811_185732.tar.gz
autodl-container-fa464e9524-aeb03d7c_seed_eval_20260812_095310.tar.gz
autodl-container-fxqnqh00m3-ad39220d_seed_eval_20260811_142539.tar.gz
autodl-container-fxqnqh00m3-ad39220d_seed_eval_20260811_185738.tar.gz
autodl-container-fxqnqh00m3-ad39220d_seed_eval_20260812_095300.tar.gz
```

Optional, if compute is available:

- Repeat `VBLL-TF-CRPS` for the same seed matrix.
- Repeat fair-CRPS sensitivity for at least one additional seed.

Do not delay the main paper on optional rows.

## Fixed Experimental Conditions

Keep these fixed for every run:

- processed Severson-derived corpus
- split file: `cell_random_seed_42.jsonl`
- train-only V/I/T normalizer
- encoder architecture: scale-256 fast encoder
- predictor architecture: scale-256 latent Transformer + VBLL head
- history window: `30`
- forecast horizon: `80`
- stochastic rollout training objective: CRPS
- stochastic rollout training samples: `K=4`
- scheduled sampling / rollout-aligned setting as in the main
  `VBLL-Rollout-CRPS` run
- VBLL prior variance: `0.5`
- final evaluation samples: `128`
- test split evaluation with free stochastic rollout

## Naming

Use explicit run names:

```text
severson_scale256_fastenc_seed{ENC}_vbll_rollout_crps_predseed{PRED}_{RUN_TAG}
```

Examples:

```text
severson_scale256_fastenc_seed42_vbll_rollout_crps_predseed8191_YYYYMMDD_HHMMSS
severson_scale256_fastenc_seed8161_vbll_rollout_crps_predseed8209_YYYYMMDD_HHMMSS
```

For encoder seed `42`, use the existing cache:

```text
severson_scale256_fast_encoder_20260808_002315_best/latents.npz
```

For encoder seed `8161`, use the new cache generated from the in-progress
encoder run:

```text
severson_scale256_fast_encoder_seed8161_*_best/latents.npz
```

## Required Outputs Per Predictor Run

Save:

- generated predictor config YAML
- training log
- `best.pt`
- `last.pt`
- `rollout_test_best_samples_128/metrics.json`
- `rollout_test_last_samples_128/metrics.json`
- `rollout_test_best_samples_128/predictions.npz`, if storage permits

If storage is tight, keep all `metrics.json` files and at least the best
checkpoint metrics. The aggregate table can be built from metrics alone.

## Evaluation Commands

After each predictor training run, evaluate both `best` and `last`:

```bash
RUN_DIR="outputs/severson_latent_tf_vbll/runs/<RUN_NAME>"
CACHE="/root/autodl-tmp/data/severson-2019/processed/latent_cache/<ENC_CACHE_DIR>/latents.npz"
SPLIT="/root/autodl-tmp/data/severson-2019/processed/splits/cell_random_seed_42.jsonl"

for CKPT in best last; do
  PYTHONPATH=src python -u scripts/evaluate.py \
    --checkpoint "$RUN_DIR/${CKPT}.pt" \
    --split test \
    --latent-cache "$CACHE" \
    --split-file "$SPLIT" \
    --num-samples 128 \
    --batch-size 32 \
    --num-workers 4 \
    --save-samples \
    --output-dir "$RUN_DIR/rollout_test_${CKPT}_samples_128"
done
```

## Aggregation

Build one table with each seed pair as one row:

```text
encoder_seed, predictor_seed, checkpoint, MAE, RMSE, CRPS, PICP90, MPIW90,
negative_rate, monotonic_violation_rate
```

Then report:

- mean and standard deviation over the six seed pairs
- best-checkpoint aggregate
- last-checkpoint aggregate as a diagnostic
- paired comparison against the completed `42 x 42` main result

Recommended output files:

```text
outputs/analysis/severson_seed_robustness_runs.csv
outputs/analysis/severson_seed_robustness_summary.csv
```

## Interpretation Rules

If the rollout-aligned method keeps better CRPS and narrower MPIW over most
seed pairs, present it as a robustness check.

If the result is mixed, report it directly:

```text
The seed study shows that the direction of the CRPS improvement is [stable /
partly seed-dependent]. This suggests that the rollout-aligned objective is
promising but should be evaluated with larger repeated runs in future work.
```

Do not claim statistical significance unless a cell-level or seed-level
confidence interval is computed and supports the statement.
