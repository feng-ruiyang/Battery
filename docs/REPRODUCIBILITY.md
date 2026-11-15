# Experiment settings

## Main configuration

| Setting | Value |
|---|---|
| Cell split | seed 42; 98 train / 21 validation / 21 test cells |
| History / forecast | 30 / 80 cycles; stride 1 |
| Latent dimension | 256 |
| History / decoder hidden dimension | 512 / 512 |
| Predictor seed | 42 unless explicitly labeled otherwise |
| Training | 120 epochs, batch 128, AdamW |
| Learning rates | base 2e-4; VBLL rho 1e-3 |
| Weight decay / KL coefficient | 1e-5 / 1e-4 |
| Auxiliary latent loss coefficient | 1.0 |
| Stochastic CRPS / negative penalty coefficients | 0.1 / 0.1 |
| Scheduled sampling | epochs 10--40: 0 to 0.25 |
| Stochastic rollout curriculum | H20 at epoch 10, H40 at 30, H80 at 50 |
| Training / validation samples | 4 / 8 |
| Validation | generated prefixes, H80, seed 42 |
| Checkpoint selection | validation CRPS + 0.1 times negative penalty |
| Probabilistic test evaluation | 128 samples, batch 32, seed 42 |

The main configuration is `configs/revision/rollout_main_seed42_reference.json`.
Cache hashes are listed in the release manifest.

## Components

- Full empirical rollout: SS enabled; generated-prefix stochastic optimization enabled.
- Fair estimator: same controlled training settings, `crps_variant: fair`.
- `no_ss`: SS disabled; stochastic optimization retained.
- `no_sro`: SS retained; stochastic training branch disabled. Its CRPS and
  associated stochastic negative penalty both contribute zero to training.
- `neither`: both training mechanisms disabled; NLL, KL and latent auxiliary
  terms remain.

`training.rollout_validation` is independent of the training switches. It
preserves the full model's checkpoint-selection protocol for deletion ablations.
Validation uses no gradients and preserves the training RNG state.
All probabilistic test rows use empirical CRPS with 128 samples, including models
trained with fair CRPS.

## Archives

- `controlled`: main, seed and controlled-estimator experiments.
- `legacy_complete`: completed experiments with earlier configurations.
- `evaluation_only`: saved predictions and metrics; checkpoint unavailable.
- `pending`: experiments awaiting completion and packaging.

The release inventory records each run's configuration, checkpoint epoch and
matching latent cache.

## Evaluation

The dataset includes partial-future windows with masks. The test set has 17,317
windows from 20 cells; 15,737 have all 80 future targets. Metrics use valid targets
and support window-micro and cell-macro aggregation. Results are labeled by
best or last checkpoint.

Predictor evaluation uses the supplied frozen encoder-42 latent cache.
