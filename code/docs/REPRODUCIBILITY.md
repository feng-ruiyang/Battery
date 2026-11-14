# Controlled experiment protocol

## Fixed scientific settings

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

The default historical YAML examples are not the audited scale256 protocol.
The exact reference is `configs/revision/rollout_main_seed42_reference.json`.
Configuration paths alone are not proof that two caches are identical: compare
SHA-256 digests in the release manifest.

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
trained with fair CRPS. The two estimator names must not be confused with the
test metric definition.

## Release categories

- `controlled`: audited main, seed and new controlled-estimator experiments.
- `legacy_complete`: completed runs whose settings differ from the controlled
  protocol. Preserve their results, but do not silently mix them into its tables.
- `evaluation_only`: saved predictions/metrics are present, but a matching
  training checkpoint has not been recovered.
- `pending`: a live pipeline; ZIP creation waits for its completion state.
- Incomplete or unverified snapshots remain in the server's original archive
  and are listed in the inventory, not promoted to completed experiments.

Known provenance findings are included in the release inventory. In particular,
`severson_scale256_fast_encoder_seed8161_20260813_112351` exists as both epoch-2
and epoch-120 snapshots on different machines. Use the completed snapshot and
its matching cache. The old uncontrolled fair run must not be substituted for
the new 512-dimensional controlled fair experiment.

## Evaluation scope

The dataset includes partial-future windows with masks. The test set has 17,317
windows from 20 cells; 15,737 have all 80 future targets. Always respect masks,
cell identities and forecast origins when comparing predictions. Report whether
results are window-micro or cell-macro and whether they use best or last weights.

A frozen encoder-42 latent cache is available. Do not claim raw-curve-to-output
reproduction unless the matching original encoder checkpoint is present in the
chosen release. Predictor reproduction from the provided cache is supported.
