# Revision Experiment Plan for Rollout-Aligned SoH Paper

This document is for the implementation/experiment Codex branch. The goal is
to address the strongest review concerns without changing the paper's core
story.

## Current Main Claim

The current paper compares:

1. `Deterministic-TF`
2. `VBLL-TF-NLL`
3. `VBLL-TF-CRPS`
4. `VBLL-Rollout-CRPS`

The main result is that `VBLL-Rollout-CRPS` improves marginal probabilistic
rollout quality compared with `VBLL-TF-CRPS`:

- CRPS: `0.00416 -> 0.00281`
- 90% MPIW: `0.04914 -> 0.02088`
- 90% PICP: `0.99918 -> 0.96342`
- RMSE: `0.01597 -> 0.01530`
- MAE is worse: `0.00317 -> 0.00373`

The revised paper now states that rollout intervals are last-layer
weight-posterior-induced trajectory intervals, not full outcome predictive
intervals with extra observation-noise sampling.

## Priority 1: Fair Ensemble CRPS

### Motivation

The current rollout training uses the empirical sample CRPS:

```text
1/K sum_k |x_k - y| - 1/(2K^2) sum_{k,j} |x_k - x_j|
```

With small `K=4`, this is biased for the underlying continuous predictive law
because self-pairs have zero distance. The reviewer identified this as the
strongest technical concern.

### Required Implementation

Add an option for fair ensemble CRPS:

```text
1/K sum_k |x_k - y| - 1/(2K(K-1)) sum_{k != j} |x_k - x_j|
```

Keep the existing empirical CRPS available for backward compatibility.

Suggested config:

```yaml
training:
  stochastic_rollout:
    objective: crps
    crps_variant: fair   # empirical | fair
```

### Experiments

Run at least:

1. `VBLL-Rollout-CRPS-fair-K4`
2. `VBLL-TF-CRPS-fair-K4`

If compute allows:

3. `VBLL-Rollout-CRPS-fair-K8`
4. `VBLL-Rollout-CRPS-fair-K16`

Keep everything else fixed:

- same processed Severson-derived dataset
- same cell split
- same pretrained encoder and latent cache
- same model size
- same validation selection rule
- same evaluation with 128 stochastic rollout samples

### Outputs

For every run, save:

- `best.pt`, `last.pt`
- training log
- `rollout_test_best_samples_128/metrics.json`
- `rollout_test_last_samples_128/metrics.json`
- `predictions.npz` if storage permits

### Acceptance Criteria

Report a table comparing:

```text
method                         MAE     RMSE    CRPS    PICP90   MPIW90
VBLL-TF-CRPS-empirical-K4
VBLL-Rollout-CRPS-empirical-K4
VBLL-TF-CRPS-fair-K4
VBLL-Rollout-CRPS-fair-K4
```

The key question is whether rollout-aligned training still improves CRPS and
MPIW after switching to fair ensemble CRPS.

## Priority 2: Per-Cell Macro Statistics and Confidence Intervals

### Motivation

The test set has 17,317 sliding windows but only about 20 effective test
cells. Windows from the same cell are correlated. The paper should not imply
that all windows are independent evidence.

### Required Analysis

Using existing `predictions.npz` files, compute per-cell metrics:

- MAE
- RMSE
- CRPS
- 50/80/90/95% PICP
- 50/80/90/95% MPIW
- negative sample rate

Then compute:

- cell-macro mean
- cell-level standard deviation
- paired per-cell differences:
  - `VBLL-Rollout-CRPS - VBLL-TF-CRPS`
  - `VBLL-TF-CRPS - VBLL-TF-NLL`
- bootstrap confidence intervals over cells, if feasible

### Outputs

Save:

- `outputs/analysis/severson_per_cell_metrics.csv`
- `outputs/analysis/severson_per_cell_pairwise_differences.csv`
- `outputs/analysis/severson_cell_macro_summary.csv`

### Acceptance Criteria

The paper needs at least one sentence like:

```text
Cell-macro results show the same direction as the window-micro aggregate ...
```

or, if not true:

```text
The improvement is mainly driven by window-micro averaging and is less stable
at the cell level ...
```

Do not hide unfavorable cell-level results.

## Priority 3: Horizon Cohort and Common-Cohort Analysis

### Motivation

At longer horizons, only windows/cells with enough remaining cycles contribute.
The set of evaluated windows may change with horizon.

### Required Analysis

For each evaluated horizon `h = 1..80`, compute:

- number of valid windows
- number of participating cells
- per-method MAE, RMSE, CRPS, PICP90, MPIW90

Then compute a common-cohort version using only windows valid through all 80
future steps.

### Outputs

Save:

- `outputs/analysis/severson_horizon_counts.csv`
- `outputs/analysis/severson_horizon_metrics_all_valid.csv`
- `outputs/analysis/severson_horizon_metrics_common80.csv`

### Acceptance Criteria

The paper should be able to say:

```text
Horizon-wise comparisons use N_h windows and C_h cells at each horizon.
The common-80 cohort shows [consistent / different] trends.
```

## Priority 4: Data Provenance and Split Audit

### Motivation

The paper currently uses 140 processed cells, while the original Severson
Nature Energy paper emphasizes 124 cells. This must be auditable.

### Required Analysis

Export:

- all cell IDs
- split membership
- batch IDs
- source file paths
- number of cycles per cell
- cells excluded from rollout because they do not yield valid windows
- exact split seed and split file path

### Outputs

Save:

- `outputs/analysis/severson_cell_inventory.csv`
- `outputs/analysis/severson_split_inventory.csv`
- `outputs/analysis/severson_rollout_excluded_cells.csv`

### Acceptance Criteria

The writing branch should be able to include a short reproducibility paragraph
and optionally an appendix/table.

## Priority 5: Additional Calibration Levels

### Motivation

The current paper mostly discusses 90% PICP/MPIW. Calibration is stronger if
multiple nominal intervals are reported.

### Required Analysis

From existing 128-sample predictions, compute:

- 50% PICP/MPIW
- 80% PICP/MPIW
- 90% PICP/MPIW
- 95% PICP/MPIW

Do this overall, per cell, and per representative horizon if possible.

### Outputs

Save:

- `outputs/analysis/severson_interval_levels_summary.csv`

### Acceptance Criteria

The paper can report a small table or a sentence showing whether intervals are
systematically conservative.

## Priority 6: Optional Minimal Ablations

Only run these if compute/time allows.

1. `VBLL-Rollout-CRPS` without non-negativity penalty.
2. `VBLL-Rollout-CRPS` without horizon curriculum.
3. `VBLL-Rollout-CRPS` without latent auxiliary loss, if easy.

Keep these as secondary evidence. Do not let them delay Priority 1--4.

## Notes for Implementation

- Do not change the main model architecture while running revision experiments.
- Do not mix different encoder caches in a controlled comparison.
- Report whether results use `best.pt` or `last.pt`.
- Prefer `best.pt` selected by the existing validation selection metric, but
  also save `last.pt` evaluations for stability checks.
- Keep all output directories timestamped.
- Store final summary CSV files under `outputs/analysis/`.
- Archive final run directories before shutting down cloud instances.

## Deliverables Back to Writing Branch

Send back:

1. A concise terminal summary table.
2. All generated CSV files.
3. The exact run directories used.
4. Any deviations from this plan.
5. Whether fair-CRPS changes the main conclusion.
