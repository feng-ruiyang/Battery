from __future__ import annotations

import numpy as np


def sample_gaussian_trajectories(
    mean: np.ndarray,
    covariance: np.ndarray,
    num_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.ndim == 1:
        return rng.multivariate_normal(mean, covariance, size=num_samples)
    samples = [
        rng.multivariate_normal(mean[i], covariance[i], size=num_samples)
        for i in range(mean.shape[0])
    ]
    return np.stack(samples, axis=0)

