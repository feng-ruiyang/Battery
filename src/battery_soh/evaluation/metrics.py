from __future__ import annotations

import numpy as np

from battery_soh.training.losses import masked_mae, masked_rmse


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    return {
        "mae": masked_mae(y_true, y_pred, mask),
        "rmse": masked_rmse(y_true, y_pred, mask),
    }


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((y_true[valid] >= lower[valid]) & (y_true[valid] <= upper[valid])))

