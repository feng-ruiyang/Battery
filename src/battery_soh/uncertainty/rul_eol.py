from __future__ import annotations

import numpy as np


def first_threshold_crossing(trajectories: np.ndarray, threshold: float) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=float)
    crossed = trajectories < threshold
    first = np.argmax(crossed, axis=-1)
    has_crossed = crossed.any(axis=-1)
    return np.where(has_crossed, first, -1)

