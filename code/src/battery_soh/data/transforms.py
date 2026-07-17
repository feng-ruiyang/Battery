from __future__ import annotations

import numpy as np


def resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return np.full((target_length,), np.nan)
    if len(values) == target_length:
        return values
    src = np.linspace(0.0, 1.0, num=len(values))
    dst = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(dst, src, values)


def stack_cycle_channels(arrays: dict[str, np.ndarray], channels: tuple[str, ...], length: int) -> np.ndarray:
    stacked = []
    for name in channels:
        values = arrays.get(name)
        if values is None:
            stacked.append(np.full((length,), np.nan, dtype=float))
        else:
            stacked.append(resample_1d(values, length))
    return np.stack(stacked, axis=-1)
