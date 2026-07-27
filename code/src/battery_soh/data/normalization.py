from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChannelNormalizer:
    channels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        size = len(self.channels)
        if self.mean.shape != (size,) or self.std.shape != (size,) or self.count.shape != (size,):
            raise ValueError("Normalizer mean/std/count must have one value per channel.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Normalizer mean/std must be finite.")
        if np.any(self.std <= 0):
            raise ValueError("Normalizer standard deviations must be positive.")

    def normalize(self, values: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
        self._check_channels(channels)
        values = np.asarray(values, dtype=float)
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
        self._check_channels(channels)
        values = np.asarray(values, dtype=float)
        return values * self.std + self.mean

    def to_json(self) -> dict[str, Any]:
        return {
            "channels": list(self.channels),
            "mean": {name: float(self.mean[index]) for index, name in enumerate(self.channels)},
            "std": {name: float(self.std[index]) for index, name in enumerate(self.channels)},
            "count": {name: int(self.count[index]) for index, name in enumerate(self.channels)},
            "metadata": self.metadata,
        }

    def _check_channels(self, channels: tuple[str, ...]) -> None:
        if tuple(channels) != self.channels:
            raise ValueError(
                f"Normalizer channels {self.channels} do not match requested channels {channels}."
            )


def load_channel_normalizer(path: str | Path | None) -> ChannelNormalizer | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    channels = tuple(str(value) for value in payload["channels"])
    return ChannelNormalizer(
        channels=channels,
        mean=np.asarray([payload["mean"][name] for name in channels], dtype=float),
        std=np.asarray([payload["std"][name] for name in channels], dtype=float),
        count=np.asarray([payload["count"][name] for name in channels], dtype=np.int64),
        metadata=dict(payload.get("metadata", {})),
    )


def save_channel_normalizer(path: str | Path, normalizer: ChannelNormalizer) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalizer.to_json(), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
