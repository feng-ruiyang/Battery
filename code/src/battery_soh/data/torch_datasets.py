from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.io import read_jsonl
from battery_soh.data.transforms import stack_cycle_channels


def load_split_ids(split_file: str | Path | None, split_name: str) -> list[str] | None:
    if split_file is None:
        return None
    rows = read_jsonl(Path(split_file))
    if not rows:
        raise ValueError(f"Split file is empty: {split_file}")
    split = rows[0]
    if split_name not in split:
        raise KeyError(f"Split {split_name!r} not found in {split_file}.")
    return list(split[split_name])


def resolve_cell_ids(
    store: CanonicalCycleStore,
    split_file: str | Path | None = None,
    split_name: str | None = None,
) -> list[str]:
    if split_file is not None and split_name is not None:
        return load_split_ids(split_file, split_name) or []
    return sorted({row["cell_id"] for row in store.cells})


class CycleTensorDataset(Dataset):
    """One fixed-length cycle tensor per canonical cycle row."""

    def __init__(
        self,
        store: CanonicalCycleStore,
        cell_ids: list[str],
        channels: tuple[str, ...],
        sequence_length: int,
    ) -> None:
        self.store = store
        self.channels = channels
        self.sequence_length = sequence_length
        self.rows: list[dict[str, Any]] = []
        for cell_id in cell_ids:
            for row in store.cycles_for_cell(cell_id):
                if row.get("array_path"):
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays = self.store.load_cycle_arrays(row)
        x_cycle = stack_cycle_channels(arrays, self.channels, self.sequence_length)
        return {
            "x_cycle": _float_tensor(x_cycle),
            "cell_id": row["cell_id"],
            "dataset_id": row["dataset_id"],
            "batch_id": row.get("batch_id"),
            "cycle_index": int(row["cycle_index"]),
            "soh": _float_or_nan(row.get("soh")),
            "summaries": {
                "q_discharge": _float_or_nan(row.get("q_discharge")),
                "internal_resistance": _float_or_nan(row.get("internal_resistance")),
                "t_avg": _float_or_nan(row.get("t_avg")),
            },
        }


def collate_cycle_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_cycle": torch.stack([item["x_cycle"] for item in items], dim=0),
        "cell_id": [item["cell_id"] for item in items],
        "dataset_id": [item["dataset_id"] for item in items],
        "batch_id": [item["batch_id"] for item in items],
        "cycle_index": torch.as_tensor([item["cycle_index"] for item in items], dtype=torch.long),
        "soh": torch.as_tensor([item["soh"] for item in items], dtype=torch.float32),
        "summaries": {
            key: torch.as_tensor([item["summaries"][key] for item in items], dtype=torch.float32)
            for key in ("q_discharge", "internal_resistance", "t_avg")
        },
    }


class LatentCacheStore:
    """Reader for cycle-level latent vectors saved by `scripts/cache_latents.py`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.z = np.asarray(data["z"], dtype=np.float32)
            self.cell_ids = np.asarray(data["cell_id"]).astype(str)
            self.dataset_ids = np.asarray(data["dataset_id"]).astype(str)
            self.batch_ids = np.asarray(data["batch_id"]).astype(str)
            self.cycle_indices = np.asarray(data["cycle_index"], dtype=np.int64)
            self.soh = np.asarray(data["soh"], dtype=np.float32)
            self.summary_names = tuple(np.asarray(data["summary_names"]).astype(str).tolist())
            self.summaries = np.asarray(data["summaries"], dtype=np.float32)

    def cell_ids_unique(self) -> list[str]:
        return sorted(set(self.cell_ids.tolist()))

    def rows_for_cell(self, cell_id: str) -> dict[str, np.ndarray]:
        idx = np.nonzero(self.cell_ids == cell_id)[0]
        order = np.argsort(self.cycle_indices[idx])
        selected = idx[order]
        return {
            "z": self.z[selected],
            "cell_id": self.cell_ids[selected],
            "dataset_id": self.dataset_ids[selected],
            "batch_id": self.batch_ids[selected],
            "cycle_index": self.cycle_indices[selected],
            "soh": self.soh[selected],
            "summaries": self.summaries[selected],
        }


class LatentWindowDataset(Dataset):
    """Sliding windows over cached cycle latents for SoH trajectory training."""

    def __init__(
        self,
        cache: LatentCacheStore,
        cell_ids: list[str],
        window: int,
        horizon: int,
        summary_channels: tuple[str, ...] = (),
        target: str = "soh",
    ) -> None:
        if target != "soh":
            raise ValueError("LatentWindowDataset currently supports target='soh' only.")
        self.cache = cache
        self.window = window
        self.horizon = horizon
        self.summary_channels = summary_channels
        self.summary_indices = tuple(cache.summary_names.index(name) for name in summary_channels)
        self.by_cell = {cell_id: cache.rows_for_cell(cell_id) for cell_id in cell_ids}
        self.index: list[tuple[str, int]] = []
        for cell_id, rows in self.by_cell.items():
            max_start = len(rows["z"]) - window - 1
            for start in range(max(0, max_start + 1)):
                hist = rows["soh"][start : start + window]
                future = rows["soh"][start + window : start + window + horizon]
                if np.isfinite(hist).all() and np.isfinite(future).any():
                    self.index.append((cell_id, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cell_id, start = self.index[index]
        rows = self.by_cell[cell_id]
        hist = slice(start, start + self.window)
        future = slice(start + self.window, start + self.window + self.horizon)
        soh_future = np.full((self.horizon,), np.nan, dtype=np.float32)
        mask = np.zeros((self.horizon,), dtype=bool)
        future_values = rows["soh"][future]
        soh_future[: len(future_values)] = future_values
        mask[: len(future_values)] = np.isfinite(future_values)

        batch = {
            "z_window": _float_tensor(rows["z"][hist]),
            "soh_history": _float_tensor(rows["soh"][hist]),
            "soh_future": _float_tensor(soh_future),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
            "history_mask": torch.zeros((self.window,), dtype=torch.bool),
            "cell_id": cell_id,
            "forecast_cycle_index": int(rows["cycle_index"][start + self.window - 1]),
        }
        if self.summary_indices:
            batch["u_window"] = _float_tensor(rows["summaries"][hist][:, self.summary_indices])
        return batch


def save_latent_cache(
    path: str | Path,
    z: np.ndarray,
    metadata: dict[str, np.ndarray],
    summary_names: tuple[str, ...],
    summaries: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z=np.asarray(z, dtype=np.float32),
        cell_id=np.asarray(metadata["cell_id"], dtype=str),
        dataset_id=np.asarray(metadata["dataset_id"], dtype=str),
        batch_id=np.asarray(metadata["batch_id"], dtype=str),
        cycle_index=np.asarray(metadata["cycle_index"], dtype=np.int64),
        soh=np.asarray(metadata["soh"], dtype=np.float32),
        summary_names=np.asarray(summary_names, dtype=str),
        summaries=np.asarray(summaries, dtype=np.float32),
    )


def _float_tensor(values: np.ndarray) -> Tensor:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(values)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
