from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from battery_soh.data.io import read_jsonl
from battery_soh.data.schema import TrainingWindowSpec
from battery_soh.data.transforms import stack_cycle_channels


@dataclass(frozen=True)
class TrainingWindow:
    cell_id: str
    dataset_id: str
    batch_id: str | None
    start_cycle_index: int
    x_window: np.ndarray | None
    u_window: np.ndarray | None
    soh_history: np.ndarray
    soh_future: np.ndarray
    mask: np.ndarray


class CanonicalCycleStore:
    """Reader for processed canonical cell/cycle files."""

    def __init__(self, processed_dir: str | Path) -> None:
        self.processed_dir = Path(processed_dir)
        self.cells = read_jsonl(self.processed_dir / "cells.jsonl")
        self.cycles = read_jsonl(self.processed_dir / "cycles.jsonl")

    def cycles_for_cell(self, cell_id: str) -> list[dict[str, Any]]:
        rows = [row for row in self.cycles if row["cell_id"] == cell_id]
        return sorted(rows, key=lambda row: row["cycle_index"])

    def load_cycle_arrays(self, cycle_row: dict[str, Any]) -> dict[str, np.ndarray]:
        array_path = cycle_row.get("array_path")
        if not array_path:
            return {}
        with np.load(self.processed_dir / array_path) as data:
            return {key: data[key] for key in data.files}


class SOHTrajectoryDataset:
    """Numpy-first dataset that can later be wrapped for PyTorch DataLoader."""

    def __init__(self, store: CanonicalCycleStore, cell_ids: list[str], spec: TrainingWindowSpec) -> None:
        self.store = store
        self.spec = spec
        self.index: list[tuple[str, int]] = []
        self._cycles_by_cell = {cell_id: store.cycles_for_cell(cell_id) for cell_id in cell_ids}
        for cell_id, cycles in self._cycles_by_cell.items():
            max_start = len(cycles) - spec.window - 1
            for start in range(max(0, max_start + 1)):
                self.index.append((cell_id, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> TrainingWindow:
        cell_id, start = self.index[index]
        cycles = self._cycles_by_cell[cell_id]
        hist = cycles[start : start + self.spec.window]
        future = cycles[start + self.spec.window : start + self.spec.window + self.spec.horizon]
        x_window = self._build_x_window(hist)
        u_window = self._build_u_window(hist)
        soh_history = np.asarray([row.get(self.spec.target, np.nan) for row in hist], dtype=float)
        soh_future = np.full((self.spec.horizon,), np.nan, dtype=float)
        mask = np.zeros((self.spec.horizon,), dtype=bool)
        for i, row in enumerate(future[: self.spec.horizon]):
            value = row.get(self.spec.target)
            if value is not None:
                soh_future[i] = float(value)
                mask[i] = True

        return TrainingWindow(
            cell_id=cell_id,
            dataset_id=hist[-1]["dataset_id"],
            batch_id=hist[-1].get("batch_id"),
            start_cycle_index=hist[0]["cycle_index"],
            x_window=x_window,
            u_window=u_window,
            soh_history=soh_history,
            soh_future=soh_future,
            mask=mask,
        )

    def _build_x_window(self, hist: list[dict[str, Any]]) -> np.ndarray | None:
        if not self.spec.cycle_channels:
            return None
        target_length = self._infer_sequence_length(hist)
        if target_length is None:
            return None

        cycles = []
        for row in hist:
            arrays = self.store.load_cycle_arrays(row)
            cycles.append(stack_cycle_channels(arrays, self.spec.cycle_channels, target_length))
        return np.stack(cycles, axis=0)

    def _build_u_window(self, hist: list[dict[str, Any]]) -> np.ndarray | None:
        if not self.spec.summary_channels:
            return None
        rows = []
        for row in hist:
            rows.append([_to_float_or_nan(row.get(name)) for name in self.spec.summary_channels])
        return np.asarray(rows, dtype=float)

    def _infer_sequence_length(self, hist: list[dict[str, Any]]) -> int | None:
        if self.spec.sequence_length is not None:
            return self.spec.sequence_length
        for row in hist:
            arrays = self.store.load_cycle_arrays(row)
            for name in self.spec.cycle_channels:
                values = arrays.get(name)
                if values is not None:
                    return int(np.asarray(values).reshape(-1).shape[0])
        return None


def _to_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
