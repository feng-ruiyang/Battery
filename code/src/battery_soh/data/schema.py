from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class CanonicalCell:
    """Cell-level metadata shared by downstream preprocessing and splits."""

    dataset_id: str
    cell_id: str
    batch_id: str | None = None
    source_file: str | None = None
    cycle_life: float | None = None
    charge_policy: str | None = None
    barcode: str | None = None
    channel: str | None = None
    num_cycles: int | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalCycle:
    """One row per cell-cycle pair.

    Large variable-length arrays are stored outside this record as a `.npz` file.
    This record keeps only scalar summary values and the relative array path.
    """

    dataset_id: str
    cell_id: str
    cycle_index: int
    batch_id: str | None = None
    array_path: str | None = None
    cycle_number: float | None = None
    q_discharge: float | None = None
    q_charge: float | None = None
    internal_resistance: float | None = None
    t_min: float | None = None
    t_avg: float | None = None
    t_max: float | None = None
    charge_time: float | None = None
    soh: float | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingWindowSpec:
    """Configuration for converting canonical cycles to model tensors."""

    window: int
    horizon: int
    cycle_channels: tuple[str, ...] = ("V", "I", "T")
    summary_channels: tuple[str, ...] = ("q_discharge", "internal_resistance", "t_avg")
    target: str = "soh"
    sequence_length: int | None = None


@dataclass(frozen=True)
class ParserScanResult:
    dataset_id: str
    raw_dir: str
    files: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


def normalize_path(path: Path) -> str:
    return path.as_posix()

