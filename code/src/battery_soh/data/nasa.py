from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from battery_soh.data.io import ensure_dir, write_cells, write_cycle_arrays, write_cycles, write_jsonl
from battery_soh.data.parsers import BaseBatteryParser
from battery_soh.data.schema import CanonicalCell, CanonicalCycle, ParserScanResult, normalize_path
from battery_soh.data.splits import assert_no_cell_leakage, make_cell_level_split


NASA_DATASET_ID = "nasa_ames_li_ion"
MODEL_OPERATION_TYPE = "discharge"


class NasaParser(BaseBatteryParser):
    """Convert NASA Ames battery aging MAT files to canonical discharge cycles."""

    dataset_id = NASA_DATASET_ID

    def __init__(
        self,
        max_cells: int | None = None,
        archive_dir: Path | None = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> None:
        self.max_cells = max_cells
        self.archive_dir = Path(archive_dir).expanduser() if archive_dir is not None else None
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def scan(self, raw_dir: Path) -> ParserScanResult:
        raw_dir = Path(raw_dir).expanduser()
        files = _inventory(raw_dir, location="raw")
        if self.archive_dir is not None:
            files.extend(_inventory(self.archive_dir, location="archive"))
        raw_mat_count = sum(
            row["suffix"] == ".mat" and row["location"] == "raw" for row in files
        )
        archive_mat_count = sum(
            row["suffix"] == ".mat" and row["location"] == "archive" for row in files
        )
        zip_count = sum(row["suffix"] == ".zip" for row in files)
        notes = [
            "All NASA operations are indexed, but only discharge operations populate cycles.jsonl.",
            "Each cell's first discharge Capacity is the SoH denominator.",
            f"Found {raw_mat_count} MAT files under raw_dir, "
            f"{archive_mat_count} MAT files under archive_dir, and {zip_count} ZIP archives.",
        ]
        if raw_mat_count == 0 and archive_mat_count:
            notes.append("raw_dir is empty; parsing will use extracted MAT files from archive_dir.")
        elif raw_mat_count == 0:
            notes.append("No MAT files found. Extract the NASA ZIP archives before parsing.")
        return ParserScanResult(
            dataset_id=self.dataset_id,
            raw_dir=normalize_path(raw_dir),
            files=files,
            notes=notes,
        )

    def parse(self, raw_dir: Path, output_dir: Path) -> None:
        raw_dir = Path(raw_dir).expanduser()
        output_dir = ensure_dir(Path(output_dir).expanduser())
        paths = _mat_paths(raw_dir)
        source_root = raw_dir
        if not paths and self.archive_dir is not None:
            paths = _mat_paths(self.archive_dir)
            source_root = self.archive_dir
        paths, duplicate_sources = _deduplicate_mat_paths(paths)
        if self.max_cells is not None:
            paths = paths[: self.max_cells]
        if not paths:
            raise FileNotFoundError(
                f"No NASA MAT files found under {raw_dir}"
                + (
                    f" or {self.archive_dir}."
                    if self.archive_dir is not None
                    else ". Extract the downloaded ZIP files first."
                )
            )
        if duplicate_sources:
            print(
                f"Skipped {len(duplicate_sources)} byte-identical duplicate MAT files.",
                flush=True,
            )

        cells: list[CanonicalCell] = []
        cycles: list[CanonicalCycle] = []
        operations: list[dict[str, Any]] = []
        started_at = time.monotonic()
        for file_index, path in enumerate(paths, start=1):
            file_started_at = time.monotonic()
            print(
                f"[NASA {file_index}/{len(paths)}] start {path.name} "
                f"({path.stat().st_size / 1_000_000:.1f} MB)",
                flush=True,
            )
            cell, cell_cycles, cell_operations = self._parse_file(
                path,
                source_root,
                output_dir,
            )
            cells.append(cell)
            cycles.extend(cell_cycles)
            operations.extend(cell_operations)
            print(
                f"[NASA {file_index}/{len(paths)}] done {cell.cell_id}: "
                f"discharges={len(cell_cycles)}, operations={len(cell_operations)}, "
                f"elapsed={time.monotonic() - file_started_at:.1f}s",
                flush=True,
            )

        cell_ids = [cell.cell_id for cell in cells]
        if len(cell_ids) != len(set(cell_ids)):
            duplicates = sorted(cell_id for cell_id, count in Counter(cell_ids).items() if count > 1)
            raise ValueError(
                "Duplicate NASA cell IDs were found. Keep repeated battery IDs in distinct "
                f"batch directories: {duplicates}"
            )

        write_cells(output_dir / "cells.jsonl", cells)
        write_cycles(output_dir / "cycles.jsonl", cycles)
        write_jsonl(output_dir / "operations.jsonl", operations)
        write_jsonl(output_dir / "splits" / "all_external_test.jsonl", [{"test": cell_ids}])
        split = make_cell_level_split(
            cell_ids,
            seed=self.split_seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        ).to_dict()
        assert_no_cell_leakage(split)
        write_jsonl(
            output_dir / "splits" / f"cell_random_seed_{self.split_seed}.jsonl",
            [split],
        )
        write_jsonl(
            output_dir / "scan_summary.jsonl",
            [
                {
                    "dataset_id": self.dataset_id,
                    "num_cells": len(cells),
                    "num_discharge_cycles": len(cycles),
                    "num_operations": len(operations),
                    "operation_counts": dict(Counter(row["operation_type"] for row in operations)),
                    "soh_capacity_reference": "first_discharge_capacity",
                    "model_operation_type": MODEL_OPERATION_TYPE,
                    "num_duplicate_sources_skipped": len(duplicate_sources),
                    "duplicate_sources": duplicate_sources,
                    "split_seed": self.split_seed,
                    "split_counts": {name: len(ids) for name, ids in split.items()},
                    "source_files": [normalize_path(path) for path in paths],
                }
            ],
        )
        print(
            f"Finished parsing {len(cells)} NASA cells, {len(cycles)} discharge cycles, "
            f"and {len(operations)} operations in {time.monotonic() - started_at:.1f}s.",
            flush=True,
        )

    def _parse_file(
        self,
        path: Path,
        raw_dir: Path,
        output_dir: Path,
    ) -> tuple[CanonicalCell, list[CanonicalCycle], list[dict[str, Any]]]:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        battery_keys = [key for key in payload if not key.startswith("__")]
        if len(battery_keys) != 1:
            raise ValueError(f"Expected one battery variable in {path}, found {battery_keys}.")

        battery_id = battery_keys[0]
        battery = payload[battery_id]
        raw_operations = np.atleast_1d(battery.cycle).reshape(-1)
        batch_id = _infer_batch_id(path, raw_dir)
        cell_id = _cell_id(batch_id, battery_id)

        parsed_operations: list[dict[str, Any]] = []
        discharge_capacities: list[float | None] = []
        operation_counts: Counter[str] = Counter()
        for operation_index, raw_operation in enumerate(raw_operations):
            operation_type = _operation_type(raw_operation.type)
            operation_counts[operation_type] += 1
            type_index = operation_counts[operation_type]
            arrays, scalars = _operation_data(raw_operation.data, operation_type)
            _validate_core_arrays(arrays, path, operation_index, operation_type)

            array_path = (
                Path("operations")
                / cell_id
                / f"{operation_index:05d}_{operation_type}_{type_index:05d}.npz"
            )
            write_cycle_arrays(output_dir / array_path, arrays)
            capacity = _positive_finite_or_none(scalars.get("capacity_ah"))
            if operation_type == MODEL_OPERATION_TYPE:
                discharge_capacities.append(capacity)

            parsed_operations.append(
                {
                    "dataset_id": self.dataset_id,
                    "cell_id": cell_id,
                    "batch_id": batch_id,
                    "battery_id": battery_id,
                    "operation_index": operation_index,
                    "operation_type": operation_type,
                    "type_index": type_index,
                    "ambient_temperature_c": _finite_scalar_or_none(
                        raw_operation.ambient_temperature
                    ),
                    "timestamp": _parse_matlab_time(raw_operation.time),
                    "array_path": normalize_path(array_path),
                    "array_fields": sorted(arrays),
                    "capacity_ah": capacity,
                    "metadata": {"raw_scalar_fields": scalars},
                }
            )

        initial_capacity = discharge_capacities[0] if discharge_capacities else None
        if initial_capacity is None:
            raise ValueError(
                f"The first discharge operation has no positive finite Capacity in {path}."
            )

        canonical_cycles: list[CanonicalCycle] = []
        for operation in parsed_operations:
            if operation["operation_type"] != MODEL_OPERATION_TYPE:
                continue
            capacity = operation["capacity_ah"]
            array_path = operation["array_path"]
            with np.load(output_dir / array_path, allow_pickle=False) as stored:
                temperature_stats = _finite_stats(stored["T"] if "T" in stored else None)
            discharge_index = int(operation["type_index"])
            canonical_cycles.append(
                CanonicalCycle(
                    dataset_id=self.dataset_id,
                    cell_id=cell_id,
                    batch_id=batch_id,
                    cycle_index=discharge_index,
                    array_path=array_path,
                    cycle_number=float(discharge_index),
                    q_discharge=capacity,
                    t_min=temperature_stats["min"],
                    t_avg=temperature_stats["mean"],
                    t_max=temperature_stats["max"],
                    soh=(capacity / initial_capacity) if capacity is not None else None,
                    metadata={
                        "operation_type": MODEL_OPERATION_TYPE,
                        "operation_index": operation["operation_index"],
                        "ambient_temperature_c": operation["ambient_temperature_c"],
                        "timestamp": operation["timestamp"],
                        "initial_capacity_ah": initial_capacity,
                        "initial_capacity_source": "first_discharge_capacity",
                    },
                )
            )

        cell = CanonicalCell(
            dataset_id=self.dataset_id,
            cell_id=cell_id,
            batch_id=batch_id,
            source_file=normalize_path(path),
            cycle_life=float(len(canonical_cycles)),
            num_cycles=len(canonical_cycles),
            metadata={
                "battery_id": battery_id,
                "initial_capacity_ah": initial_capacity,
                "initial_capacity_source": "first_discharge_capacity",
                "model_operation_type": MODEL_OPERATION_TYPE,
                "operation_counts": dict(operation_counts),
                "num_operations": len(parsed_operations),
            },
        )
        return cell, canonical_cycles, parsed_operations


def _operation_data(data: Any, operation_type: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fields = set(getattr(data, "_fieldnames", ()) or ())
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}

    canonical_fields = {
        "Voltage_measured": "V",
        "Current_measured": "I",
        "Temperature_measured": "T",
        "Time": "t",
    }
    auxiliary_fields = {
        "Current_charge": "I_charge",
        "Voltage_charge": "V_charge",
        "Current_load": "I_load",
        "Voltage_load": "V_load",
    }
    for source, target in {**canonical_fields, **auxiliary_fields}.items():
        if source in fields:
            arrays[target] = np.asarray(getattr(data, source), dtype=np.float64).reshape(-1)

    if operation_type == "discharge" and "Capacity" in fields:
        scalars["capacity_ah"] = _finite_scalar_or_none(getattr(data, "Capacity"))

    known_fields = set(canonical_fields) | set(auxiliary_fields) | {"Capacity"}
    for field_name in sorted(fields - known_fields):
        value = np.asarray(getattr(data, field_name)).squeeze()
        if value.ndim == 0 and not np.iscomplexobj(value):
            scalars[field_name] = _json_scalar(value)
        else:
            arrays[f"raw_{field_name}"] = value
    return arrays, scalars


def _validate_core_arrays(
    arrays: dict[str, np.ndarray],
    path: Path,
    operation_index: int,
    operation_type: str,
) -> None:
    if operation_type not in {"charge", "discharge"}:
        return
    missing = sorted({"I", "V", "T", "t"} - set(arrays))
    if missing:
        raise ValueError(
            f"Missing NASA core arrays {missing} in {path}, operation={operation_index}."
        )
    lengths = {name: len(arrays[name]) for name in ("I", "V", "T", "t")}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Unequal NASA core array lengths in {path}, operation={operation_index}: {lengths}"
        )
    if not lengths["I"]:
        raise ValueError(f"Empty NASA operation in {path}, operation={operation_index}.")


def _parse_matlab_time(value: Any) -> str | None:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(values) < 6 or not np.isfinite(values[:6]).all():
        return None
    try:
        base = datetime(
            int(values[0]),
            int(values[1]),
            int(values[2]),
            int(values[3]),
            int(values[4]),
        )
        return (base + timedelta(seconds=float(values[5]))).isoformat()
    except (OverflowError, ValueError):
        return None


def _operation_type(value: Any) -> str:
    operation_type = str(np.asarray(value).squeeze()).strip().lower()
    return operation_type or "unknown"


def _infer_batch_id(path: Path, raw_dir: Path) -> str | None:
    relative_parent = path.relative_to(raw_dir).parent
    if relative_parent == Path("."):
        return None
    return normalize_path(relative_parent)


def _cell_id(batch_id: str | None, battery_id: str) -> str:
    if batch_id is None:
        return battery_id
    safe_batch = re.sub(r"[^A-Za-z0-9._-]+", "_", batch_id).strip("_")
    return f"{safe_batch}__{battery_id}"


def _inventory(root: Path, location: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {
            "path": normalize_path(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
            "location": location,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _mat_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.mat")
        if not path.name.startswith(("._", "."))
    )


def _deduplicate_mat_paths(
    paths: list[Path],
) -> tuple[list[Path], list[dict[str, str]]]:
    """Drop byte-identical files repeated across NASA archive subdirectories."""
    by_size: dict[int, list[Path]] = {}
    for path in paths:
        by_size.setdefault(path.stat().st_size, []).append(path)

    duplicate_to_kept: dict[Path, Path] = {}
    for same_size_paths in by_size.values():
        if len(same_size_paths) < 2:
            continue
        by_digest: dict[str, Path] = {}
        for path in same_size_paths:
            digest = _file_sha256(path)
            kept = by_digest.setdefault(digest, path)
            if kept != path:
                duplicate_to_kept[path] = kept

    unique_paths = [path for path in paths if path not in duplicate_to_kept]
    duplicates = [
        {
            "path": normalize_path(path),
            "duplicate_of": normalize_path(duplicate_to_kept[path]),
        }
        for path in paths
        if path in duplicate_to_kept
    ]
    return unique_paths, duplicates


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_finite_or_none(value: Any) -> float | None:
    result = _finite_scalar_or_none(value)
    return result if result is not None and result > 0 else None


def _finite_scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if len(array) != 1 or np.iscomplexobj(array):
        return None
    result = float(array[0])
    return result if np.isfinite(result) else None


def _json_scalar(value: Any) -> str | int | float | bool | None:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (np.bool_, bool)):
        return bool(scalar)
    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    if isinstance(scalar, (np.floating, float)):
        result = float(scalar)
        return result if np.isfinite(result) else None
    if isinstance(scalar, (str, np.str_)):
        return str(scalar)
    return str(scalar)


def _finite_stats(value: Any) -> dict[str, float | None]:
    if value is None:
        return {"min": None, "mean": None, "max": None}
    array = np.asarray(value, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
    }
