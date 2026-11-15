from __future__ import annotations

import json
import pickle
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from battery_soh.data.io import ensure_dir, write_cells, write_cycle_arrays, write_cycles, write_jsonl
from battery_soh.data.parsers import BaseBatteryParser
from battery_soh.data.schema import CanonicalCell, CanonicalCycle, ParserScanResult, normalize_path


HUST_DATASET_ID = "hust_lfp"


class HustParser(BaseBatteryParser):
    """Convert standardized BatteryLife HUST pickle files to canonical cycles."""

    dataset_id = HUST_DATASET_ID

    def __init__(self, max_cells: int | None = None) -> None:
        self.max_cells = max_cells

    def scan(self, raw_dir: Path) -> ParserScanResult:
        raw_dir = Path(raw_dir)
        files = [
            {
                "path": normalize_path(path.relative_to(raw_dir)),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
            for path in sorted(raw_dir.rglob("*"))
            if path.is_file()
        ]
        notes = []
        if not files:
            notes.append("No files found.")
        if not any(record["suffix"] == ".pkl" for record in files):
            notes.append("No BatteryLife HUST pickle files found.")
        notes.append("HUST temperature is optional and remains missing when absent.")
        return ParserScanResult(
            dataset_id=self.dataset_id,
            raw_dir=normalize_path(raw_dir),
            files=files,
            notes=notes,
        )

    def parse(self, raw_dir: Path, output_dir: Path) -> None:
        raw_dir = Path(raw_dir)
        output_dir = ensure_dir(Path(output_dir))
        paths = sorted(raw_dir.rglob("*.pkl"))
        if self.max_cells is not None:
            paths = paths[: self.max_cells]
        if not paths:
            raise FileNotFoundError(f"No HUST pickle files found under {raw_dir}.")

        cells: list[CanonicalCell] = []
        cycles: list[CanonicalCycle] = []
        started_at = time.monotonic()
        for file_index, path in enumerate(paths, start=1):
            file_started_at = time.monotonic()
            print(
                f"[HUST {file_index}/{len(paths)}] start {path.name} "
                f"({path.stat().st_size / 1_000_000:.1f} MB)",
                flush=True,
            )
            cell, cell_cycles = self._parse_file(path, output_dir)
            cells.append(cell)
            cycles.extend(cell_cycles)
            print(
                f"[HUST {file_index}/{len(paths)}] done {cell.cell_id}: "
                f"cycles={len(cell_cycles)}, elapsed={time.monotonic() - file_started_at:.1f}s",
                flush=True,
            )

        cell_ids = [cell.cell_id for cell in cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Duplicate HUST cell_id values were found.")
        write_cells(output_dir / "cells.jsonl", cells)
        write_cycles(output_dir / "cycles.jsonl", cycles)
        write_jsonl(output_dir / "splits" / "all_external_test.jsonl", [{"test": cell_ids}])
        write_jsonl(
            output_dir / "scan_summary.jsonl",
            [
                {
                    "dataset_id": self.dataset_id,
                    "num_cells": len(cells),
                    "num_cycles": len(cycles),
                    "temperature_available_cells": sum(
                        bool(cell.metadata.get("temperature_available")) for cell in cells
                    ),
                    "source_files": [normalize_path(path) for path in paths],
                }
            ],
        )
        print(
            f"Finished parsing {len(cells)} HUST cells and {len(cycles)} cycles "
            f"in {time.monotonic() - started_at:.1f}s.",
            flush=True,
        )

    def _parse_file(
        self,
        path: Path,
        output_dir: Path,
    ) -> tuple[CanonicalCell, list[CanonicalCycle]]:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Expected a mapping in {path}, got {type(payload)!r}.")
        raw_cycles = payload.get("cycle_data")
        if not isinstance(raw_cycles, list) or not raw_cycles:
            raise ValueError(f"Expected a non-empty cycle_data list in {path}.")

        ordered_cycles = sorted(raw_cycles, key=lambda row: int(row["cycle_number"]))
        cycle_numbers = [int(row["cycle_number"]) for row in ordered_cycles]
        if len(cycle_numbers) != len(set(cycle_numbers)):
            raise ValueError(f"Duplicate cycle_number values in {path}.")

        q_discharge_values = [_finite_max(row.get("discharge_capacity_in_Ah")) for row in ordered_cycles]
        initial_q_discharge = next(
            (value for value in q_discharge_values if value is not None and value > 0),
            None,
        )
        if initial_q_discharge is None:
            raise ValueError(f"No positive discharge capacity found in {path}.")

        cell_id = str(payload.get("cell_id") or path.stem)
        batch_id = cell_id.rsplit("-", 1)[0] if "-" in cell_id else None
        canonical_cycles: list[CanonicalCycle] = []
        temperature_available = False
        for raw_cycle, q_discharge in zip(ordered_cycles, q_discharge_values):
            cycle_number = int(raw_cycle["cycle_number"])
            arrays = _cycle_arrays(raw_cycle, path=path, cycle_number=cycle_number)
            temperature = arrays.get("T")
            temperature_stats = _finite_stats(temperature)
            temperature_available = temperature_available or temperature_stats["mean"] is not None
            q_charge = _finite_max(raw_cycle.get("charge_capacity_in_Ah"))
            internal_resistance = _finite_mean(raw_cycle.get("internal_resistance_in_ohm"))
            array_path = Path("arrays") / cell_id / f"cycle_{cycle_number:05d}.npz"
            write_cycle_arrays(output_dir / array_path, arrays)
            canonical_cycles.append(
                CanonicalCycle(
                    dataset_id=self.dataset_id,
                    cell_id=cell_id,
                    batch_id=batch_id,
                    cycle_index=cycle_number,
                    array_path=normalize_path(array_path),
                    cycle_number=float(cycle_number),
                    q_discharge=q_discharge,
                    q_charge=q_charge,
                    internal_resistance=internal_resistance,
                    t_min=temperature_stats["min"],
                    t_avg=temperature_stats["mean"],
                    t_max=temperature_stats["max"],
                    soh=(q_discharge / initial_q_discharge) if q_discharge is not None else None,
                    metadata={"temperature_available": temperature_stats["mean"] is not None},
                )
            )

        cell = CanonicalCell(
            dataset_id=self.dataset_id,
            cell_id=cell_id,
            batch_id=batch_id,
            source_file=normalize_path(path),
            charge_policy=_protocol_text(payload.get("charge_protocol")),
            num_cycles=len(canonical_cycles),
            metadata={
                "form_factor": _json_value(payload.get("form_factor")),
                "anode_material": _json_value(payload.get("anode_material")),
                "cathode_material": _json_value(payload.get("cathode_material")),
                "nominal_capacity_in_Ah": _json_value(payload.get("nominal_capacity_in_Ah")),
                "depth_of_charge": _json_value(payload.get("depth_of_charge")),
                "depth_of_discharge": _json_value(payload.get("depth_of_discharge")),
                "max_voltage_limit_in_V": _json_value(payload.get("max_voltage_limit_in_V")),
                "min_voltage_limit_in_V": _json_value(payload.get("min_voltage_limit_in_V")),
                "charge_protocol": _json_value(payload.get("charge_protocol")),
                "discharge_protocol": _json_value(payload.get("discharge_protocol")),
                "soc_interval": _json_value(payload.get("SOC_interval")),
                "temperature_available": temperature_available,
            },
        )
        return cell, canonical_cycles


def _cycle_arrays(
    cycle: Mapping[str, Any],
    path: Path,
    cycle_number: int,
) -> dict[str, np.ndarray]:
    aliases = {
        "I": "current_in_A",
        "V": "voltage_in_V",
        "Qc": "charge_capacity_in_Ah",
        "Qd": "discharge_capacity_in_Ah",
        "t": "time_in_s",
        "T": "temperature_in_C",
        "IR": "internal_resistance_in_ohm",
    }
    arrays: dict[str, np.ndarray] = {}
    for target, source in aliases.items():
        value = cycle.get(source)
        if value is None:
            continue
        array = np.asarray(value, dtype=float).reshape(-1)
        if len(array) > 0:
            arrays[target] = array

    if "I" not in arrays or "V" not in arrays:
        raise ValueError(f"Missing I/V in {path}, cycle={cycle_number}.")
    expected_length = len(arrays["I"])
    mismatched = {
        name: len(array)
        for name, array in arrays.items()
        if len(array) != expected_length
    }
    if mismatched:
        raise ValueError(
            f"Unequal HUST array lengths in {path}, cycle={cycle_number}: "
            f"expected={expected_length}, mismatched={mismatched}"
        )
    if "t" in arrays:
        finite_time = arrays["t"][np.isfinite(arrays["t"])]
        if len(finite_time):
            arrays["t"] = arrays["t"] - finite_time[0]
    return arrays


def _finite_max(values: Any) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(finite.max()) if len(finite) else None


def _finite_mean(values: Any) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else None


def _finite_stats(values: Any) -> dict[str, float | None]:
    if values is None:
        return {"min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
    }


def _protocol_text(protocol: Any) -> str | None:
    if protocol is None:
        return None
    return json.dumps(_json_value(protocol), ensure_ascii=True, sort_keys=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
