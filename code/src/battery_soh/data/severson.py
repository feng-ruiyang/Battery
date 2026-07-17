from __future__ import annotations

import json
import pickle
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from battery_soh.data.io import ensure_dir, write_cells, write_cycle_arrays, write_cycles, write_jsonl
from battery_soh.data.parsers import BaseBatteryParser
from battery_soh.data.raw_cleaning import clean_raw_json, load_and_clean_raw_json
from battery_soh.data.schema import CanonicalCell, CanonicalCycle, ParserScanResult, normalize_path
from battery_soh.data.splits import make_cell_level_split


SEVERSON_DATASET_ID = "severson_lfp_fast_charge"
SUMMARY_ALIASES = {
    "cycle": ("cycle", "cycle_index"),
    "q_discharge": ("QD", "QDischarge", "discharge_capacity"),
    "q_charge": ("QC", "QCharge", "charge_capacity"),
    "internal_resistance": ("IR", "dc_internal_resistance"),
    "t_avg": ("Tavg", "temperature_average"),
    "t_min": ("Tmin", "temperature_minimum"),
    "t_max": ("Tmax", "temperature_maximum"),
    "charge_time": ("chargetime", "charge_time", "charge_duration"),
}
JSON_ARRAY_ALIASES = {
    "I": "current",
    "V": "voltage",
    "T": "temperature",
    "Qc": "charge_capacity",
    "Qd": "discharge_capacity",
    "IR": "internal_resistance",
    "step_type": "step_type",
    "point_index": "point_index",
}
DEFAULT_CYCLE_ARRAYS = (
    "I",
    "V",
    "T",
    "Qc",
    "Qd",
    "IR",
    "Qdlin",
    "Tdlin",
    "dQdV",
    "t",
    "step_type",
    "point_index",
)


class SeversonParser(BaseBatteryParser):
    dataset_id = SEVERSON_DATASET_ID

    def __init__(
        self,
        include_arrays: tuple[str, ...] = DEFAULT_CYCLE_ARRAYS,
        split_seed: int = 42,
        max_cells: int | None = None,
    ) -> None:
        self.include_arrays = include_arrays
        self.split_seed = split_seed
        self.max_cells = max_cells

    def scan(self, raw_dir: Path) -> ParserScanResult:
        raw_dir = Path(raw_dir)
        files: list[dict[str, Any]] = []
        notes: list[str] = []
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(raw_dir)
            record: dict[str, Any] = {
                "path": normalize_path(rel),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
            if path.suffix.lower() == ".zip":
                record["zip_members_preview"] = _preview_zip(path)
                notes.append(f"Found zip archive: {rel}")
            files.append(record)

        if not files:
            notes.append("No files found. Put FastCharge.zip or extracted .pkl/.mat files here.")
        if not any(f["suffix"] in {".pkl", ".json", ".zip"} for f in files):
            notes.append("No .pkl, .json, or .zip file found.")
        return ParserScanResult(
            dataset_id=self.dataset_id,
            raw_dir=normalize_path(raw_dir),
            files=files,
            notes=notes,
        )

    def parse(self, raw_dir: Path, output_dir: Path) -> None:
        raw_dir = Path(raw_dir)
        output_dir = ensure_dir(Path(output_dir))

        pkl_files = sorted(raw_dir.rglob("*.pkl"))
        json_files = sorted(raw_dir.rglob("*_structure.json"))
        zip_files = sorted(raw_dir.rglob("*.zip"))
        if not pkl_files and not json_files and not zip_files:
            mat_files = sorted(raw_dir.rglob("*.mat"))
            if mat_files:
                raise NotImplementedError(
                    "Found .mat files but no .pkl files. MAT parsing will be implemented after "
                    "we inspect the downloaded sample. For now, use the author notebooks to build pkl."
                )
            raise FileNotFoundError(f"No .pkl, *_structure.json, .zip, or .mat files found under {raw_dir}.")

        cells: list[CanonicalCell] = []
        cycles: list[CanonicalCycle] = []
        source_files: list[str] = []
        started_at = time.monotonic()
        print(
            "Parsing Severson files: "
            f"{len(json_files)} json, {len(zip_files)} zip, {len(pkl_files)} pkl "
            f"from {raw_dir}",
            flush=True,
        )
        for file_index, json_path in enumerate(json_files, start=1):
            if self._hit_cell_limit(cells):
                print(f"Reached max_cells={self.max_cells}; stopping JSON parsing.", flush=True)
                break
            file_started_at = time.monotonic()
            print(
                f"[json {file_index}/{len(json_files)}] start {json_path.name} "
                f"({json_path.stat().st_size / 1_000_000:.1f} MB)",
                flush=True,
            )
            batch_cells, batch_cycles = self._parse_json_file(json_path, output_dir)
            cells.extend(batch_cells)
            cycles.extend(batch_cycles)
            source_files.append(normalize_path(json_path))
            _log_parse_progress(
                source_name=json_path.name,
                file_started_at=file_started_at,
                started_at=started_at,
                batch_cells=len(batch_cells),
                batch_cycles=len(batch_cycles),
                total_cells=len(cells),
                total_cycles=len(cycles),
            )

        for file_index, zip_path in enumerate(zip_files, start=1):
            if self._hit_cell_limit(cells):
                print(f"Reached max_cells={self.max_cells}; stopping zip parsing.", flush=True)
                break
            file_started_at = time.monotonic()
            print(
                f"[zip {file_index}/{len(zip_files)}] start {zip_path.name} "
                f"({zip_path.stat().st_size / 1_000_000:.1f} MB)",
                flush=True,
            )
            batch_cells, batch_cycles, parsed_members = self._parse_json_zip(zip_path, output_dir, len(cells))
            cells.extend(batch_cells)
            cycles.extend(batch_cycles)
            source_files.extend(f"{normalize_path(zip_path)}::{member}" for member in parsed_members)
            _log_parse_progress(
                source_name=zip_path.name,
                file_started_at=file_started_at,
                started_at=started_at,
                batch_cells=len(batch_cells),
                batch_cycles=len(batch_cycles),
                total_cells=len(cells),
                total_cycles=len(cycles),
            )

        for file_index, pkl_path in enumerate(pkl_files, start=1):
            if self._hit_cell_limit(cells):
                print(f"Reached max_cells={self.max_cells}; stopping pkl parsing.", flush=True)
                break
            file_started_at = time.monotonic()
            print(
                f"[pkl {file_index}/{len(pkl_files)}] start {pkl_path.name} "
                f"({pkl_path.stat().st_size / 1_000_000:.1f} MB)",
                flush=True,
            )
            batch_id = _infer_batch_id(pkl_path)
            batch_cells, batch_cycles = self._parse_pkl_file(pkl_path, output_dir, batch_id)
            cells.extend(batch_cells)
            cycles.extend(batch_cycles)
            source_files.append(normalize_path(pkl_path))
            _log_parse_progress(
                source_name=pkl_path.name,
                file_started_at=file_started_at,
                started_at=started_at,
                batch_cells=len(batch_cells),
                batch_cycles=len(batch_cycles),
                total_cells=len(cells),
                total_cycles=len(cycles),
            )

        if not cells:
            raise FileNotFoundError(f"No parseable Severson/BEEP cell records found under {raw_dir}.")

        write_cells(output_dir / "cells.jsonl", cells)
        write_cycles(output_dir / "cycles.jsonl", cycles)
        split = make_cell_level_split([cell.cell_id for cell in cells], seed=self.split_seed)
        write_jsonl(output_dir / "splits" / f"cell_random_seed_{self.split_seed}.jsonl", [split.to_dict()])
        write_jsonl(
            output_dir / "scan_summary.jsonl",
            [
                {
                    "dataset_id": self.dataset_id,
                    "num_cells": len(cells),
                    "num_cycles": len(cycles),
                    "source_files": source_files,
                }
            ],
        )
        print(
            f"Finished parsing {len(cells)} cells and {len(cycles)} cycles "
            f"in {_format_duration(time.monotonic() - started_at)}.",
            flush=True,
        )

    def _hit_cell_limit(self, cells: list[CanonicalCell]) -> bool:
        return self.max_cells is not None and len(cells) >= self.max_cells

    def _parse_json_file(
        self,
        json_path: Path,
        output_dir: Path,
    ) -> tuple[list[CanonicalCell], list[CanonicalCycle]]:
        data = load_and_clean_raw_json(json_path)
        return self._parse_beep_structure(data, output_dir, source_file=normalize_path(json_path))

    def _parse_json_zip(
        self,
        zip_path: Path,
        output_dir: Path,
        cells_seen: int,
    ) -> tuple[list[CanonicalCell], list[CanonicalCycle], list[str]]:
        cells: list[CanonicalCell] = []
        cycles: list[CanonicalCycle] = []
        parsed_members: list[str] = []
        with zipfile.ZipFile(zip_path) as zf:
            members = [info for info in zf.infolist() if info.filename.endswith("_structure.json")]
            for info in sorted(members, key=lambda item: item.file_size):
                if self.max_cells is not None and cells_seen + len(cells) >= self.max_cells:
                    break
                with zf.open(info) as f:
                    data = json.load(f)
                data = clean_raw_json(f"{normalize_path(zip_path)}::{info.filename}", data)
                batch_cells, batch_cycles = self._parse_beep_structure(
                    data,
                    output_dir,
                    source_file=f"{normalize_path(zip_path)}::{info.filename}",
                    fallback_cell_id=Path(info.filename).stem.replace("_structure", ""),
                )
                cells.extend(batch_cells)
                cycles.extend(batch_cycles)
                parsed_members.append(info.filename)
        return cells, cycles, parsed_members

    def _parse_beep_structure(
        self,
        data: dict[str, Any],
        output_dir: Path,
        source_file: str,
        fallback_cell_id: str | None = None,
    ) -> tuple[list[CanonicalCell], list[CanonicalCycle]]:
        if not isinstance(data, dict):
            raise TypeError(f"Expected BEEP JSON dict, got {type(data)!r}: {source_file}")
        summary = data.get("summary") or {}
        interpolated = data.get("cycles_interpolated") or {}
        if not isinstance(summary, dict) or not isinstance(interpolated, dict):
            return [], []

        barcode = _to_optional_str(data.get("barcode"))
        channel = _to_optional_str(data.get("channel_id"))
        protocol = _to_optional_str(data.get("protocol"))
        cell_id = _clean_cell_id(fallback_cell_id or barcode or Path(source_file).stem)
        batch_id = _infer_batch_id_from_protocol(protocol) or _infer_batch_id(Path(source_file.split("::")[0]))
        qd = _summary_array(summary, "q_discharge")
        soh_values = _compute_soh(qd)
        cycle_indices = _summary_array(summary, "cycle")
        num_cycles = _safe_len(cycle_indices)

        cell = CanonicalCell(
            dataset_id=self.dataset_id,
            cell_id=cell_id,
            batch_id=batch_id,
            source_file=source_file,
            cycle_life=None,
            charge_policy=protocol,
            barcode=barcode,
            channel=channel,
            num_cycles=num_cycles,
            metadata={
                "module": data.get("@module"),
                "class": data.get("@class"),
                "version": data.get("@version"),
            },
        )
        cycles: list[CanonicalCycle] = []
        grouped = _group_interpolated_cycles(interpolated)
        summary_cycle_ids = [int(x) for x in np.asarray(cycle_indices if cycle_indices is not None else [], dtype=int)]
        if not summary_cycle_ids:
            summary_cycle_ids = sorted(grouped.keys())

        for row_index, cycle_id in enumerate(summary_cycle_ids):
            arrays = grouped.get(cycle_id, {})
            array_rel_path = Path("arrays") / cell_id / f"cycle_{cycle_id:05d}.npz"
            if arrays:
                write_cycle_arrays(output_dir / array_rel_path, arrays)
                array_path = normalize_path(array_rel_path)
            else:
                array_path = None
            cycles.append(
                CanonicalCycle(
                    dataset_id=self.dataset_id,
                    cell_id=cell_id,
                    batch_id=batch_id,
                    cycle_index=cycle_id,
                    array_path=array_path,
                    cycle_number=float(cycle_id),
                    q_discharge=_summary_scalar_by_row(summary, "q_discharge", row_index),
                    q_charge=_summary_scalar_by_row(summary, "q_charge", row_index),
                    internal_resistance=_summary_scalar_by_row(
                        summary, "internal_resistance", row_index
                    ),
                    t_min=_summary_scalar_by_row(summary, "t_min", row_index),
                    t_avg=_summary_scalar_by_row(summary, "t_avg", row_index),
                    t_max=_summary_scalar_by_row(summary, "t_max", row_index),
                    charge_time=_summary_scalar_by_row(summary, "charge_time", row_index),
                    soh=_array_scalar(soh_values, row_index),
                    metadata={
                        "date_time_iso": _summary_raw_scalar(summary, "date_time_iso", row_index),
                        "paused": _summary_raw_scalar(summary, "paused", row_index),
                    },
                )
            )
        return [cell], cycles

    def _parse_pkl_file(
        self,
        pkl_path: Path,
        output_dir: Path,
        batch_id: str | None,
    ) -> tuple[list[CanonicalCell], list[CanonicalCycle]]:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected pkl to contain a dict, got {type(data)!r}: {pkl_path}")

        cells: list[CanonicalCell] = []
        cycles: list[CanonicalCycle] = []
        for raw_cell_id, cell_data in sorted(data.items(), key=lambda item: str(item[0])):
            cell_id = _clean_cell_id(raw_cell_id)
            if not isinstance(cell_data, dict):
                continue

            summary = cell_data.get("summary", {}) or {}
            cycle_dict = cell_data.get("cycles", {}) or {}
            qd = _summary_array(summary, "q_discharge")
            soh_values = _compute_soh(qd)
            num_cycles = len(cycle_dict) if cycle_dict else _safe_len(qd)

            cells.append(
                CanonicalCell(
                    dataset_id=self.dataset_id,
                    cell_id=cell_id,
                    batch_id=batch_id,
                    source_file=normalize_path(pkl_path),
                    cycle_life=_to_optional_float(cell_data.get("cycle_life")),
                    charge_policy=_to_optional_str(cell_data.get("charge_policy")),
                    barcode=_to_optional_str(cell_data.get("barcode")),
                    channel=_to_optional_str(cell_data.get("channel") or cell_data.get("channel_id")),
                    num_cycles=num_cycles,
                )
            )

            for cycle_key, cycle_data in sorted(cycle_dict.items(), key=lambda item: _cycle_sort_key(item[0])):
                cycle_index = _cycle_sort_key(cycle_key)
                array_rel_path = Path("arrays") / cell_id / f"cycle_{cycle_index:05d}.npz"
                arrays = {
                    name: np.asarray(cycle_data[name])
                    for name in self.include_arrays
                    if isinstance(cycle_data, dict) and name in cycle_data
                }
                write_cycle_arrays(output_dir / array_rel_path, arrays)
                cycles.append(
                    CanonicalCycle(
                        dataset_id=self.dataset_id,
                        cell_id=cell_id,
                        batch_id=batch_id,
                        cycle_index=cycle_index,
                        array_path=normalize_path(array_rel_path),
                        cycle_number=_summary_scalar(summary, "cycle", cycle_index),
                        q_discharge=_summary_scalar(summary, "q_discharge", cycle_index),
                        q_charge=_summary_scalar(summary, "q_charge", cycle_index),
                        internal_resistance=_summary_scalar(
                            summary, "internal_resistance", cycle_index
                        ),
                        t_min=_summary_scalar(summary, "t_min", cycle_index),
                        t_avg=_summary_scalar(summary, "t_avg", cycle_index),
                        t_max=_summary_scalar(summary, "t_max", cycle_index),
                        charge_time=_summary_scalar(summary, "charge_time", cycle_index),
                        soh=_array_scalar(soh_values, cycle_index),
                    )
                )
        return cells, cycles


def _preview_zip(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return [
            {"filename": info.filename, "size_bytes": info.file_size}
            for info in zf.infolist()[:limit]
        ]


def _log_parse_progress(
    source_name: str,
    file_started_at: float,
    started_at: float,
    batch_cells: int,
    batch_cycles: int,
    total_cells: int,
    total_cycles: int,
) -> None:
    file_elapsed = time.monotonic() - file_started_at
    total_elapsed = time.monotonic() - started_at
    print(
        f"  done {source_name}: +{batch_cells} cells, +{batch_cycles} cycles "
        f"in {_format_duration(file_elapsed)}; totals={total_cells} cells/"
        f"{total_cycles} cycles; elapsed={_format_duration(total_elapsed)}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    seconds_int = int(seconds)
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{seconds:.1f}s"


def _extract_pkl_members(zip_files: list[Path], output_dir: Path) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    extracted: list[Path] = []
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if not info.filename.lower().endswith(".pkl"):
                    continue
                target = output_dir / Path(info.filename).name
                if not target.exists():
                    with zf.open(info) as src, target.open("wb") as dst:
                        dst.write(src.read())
                extracted.append(target)
    return sorted(extracted)


def _infer_batch_id(path: Path) -> str | None:
    name = path.name.lower()
    match = re.search(r"(batch[_-]?\d+|b\d+)", name)
    if match:
        return match.group(1).replace("_", "").replace("-", "")
    if "2017-05-12" in name:
        return "batch1"
    if "2017-06-30" in name:
        return "batch2"
    if "2018-04-12" in name:
        return "batch3"
    return None


def _infer_batch_id_from_protocol(protocol: str | None) -> str | None:
    if protocol is None:
        return None
    text = protocol.lower()
    match = re.search(r"batch\s*([0-9]+)", text)
    if match:
        return f"batch{match.group(1)}"
    match = re.search(r"batch([0-9]+)", text)
    if match:
        return f"batch{match.group(1)}"
    return None


def _clean_cell_id(raw_cell_id: Any) -> str:
    text = str(raw_cell_id).strip()
    text = text.replace("/", "_").replace(" ", "_")
    return text


def _summary_array(summary: dict[str, Any], canonical_name: str) -> np.ndarray | None:
    for key in SUMMARY_ALIASES[canonical_name]:
        if key in summary:
            return np.asarray(summary[key]).reshape(-1)
    return None


def _summary_scalar(summary: dict[str, Any], canonical_name: str, cycle_index: int) -> float | None:
    return _array_scalar(_summary_array(summary, canonical_name), cycle_index)


def _summary_scalar_by_row(summary: dict[str, Any], canonical_name: str, row_index: int) -> float | None:
    return _array_scalar(_summary_array(summary, canonical_name), row_index)


def _summary_raw_scalar(summary: dict[str, Any], key: str, row_index: int) -> Any:
    if key not in summary:
        return None
    values = summary[key]
    if not isinstance(values, list) or row_index >= len(values):
        return None
    return values[row_index]


def _array_scalar(array: np.ndarray | None, index: int) -> float | None:
    if array is None:
        return None
    if index < 0 or index >= len(array):
        return None
    return _to_optional_float(array[index])


def _compute_soh(q_discharge: np.ndarray | None) -> np.ndarray | None:
    if q_discharge is None or len(q_discharge) == 0:
        return None
    q = np.asarray(q_discharge, dtype=float)
    valid = q[np.isfinite(q) & (q > 0)]
    if len(valid) == 0:
        return None
    return q / valid[0]


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if len(array) == 0:
        return None
    try:
        result = float(array[0])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    array = np.asarray(value)
    if array.shape:
        value = array.reshape(-1)[0]
    return str(value)


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(len(value))
    except TypeError:
        return None


def _cycle_sort_key(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0))
        return 0


def _group_interpolated_cycles(interpolated: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    cycle_ids = np.asarray(interpolated.get("cycle_index", []), dtype=int)
    if cycle_ids.size == 0:
        return {}
    grouped: dict[int, dict[str, np.ndarray]] = {}
    unique_ids = np.unique(cycle_ids)
    for cycle_id in unique_ids:
        mask = cycle_ids == cycle_id
        arrays: dict[str, np.ndarray] = {}
        for canonical_name, json_name in JSON_ARRAY_ALIASES.items():
            if json_name not in interpolated:
                continue
            values = np.asarray(interpolated[json_name])
            if values.shape[0] != cycle_ids.shape[0]:
                continue
            arrays[canonical_name] = values[mask]
        arrays["point_index"] = np.arange(mask.sum(), dtype=np.int32)
        grouped[int(cycle_id)] = arrays
    return grouped
