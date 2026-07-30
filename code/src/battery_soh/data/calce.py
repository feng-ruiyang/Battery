from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from battery_soh.data.io import ensure_dir, write_cells, write_cycle_arrays, write_cycles, write_jsonl
from battery_soh.data.parsers import BaseBatteryParser
from battery_soh.data.schema import CanonicalCell, CanonicalCycle, ParserScanResult, normalize_path
from battery_soh.data.splits import assert_no_cell_leakage, make_cell_level_split


CALCE_DATASET_ID = "calce_cs2"
CS2_NOMINAL_CAPACITY_AH = 1.1


@dataclass(frozen=True)
class CalceFilterConfig:
    min_q_ah: float = 0.03
    max_charge_discharge_mismatch: float = 0.25
    min_voltage_max_v: float = 4.0
    max_voltage_min_v: float = 2.9
    min_points: int = 50
    local_outlier_window: int = 3
    local_outlier_drop: float = 0.08


@dataclass(frozen=True)
class _ExcelSource:
    cell_id: str
    source_path: Path
    member_name: str | None
    sort_time: datetime


@dataclass
class _CandidateCycle:
    local_cycle: int
    arrays: dict[str, np.ndarray]
    q_discharge: float
    q_charge: float
    cap_mismatch: float
    voltage_min: float
    voltage_max: float
    internal_resistance: float | None
    source_file: str
    source_member: str | None
    source_date: str | None
    n_points: int
    keep_complete: bool
    drop_reason: str | None
    soh: float
    is_local_outlier: bool = False


class CalceParser(BaseBatteryParser):
    """Convert CALCE CS2 Arbin Excel logs into canonical full-cycle records."""

    dataset_id = CALCE_DATASET_ID

    def __init__(
        self,
        max_cells: int | None = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        nominal_capacity_ah: float = CS2_NOMINAL_CAPACITY_AH,
        filters: CalceFilterConfig | None = None,
        cell_pattern: str = r"CS2[_-]\d+",
    ) -> None:
        self.max_cells = max_cells
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.nominal_capacity_ah = nominal_capacity_ah
        self.filters = filters or CalceFilterConfig()
        self.cell_pattern = re.compile(cell_pattern, re.IGNORECASE)

    def scan(self, raw_dir: Path) -> ParserScanResult:
        raw_dir = Path(raw_dir).expanduser()
        files = _inventory(raw_dir)
        cell_ids = sorted(
            cell_id
            for row in files
            if (cell_id := _cell_id_from_path(Path(row["path"]))) is not None
        )
        notes = [
            "CALCE CS2 logs store cumulative charge/discharge capacities inside each Excel file; "
            "per-cycle capacity is computed as max-min within Cycle_Index.",
            f"SoH is discharge capacity divided by the {self.nominal_capacity_ah:g} Ah nominal CS2 capacity.",
            "Incomplete boundary cycles and local diagnostic outliers are filtered before writing cycles.jsonl.",
            f"Detected {len(cell_ids)} candidate CS2 cells.",
        ]
        if not files:
            notes.append("No files found.")
        if not any(row["suffix"] in {".zip", ".xlsx", ".xls"} for row in files):
            notes.append("No CALCE ZIP or Excel files found.")
        return ParserScanResult(
            dataset_id=self.dataset_id,
            raw_dir=normalize_path(raw_dir),
            files=files,
            notes=notes,
        )

    def parse(self, raw_dir: Path, output_dir: Path) -> None:
        raw_dir = Path(raw_dir).expanduser()
        output_dir = ensure_dir(Path(output_dir).expanduser())
        sources_by_cell = _discover_excel_sources(raw_dir, self.cell_pattern)
        if self.max_cells is not None:
            sources_by_cell = dict(list(sorted(sources_by_cell.items()))[: self.max_cells])
        if not sources_by_cell:
            raise FileNotFoundError(f"No CALCE CS2 Excel sources found under {raw_dir}.")

        cells: list[CanonicalCell] = []
        cycles: list[CanonicalCycle] = []
        diagnostics: list[dict[str, Any]] = []
        started_at = time.monotonic()
        for cell_index, (cell_id, sources) in enumerate(sorted(sources_by_cell.items()), start=1):
            file_started_at = time.monotonic()
            print(
                f"[CALCE {cell_index}/{len(sources_by_cell)}] start {cell_id}: "
                f"excel_files={len(sources)}",
                flush=True,
            )
            cell, cell_cycles, cell_diagnostics = self._parse_cell(
                cell_id=cell_id,
                sources=sources,
                output_dir=output_dir,
            )
            cells.append(cell)
            cycles.extend(cell_cycles)
            diagnostics.extend(cell_diagnostics)
            print(
                f"[CALCE {cell_index}/{len(sources_by_cell)}] done {cell_id}: "
                f"cycles={len(cell_cycles)}, dropped={len(cell_diagnostics)}, "
                f"elapsed={time.monotonic() - file_started_at:.1f}s",
                flush=True,
            )

        cell_ids = [cell.cell_id for cell in cells]
        write_cells(output_dir / "cells.jsonl", cells)
        write_cycles(output_dir / "cycles.jsonl", cycles)
        write_jsonl(output_dir / "cycle_filter_diagnostics.jsonl", diagnostics)
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
                    "num_cycles": len(cycles),
                    "num_dropped_cycles": len(diagnostics),
                    "drop_reasons": dict(Counter(row["reason"] for row in diagnostics)),
                    "nominal_capacity_ah": self.nominal_capacity_ah,
                    "soh_capacity_reference": "nominal_capacity_1p1ah",
                    "filters": self.filters.__dict__,
                    "split_seed": self.split_seed,
                    "split_counts": {name: len(ids) for name, ids in split.items()},
                    "source_files": sorted(
                        {
                            normalize_path(source.source_path)
                            for sources in sources_by_cell.values()
                            for source in sources
                        }
                    ),
                    "source_members": sorted(
                        {
                            source.member_name
                            for sources in sources_by_cell.values()
                            for source in sources
                            if source.member_name is not None
                        }
                    ),
                }
            ],
        )
        print(
            f"Finished parsing {len(cells)} CALCE cells and {len(cycles)} cycles "
            f"in {time.monotonic() - started_at:.1f}s.",
            flush=True,
        )

    def _parse_cell(
        self,
        cell_id: str,
        sources: list[_ExcelSource],
        output_dir: Path,
    ) -> tuple[CanonicalCell, list[CanonicalCycle], list[dict[str, Any]]]:
        candidates: list[_CandidateCycle] = []
        for source in sorted(sources, key=lambda item: (item.sort_time, item.member_name or "")):
            candidates.extend(self._read_excel_cycles(source))

        _mark_local_outliers(candidates, self.filters)

        canonical_cycles: list[CanonicalCycle] = []
        diagnostics: list[dict[str, Any]] = []
        for raw_index, candidate in enumerate(candidates, start=1):
            reason = candidate.drop_reason
            if candidate.keep_complete and candidate.is_local_outlier:
                reason = "local_soh_outlier"
            if reason is not None:
                diagnostics.append(_diagnostic_row(cell_id, raw_index, candidate, reason))
                continue

            cycle_index = len(canonical_cycles) + 1
            array_path = Path("arrays") / cell_id / f"cycle_{cycle_index:05d}.npz"
            write_cycle_arrays(output_dir / array_path, candidate.arrays)
            canonical_cycles.append(
                CanonicalCycle(
                    dataset_id=self.dataset_id,
                    cell_id=cell_id,
                    batch_id="CS2",
                    cycle_index=cycle_index,
                    array_path=normalize_path(array_path),
                    cycle_number=float(cycle_index),
                    q_discharge=candidate.q_discharge,
                    q_charge=candidate.q_charge,
                    internal_resistance=candidate.internal_resistance,
                    soh=candidate.soh,
                    metadata={
                        "raw_cycle_index": raw_index,
                        "local_cycle": candidate.local_cycle,
                        "source_file": candidate.source_file,
                        "source_member": candidate.source_member,
                        "source_date": candidate.source_date,
                        "nominal_capacity_ah": self.nominal_capacity_ah,
                        "soh_capacity_reference": "nominal_capacity_1p1ah",
                        "cap_mismatch": candidate.cap_mismatch,
                    },
                )
            )

        if not canonical_cycles:
            raise ValueError(f"No usable CALCE cycles found for {cell_id}.")

        cell = CanonicalCell(
            dataset_id=self.dataset_id,
            cell_id=cell_id,
            batch_id="CS2",
            source_file=normalize_path(sources[0].source_path),
            cycle_life=float(len(canonical_cycles)),
            num_cycles=len(canonical_cycles),
            charge_policy="CC-CV charge to 4.2V; CS2 nominal 1.1Ah; CALCE CS2 logs",
            metadata={
                "nominal_capacity_ah": self.nominal_capacity_ah,
                "soh_capacity_reference": "nominal_capacity_1p1ah",
                "num_excel_files": len(sources),
                "num_raw_cycles": len(candidates),
                "num_dropped_cycles": len(diagnostics),
                "drop_reasons": dict(Counter(row["reason"] for row in diagnostics)),
                "temperature_available": False,
                "source_members": [
                    source.member_name for source in sources if source.member_name is not None
                ],
            },
        )
        return cell, canonical_cycles, diagnostics

    def _read_excel_cycles(self, source: _ExcelSource) -> list[_CandidateCycle]:
        if source.member_name is None:
            with source.source_path.open("rb") as f:
                payload = f.read()
        else:
            with ZipFile(source.source_path) as archive:
                payload = archive.read(source.member_name)

        xls = pd.ExcelFile(BytesIO(payload))
        channel_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Channel")]
        if not channel_sheets:
            raise ValueError(f"No Channel_* sheet found in {source.source_path}.")
        df = pd.read_excel(BytesIO(payload), sheet_name=channel_sheets[0])
        return [
            self._candidate_from_group(source, local_cycle, group)
            for local_cycle, group in df.groupby("Cycle_Index", sort=True)
            if _finite_number(local_cycle) is not None
        ]

    def _candidate_from_group(
        self,
        source: _ExcelSource,
        local_cycle: Any,
        group: pd.DataFrame,
    ) -> _CandidateCycle:
        numeric = {}
        for column in _REQUIRED_COLUMNS:
            if column not in group.columns:
                raise ValueError(
                    f"Missing CALCE column {column!r} in {source.source_path}"
                    + (f":{source.member_name}" if source.member_name else "")
                )
            numeric[column] = pd.to_numeric(group[column], errors="coerce").to_numpy(np.float64)

        q_discharge_raw = numeric["Discharge_Capacity(Ah)"]
        q_charge_raw = numeric["Charge_Capacity(Ah)"]
        voltage = numeric["Voltage(V)"]
        current = numeric["Current(A)"]
        q_discharge = _finite_range(q_discharge_raw)
        q_charge = _finite_range(q_charge_raw)
        voltage_min = _finite_min(voltage)
        voltage_max = _finite_max(voltage)
        cap_mismatch = abs(q_discharge - q_charge) / max(q_discharge, q_charge, 1e-12)

        arrays = {
            "V": voltage,
            "I": current,
            "Qc": q_charge_raw - _finite_min(q_charge_raw),
            "Qd": q_discharge_raw - _finite_min(q_discharge_raw),
            "t": numeric["Test_Time(s)"] - _finite_min(numeric["Test_Time(s)"]),
        }
        if "Internal_Resistance(Ohm)" in group.columns:
            resistance = pd.to_numeric(
                group["Internal_Resistance(Ohm)"], errors="coerce"
            ).to_numpy(np.float64)
            arrays["IR"] = resistance
        else:
            resistance = np.asarray([], dtype=np.float64)

        keep, reason = _complete_filter_reason(
            n_points=len(group),
            q_discharge=q_discharge,
            q_charge=q_charge,
            cap_mismatch=cap_mismatch,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            filters=self.filters,
        )
        return _CandidateCycle(
            local_cycle=int(float(local_cycle)),
            arrays=arrays,
            q_discharge=q_discharge,
            q_charge=q_charge,
            cap_mismatch=cap_mismatch,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            internal_resistance=_finite_mean(resistance),
            source_file=normalize_path(source.source_path),
            source_member=source.member_name,
            source_date=source.sort_time.date().isoformat()
            if source.sort_time != datetime.max
            else None,
            n_points=len(group),
            keep_complete=keep,
            drop_reason=reason,
            soh=q_discharge / self.nominal_capacity_ah,
        )


_REQUIRED_COLUMNS = (
    "Cycle_Index",
    "Test_Time(s)",
    "Current(A)",
    "Voltage(V)",
    "Charge_Capacity(Ah)",
    "Discharge_Capacity(Ah)",
)


def _discover_excel_sources(
    raw_dir: Path,
    cell_pattern: re.Pattern[str],
) -> dict[str, list[_ExcelSource]]:
    sources: dict[str, list[_ExcelSource]] = {}
    if not raw_dir.exists():
        return sources

    for zip_path in sorted(raw_dir.rglob("*.zip")):
        cell_id = _cell_id_from_path(zip_path)
        if cell_id is None or not cell_pattern.fullmatch(cell_id):
            continue
        with ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if Path(member).suffix.lower() not in {".xlsx", ".xls"}:
                    continue
                sources.setdefault(cell_id, []).append(
                    _ExcelSource(
                        cell_id=cell_id,
                        source_path=zip_path,
                        member_name=member,
                        sort_time=_parse_calce_date(Path(member).name),
                    )
                )

    zip_cell_ids = set(sources)
    for excel_path in sorted(raw_dir.rglob("*")):
        if excel_path.suffix.lower() not in {".xlsx", ".xls"}:
            continue
        cell_id = _cell_id_from_path(excel_path)
        if cell_id is None or not cell_pattern.fullmatch(cell_id):
            continue
        if cell_id in zip_cell_ids:
            continue
        sources.setdefault(cell_id, []).append(
            _ExcelSource(
                cell_id=cell_id,
                source_path=excel_path,
                member_name=None,
                sort_time=_parse_calce_date(excel_path.name),
            )
        )
    return sources


def _mark_local_outliers(candidates: list[_CandidateCycle], filters: CalceFilterConfig) -> None:
    complete_indices = [
        index for index, candidate in enumerate(candidates) if candidate.keep_complete
    ]
    soh = np.asarray([candidates[index].soh for index in complete_indices], dtype=np.float64)
    if len(soh) == 0:
        return
    for position, candidate_index in enumerate(complete_indices):
        lo = max(0, position - filters.local_outlier_window)
        hi = min(len(complete_indices), position + filters.local_outlier_window + 1)
        neighbors = np.delete(soh[lo:hi], position - lo)
        if len(neighbors) < filters.local_outlier_window:
            continue
        if float(np.median(neighbors)) - soh[position] > filters.local_outlier_drop:
            candidates[candidate_index].is_local_outlier = True


def _complete_filter_reason(
    n_points: int,
    q_discharge: float,
    q_charge: float,
    cap_mismatch: float,
    voltage_min: float,
    voltage_max: float,
    filters: CalceFilterConfig,
) -> tuple[bool, str | None]:
    if q_discharge <= filters.min_q_ah:
        return False, "low_discharge_capacity"
    if q_charge <= filters.min_q_ah:
        return False, "low_charge_capacity"
    if cap_mismatch > filters.max_charge_discharge_mismatch:
        return False, "charge_discharge_mismatch"
    if voltage_min > filters.max_voltage_min_v:
        return False, "incomplete_discharge_voltage"
    if voltage_max < filters.min_voltage_max_v:
        return False, "incomplete_charge_voltage"
    if n_points < filters.min_points:
        return False, "too_few_points"
    return True, None


def _diagnostic_row(
    cell_id: str,
    raw_index: int,
    candidate: _CandidateCycle,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset_id": CALCE_DATASET_ID,
        "cell_id": cell_id,
        "raw_cycle_index": raw_index,
        "local_cycle": candidate.local_cycle,
        "source_file": candidate.source_file,
        "source_member": candidate.source_member,
        "reason": reason,
        "n_points": candidate.n_points,
        "q_discharge": candidate.q_discharge,
        "q_charge": candidate.q_charge,
        "soh": candidate.soh,
        "cap_mismatch": candidate.cap_mismatch,
        "voltage_min": candidate.voltage_min,
        "voltage_max": candidate.voltage_max,
    }


def _inventory(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [
        {
            "path": normalize_path(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _cell_id_from_path(path: Path) -> str | None:
    match = re.search(r"(CS2)[_-](\d+)", path.as_posix(), flags=re.IGNORECASE)
    if not match:
        return None
    return f"{match.group(1).upper()}_{int(match.group(2))}"


def _parse_calce_date(name: str) -> datetime:
    match = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})(?:\D|$)", name)
    if not match:
        return datetime.max
    month, day, year = (int(value) for value in match.groups())
    try:
        return datetime(2000 + year, month, day)
    except ValueError:
        return datetime.max


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _finite_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _finite_range(values: Any) -> float:
    finite = _finite_values(values)
    return float(finite.max() - finite.min()) if len(finite) else 0.0


def _finite_min(values: Any) -> float:
    finite = _finite_values(values)
    return float(finite.min()) if len(finite) else 0.0


def _finite_max(values: Any) -> float:
    finite = _finite_values(values)
    return float(finite.max()) if len(finite) else 0.0


def _finite_mean(values: Any) -> float | None:
    finite = _finite_values(values)
    return float(finite.mean()) if len(finite) else None
