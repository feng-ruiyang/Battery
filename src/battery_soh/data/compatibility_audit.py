from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.torch_datasets import load_split_ids


def audit_hust_pickle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {path}, got {type(payload)!r}.")

    cycles = payload.get("cycle_data")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError(f"Expected a non-empty cycle_data list in {path}.")

    direction = CurrentDirectionAccumulator()
    temperature = NumericAccumulator()
    unequal_length_cycles = 0
    nonfinite_values = 0
    cycle_numbers: list[int] = []
    cycle_lengths: list[int] = []

    for cycle in cycles:
        if not isinstance(cycle, Mapping):
            raise TypeError(f"Expected each HUST cycle to be a mapping, got {type(cycle)!r}.")
        current = _numeric_1d(cycle.get("current_in_A"))
        voltage = _numeric_1d(cycle.get("voltage_in_V"))
        charge = _numeric_1d(cycle.get("charge_capacity_in_Ah"))
        discharge = _numeric_1d(cycle.get("discharge_capacity_in_Ah"))
        time = _numeric_1d(cycle.get("time_in_s"))
        lengths = {len(value) for value in (current, voltage, charge, discharge, time)}
        if len(lengths) != 1:
            unequal_length_cycles += 1
            continue

        cycle_lengths.append(len(current))
        cycle_numbers.append(int(cycle["cycle_number"]))
        nonfinite_values += sum(
            int((~np.isfinite(value)).sum())
            for value in (current, voltage, charge, discharge, time)
        )
        direction.update(current, charge, discharge)
        temperature.update(cycle.get("temperature_in_C"))

    return {
        "path": str(path),
        "cell_id": str(payload.get("cell_id", path.stem)),
        "num_cycles": len(cycles),
        "cycle_numbers_strictly_increasing": _strictly_increasing(cycle_numbers),
        "cycle_length": _distribution(cycle_lengths),
        "unequal_length_cycles": unequal_length_cycles,
        "nonfinite_core_values": nonfinite_values,
        "temperature": temperature.summary(),
        "current_direction": direction.summary(),
    }


def audit_severson_training(
    processed_dir: str | Path,
    split_file: str | Path,
    split_name: str = "train",
    log_every: int = 10_000,
) -> dict[str, Any]:
    store = CanonicalCycleStore(processed_dir)
    selected_ids = set(load_split_ids(split_file, split_name) or [])
    rows = [row for row in store.cycles if row["cell_id"] in selected_ids]
    direction = CurrentDirectionAccumulator()
    temperature = NumericAccumulator()
    cycles_with_arrays = 0
    cycles_with_temperature = 0

    for index, row in enumerate(rows, start=1):
        arrays = store.load_cycle_arrays(row)
        if not arrays:
            continue
        cycles_with_arrays += 1
        direction.update(arrays.get("I"), arrays.get("Qc"), arrays.get("Qd"))
        if temperature.update(arrays.get("T")):
            cycles_with_temperature += 1
        if log_every > 0 and (index == 1 or index % log_every == 0 or index == len(rows)):
            print(f"severson audit cycle={index}/{len(rows)}", flush=True)

    return {
        "processed_dir": str(processed_dir),
        "split_file": str(split_file),
        "split_name": split_name,
        "num_cells": len(selected_ids),
        "num_cycle_rows": len(rows),
        "cycles_with_arrays": cycles_with_arrays,
        "cycles_with_temperature": cycles_with_temperature,
        "temperature": temperature.summary(),
        "current_direction": direction.summary(),
    }


def compare_compatibility(
    hust: Mapping[str, Any],
    severson: Mapping[str, Any],
) -> dict[str, Any]:
    hust_direction = hust["current_direction"]
    severson_direction = severson["current_direction"]
    charge_aligned = _same_known_sign(
        hust_direction["charge_current_sign"],
        severson_direction["charge_current_sign"],
    )
    discharge_aligned = _same_known_sign(
        hust_direction["discharge_current_sign"],
        severson_direction["discharge_current_sign"],
    )
    temperature_available = int(hust["temperature"]["finite_count"]) > 0
    severson_temperature_mean = severson["temperature"]["mean"]
    return {
        "charge_sign_aligned": charge_aligned,
        "discharge_sign_aligned": discharge_aligned,
        "flip_hust_current": (
            charge_aligned is False
            and discharge_aligned is False
        ),
        "hust_temperature_available": temperature_available,
        "t_zero_fill": 0.0,
        "t_severson_mean_fill": severson_temperature_mean,
        "warnings": _compatibility_warnings(
            charge_aligned=charge_aligned,
            discharge_aligned=discharge_aligned,
            temperature_available=temperature_available,
            severson_temperature_mean=severson_temperature_mean,
        ),
    }


class CurrentDirectionAccumulator:
    def __init__(self) -> None:
        self.charge_current_sum = 0.0
        self.charge_capacity_gain = 0.0
        self.discharge_current_sum = 0.0
        self.discharge_capacity_gain = 0.0
        self.cycles_used = 0

    def update(
        self,
        current: Any,
        charge_capacity: Any,
        discharge_capacity: Any,
    ) -> bool:
        current_values = _numeric_1d(current)
        charge_values = _numeric_1d(charge_capacity)
        discharge_values = _numeric_1d(discharge_capacity)
        if (
            len(current_values) < 2
            or len(current_values) != len(charge_values)
            or len(current_values) != len(discharge_values)
        ):
            return False

        midpoint_current = 0.5 * (current_values[:-1] + current_values[1:])
        charge_gain = np.diff(charge_values)
        discharge_gain = np.diff(discharge_values)
        self.charge_current_sum, self.charge_capacity_gain = _accumulate_direction(
            midpoint_current,
            charge_gain,
            self.charge_current_sum,
            self.charge_capacity_gain,
        )
        self.discharge_current_sum, self.discharge_capacity_gain = _accumulate_direction(
            midpoint_current,
            discharge_gain,
            self.discharge_current_sum,
            self.discharge_capacity_gain,
        )
        self.cycles_used += 1
        return True

    def summary(self) -> dict[str, Any]:
        charge_mean = _safe_ratio(self.charge_current_sum, self.charge_capacity_gain)
        discharge_mean = _safe_ratio(
            self.discharge_current_sum,
            self.discharge_capacity_gain,
        )
        return {
            "cycles_used": self.cycles_used,
            "charge_current_during_capacity_gain_A": charge_mean,
            "discharge_current_during_capacity_gain_A": discharge_mean,
            "charge_current_sign": _sign_label(charge_mean),
            "discharge_current_sign": _sign_label(discharge_mean),
        }


class NumericAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: Any) -> bool:
        if values is None:
            return False
        array = _numeric_1d(values)
        finite = array[np.isfinite(array)]
        if len(finite) == 0:
            return False
        self.total += float(finite.sum(dtype=np.float64))
        self.count += int(len(finite))
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        return True

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "finite_count": 0,
                "mean": None,
                "min": None,
                "max": None,
            }
        return {
            "finite_count": self.count,
            "mean": self.total / self.count,
            "min": self.minimum,
            "max": self.maximum,
        }


def _accumulate_direction(
    current: np.ndarray,
    capacity_gain: np.ndarray,
    current_sum: float,
    gain_sum: float,
) -> tuple[float, float]:
    valid = (
        np.isfinite(current)
        & np.isfinite(capacity_gain)
        & (capacity_gain > 1e-10)
    )
    if not np.any(valid):
        return current_sum, gain_sum
    gains = capacity_gain[valid]
    return (
        current_sum + float(np.sum(current[valid] * gains, dtype=np.float64)),
        gain_sum + float(np.sum(gains, dtype=np.float64)),
    )


def _numeric_1d(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _sign_label(value: float | None, tolerance: float = 1e-6) -> str:
    if value is None or not np.isfinite(value) or abs(value) <= tolerance:
        return "unknown"
    return "positive" if value > 0 else "negative"


def _same_known_sign(left: str, right: str) -> bool | None:
    if "unknown" in {left, right}:
        return None
    return left == right


def _strictly_increasing(values: Iterable[int]) -> bool:
    values = list(values)
    return all(right > left for left, right in zip(values, values[1:]))


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    array = np.asarray(values)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "max": int(array.max()),
    }


def _compatibility_warnings(
    charge_aligned: bool | None,
    discharge_aligned: bool | None,
    temperature_available: bool,
    severson_temperature_mean: float | None,
) -> list[str]:
    warnings = []
    if charge_aligned is None or discharge_aligned is None:
        warnings.append("Could not infer both current directions from capacity increments.")
    elif charge_aligned != discharge_aligned:
        warnings.append("Charge and discharge direction checks disagree; inspect phase semantics.")
    elif not charge_aligned:
        warnings.append("HUST current must be sign-flipped to match Severson.")
    if not temperature_available:
        warnings.append("HUST temperature is missing; T=0 and train-mean fills test different shifts.")
    if severson_temperature_mean is None:
        warnings.append("No finite Severson training temperature was found.")
    return warnings
