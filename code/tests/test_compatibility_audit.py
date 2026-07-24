import pickle

import numpy as np

from battery_soh.data.compatibility_audit import (
    CurrentDirectionAccumulator,
    NumericAccumulator,
    audit_hust_pickle,
    compare_compatibility,
)


def test_current_direction_uses_capacity_increments():
    accumulator = CurrentDirectionAccumulator()
    accumulator.update(
        current=np.asarray([2.0, 2.0, -3.0, -3.0]),
        charge_capacity=np.asarray([0.0, 0.5, 0.5, 0.5]),
        discharge_capacity=np.asarray([0.0, 0.0, 0.4, 0.8]),
    )

    result = accumulator.summary()

    assert result["charge_current_sign"] == "positive"
    assert result["discharge_current_sign"] == "negative"


def test_numeric_accumulator_ignores_nonfinite_values():
    accumulator = NumericAccumulator()
    accumulator.update(np.asarray([20.0, np.nan, 30.0, np.inf]))

    assert accumulator.summary() == {
        "finite_count": 2,
        "mean": 25.0,
        "min": 20.0,
        "max": 30.0,
    }


def test_hust_audit_and_comparison(tmp_path):
    path = tmp_path / "cell.pkl"
    cycle = {
        "cycle_number": 1,
        "current_in_A": np.asarray([2.0, 2.0, -3.0, -3.0]),
        "voltage_in_V": np.asarray([3.0, 3.5, 3.4, 2.0]),
        "charge_capacity_in_Ah": np.asarray([0.0, 0.5, 0.5, 0.5]),
        "discharge_capacity_in_Ah": np.asarray([0.0, 0.0, 0.4, 0.8]),
        "time_in_s": np.arange(4),
        "temperature_in_C": None,
    }
    with path.open("wb") as f:
        pickle.dump({"cell_id": "test", "cycle_data": [cycle]}, f)

    hust = audit_hust_pickle(path)
    severson = {
        "temperature": {"finite_count": 2, "mean": 25.0, "min": 20.0, "max": 30.0},
        "current_direction": {
            "charge_current_sign": "negative",
            "discharge_current_sign": "positive",
        },
    }
    comparison = compare_compatibility(hust, severson)

    assert hust["temperature"]["finite_count"] == 0
    assert comparison["flip_hust_current"] is True
    assert comparison["t_severson_mean_fill"] == 25.0
