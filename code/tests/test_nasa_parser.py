import numpy as np
from scipy.io import savemat

from battery_soh.data.io import read_jsonl
from battery_soh.data.nasa import NasaParser


def test_nasa_parser_uses_first_discharge_capacity_and_indexes_all_operations(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    batch_dir = raw_dir / "FY08Q4"
    batch_dir.mkdir(parents=True)
    _write_battery(batch_dir / "B0005.mat")

    NasaParser().parse(raw_dir, output_dir)

    cells = read_jsonl(output_dir / "cells.jsonl")
    cycles = read_jsonl(output_dir / "cycles.jsonl")
    operations = read_jsonl(output_dir / "operations.jsonl")
    split = read_jsonl(output_dir / "splits" / "all_external_test.jsonl")[0]
    summary = read_jsonl(output_dir / "scan_summary.jsonl")[0]

    assert cells[0]["cell_id"] == "FY08Q4__B0005"
    assert cells[0]["metadata"]["initial_capacity_ah"] == 1.8
    assert cells[0]["metadata"]["operation_counts"] == {
        "charge": 1,
        "discharge": 2,
        "impedance": 1,
    }
    assert split == {"test": ["FY08Q4__B0005"]}
    assert summary["num_operations"] == 4
    assert summary["num_discharge_cycles"] == 2
    assert [row["operation_type"] for row in operations] == [
        "charge",
        "discharge",
        "impedance",
        "discharge",
    ]

    assert len(cycles) == 2
    assert cycles[0]["q_discharge"] == 1.8
    assert cycles[0]["soh"] == 1.0
    assert cycles[1]["q_discharge"] == 1.62
    assert cycles[1]["soh"] == 1.62 / 1.8
    assert cycles[0]["metadata"]["initial_capacity_source"] == "first_discharge_capacity"

    with np.load(output_dir / cycles[0]["array_path"], allow_pickle=False) as arrays:
        assert {"I", "V", "T", "t", "I_load", "V_load"} <= set(arrays.files)
        np.testing.assert_allclose(arrays["I"], [-2.0, -2.0, -2.0])
        np.testing.assert_allclose(arrays["T"], [25.0, 26.0, 27.0])

    impedance = operations[2]
    with np.load(output_dir / impedance["array_path"], allow_pickle=False) as arrays:
        assert "raw_Battery_impedance" in arrays.files
        assert np.iscomplexobj(arrays["raw_Battery_impedance"])


def test_nasa_parser_rejects_missing_first_discharge_capacity(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    operations = np.empty(2, dtype=object)
    operations[0] = _operation("discharge", _discharge_data(np.nan), second=0.0)
    operations[1] = _operation("discharge", _discharge_data(1.62), second=1.0)
    savemat(raw_dir / "B0006.mat", {"B0006": {"cycle": operations}})

    try:
        NasaParser().parse(raw_dir, output_dir)
    except ValueError as error:
        assert "first discharge operation" in str(error)
    else:
        raise AssertionError("Expected missing first-discharge Capacity to be rejected.")


def _write_battery(path):
    operations = np.empty(4, dtype=object)
    operations[0] = _operation("charge", _charge_data(), second=0.0)
    operations[1] = _operation("discharge", _discharge_data(1.8), second=1.5)
    operations[2] = _operation("impedance", _impedance_data(), second=2.0)
    operations[3] = _operation("discharge", _discharge_data(1.62), second=3.0)
    savemat(path, {"B0005": {"cycle": operations}})


def _operation(operation_type, data, second):
    return {
        "type": operation_type,
        "ambient_temperature": 24.0,
        "time": np.asarray([2008, 4, 2, 13, 8, second]),
        "data": data,
    }


def _core_data():
    return {
        "Voltage_measured": np.asarray([4.0, 3.5, 3.0]),
        "Current_measured": np.asarray([-2.0, -2.0, -2.0]),
        "Temperature_measured": np.asarray([25.0, 26.0, 27.0]),
        "Time": np.asarray([0.0, 1.0, 2.0]),
    }


def _charge_data():
    data = _core_data()
    data["Current_measured"] = np.asarray([1.0, 1.0, 1.0])
    data["Current_charge"] = np.asarray([1.0, 1.0, 1.0])
    data["Voltage_charge"] = np.asarray([4.0, 4.1, 4.2])
    return data


def _discharge_data(capacity):
    data = _core_data()
    data["Current_load"] = np.asarray([-2.0, -2.0, -2.0])
    data["Voltage_load"] = np.asarray([4.0, 3.5, 3.0])
    data["Capacity"] = capacity
    return data


def _impedance_data():
    return {
        "Sense_current": np.asarray([1.0 + 2.0j]),
        "Battery_impedance": np.asarray([0.1 + 0.2j, 0.2 + 0.3j]),
        "Re": 0.05,
        "Rct": 0.1,
    }
