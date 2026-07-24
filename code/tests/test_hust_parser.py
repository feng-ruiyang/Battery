import pickle

import numpy as np

from battery_soh.data.hust import HustParser
from battery_soh.data.io import read_jsonl


def test_hust_parser_writes_external_test_canonical_data(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    payload = {
        "cell_id": "HUST_1-1",
        "cycle_data": [
            _cycle(1, discharge_capacity=1.2),
            _cycle(2, discharge_capacity=0.9),
        ],
        "form_factor": "cylindrical_18650",
        "anode_material": "graphite",
        "cathode_material": "LFP",
        "nominal_capacity_in_Ah": 1.1,
        "charge_protocol": [{"rate_in_C": 5.0}],
        "discharge_protocol": [{"rate_in_C": 5.0}],
    }
    with (raw_dir / "HUST_1-1.pkl").open("wb") as f:
        pickle.dump(payload, f)

    HustParser().parse(raw_dir, output_dir)

    cells = read_jsonl(output_dir / "cells.jsonl")
    cycles = read_jsonl(output_dir / "cycles.jsonl")
    split = read_jsonl(output_dir / "splits" / "all_external_test.jsonl")[0]
    assert cells[0]["dataset_id"] == "hust_lfp"
    assert cells[0]["metadata"]["temperature_available"] is False
    assert split == {"test": ["HUST_1-1"]}
    assert cycles[0]["q_discharge"] == 1.2
    assert cycles[0]["soh"] == 1.0
    assert cycles[1]["soh"] == 0.9 / 1.2
    assert cycles[0]["t_avg"] is None

    with np.load(output_dir / cycles[0]["array_path"]) as arrays:
        assert set(arrays.files) == {"I", "V", "Qc", "Qd", "t"}
        np.testing.assert_allclose(arrays["t"], [0.0, 1.0, 2.0, 3.0])


def _cycle(cycle_number: int, discharge_capacity: float) -> dict:
    return {
        "cycle_number": cycle_number,
        "current_in_A": np.asarray([2.0, 2.0, -3.0, -3.0]),
        "voltage_in_V": np.asarray([3.0, 3.5, 3.4, 2.0]),
        "charge_capacity_in_Ah": np.asarray([0.0, 0.5, 0.5, 0.5]),
        "discharge_capacity_in_Ah": np.asarray(
            [0.0, 0.0, 0.5 * discharge_capacity, discharge_capacity]
        ),
        "time_in_s": np.asarray([100.0, 101.0, 102.0, 103.0]),
        "temperature_in_C": None,
        "internal_resistance_in_ohm": None,
    }
