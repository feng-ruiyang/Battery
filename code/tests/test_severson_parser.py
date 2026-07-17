import pickle

import numpy as np

import battery_soh.data.severson as severson
from battery_soh.data.io import read_jsonl
from battery_soh.data.severson import SeversonParser


def test_severson_parser_pkl(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    cell = {
        "cycle_life": np.asarray([1200]),
        "charge_policy": "4C",
        "summary": {
            "cycle": np.asarray([0, 1]),
            "QD": np.asarray([1.1, 1.0]),
            "QC": np.asarray([1.12, 1.03]),
            "IR": np.asarray([0.01, 0.011]),
            "Tavg": np.asarray([30.0, 31.0]),
            "Tmin": np.asarray([29.0, 30.0]),
            "Tmax": np.asarray([32.0, 33.0]),
            "chargetime": np.asarray([10.0, 11.0]),
        },
        "cycles": {
            "0": {"I": np.asarray([1, 2]), "V": np.asarray([3, 4]), "T": np.asarray([30, 31])},
            "1": {"I": np.asarray([2, 3]), "V": np.asarray([4, 5]), "T": np.asarray([31, 32])},
        },
    }
    with (raw_dir / "batch1.pkl").open("wb") as f:
        pickle.dump({"b1c0": cell}, f)

    parser = SeversonParser()
    parser.parse(raw_dir, out_dir)

    cells = read_jsonl(out_dir / "cells.jsonl")
    cycles = read_jsonl(out_dir / "cycles.jsonl")
    assert len(cells) == 1
    assert len(cycles) == 2
    assert cycles[0]["soh"] == 1.0
    assert cycles[1]["soh"] == 1.0 / 1.1
    assert (out_dir / cycles[0]["array_path"]).exists()


def test_severson_parser_beep_json(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    data = {
        "@module": "beep.structure",
        "@class": "ProcessedCyclerRun",
        "barcode": "el123",
        "protocol": "2018-04-12_batch8/test.sdu",
        "channel_id": 25,
        "summary": {
            "cycle_index": [0, 1],
            "discharge_capacity": [1.5, 1.2],
            "charge_capacity": [1.4, 1.1],
            "dc_internal_resistance": [0.08, 0.09],
            "temperature_maximum": [33.0, 34.0],
            "temperature_average": [31.0, 32.0],
            "temperature_minimum": [25.0, 26.0],
            "charge_duration": [100.0, 120.0],
        },
        "cycles_interpolated": {
            "cycle_index": [0, 0, 1, 1],
            "voltage": [2.8, 3.0, 2.9, 3.1],
            "temperature": [30.0, 31.0, 32.0, 33.0],
            "charge_capacity": [0.0, 1.0, 0.0, 1.1],
            "discharge_capacity": [0.0, 1.2, 0.0, 1.0],
            "internal_resistance": [0.08, 0.08, 0.09, 0.09],
            "current": [-1.0, -1.0, -1.1, -1.1],
            "step_type": ["discharge", "discharge", "discharge", "discharge"],
        },
        "@version": "test",
    }
    import json

    (raw_dir / "FastCharge_000000_CH25_structure.json").write_text(json.dumps(data))

    parser = SeversonParser()
    parser.parse(raw_dir, out_dir)

    cells = read_jsonl(out_dir / "cells.jsonl")
    cycles = read_jsonl(out_dir / "cycles.jsonl")
    assert len(cells) == 1
    assert cells[0]["barcode"] == "el123"
    assert cells[0]["batch_id"] == "batch8"
    assert len(cycles) == 2
    assert cycles[0]["q_discharge"] == 1.5
    assert cycles[1]["soh"] == 1.2 / 1.5
    assert (out_dir / cycles[0]["array_path"]).exists()


def test_severson_parser_uses_raw_json_cleaning_hook(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    data = {
        "barcode": "before-cleaning",
        "summary": {
            "cycle_index": [0],
            "discharge_capacity": [1.0],
        },
        "cycles_interpolated": {
            "cycle_index": [0],
            "voltage": [3.0],
        },
    }

    def fake_load_and_clean_raw_json(json_path):
        assert json_path.name == "FastCharge_000000_CH25_structure.json"
        cleaned = dict(data)
        cleaned["barcode"] = "after-cleaning"
        return cleaned

    import json

    (raw_dir / "FastCharge_000000_CH25_structure.json").write_text(json.dumps(data))
    monkeypatch.setattr(severson, "load_and_clean_raw_json", fake_load_and_clean_raw_json)

    parser = SeversonParser()
    parser.parse(raw_dir, out_dir)

    cells = read_jsonl(out_dir / "cells.jsonl")
    assert cells[0]["barcode"] == "after-cleaning"
