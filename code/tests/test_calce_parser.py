from __future__ import annotations

from zipfile import ZipFile

import numpy as np
import pandas as pd

from battery_soh.data.calce import CalceFilterConfig, CalceParser
from battery_soh.data.io import read_jsonl


def test_calce_scan_ignores_non_cs2_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "notes.txt").write_text("metadata", encoding="utf-8")
    (raw_dir / "CS2_33.zip").write_bytes(b"not inspected by scan")

    scan = CalceParser().scan(raw_dir)

    assert any("Detected 1 candidate CS2 cells" in note for note in scan.notes)


def test_calce_parser_differences_cumulative_capacity_and_filters_diagnostics(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    sheet = pd.concat(
        [
            _cycle(1, q_discharge=1.10, q_charge=1.10),
            _cycle(2, q_discharge=1.08, q_charge=1.08),
            _cycle(3, q_discharge=0.98, q_charge=0.98),
            _cycle(4, q_discharge=1.06, q_charge=1.06),
            _cycle(5, q_discharge=1.04, q_charge=1.04),
            _cycle(6, q_discharge=0.12, q_charge=1.02, voltage_min=3.2),
        ],
        ignore_index=True,
    )

    class FakeExcelFile:
        sheet_names = ["Info", "Channel_1-006"]

        def __init__(self, _payload):
            pass

    monkeypatch.setattr("battery_soh.data.calce.pd.ExcelFile", FakeExcelFile)
    monkeypatch.setattr(
        "battery_soh.data.calce.pd.read_excel",
        lambda *_args, **_kwargs: sheet.copy(),
    )

    with ZipFile(raw_dir / "CS2_33.zip", "w") as archive:
        archive.writestr("CS2_33/CS2_33_8_17_10.xlsx", b"dummy xlsx bytes")

    parser = CalceParser(
        filters=CalceFilterConfig(
            min_points=3,
            local_outlier_window=1,
            local_outlier_drop=0.06,
        ),
    )
    parser.parse(raw_dir, output_dir)

    cells = read_jsonl(output_dir / "cells.jsonl")
    cycles = read_jsonl(output_dir / "cycles.jsonl")
    diagnostics = read_jsonl(output_dir / "cycle_filter_diagnostics.jsonl")
    split = read_jsonl(output_dir / "splits" / "all_external_test.jsonl")[0]
    summary = read_jsonl(output_dir / "scan_summary.jsonl")[0]

    assert cells[0]["cell_id"] == "CS2_33"
    assert cells[0]["metadata"]["nominal_capacity_ah"] == 1.1
    assert split == {"test": ["CS2_33"]}
    assert summary["drop_reasons"] == {
        "charge_discharge_mismatch": 1,
        "local_soh_outlier": 1,
    }

    assert [cycle["cycle_index"] for cycle in cycles] == [1, 2, 3, 4]
    np.testing.assert_allclose(
        [cycle["q_discharge"] for cycle in cycles],
        [1.10, 1.08, 1.06, 1.04],
    )
    np.testing.assert_allclose(
        [cycle["soh"] for cycle in cycles],
        [1.10 / 1.1, 1.08 / 1.1, 1.06 / 1.1, 1.04 / 1.1],
    )
    assert cycles[0]["metadata"]["source_member"] == "CS2_33/CS2_33_8_17_10.xlsx"
    assert cycles[0]["metadata"]["soh_capacity_reference"] == "nominal_capacity_1p1ah"

    assert [row["reason"] for row in diagnostics] == [
        "local_soh_outlier",
        "charge_discharge_mismatch",
    ]

    with np.load(output_dir / cycles[0]["array_path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"I", "IR", "Qc", "Qd", "V", "t"}
        np.testing.assert_allclose(arrays["Qd"], [0.0, 0.55, 1.10])
        np.testing.assert_allclose(arrays["Qc"], [0.0, 0.55, 1.10])
        np.testing.assert_allclose(arrays["t"], [0.0, 1.0, 2.0])


def _cycle(
    cycle_index: int,
    q_discharge: float,
    q_charge: float,
    voltage_min: float = 2.7,
) -> pd.DataFrame:
    discharge_base = 10.0 * cycle_index
    charge_base = 20.0 * cycle_index
    return pd.DataFrame(
        {
            "Data_Point": [cycle_index * 10 + 1, cycle_index * 10 + 2, cycle_index * 10 + 3],
            "Test_Time(s)": [cycle_index * 100.0, cycle_index * 100.0 + 1.0, cycle_index * 100.0 + 2.0],
            "Step_Time(s)": [0.0, 1.0, 2.0],
            "Step_Index": [1, 2, 3],
            "Cycle_Index": [cycle_index, cycle_index, cycle_index],
            "Current(A)": [0.55, -0.55, -0.55],
            "Voltage(V)": [4.2, 3.5, voltage_min],
            "Charge_Capacity(Ah)": [
                charge_base,
                charge_base + 0.5 * q_charge,
                charge_base + q_charge,
            ],
            "Discharge_Capacity(Ah)": [
                discharge_base,
                discharge_base + 0.5 * q_discharge,
                discharge_base + q_discharge,
            ],
            "Internal_Resistance(Ohm)": [0.0, 0.01, 0.02],
        }
    )
