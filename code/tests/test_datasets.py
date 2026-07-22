import numpy as np

from battery_soh.data.datasets import CanonicalCycleStore, SOHTrajectoryDataset
from battery_soh.data.io import write_jsonl
from battery_soh.data.schema import TrainingWindowSpec
from battery_soh.data.transforms import fill_nonfinite_1d, resample_1d


def test_soh_dataset_loads_curve_and_summary_windows(tmp_path):
    processed = tmp_path / "processed"
    array_dir = processed / "arrays" / "cell_a"
    array_dir.mkdir(parents=True)

    write_jsonl(
        processed / "cells.jsonl",
        [{"dataset_id": "test", "cell_id": "cell_a", "batch_id": "batch1"}],
    )
    cycles = []
    for i in range(4):
        array_path = f"arrays/cell_a/cycle_{i:05d}.npz"
        np.savez(
            processed / array_path,
            V=np.asarray([3.0 + i, 3.1 + i]),
            I=np.asarray([1.0, 2.0, 3.0]),
            T=np.asarray([25.0 + i]),
        )
        cycles.append(
            {
                "dataset_id": "test",
                "cell_id": "cell_a",
                "batch_id": "batch1",
                "cycle_index": i,
                "array_path": array_path,
                "q_discharge": 1.0 - 0.1 * i,
                "internal_resistance": 0.01 + 0.001 * i,
                "t_avg": 30.0 + i,
                "soh": 1.0 - 0.05 * i,
            }
        )
    write_jsonl(processed / "cycles.jsonl", cycles)

    store = CanonicalCycleStore(processed)
    spec = TrainingWindowSpec(window=2, horizon=2, cycle_channels=("V", "I", "T"), sequence_length=4)
    dataset = SOHTrajectoryDataset(store, ["cell_a"], spec)

    item = dataset[0]

    assert item.x_window is not None
    assert item.u_window is not None
    assert item.x_window.shape == (2, 4, 3)
    assert item.u_window.shape == (2, 3)
    np.testing.assert_allclose(item.soh_history, [1.0, 0.95])
    np.testing.assert_allclose(item.soh_future, [0.9, 0.85])
    np.testing.assert_array_equal(item.mask, [True, True])
    np.testing.assert_allclose(item.u_window[0], [1.0, 0.01, 30.0])


def test_soh_dataset_marks_missing_curve_channels_as_nan(tmp_path):
    processed = tmp_path / "processed"
    array_dir = processed / "arrays" / "cell_a"
    array_dir.mkdir(parents=True)

    write_jsonl(processed / "cells.jsonl", [{"dataset_id": "test", "cell_id": "cell_a"}])
    cycles = []
    for i in range(3):
        array_path = f"arrays/cell_a/cycle_{i:05d}.npz"
        np.savez(processed / array_path, V=np.asarray([3.0, 3.1]))
        cycles.append(
            {
                "dataset_id": "test",
                "cell_id": "cell_a",
                "cycle_index": i,
                "array_path": array_path,
                "soh": 1.0 - 0.1 * i,
            }
        )
    write_jsonl(processed / "cycles.jsonl", cycles)

    store = CanonicalCycleStore(processed)
    spec = TrainingWindowSpec(window=2, horizon=1, cycle_channels=("V", "I"), sequence_length=2)
    item = SOHTrajectoryDataset(store, ["cell_a"], spec)[0]

    assert item.x_window is not None
    assert item.x_window.shape == (2, 2, 2)
    assert np.isnan(item.x_window[:, :, 1]).all()


def test_resample_fills_nonfinite_edges_and_middle_gaps():
    values = np.asarray([np.nan, np.inf, 2.0, np.nan, 6.0, -np.inf])

    filled = fill_nonfinite_1d(values)
    resampled = resample_1d(values, target_length=3)

    np.testing.assert_allclose(filled, [2.0, 2.0, 2.0, 4.0, 6.0, 6.0])
    np.testing.assert_allclose(resampled, [2.0, 3.0, 6.0])
