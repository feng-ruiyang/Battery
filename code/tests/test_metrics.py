import numpy as np

from battery_soh.evaluation.metrics import interval_coverage, point_metrics


def test_point_metrics_with_mask():
    y_true = np.asarray([1.0, 2.0, 3.0])
    y_pred = np.asarray([1.0, 4.0, 3.0])
    mask = np.asarray([True, False, True])
    metrics = point_metrics(y_true, y_pred, mask)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_interval_coverage():
    y = np.asarray([1.0, 2.0, 3.0])
    lo = np.asarray([0.0, 2.5, 2.5])
    hi = np.asarray([1.5, 3.5, 3.5])
    assert interval_coverage(y, lo, hi) == 2 / 3

