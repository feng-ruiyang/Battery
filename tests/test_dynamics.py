import numpy as np

from battery_soh.models.degradation_dynamics import (
    make_piecewise_linear_basis,
    theta_to_trajectory_mean,
)


def test_basis_shape_and_row_sum():
    basis = make_piecewise_linear_basis(horizon=8, num_basis=3)
    assert basis.shape == (8, 3)
    np.testing.assert_allclose(basis.sum(axis=1), np.ones(8))


def test_theta_to_trajectory_mean_shape():
    basis = make_piecewise_linear_basis(horizon=5, num_basis=2)
    theta = np.asarray([[1.0, 0.01, 0.02]])
    y = theta_to_trajectory_mean(theta, basis)
    assert y.shape == (1, 5)
    assert y[0, -1] < y[0, 0]

