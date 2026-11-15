from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def make_piecewise_linear_basis(horizon: int, num_basis: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if num_basis <= 0:
        raise ValueError("num_basis must be positive.")
    grid = np.linspace(0.0, 1.0, num=horizon)
    knots = np.linspace(0.0, 1.0, num=num_basis)
    basis = np.zeros((horizon, num_basis), dtype=float)
    for i, knot in enumerate(knots):
        if num_basis == 1:
            basis[:, i] = 1.0
        else:
            width = knots[1] - knots[0]
            basis[:, i] = np.maximum(1.0 - np.abs(grid - knot) / width, 0.0)
    row_sum = basis.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return basis / row_sum


def make_trajectory_matrix(horizon: int, basis: np.ndarray) -> np.ndarray:
    """Build G where y = G theta and theta = [s_t, c_1, ..., c_K]."""
    cumulative_basis = np.cumsum(basis, axis=0)
    ones = np.ones((horizon, 1), dtype=float)
    return np.concatenate([ones, -cumulative_basis], axis=1)


def theta_to_trajectory_mean(theta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    s_t = theta[..., :1]
    coeffs = theta[..., 1:]
    rates = coeffs @ basis.T
    return s_t - np.cumsum(rates, axis=-1)


class DegradationDynamics(nn.Module):
    """Analytically propagate theta distribution to future SoH trajectory."""

    def __init__(
        self,
        horizon: int,
        num_basis: int,
        basis_kind: str = "piecewise_linear",
        aleatoric_init: float = 1e-3,
        min_variance: float = 1e-6,
    ) -> None:
        super().__init__()
        if basis_kind != "piecewise_linear":
            raise ValueError(f"Unsupported basis_kind {basis_kind!r}.")
        basis = make_piecewise_linear_basis(horizon, num_basis)
        trajectory_matrix = make_trajectory_matrix(horizon, basis)
        self.horizon = horizon
        self.num_basis = num_basis
        self.min_variance = min_variance
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.register_buffer("trajectory_matrix", torch.as_tensor(trajectory_matrix, dtype=torch.float32))
        self.log_aleatoric_var = nn.Parameter(torch.full((horizon,), float(np.log(aleatoric_init))))

    @property
    def theta_dim(self) -> int:
        return 1 + self.num_basis

    def forward(self, theta_mean: Tensor, theta_var: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        g = self.trajectory_matrix.to(dtype=theta_mean.dtype, device=theta_mean.device)
        basis = self.basis.to(dtype=theta_mean.dtype, device=theta_mean.device)
        y_mean = theta_mean @ g.T
        epistemic_cov = torch.einsum("hk,bk,lk->bhl", g, theta_var, g)
        aleatoric_var = self.log_aleatoric_var.exp().to(dtype=theta_mean.dtype, device=theta_mean.device)
        eye = torch.eye(self.horizon, dtype=theta_mean.dtype, device=theta_mean.device)
        y_cov = epistemic_cov + eye.unsqueeze(0) * aleatoric_var.view(1, 1, -1)
        y_cov = y_cov + eye.unsqueeze(0) * self.min_variance
        coeffs = theta_mean[:, 1:]
        rate_mean = coeffs @ basis.T
        return {
            "y_mean": y_mean,
            "y_cov": y_cov,
            "rate_mean": rate_mean,
            "epistemic_cov": epistemic_cov,
            "aleatoric_var": aleatoric_var,
            "regularization": {
                "negative_rate": torch.relu(-rate_mean).pow(2).mean(),
                "smoothness": _smoothness_penalty(rate_mean),
            },
        }


def _smoothness_penalty(rate: Tensor) -> Tensor:
    if rate.shape[-1] < 3:
        return rate.new_tensor(0.0)
    second_diff = rate[:, 2:] - 2 * rate[:, 1:-1] + rate[:, :-2]
    return second_diff.pow(2).mean()
