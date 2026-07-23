from __future__ import annotations

import torch

from battery_soh.models.soh_predictor import SOHProbabilisticPredictor
from battery_soh.models.trajectory_decoder import TeacherForcingTrajectoryDecoder
from battery_soh.training.losses import masked_trajectory_mse, total_loss


def test_deterministic_decoder_returns_zero_predictive_variance() -> None:
    decoder = TeacherForcingTrajectoryDecoder(
        state_dim=8,
        horizon=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        head_kind="deterministic",
    )
    outputs = decoder(
        state=torch.randn(2, 8),
        decoder_inputs=torch.randn(2, 3),
    )

    assert outputs["y_mean"].shape == (2, 3)
    assert outputs["y_cov"].shape == (2, 3, 3)
    assert torch.count_nonzero(outputs["y_var"]) == 0
    assert outputs["kl"].item() == 0.0


def test_deterministic_predictor_uses_shared_backbone_shapes() -> None:
    model = SOHProbabilisticPredictor.deterministic(
        latent_dim=4,
        aux_dim=0,
        horizon=3,
        history_dim=8,
        decoder_dim=8,
    )
    outputs = model(
        z_window=torch.randn(2, 5, 4),
        soh_history=torch.ones(2, 5),
        soh_future=torch.full((2, 3), 0.9),
    )

    assert outputs["next_z_pred"].shape == (2, 4, 4)
    assert outputs["state"].shape == (2, 8)
    assert outputs["y_mean"].shape == (2, 3)
    assert torch.count_nonzero(outputs["y_var"]) == 0


def test_masked_trajectory_mse_ignores_missing_targets() -> None:
    y_true = torch.tensor([[1.0, 2.0, float("nan")]])
    y_mean = torch.zeros_like(y_true)
    mask = torch.tensor([[True, True, False]])

    loss = masked_trajectory_mse(y_true, y_mean, mask)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_deterministic_total_loss_uses_mse_objective() -> None:
    y_mean = torch.tensor([[0.5, 0.25]], requires_grad=True)
    outputs = {
        "y_mean": y_mean,
        "y_cov": torch.zeros(1, 2, 2),
        "kl": torch.tensor(0.0),
        "next_z_pred": torch.empty(1, 0, 2),
    }
    batch = {
        "soh_future": torch.tensor([[1.0, 0.5]]),
        "mask": torch.tensor([[True, True]]),
        "z_window": torch.zeros(1, 1, 2),
    }
    cfg = {
        "training": {
            "objective": "mse",
            "lambda_state": 0.0,
            "beta_kl": 0.0,
        }
    }

    losses = total_loss(outputs, batch, cfg)
    losses["loss"].backward()

    assert torch.isclose(losses["loss"], torch.tensor(0.15625))
    assert torch.isclose(losses["mse"], losses["data_fit"])
    assert y_mean.grad is not None
