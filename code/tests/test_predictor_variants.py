from __future__ import annotations

import torch

from battery_soh.models.soh_predictor import SOHProbabilisticPredictor
from battery_soh.models.trajectory_decoder import TeacherForcingTrajectoryDecoder
from battery_soh.training.losses import (
    masked_trajectory_mse,
    sample_based_crps,
    stochastic_rollout_negative_loss,
    total_loss,
)
from battery_soh.training.trainer import (
    scheduled_sampling_probability,
    stochastic_rollout_settings,
)


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


def test_vbll_rollout_samples_fixed_weights_in_parallel() -> None:
    decoder = TeacherForcingTrajectoryDecoder(
        state_dim=8,
        horizon=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        head_kind="vbll",
    )
    decoder.eval()
    samples = decoder.sample(
        state=torch.randn(2, 8),
        start_soh=torch.ones(2),
        num_samples=3,
    )

    assert samples.shape == (3, 2, 4)
    assert not torch.equal(samples[0], samples[1])


def test_vbll_predictor_propagates_prior_standard_deviation() -> None:
    model = SOHProbabilisticPredictor.vbll(
        latent_dim=4,
        aux_dim=0,
        horizon=3,
        history_dim=8,
        decoder_dim=8,
        vbll_prior_std=0.5**0.5,
    )

    assert model.decoder.vbll_head.prior_std == 0.5**0.5


def test_vbll_rollout_can_use_posterior_mean_weights() -> None:
    decoder = TeacherForcingTrajectoryDecoder(
        state_dim=8,
        horizon=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        head_kind="vbll",
    )
    decoder.eval()
    state = torch.randn(2, 8)
    start_soh = torch.ones(2)

    first = decoder.sample(state, start_soh, num_samples=128, use_mean=True)
    second = decoder.sample(state, start_soh, num_samples=128, use_mean=True)

    assert first.shape == (1, 2, 4)
    torch.testing.assert_close(first, second)


def test_stochastic_rollout_backpropagates_to_posterior_scale() -> None:
    decoder = TeacherForcingTrajectoryDecoder(
        state_dim=8,
        horizon=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        head_kind="vbll",
    )
    samples = decoder.stochastic_rollout(
        state=torch.randn(2, 8),
        start_soh=torch.ones(2),
        num_samples=3,
        horizon=4,
    )
    samples.square().mean().backward()

    assert samples.shape == (3, 2, 4)
    assert decoder.vbll_head.weight_rho.grad is not None
    assert torch.count_nonzero(decoder.vbll_head.weight_rho.grad) > 0


def test_sample_based_crps_balances_target_error_and_sample_spread() -> None:
    samples = torch.tensor([[[0.0]], [[2.0]]])
    target = torch.tensor([[1.0]])

    loss = sample_based_crps(samples, target)

    assert torch.isclose(loss, torch.tensor(0.5))


def test_stochastic_rollout_negative_loss_penalizes_only_negative_samples() -> None:
    outputs = {
        "y_mean": torch.zeros(1, 1),
        "stochastic_rollout_samples": torch.tensor([[[-2.0, 1.0]]]),
    }
    batch = {"mask": torch.tensor([[True, True]])}

    loss = stochastic_rollout_negative_loss(outputs, batch)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_scheduled_sampling_uses_model_inputs_at_probability_one() -> None:
    model = SOHProbabilisticPredictor.vbll(
        latent_dim=4,
        aux_dim=0,
        horizon=4,
        history_dim=8,
        decoder_dim=8,
    )
    model.eval()
    outputs = model(
        z_window=torch.randn(2, 5, 4),
        soh_history=torch.ones(2, 5),
        soh_future=torch.full((2, 4), 0.9),
        scheduled_sampling_probability=1.0,
    )

    assert outputs["scheduled_sampling_probability"].item() == 1.0
    assert outputs["scheduled_sampling_fraction"].item() == 1.0
    assert torch.isfinite(outputs["y_mean"]).all()


def test_scheduled_sampling_probability_uses_linear_ramp() -> None:
    cfg = {
        "training": {
            "scheduled_sampling": {
                "enabled": True,
                "start_epoch": 10,
                "end_epoch": 70,
                "start_probability": 0.0,
                "end_probability": 0.5,
            }
        }
    }

    assert scheduled_sampling_probability(cfg, 1) == 0.0
    assert scheduled_sampling_probability(cfg, 10) == 0.0
    assert scheduled_sampling_probability(cfg, 40) == 0.25
    assert scheduled_sampling_probability(cfg, 70) == 0.5
    assert scheduled_sampling_probability(cfg, 100) == 0.5


def test_stochastic_rollout_settings_respect_start_epoch() -> None:
    cfg = {
        "training": {
            "stochastic_rollout": {
                "enabled": True,
                "start_epoch": 10,
                "num_samples": 4,
                "horizon": 20,
            }
        }
    }

    assert stochastic_rollout_settings(cfg, 9) == (0, 0)
    assert stochastic_rollout_settings(cfg, 10) == (4, 20)


def test_total_loss_combines_nll_and_differentiable_rollout_crps() -> None:
    model = SOHProbabilisticPredictor.vbll(
        latent_dim=4,
        aux_dim=0,
        horizon=4,
        history_dim=8,
        decoder_dim=8,
    )
    batch = {
        "z_window": torch.randn(2, 5, 4),
        "soh_history": torch.ones(2, 5),
        "soh_future": torch.full((2, 4), 0.9),
        "mask": torch.ones(2, 4, dtype=torch.bool),
    }
    outputs = model(
        z_window=batch["z_window"],
        soh_history=batch["soh_history"],
        soh_future=batch["soh_future"],
        stochastic_rollout_num_samples=3,
        stochastic_rollout_horizon=3,
    )
    cfg = {
        "training": {
            "objective": "nll",
            "beta_kl": 1.0e-4,
            "lambda_state": 1.0,
            "stochastic_rollout": {
                "weight": 0.1,
                "negative_weight": 0.1,
                "objective": "crps",
            },
        }
    }

    losses = total_loss(outputs, batch, cfg)
    losses["loss"].backward()

    assert losses["stochastic_rollout"].item() > 0.0
    assert losses["stochastic_rollout_negative"].item() >= 0.0
    assert model.decoder.vbll_head.weight_rho.grad is not None
    assert torch.count_nonzero(model.decoder.vbll_head.weight_rho.grad) > 0


def test_incremental_decoder_matches_full_prefix_decoder() -> None:
    decoder = TeacherForcingTrajectoryDecoder(
        state_dim=8,
        horizon=5,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        head_kind="vbll",
    )
    decoder.eval()
    state = torch.randn(3, 8)
    decoder_inputs = torch.randn(3, 5)

    expected = decoder._decode_prefix(state, decoder_inputs)
    memory = decoder.state_proj(state).unsqueeze(1)
    cache = [None] * len(decoder.decoder.layers)
    actual_steps = [
        decoder._decode_incremental_step(
            memory=memory,
            decoder_input=decoder_inputs[:, position : position + 1],
            position=position,
            self_attention_cache=cache,
        )
        for position in range(decoder_inputs.shape[1])
    ]
    actual = torch.cat(actual_steps, dim=1)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_deterministic_rollout_uses_single_mean_trajectory() -> None:
    decoder = TeacherForcingTrajectoryDecoder(
        state_dim=8,
        horizon=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        head_kind="deterministic",
    )
    decoder.eval()
    samples = decoder.sample(
        state=torch.randn(2, 8),
        start_soh=torch.ones(2),
        num_samples=3,
    )

    assert samples.shape == (1, 2, 4)


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
