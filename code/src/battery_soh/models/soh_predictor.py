from __future__ import annotations

from torch import Tensor, nn

from battery_soh.models.history_aggregator import LatentStateTransformer
from battery_soh.models.trajectory_decoder import (
    TeacherForcingTrajectoryDecoder,
    make_teacher_forcing_inputs,
)


class SOHProbabilisticPredictor(nn.Module):
    """Latent degradation state learner + teacher-forcing VBLL trajectory decoder.

    The state transformer learns `X_1:t-1 -> X_t` in latent space. The final
    causal hidden state is then used by an autoregressive decoder whose VBLL
    head predicts the future SoH trajectory distribution.
    """

    def __init__(
        self,
        state_model: LatentStateTransformer,
        decoder: TeacherForcingTrajectoryDecoder,
    ) -> None:
        super().__init__()
        self.state_model = state_model
        self.decoder = decoder

    @classmethod
    def vbll(
        cls,
        latent_dim: int,
        aux_dim: int,
        horizon: int,
        num_basis: int | None = None,
        history_dim: int = 256,
        decoder_dim: int | None = None,
        vbll_prior_std: float = 1.0,
        vbll_init_rho: float = -5.0,
    ) -> "SOHProbabilisticPredictor":
        del num_basis
        state_model = LatentStateTransformer(
            latent_dim=latent_dim,
            aux_dim=aux_dim,
            hidden_dim=history_dim,
        )
        decoder = TeacherForcingTrajectoryDecoder(
            state_dim=history_dim,
            horizon=horizon,
            hidden_dim=decoder_dim or history_dim,
            vbll_prior_std=vbll_prior_std,
            vbll_init_rho=vbll_init_rho,
        )
        return cls(state_model=state_model, decoder=decoder)

    @classmethod
    def deterministic(
        cls,
        latent_dim: int,
        aux_dim: int,
        horizon: int,
        num_basis: int | None = None,
        history_dim: int = 256,
        decoder_dim: int | None = None,
    ) -> "SOHProbabilisticPredictor":
        del num_basis
        state_model = LatentStateTransformer(
            latent_dim=latent_dim,
            aux_dim=aux_dim,
            hidden_dim=history_dim,
        )
        decoder = TeacherForcingTrajectoryDecoder(
            state_dim=history_dim,
            horizon=horizon,
            hidden_dim=decoder_dim or history_dim,
            head_kind="deterministic",
        )
        return cls(state_model=state_model, decoder=decoder)

    def forward(
        self,
        z_window: Tensor,
        soh_history: Tensor,
        soh_future: Tensor | None = None,
        u_window: Tensor | None = None,
        history_mask: Tensor | None = None,
        num_samples: int = 0,
        scheduled_sampling_probability: float = 0.0,
        stochastic_rollout_num_samples: int = 0,
        stochastic_rollout_horizon: int = 0,
    ) -> dict:
        state_outputs = self.state_model(z_window=z_window, u_window=u_window, mask=history_mask)
        state = state_outputs["state"]
        scheduled_fraction = state.new_tensor(0.0)
        if soh_future is None:
            decoder_inputs = _repeat_last_soh(soh_history, self.decoder.horizon)
        elif scheduled_sampling_probability > 0.0:
            decoder_was_training = self.decoder.training
            self.decoder.eval()
            try:
                decoder_inputs, selection_mask = self.decoder.scheduled_sampling_inputs(
                    state=state.detach(),
                    start_soh=soh_history[:, -1],
                    soh_future=soh_future,
                    probability=scheduled_sampling_probability,
                )
            finally:
                self.decoder.train(decoder_was_training)
            if selection_mask.numel() > 0:
                scheduled_fraction = selection_mask.to(dtype=state.dtype).mean()
        else:
            decoder_inputs = make_teacher_forcing_inputs(soh_history=soh_history, soh_future=soh_future)

        trajectory = self.decoder(state=state, decoder_inputs=decoder_inputs)
        outputs = {
            **trajectory,
            "state": state,
            "encoded_history": state_outputs["encoded_history"],
            "next_z_pred": state_outputs["next_z_pred"],
            "scheduled_sampling_probability": state.new_tensor(
                scheduled_sampling_probability
            ),
            "scheduled_sampling_fraction": scheduled_fraction,
        }
        if num_samples > 0:
            outputs["trajectory_samples"] = self.decoder.sample(
                state=state,
                start_soh=soh_history[:, -1],
                num_samples=num_samples,
            )
        if stochastic_rollout_num_samples > 0:
            outputs["stochastic_rollout_samples"] = self.decoder.stochastic_rollout(
                state=state,
                start_soh=soh_history[:, -1],
                num_samples=stochastic_rollout_num_samples,
                horizon=stochastic_rollout_horizon,
            )
        return outputs


def _repeat_last_soh(soh_history: Tensor, horizon: int) -> Tensor:
    return soh_history[:, -1:].expand(-1, horizon)
