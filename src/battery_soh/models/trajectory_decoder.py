from __future__ import annotations

import torch
from torch import Tensor, nn

from battery_soh.models.encoder_decoder import SinusoidalPositionalEncoding
from battery_soh.models.vbll import VBLLLinear


class TeacherForcingTrajectoryDecoder(nn.Module):
    """Autoregressive SoH decoder with a VBLL one-step prediction head.

    During training, `decoder_inputs` should contain the previous true SoH for
    each forecast step: `[s_t, y_{t+1}, ..., y_{t+H-1}]`. During inference,
    `sample()` feeds sampled or mean predictions back into the decoder.
    """

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        head_kind: str = "vbll",
        vbll_prior_std: float = 1.0,
        vbll_init_rho: float = -5.0,
    ) -> None:
        super().__init__()
        if head_kind not in {"vbll", "deterministic"}:
            raise ValueError(f"Unsupported trajectory head kind: {head_kind!r}.")
        self.horizon = horizon
        self.head_kind = head_kind
        self.soh_proj = nn.Linear(1, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.positional = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        if head_kind == "vbll":
            self.vbll_head = VBLLLinear(
                input_dim=hidden_dim,
                output_dim=1,
                prior_std=vbll_prior_std,
                init_rho=vbll_init_rho,
            )
        else:
            self.deterministic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor, decoder_inputs: Tensor) -> dict[str, Tensor]:
        if decoder_inputs.shape[1] != self.horizon:
            raise ValueError(
                f"decoder_inputs length must equal horizon={self.horizon}, "
                f"got {decoder_inputs.shape[1]}."
            )
        decoded = self._decode_prefix(state, decoder_inputs)
        flat = decoded.reshape(-1, decoded.shape[-1])
        if self.head_kind == "vbll":
            step_dist = self.vbll_head(flat)
            y_mean = step_dist["theta_mean"].reshape(decoded.shape[0], self.horizon)
            y_var = step_dist["theta_var"].reshape(decoded.shape[0], self.horizon).clamp_min(1e-12)
            kl = step_dist["kl"]
        else:
            y_mean = self.deterministic_head(flat).reshape(decoded.shape[0], self.horizon)
            y_var = torch.zeros_like(y_mean)
            kl = y_mean.new_tensor(0.0)
        return {
            "y_mean": y_mean,
            "y_var": y_var,
            "y_cov": torch.diag_embed(y_var),
            "kl": kl,
            "decoder_state": decoded,
        }

    @torch.no_grad()
    def sample(
        self,
        state: Tensor,
        start_soh: Tensor,
        num_samples: int,
        use_mean: bool = False,
    ) -> Tensor:
        return self._rollout(
            state=state,
            start_soh=start_soh,
            num_samples=num_samples,
            horizon=self.horizon,
            use_mean=use_mean,
        )

    def stochastic_rollout(
        self,
        state: Tensor,
        start_soh: Tensor,
        num_samples: int,
        horizon: int,
    ) -> Tensor:
        """Differentiable fixed-weight trajectories for multi-step training."""

        return self._rollout(
            state=state,
            start_soh=start_soh,
            num_samples=num_samples,
            horizon=horizon,
            use_mean=False,
        )

    def teacher_forced_stochastic_samples(
        self,
        state: Tensor,
        decoder_inputs: Tensor,
        num_samples: int,
        horizon: int,
    ) -> Tensor:
        """Differentiable samples using true autoregressive prefixes.

        This keeps the prefix regime teacher-forced while still sampling the
        VBLL output head, which lets CRPS training match the probabilistic
        objective without feeding sampled predictions back as future inputs.
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"horizon must be in [1, {self.horizon}], got {horizon}.")
        if decoder_inputs.shape[1] < horizon:
            raise ValueError(
                f"decoder_inputs length must be at least horizon={horizon}, "
                f"got {decoder_inputs.shape[1]}."
            )
        if self.head_kind == "deterministic":
            sample_count = 1
        else:
            sample_count = num_samples
        batch_size = state.shape[0]
        repeated_state = (
            state.unsqueeze(0)
            .expand(sample_count, -1, -1)
            .reshape(sample_count * batch_size, state.shape[-1])
        )
        repeated_inputs = (
            decoder_inputs[:, :horizon]
            .unsqueeze(0)
            .expand(sample_count, -1, -1)
            .reshape(sample_count * batch_size, horizon)
        )
        decoded = self._decode_prefix(repeated_state, repeated_inputs)
        sampled_weight, sampled_bias = self._sample_output_parameters(
            sample_count=sample_count,
            batch_size=batch_size,
            use_mean=False,
        )
        flat_features = decoded.reshape(sample_count * batch_size * horizon, decoded.shape[-1])
        repeated_weight = sampled_weight.repeat_interleave(horizon, dim=0)
        repeated_bias = sampled_bias.repeat_interleave(horizon, dim=0)
        samples = (flat_features * repeated_weight).sum(dim=-1) + repeated_bias
        return samples.reshape(sample_count, batch_size, horizon)

    def _rollout(
        self,
        state: Tensor,
        start_soh: Tensor,
        num_samples: int,
        horizon: int,
        use_mean: bool,
    ) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"horizon must be in [1, {self.horizon}], got {horizon}.")
        sample_count = 1 if use_mean or self.head_kind == "deterministic" else num_samples
        batch_size = state.shape[0]
        repeated_state = (
            state.unsqueeze(0)
            .expand(sample_count, -1, -1)
            .reshape(sample_count * batch_size, state.shape[-1])
        )
        prefix = (
            start_soh.unsqueeze(0)
            .expand(sample_count, -1)
            .reshape(sample_count * batch_size, 1)
        )
        sampled_weight, sampled_bias = self._sample_output_parameters(
            sample_count=sample_count,
            batch_size=batch_size,
            use_mean=use_mean,
        )

        memory = self.state_proj(repeated_state).unsqueeze(1)
        self_attention_cache: list[Tensor | None] = [None] * len(self.decoder.layers)
        trajectory = []
        decoder_input = prefix
        for position in range(horizon):
            decoded = self._decode_incremental_step(
                memory=memory,
                decoder_input=decoder_input,
                position=position,
                self_attention_cache=self_attention_cache,
            )
            features = decoded[:, 0]
            y_step = (features * sampled_weight).sum(dim=-1) + sampled_bias
            trajectory.append(y_step)
            decoder_input = y_step.unsqueeze(-1)

        samples = torch.stack(trajectory, dim=-1)
        return samples.reshape(sample_count, batch_size, horizon)

    @torch.no_grad()
    def scheduled_sampling_inputs(
        self,
        state: Tensor,
        start_soh: Tensor,
        soh_future: Tensor,
        probability: float,
    ) -> tuple[Tensor, Tensor]:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Scheduled-sampling probability must be in [0, 1], got {probability}.")
        if soh_future.shape[1] != self.horizon:
            raise ValueError(
                f"soh_future length must equal horizon={self.horizon}, got {soh_future.shape[1]}."
            )

        memory = self.state_proj(state).unsqueeze(1)
        self_attention_cache: list[Tensor | None] = [None] * len(self.decoder.layers)
        decoder_input = start_soh.unsqueeze(-1)
        decoder_inputs = [decoder_input]
        selections = []

        for position in range(self.horizon - 1):
            decoded = self._decode_incremental_step(
                memory=memory,
                decoder_input=decoder_input,
                position=position,
                self_attention_cache=self_attention_cache,
            )
            prediction = self._posterior_mean(decoded[:, 0]).unsqueeze(-1)
            target = soh_future[:, position : position + 1]
            valid_target = torch.isfinite(target)
            selected = (torch.rand_like(target) < probability) & valid_target
            decoder_input = torch.where(selected, prediction, torch.nan_to_num(target))
            decoder_inputs.append(decoder_input)
            selections.append(selected)

        selection_mask = (
            torch.cat(selections, dim=1)
            if selections
            else torch.empty(
                soh_future.shape[0],
                0,
                dtype=torch.bool,
                device=soh_future.device,
            )
        )
        return torch.cat(decoder_inputs, dim=1), selection_mask

    def _decode_prefix(self, state: Tensor, decoder_inputs: Tensor) -> Tensor:
        tgt = self.positional(self.soh_proj(decoder_inputs.unsqueeze(-1)))
        memory = self.state_proj(state).unsqueeze(1)
        causal_mask = _causal_mask(decoder_inputs.shape[1], decoder_inputs.device)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        return self.norm(decoded)

    def _decode_incremental_step(
        self,
        memory: Tensor,
        decoder_input: Tensor,
        position: int,
        self_attention_cache: list[Tensor | None],
    ) -> Tensor:
        token = self.soh_proj(decoder_input.unsqueeze(-1))
        token = token + self.positional.encoding[:, position : position + 1]

        for index, layer in enumerate(self.decoder.layers):
            if not layer.norm_first:
                raise RuntimeError("Incremental rollout requires pre-norm decoder layers.")

            query = layer.norm1(token)
            cached = self_attention_cache[index]
            key_value = query if cached is None else torch.cat([cached, query], dim=1)
            self_attention_cache[index] = key_value
            self_attention = layer.self_attn(
                query,
                key_value,
                key_value,
                need_weights=False,
            )[0]
            token = token + layer.dropout1(self_attention)

            cross_query = layer.norm2(token)
            cross_attention = layer.multihead_attn(
                cross_query,
                memory,
                memory,
                need_weights=False,
            )[0]
            token = token + layer.dropout2(cross_attention)

            feed_forward_input = layer.norm3(token)
            feed_forward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(feed_forward_input)))
            )
            token = token + layer.dropout3(feed_forward)

        return self.norm(token)

    def _sample_output_parameters(
        self,
        sample_count: int,
        batch_size: int,
        use_mean: bool,
    ) -> tuple[Tensor, Tensor]:
        if self.head_kind == "deterministic":
            weight = self.deterministic_head.weight
            bias = self.deterministic_head.bias
        else:
            weight = self.vbll_head.weight_mu
            bias = self.vbll_head.bias_mu
            if not use_mean:
                weight_std = torch.nn.functional.softplus(self.vbll_head.weight_rho)
                weight = weight.unsqueeze(0) + torch.randn(
                    sample_count,
                    *weight.shape,
                    dtype=weight.dtype,
                    device=weight.device,
                ) * weight_std.unsqueeze(0)
                if bias is not None and self.vbll_head.bias_rho is not None:
                    bias_std = torch.nn.functional.softplus(self.vbll_head.bias_rho)
                    bias = bias.unsqueeze(0) + torch.randn(
                        sample_count,
                        *bias.shape,
                        dtype=bias.dtype,
                        device=bias.device,
                    ) * bias_std.unsqueeze(0)

        if weight.ndim == 2:
            weight = weight.unsqueeze(0)
        weight = weight[:, 0].repeat_interleave(batch_size, dim=0)
        if bias is None:
            sampled_bias = weight.new_zeros(weight.shape[0])
        else:
            if bias.ndim == 1:
                bias = bias.unsqueeze(0)
            sampled_bias = bias[:, 0].repeat_interleave(batch_size, dim=0)
        return weight, sampled_bias

    def _posterior_mean(self, features: Tensor) -> Tensor:
        if self.head_kind == "vbll":
            return torch.nn.functional.linear(
                features,
                self.vbll_head.weight_mu,
                self.vbll_head.bias_mu,
            ).squeeze(-1)
        return self.deterministic_head(features).squeeze(-1)


def make_teacher_forcing_inputs(soh_history: Tensor, soh_future: Tensor) -> Tensor:
    """Build `[s_t, y_{t+1}, ..., y_{t+H-1}]` from history and future labels."""

    start = soh_history[:, -1:]
    if soh_future.shape[1] == 1:
        return start
    future_prefix = torch.nan_to_num(soh_future[:, :-1], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat([start, future_prefix], dim=1)


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
