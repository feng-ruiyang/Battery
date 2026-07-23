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
            self.vbll_head = VBLLLinear(input_dim=hidden_dim, output_dim=1)
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
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
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

        trajectory = []
        for _ in range(self.horizon):
            decoded = self._decode_prefix(repeated_state, prefix)
            features = decoded[:, -1]
            y_step = (features * sampled_weight).sum(dim=-1) + sampled_bias
            trajectory.append(y_step)
            prefix = torch.cat([prefix, y_step.unsqueeze(-1)], dim=1)

        samples = torch.stack(trajectory, dim=-1)
        return samples.reshape(sample_count, batch_size, self.horizon)

    def _decode_prefix(self, state: Tensor, decoder_inputs: Tensor) -> Tensor:
        tgt = self.positional(self.soh_proj(decoder_inputs.unsqueeze(-1)))
        memory = self.state_proj(state).unsqueeze(1)
        causal_mask = _causal_mask(decoder_inputs.shape[1], decoder_inputs.device)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        return self.norm(decoded)

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


def make_teacher_forcing_inputs(soh_history: Tensor, soh_future: Tensor) -> Tensor:
    """Build `[s_t, y_{t+1}, ..., y_{t+H-1}]` from history and future labels."""

    start = soh_history[:, -1:]
    if soh_future.shape[1] == 1:
        return start
    future_prefix = torch.nan_to_num(soh_future[:, :-1], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat([start, future_prefix], dim=1)


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
