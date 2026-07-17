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
    ) -> None:
        super().__init__()
        self.horizon = horizon
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
        self.vbll_head = VBLLLinear(input_dim=hidden_dim, output_dim=1)

    def forward(self, state: Tensor, decoder_inputs: Tensor) -> dict[str, Tensor]:
        if decoder_inputs.shape[1] != self.horizon:
            raise ValueError(
                f"decoder_inputs length must equal horizon={self.horizon}, "
                f"got {decoder_inputs.shape[1]}."
            )
        tgt = self.soh_proj(decoder_inputs.unsqueeze(-1))
        tgt = self.positional(tgt)
        memory = self.state_proj(state).unsqueeze(1)
        causal_mask = _causal_mask(self.horizon, decoder_inputs.device)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        decoded = self.norm(decoded)
        flat = decoded.reshape(-1, decoded.shape[-1])
        step_dist = self.vbll_head(flat)
        y_mean = step_dist["theta_mean"].reshape(decoded.shape[0], self.horizon)
        y_var = step_dist["theta_var"].reshape(decoded.shape[0], self.horizon).clamp_min(1e-12)
        return {
            "y_mean": y_mean,
            "y_var": y_var,
            "y_cov": torch.diag_embed(y_var),
            "kl": step_dist["kl"],
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
        samples = []
        for _ in range(num_samples):
            trajectory = []
            for step in range(self.horizon):
                decoder_inputs = _pad_decoder_inputs(start_soh, trajectory, self.horizon)
                dist = self.forward(state=state, decoder_inputs=decoder_inputs)
                mean = dist["y_mean"][:, step]
                var = dist["y_var"][:, step]
                if use_mean:
                    y_step = mean
                else:
                    y_step = mean + torch.randn_like(mean) * var.sqrt()
                trajectory.append(y_step)
            samples.append(torch.stack(trajectory, dim=1))
        return torch.stack(samples, dim=0)


def make_teacher_forcing_inputs(soh_history: Tensor, soh_future: Tensor) -> Tensor:
    """Build `[s_t, y_{t+1}, ..., y_{t+H-1}]` from history and future labels."""

    start = soh_history[:, -1:]
    if soh_future.shape[1] == 1:
        return start
    future_prefix = torch.nan_to_num(soh_future[:, :-1], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat([start, future_prefix], dim=1)


def _pad_decoder_inputs(start_soh: Tensor, trajectory: list[Tensor], horizon: int) -> Tensor:
    values = [start_soh.new_zeros(start_soh.shape) for _ in range(horizon)]
    values[0] = start_soh
    for i, value in enumerate(trajectory, start=1):
        if i >= horizon:
            break
        values[i] = value
    return torch.stack(values, dim=1)


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
