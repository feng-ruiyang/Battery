from __future__ import annotations

import torch
from torch import Tensor, nn

from battery_soh.models.encoder_decoder import SinusoidalPositionalEncoding


class LatentStateTransformer(nn.Module):
    """Causal transformer that learns degradation state by predicting `X_t` from `X_1:t-1`.

    `z_window` is the encoded cycle sequence. At position `i`, the causal
    transformer can only attend to positions `<= i`; `next_latent_head` then
    predicts the next latent `z_{i+1}`. The final hidden state is used as the
    degradation state for future trajectory decoding.
    """

    def __init__(
        self,
        latent_dim: int,
        aux_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(latent_dim + aux_dim, hidden_dim)
        self.positional = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_window: Tensor,
        u_window: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if z_window.shape[1] < 1:
            raise ValueError("z_window must contain at least one cycle.")
        if u_window is not None:
            x = torch.cat([z_window, u_window], dim=-1)
        else:
            x = z_window

        hidden = self.positional(self.input_proj(x))
        causal_mask = _causal_mask(hidden.shape[1], hidden.device)
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=mask)
        encoded = self.norm(encoded)
        state = _last_valid(encoded, mask)
        if encoded.shape[1] > 1:
            next_z_pred = self.next_latent_head(encoded[:, :-1])
        else:
            next_z_pred = encoded.new_empty(encoded.shape[0], 0, self.latent_dim)
        return {
            "state": state,
            "encoded_history": encoded,
            "next_z_pred": next_z_pred,
        }


class HistoryAggregator(LatentStateTransformer):
    """Compatibility wrapper returning only the final latent degradation state."""

    def forward(
        self,
        z_window: Tensor,
        u_window: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        return super().forward(z_window=z_window, u_window=u_window, mask=mask)["state"]


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def _last_valid(encoded: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return encoded[:, -1]
    lengths = (~mask).sum(dim=1).clamp_min(1)
    return encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths - 1]
