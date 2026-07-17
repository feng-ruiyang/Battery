from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.shape[1]]


class CycleEncoder(nn.Module):
    """Transformer encoder for one cycle signal.

    Input shape: `[batch, length, input_dim]`.
    Output shape: `[batch, latent_dim]`.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.latent_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x_cycle: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.input_proj(x_cycle)
        x = self.positional(x)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        if mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = (~mask).to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.latent_proj(self.norm(pooled))


class CycleDecoder(nn.Module):
    """Simple reconstruction decoder conditioned on a cycle latent vector.

    This first version is non-autoregressive to keep pretraining cheap. It predicts
    the full fixed-length signal from learned positions plus the latent state.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 128,
        d_model: int = 128,
        sequence_length: int = 1000,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.position = nn.Parameter(torch.randn(1, sequence_length, d_model) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        batch = z.shape[0]
        latent = self.latent_proj(z).unsqueeze(1)
        x = self.position.expand(batch, -1, -1) + latent
        return self.output_proj(self.decoder(x))


class CycleAutoencoder(nn.Module):
    def __init__(self, encoder: CycleEncoder, decoder: CycleDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_cycle: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        z = self.encoder(x_cycle, mask=mask)
        x_recon = self.decoder(z)
        return {"z": z, "x_recon": x_recon}
