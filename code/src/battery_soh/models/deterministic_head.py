from __future__ import annotations

from torch import Tensor, nn


class DeterministicHead(nn.Module):
    """MLP head that outputs deterministic dynamics parameters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        theta_mean = self.net(features)
        theta_var = theta_mean.new_zeros(theta_mean.shape)
        return {"theta_mean": theta_mean, "theta_var": theta_var, "kl": theta_mean.new_tensor(0.0)}
