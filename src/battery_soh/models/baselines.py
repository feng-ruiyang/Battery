from __future__ import annotations

from torch import Tensor, nn


class DirectMultiStepMLP(nn.Module):
    """Simple direct multi-step baseline for cached history features."""

    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        y_mean = self.net(features)
        return {"y_mean": y_mean}
