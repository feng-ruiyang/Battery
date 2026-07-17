from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    best: float | None = None
    num_bad_epochs: int = 0

    def step(self, value: float) -> bool:
        improved = self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        )
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

