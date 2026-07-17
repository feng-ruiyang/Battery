from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn

from battery_soh.training.losses import total_loss


Batch = dict[str, Tensor]


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    device: torch.device,
    forward_fn: Callable[[nn.Module, Batch], dict[str, Any]] | None = None,
) -> EpochResult:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = forward_fn(model, batch) if forward_fn else model(**_model_inputs(batch))
        losses = total_loss(outputs, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        _accumulate(totals, losses)
        count += 1
    return _epoch_result(totals, count)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    cfg: dict[str, Any],
    device: torch.device,
    forward_fn: Callable[[nn.Module, Batch], dict[str, Any]] | None = None,
) -> EpochResult:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = forward_fn(model, batch) if forward_fn else model(**_model_inputs(batch))
        losses = total_loss(outputs, batch, cfg)
        _accumulate(totals, losses)
        count += 1
    return _epoch_result(totals, count)


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _model_inputs(batch: Batch) -> dict[str, Tensor | None]:
    return {
        "z_window": batch["z_window"],
        "soh_history": batch["soh_history"],
        "soh_future": batch.get("soh_future"),
        "u_window": batch.get("u_window"),
        "history_mask": batch.get("history_mask"),
    }


def _accumulate(totals: dict[str, float], losses: dict[str, Tensor]) -> None:
    for key, value in losses.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())


def _epoch_result(totals: dict[str, float], count: int) -> EpochResult:
    denom = max(count, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    return EpochResult(loss=metrics.get("loss", float("nan")), metrics=metrics)
