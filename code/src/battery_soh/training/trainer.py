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
    epoch: int | None = None,
) -> EpochResult:
    model.train()
    totals: dict[str, float] = {}
    running_totals: dict[str, float] = {}
    running_count = 0
    count = 0
    log_every = int(cfg.get("training", {}).get("log_every_n_steps", 0))
    total_steps = _safe_len(dataloader)
    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = forward_fn(model, batch) if forward_fn else model(**_model_inputs(batch))
        losses = total_loss(outputs, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        _accumulate(totals, losses)
        _accumulate(running_totals, losses)
        count += 1
        running_count += 1
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == total_steps):
            running = _average(running_totals, step_count=running_count)
            epoch_metrics = _average(totals, step_count=count)
            step_label = f"{step}/{total_steps}" if total_steps is not None else str(step)
            epoch_label = f"epoch={epoch} " if epoch is not None else ""
            print(
                f"{epoch_label}step={step_label} "
                f"train_loss_running={running.get('loss', float('nan')):.6f} "
                f"train_nll_running={running.get('nll', float('nan')):.6f} "
                f"train_state_running={running.get('latent_state', float('nan')):.6f} "
                f"train_loss_epoch={epoch_metrics.get('loss', float('nan')):.6f}",
                flush=True,
            )
            running_totals = {}
            running_count = 0
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
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


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


def _average(totals: dict[str, float], step_count: int) -> dict[str, float]:
    denom = max(step_count, 1)
    return {key: value / denom for key, value in totals.items()}


def _epoch_result(totals: dict[str, float], count: int) -> EpochResult:
    metrics = _average(totals, count)
    return EpochResult(loss=metrics.get("loss", float("nan")), metrics=metrics)


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None
