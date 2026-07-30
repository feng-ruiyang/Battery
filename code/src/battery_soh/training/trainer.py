from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    sampling_probability = scheduled_sampling_probability(cfg, epoch)
    rollout_num_samples, rollout_horizon = stochastic_rollout_settings(cfg, epoch)
    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        outputs = (
            forward_fn(model, batch)
            if forward_fn
            else model(
                **_model_inputs(batch),
                scheduled_sampling_probability=sampling_probability,
                stochastic_rollout_num_samples=rollout_num_samples,
                stochastic_rollout_horizon=rollout_horizon,
            )
        )
        losses = total_loss(outputs, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        losses["vbll_rho_grad_norm"] = vbll_rho_grad_norm(model)
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
                f"train_data_running={running.get('data_fit', float('nan')):.6f} "
                f"train_state_running={running.get('latent_state', float('nan')):.6f} "
                f"train_ss={running.get('scheduled_sampling_fraction', 0.0):.3f} "
                f"train_rollout={running.get('stochastic_rollout', 0.0):.6f} "
                f"train_negative="
                f"{running.get('stochastic_rollout_negative', 0.0):.6f} "
                f"train_rollout_horizon="
                f"{running.get('stochastic_rollout_horizon', 0.0):.0f} "
                f"train_rho_grad="
                f"{running.get('vbll_rho_grad_norm', 0.0):.6f} "
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


def scheduled_sampling_probability(cfg: dict[str, Any], epoch: int | None) -> float:
    schedule = cfg.get("training", {}).get("scheduled_sampling", {})
    if not schedule or not bool(schedule.get("enabled", False)) or epoch is None:
        return 0.0

    start_epoch = int(schedule.get("start_epoch", 1))
    end_epoch = int(schedule.get("end_epoch", start_epoch))
    start_probability = float(schedule.get("start_probability", 0.0))
    end_probability = float(schedule.get("end_probability", 0.5))
    if not 0.0 <= start_probability <= 1.0 or not 0.0 <= end_probability <= 1.0:
        raise ValueError("Scheduled-sampling probabilities must be in [0, 1].")
    if end_epoch < start_epoch:
        raise ValueError("scheduled_sampling.end_epoch must be >= start_epoch.")
    if epoch <= start_epoch:
        return start_probability
    if epoch >= end_epoch:
        return end_probability
    progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return start_probability + progress * (end_probability - start_probability)


def stochastic_rollout_settings(
    cfg: dict[str, Any],
    epoch: int | None,
) -> tuple[int, int]:
    rollout = cfg.get("training", {}).get("stochastic_rollout", {})
    if not rollout or not bool(rollout.get("enabled", False)) or epoch is None:
        return 0, 0
    if epoch < int(rollout.get("start_epoch", 1)):
        return 0, 0

    num_samples = int(rollout.get("num_samples", 0))
    if num_samples <= 0:
        raise ValueError("stochastic_rollout.num_samples must be positive.")

    schedule = rollout.get("horizon_schedule")
    if schedule is not None:
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("stochastic_rollout.horizon_schedule must be a non-empty list.")
        parsed_schedule = []
        for stage in schedule:
            if not isinstance(stage, dict):
                raise ValueError("Each stochastic rollout horizon stage must be a mapping.")
            stage_epoch = int(stage.get("start_epoch", 0))
            stage_horizon = int(stage.get("horizon", 0))
            if stage_epoch <= 0 or stage_horizon <= 0:
                raise ValueError(
                    "Each stochastic rollout horizon stage requires positive "
                    "start_epoch and horizon."
                )
            parsed_schedule.append((stage_epoch, stage_horizon))
        if len({stage_epoch for stage_epoch, _ in parsed_schedule}) != len(parsed_schedule):
            raise ValueError("stochastic_rollout.horizon_schedule start epochs must be unique.")
        active = [
            (stage_epoch, stage_horizon)
            for stage_epoch, stage_horizon in parsed_schedule
            if epoch >= stage_epoch
        ]
        if not active:
            return 0, 0
        horizon = max(active, key=lambda stage: stage[0])[1]
    else:
        horizon = int(rollout.get("horizon", 0))
    if horizon <= 0:
        raise ValueError("stochastic_rollout.horizon must be positive.")
    return num_samples, horizon


def build_predictor_optimizer(
    model: nn.Module,
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    learning_rate = float(train_cfg.get("lr", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    rho_multiplier = float(train_cfg.get("vbll_rho_lr_multiplier", 1.0))
    if rho_multiplier <= 0.0:
        raise ValueError("training.vbll_rho_lr_multiplier must be positive.")

    base_parameters = []
    rho_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(("weight_rho", "bias_rho")):
            rho_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    parameter_groups = [
        {
            "params": base_parameters,
            "lr": learning_rate,
            "group_name": "base",
        }
    ]
    if rho_parameters:
        parameter_groups.append(
            {
                "params": rho_parameters,
                "lr": learning_rate * rho_multiplier,
                "group_name": "vbll_rho",
            }
        )
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


@torch.no_grad()
def vbll_posterior_metrics(model: nn.Module) -> dict[str, float]:
    weight_std = []
    bias_std = []
    for name, parameter in model.named_parameters():
        if name.endswith("weight_rho"):
            weight_std.append(F.softplus(parameter.detach()).reshape(-1))
        elif name.endswith("bias_rho"):
            bias_std.append(F.softplus(parameter.detach()).reshape(-1))

    metrics: dict[str, float] = {}
    if weight_std:
        values = torch.cat(weight_std)
        metrics["weight_std_mean"] = float(values.mean().cpu())
        metrics["weight_std_max"] = float(values.max().cpu())
    if bias_std:
        values = torch.cat(bias_std)
        metrics["bias_std"] = float(values.mean().cpu())
    return metrics


def vbll_rho_grad_norm(model: nn.Module) -> Tensor:
    squared_norm = None
    reference = None
    for name, parameter in model.named_parameters():
        if not name.endswith(("weight_rho", "bias_rho")):
            continue
        reference = parameter
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().pow(2).sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is not None:
        return squared_norm.sqrt()
    if reference is not None:
        return reference.new_tensor(0.0)
    return torch.tensor(0.0)
