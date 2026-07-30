from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def gaussian_trajectory_nll(
    y_true: Tensor,
    y_mean: Tensor,
    y_cov: Tensor,
    mask: Tensor | None = None,
    jitter: float = 1e-5,
) -> Tensor:
    """Multivariate Gaussian NLL with optional horizon mask.

    If `mask` is present, the first version falls back to independent masked
    diagonal likelihood. Full masked covariance can be added after data lengths
    are finalized.
    """

    if mask is not None:
        diag_var = torch.diagonal(y_cov, dim1=-2, dim2=-1).clamp_min(jitter)
        valid = mask.to(dtype=torch.bool) & torch.isfinite(y_true)
        nll = 0.5 * (((y_true - y_mean).pow(2) / diag_var) + diag_var.log())
        return nll[valid].mean()

    eye = torch.eye(y_cov.shape[-1], dtype=y_cov.dtype, device=y_cov.device)
    cov = y_cov + eye.unsqueeze(0) * jitter
    chol = torch.linalg.cholesky(cov)
    residual = (y_true - y_mean).unsqueeze(-1)
    solved = torch.cholesky_solve(residual, chol).squeeze(-1)
    mahal = ((y_true - y_mean) * solved).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    return 0.5 * (mahal + logdet).mean()


def total_loss(outputs: dict[str, Any], batch: dict[str, Tensor], cfg: dict[str, Any]) -> dict[str, Tensor]:
    y_true = batch["soh_future"]
    mask = batch.get("mask")
    mse = masked_trajectory_mse(y_true, outputs["y_mean"], mask=mask)
    objective = str(_deep_get(cfg, "training.objective", "nll"))
    if objective == "nll":
        data_fit = gaussian_trajectory_nll(y_true, outputs["y_mean"], outputs["y_cov"], mask=mask)
        nll = data_fit.detach()
    elif objective == "mse":
        data_fit = mse
        nll = mse.new_tensor(0.0)
    else:
        raise ValueError(f"Unsupported training objective: {objective!r}.")
    latent_state = latent_next_step_loss(outputs, batch)
    kl = _as_tensor(outputs.get("kl", 0.0), data_fit)
    regularization = outputs.get("regularization", {})
    negative_rate = _as_tensor(regularization.get("negative_rate", 0.0), data_fit)
    smoothness = _as_tensor(regularization.get("smoothness", 0.0), data_fit)
    stochastic_rollout = stochastic_rollout_loss(outputs, batch, cfg)
    stochastic_negative = stochastic_rollout_negative_loss(outputs, batch)
    rollout_samples = outputs.get("stochastic_rollout_samples")
    rollout_horizon = rollout_samples.shape[-1] if rollout_samples is not None else 0

    beta_kl = float(_deep_get(cfg, "training.beta_kl", 0.0))
    lambda_state = float(_deep_get(cfg, "training.lambda_state", 1.0))
    lambda_negative = float(_deep_get(cfg, "training.lambda_negative", 0.0))
    lambda_smooth = float(_deep_get(cfg, "training.lambda_smooth", 0.0))
    lambda_rollout = float(_deep_get(cfg, "training.stochastic_rollout.weight", 0.0))
    lambda_rollout_negative = float(
        _deep_get(cfg, "training.stochastic_rollout.negative_weight", 0.0)
    )
    loss = (
        data_fit
        + lambda_state * latent_state
        + beta_kl * kl
        + lambda_negative * negative_rate
        + lambda_smooth * smoothness
        + lambda_rollout * stochastic_rollout
        + lambda_rollout_negative * stochastic_negative
    )
    return {
        "loss": loss,
        "data_fit": data_fit.detach(),
        "mse": mse.detach(),
        "nll": nll,
        "latent_state": latent_state.detach(),
        "kl": kl.detach(),
        "negative_rate": negative_rate.detach(),
        "smoothness": smoothness.detach(),
        "stochastic_rollout": stochastic_rollout.detach(),
        "stochastic_rollout_negative": stochastic_negative.detach(),
        "stochastic_rollout_horizon": data_fit.new_tensor(rollout_horizon).detach(),
        "scheduled_sampling_probability": _as_tensor(
            outputs.get("scheduled_sampling_probability", 0.0),
            data_fit,
        ).detach(),
        "scheduled_sampling_fraction": _as_tensor(
            outputs.get("scheduled_sampling_fraction", 0.0),
            data_fit,
        ).detach(),
    }


def masked_trajectory_mse(
    y_true: Tensor,
    y_mean: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    valid = torch.isfinite(y_true)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return y_mean.sum() * 0.0
    return (y_true[valid] - y_mean[valid]).pow(2).mean()


def latent_next_step_loss(outputs: dict[str, Any], batch: dict[str, Tensor]) -> Tensor:
    pred = outputs.get("next_z_pred")
    if pred is None or pred.shape[1] == 0:
        return batch["soh_future"].new_tensor(0.0)
    target = batch["z_window"][:, 1 : 1 + pred.shape[1]]
    loss = (pred - target).pow(2).mean(dim=-1)
    history_mask = batch.get("history_mask")
    if history_mask is not None:
        valid = (~history_mask[:, 1 : 1 + pred.shape[1]]).to(loss.dtype)
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)
    return loss.mean()


def stochastic_rollout_loss(
    outputs: dict[str, Any],
    batch: dict[str, Tensor],
    cfg: dict[str, Any],
) -> Tensor:
    samples = outputs.get("stochastic_rollout_samples")
    if samples is None:
        return batch["soh_future"].new_tensor(0.0)

    horizon = samples.shape[-1]
    target = batch["soh_future"][:, :horizon]
    mask = batch.get("mask")
    if mask is not None:
        mask = mask[:, :horizon]
    objective = str(_deep_get(cfg, "training.stochastic_rollout.objective", "crps"))
    if objective == "crps":
        return sample_based_crps(samples, target, mask)
    if objective == "mse":
        valid = torch.isfinite(target)
        if mask is not None:
            valid = valid & mask.to(dtype=torch.bool)
        if not torch.any(valid):
            return samples.sum() * 0.0
        squared_error = (samples - target.unsqueeze(0)).pow(2)
        return squared_error[:, valid].mean()
    raise ValueError(f"Unsupported stochastic rollout objective: {objective!r}.")


def sample_based_crps(
    samples: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Empirical CRPS for samples shaped [num_samples, batch, horizon]."""

    valid = torch.isfinite(target)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return samples.sum() * 0.0

    target_distance = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    pairwise_distance = (
        samples.unsqueeze(1) - samples.unsqueeze(0)
    ).abs().mean(dim=(0, 1))
    crps = target_distance - 0.5 * pairwise_distance
    return crps[valid].mean()


def stochastic_rollout_negative_loss(
    outputs: dict[str, Any],
    batch: dict[str, Tensor],
) -> Tensor:
    samples = outputs.get("stochastic_rollout_samples")
    if samples is None:
        reference = outputs["y_mean"]
        return reference.new_tensor(0.0)
    penalty = samples.clamp_max(0.0).neg()
    mask = batch.get("mask")
    if mask is None:
        return penalty.mean()
    valid = mask[:, : samples.shape[-1]].to(dtype=torch.bool)
    if not torch.any(valid):
        return samples.sum() * 0.0
    return penalty[:, valid].mean()


def reconstruction_loss(x_recon: Tensor, x_true: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = F.mse_loss(x_recon, x_true, reduction="none")
    if mask is not None:
        valid = (~mask).to(loss.dtype).unsqueeze(-1)
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)
    return loss.mean()


def _as_tensor(value: Any, like: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return like.new_tensor(float(value))


def _deep_get(mapping: dict[str, Any], path: str, default: Any) -> Any:
    current: Any = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
