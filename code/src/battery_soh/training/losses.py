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
    nll = gaussian_trajectory_nll(y_true, outputs["y_mean"], outputs["y_cov"], mask=mask)
    latent_state = latent_next_step_loss(outputs, batch)
    kl = _as_tensor(outputs.get("kl", 0.0), nll)
    regularization = outputs.get("regularization", {})
    negative_rate = _as_tensor(regularization.get("negative_rate", 0.0), nll)
    smoothness = _as_tensor(regularization.get("smoothness", 0.0), nll)

    beta_kl = float(_deep_get(cfg, "training.beta_kl", 0.0))
    lambda_state = float(_deep_get(cfg, "training.lambda_state", 1.0))
    lambda_negative = float(_deep_get(cfg, "training.lambda_negative", 0.0))
    lambda_smooth = float(_deep_get(cfg, "training.lambda_smooth", 0.0))
    loss = (
        nll
        + lambda_state * latent_state
        + beta_kl * kl
        + lambda_negative * negative_rate
        + lambda_smooth * smoothness
    )
    return {
        "loss": loss,
        "nll": nll.detach(),
        "latent_state": latent_state.detach(),
        "kl": kl.detach(),
        "negative_rate": negative_rate.detach(),
        "smoothness": smoothness.detach(),
    }


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
