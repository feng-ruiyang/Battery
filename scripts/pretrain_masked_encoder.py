from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from battery_soh.models.encoder_decoder import CycleAutoencoder, CycleDecoder, CycleEncoder
from battery_soh.training.checkpointing import load_checkpoint, save_checkpoint
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device
from battery_soh.utils.logging import build_metric_logger
from battery_soh.utils.runs import resolve_run_output_dir
from battery_soh.utils.seed import seed_everything
from pretrain_encoder import _build_scheduler, build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain cycle autoencoder with SimMTM-style masking.")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_encoder_severson_simmtm.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    masked_cfg = train_cfg.get("masked_pretraining", {})
    channels = tuple(data_cfg.get("channels", ("V", "I", "T")))
    if int(model_cfg["input_dim"]) != len(channels):
        raise ValueError(
            f"model.input_dim={model_cfg['input_dim']} must match data.channels={len(channels)}."
        )

    device = resolve_device(train_cfg.get("device", "auto"))
    encoder = CycleEncoder(
        input_dim=int(model_cfg["input_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        d_model=int(model_cfg.get("d_model", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        num_heads=int(model_cfg.get("num_heads", 4)),
    )
    decoder = CycleDecoder(
        output_dim=int(model_cfg["input_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        d_model=int(model_cfg.get("d_model", 128)),
        sequence_length=int(data_cfg.get("sequence_length", 1000)),
    )
    model = CycleAutoencoder(encoder, decoder).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MaskedCycleAutoencoder parameters: {n_params:,}")
    print(f"Device: {device}")
    print(
        "Masked pretraining: "
        f"views={int(masked_cfg.get('num_views', 3))}, "
        f"mask_ratio={float(masked_cfg.get('mask_ratio', 0.5))}, "
        f"block_length={int(masked_cfg.get('block_length', 1))}, "
        f"contrastive_weight={float(masked_cfg.get('contrastive_weight', 0.1))}",
        flush=True,
    )

    if args.dry_run:
        x = torch.randn(2, int(data_cfg.get("sequence_length", 1000)), len(channels), device=device)
        result = masked_forward(model, x, channels, train_cfg)
        for key, value in result.items():
            if isinstance(value, Tensor):
                print(f"{key}: {tuple(value.shape) if value.ndim else float(value):}")
        return

    train_loader, val_loader = build_dataloaders(cfg)
    if len(train_loader.dataset) == 0:
        raise SystemExit("No training cycles found. Run preprocessing first and check processed_dir.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, train_cfg)
    output_dir = (
        args.resume.expanduser().resolve().parent
        if args.resume is not None
        else resolve_run_output_dir(cfg, "outputs/masked_cycle_autoencoder")
    )
    print(f"Output dir: {output_dir}", flush=True)

    best_val = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume is not None:
        payload = load_checkpoint(args.resume.expanduser(), model, optimizer=optimizer, scheduler=scheduler)
        start_epoch = int(payload.get("epoch") or 0) + 1
        state = payload.get("training_state", {})
        best_val = float(state.get("best_val", payload.get("metrics", {}).get("val_loss", float("inf"))))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        resume_lr = train_cfg.get("resume_lr")
        if resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(resume_lr)
        print(
            f"Resumed checkpoint: {args.resume}; start_epoch={start_epoch}; "
            f"best_val={best_val:.6f}; lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )

    logger = build_metric_logger(cfg, output_dir)
    max_epochs = int(train_cfg.get("max_epochs", 1))
    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    if start_epoch > max_epochs:
        raise ValueError(f"Resume starts at epoch {start_epoch}, but max_epochs={max_epochs}.")

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            train_result = train_one_epoch(model, train_loader, optimizer, device, train_cfg, epoch, channels)
            val_result = evaluate(model, val_loader, device, train_cfg, channels) if val_loader else train_result
            train_loss = train_result["loss"]
            val_loss = val_result["loss"]
            previous_best = best_val
            is_best = val_loss < previous_best
            meaningful = val_loss < previous_best - early_min_delta
            if is_best:
                best_val = val_loss
            epochs_without_improvement = 0 if meaningful else epochs_without_improvement + 1
            if scheduler is not None:
                scheduler.step(val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/masked_reconstruction": train_result["masked_reconstruction"],
                "train/contrastive": train_result["contrastive"],
                "val/loss": val_loss,
                "val/masked_reconstruction": val_result["masked_reconstruction"],
                "val/contrastive": val_result["contrastive"],
                **{f"train/masked_mse_{c}": train_result[f"mse_{c}"] for c in channels},
                **{f"val/masked_mse_{c}": val_result[f"mse_{c}"] for c in channels},
                "best/val_loss": best_val,
                "train/num_cycles": len(train_loader.dataset),
                "val/num_cycles": len(val_loader.dataset) if val_loader else 0,
                "model/parameters": n_params,
                "optim/lr": current_lr,
                "checkpoint/is_best": int(is_best),
                "early_stopping/epochs_without_improvement": epochs_without_improvement,
            }
            logger.log(metrics)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"train_masked={train_result['masked_reconstruction']:.6f} "
                f"val_loss={val_loss:.6f} val_masked={val_result['masked_reconstruction']:.6f} "
                f"lr={current_lr:.6g}",
                flush=True,
            )
            state = {
                "best_val": best_val,
                "epochs_without_improvement": epochs_without_improvement,
            }
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
                config=cfg,
                scheduler=scheduler,
                training_state=state,
            )
            if is_best:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"train_loss": train_loss, "val_loss": val_loss},
                    config=cfg,
                    scheduler=scheduler,
                    training_state=state,
                )
            if early_enabled and early_patience > 0 and epochs_without_improvement >= early_patience:
                print(
                    f"Early stopping at epoch={epoch}: no validation improvement for "
                    f"{epochs_without_improvement} epochs.",
                    flush=True,
                )
                break
    finally:
        logger.close()
    print(f"Wrote checkpoints to {output_dir}")


def train_one_epoch(
    model: CycleAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_cfg: dict[str, Any],
    epoch: int,
    channels: tuple[str, ...],
) -> dict[str, float]:
    model.train()
    totals = _empty_totals(channels)
    log_every = int(train_cfg.get("log_every_n_steps", 0))
    running_loss = 0.0
    running_count = 0
    for step, batch in enumerate(dataloader, start=1):
        x_cycle = batch["x_cycle"].to(device, non_blocking=True)
        result = masked_forward(model, x_cycle, channels, train_cfg)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        optimizer.step()
        _update_totals(totals, result, channels)
        running_loss += float(result["loss"].detach().cpu())
        running_count += 1
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == len(dataloader)):
            print(
                f"epoch={epoch} step={step}/{len(dataloader)} "
                f"train_loss_running={running_loss / max(running_count, 1):.6f} "
                f"train_loss_epoch={totals['loss'] / max(totals['count'], 1):.6f}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0
    return _finalize_totals(totals, channels)


@torch.no_grad()
def evaluate(
    model: CycleAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any],
    channels: tuple[str, ...],
) -> dict[str, float]:
    model.eval()
    totals = _empty_totals(channels)
    for batch in dataloader:
        x_cycle = batch["x_cycle"].to(device, non_blocking=True)
        result = masked_forward(model, x_cycle, channels, train_cfg)
        _update_totals(totals, result, channels)
    return _finalize_totals(totals, channels)


def masked_forward(
    model: CycleAutoencoder,
    x_cycle: Tensor,
    channels: tuple[str, ...],
    train_cfg: dict[str, Any],
) -> dict[str, Tensor]:
    cfg = train_cfg.get("masked_pretraining", {})
    num_views = int(cfg.get("num_views", 3))
    mask_ratio = float(cfg.get("mask_ratio", 0.5))
    block_length = int(cfg.get("block_length", 1))
    contrastive_weight = float(cfg.get("contrastive_weight", 0.1))
    temperature = float(cfg.get("temperature", 0.2))
    if num_views <= 0:
        raise ValueError("masked_pretraining.num_views must be positive.")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("masked_pretraining.mask_ratio must be between 0 and 1.")
    if block_length <= 0:
        raise ValueError("masked_pretraining.block_length must be positive.")

    batch, length, channel_count = x_cycle.shape
    x_views = x_cycle.repeat_interleave(num_views, dim=0)
    mask = make_artificial_mask(
        batch=batch,
        num_views=num_views,
        length=length,
        channels=channel_count,
        mask_ratio=mask_ratio,
        block_length=block_length,
        device=x_cycle.device,
    )
    corrupted = x_views.masked_fill(mask, 0.0)
    outputs = model(corrupted)
    target = x_cycle.repeat_interleave(num_views, dim=0)
    reconstruction = masked_reconstruction_loss(outputs["x_recon"], target, mask)
    contrastive = simmtm_contrastive_loss(
        outputs["z"].reshape(batch, num_views, -1).transpose(0, 1),
        temperature=temperature,
    )
    loss = reconstruction + contrastive_weight * contrastive
    channel_mse = masked_channel_mse(outputs["x_recon"], target, mask, channels)
    return {
        "loss": loss,
        "masked_reconstruction": reconstruction.detach(),
        "contrastive": contrastive.detach(),
        "channel_mse": channel_mse.detach(),
    }


def make_artificial_mask(
    batch: int,
    num_views: int,
    length: int,
    channels: int,
    mask_ratio: float,
    block_length: int,
    device: torch.device,
) -> Tensor:
    total = batch * num_views
    if block_length == 1:
        return torch.rand(total, length, channels, device=device) < mask_ratio
    block_count = (length + block_length - 1) // block_length
    block_mask = torch.rand(total, block_count, channels, device=device) < mask_ratio
    return block_mask.repeat_interleave(block_length, dim=1)[:, :length, :]


def masked_reconstruction_loss(x_recon: Tensor, x_true: Tensor, mask: Tensor) -> Tensor:
    valid = mask & torch.isfinite(x_true)
    if not torch.any(valid):
        return x_recon.sum() * 0.0
    return (x_recon[valid] - x_true[valid]).pow(2).mean()


def masked_channel_mse(
    x_recon: Tensor,
    x_true: Tensor,
    mask: Tensor,
    channels: tuple[str, ...],
) -> Tensor:
    values = []
    for index, _channel in enumerate(channels):
        valid = mask[:, :, index] & torch.isfinite(x_true[:, :, index])
        if torch.any(valid):
            values.append((x_recon[:, :, index][valid] - x_true[:, :, index][valid]).pow(2).mean())
        else:
            values.append(x_recon.sum() * 0.0)
    return torch.stack(values)


def simmtm_contrastive_loss(z_views: Tensor, temperature: float = 0.2) -> Tensor:
    """Supervised-contrastive loss where masked views of the same cycle are positives."""

    num_views, batch, dim = z_views.shape
    if num_views < 2 or batch < 2:
        return z_views.sum() * 0.0
    z = torch.nn.functional.normalize(z_views.transpose(0, 1).reshape(batch * num_views, dim), dim=-1)
    labels = torch.arange(batch, device=z.device).repeat_interleave(num_views)
    logits = z @ z.T / max(temperature, 1.0e-6)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, float("-inf"))
    positive = labels[:, None] == labels[None, :]
    positive = positive & ~eye
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    return -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_count).mean()


def _empty_totals(channels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "loss": 0.0,
        "masked_reconstruction": 0.0,
        "contrastive": 0.0,
        "channel_mse": np.zeros((len(channels),), dtype=float),
        "count": 0,
    }


def _update_totals(
    totals: dict[str, Any],
    result: dict[str, Tensor],
    channels: tuple[str, ...],
) -> None:
    totals["loss"] += float(result["loss"].detach().cpu())
    totals["masked_reconstruction"] += float(result["masked_reconstruction"].detach().cpu())
    totals["contrastive"] += float(result["contrastive"].detach().cpu())
    totals["channel_mse"] += result["channel_mse"].detach().cpu().numpy()
    totals["count"] += 1


def _finalize_totals(totals: dict[str, Any], channels: tuple[str, ...]) -> dict[str, float]:
    count = max(int(totals["count"]), 1)
    result = {
        "loss": float(totals["loss"] / count),
        "masked_reconstruction": float(totals["masked_reconstruction"] / count),
        "contrastive": float(totals["contrastive"] / count),
    }
    result.update(
        {
            f"mse_{channel}": float(totals["channel_mse"][index] / count)
            for index, channel in enumerate(channels)
        }
    )
    return result


if __name__ == "__main__":
    main()
