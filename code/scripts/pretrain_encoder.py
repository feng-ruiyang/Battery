from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.normalization import load_channel_normalizer
from battery_soh.data.torch_datasets import CycleTensorDataset, collate_cycle_batch, resolve_cell_ids
from battery_soh.models.encoder_decoder import CycleAutoencoder, CycleDecoder, CycleEncoder
from battery_soh.training.checkpointing import load_checkpoint, save_checkpoint
from battery_soh.training.losses import reconstruction_loss
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device
from battery_soh.utils.logging import build_metric_logger
from battery_soh.utils.runs import resolve_run_output_dir
from battery_soh.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain cycle encoder-decoder.")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_encoder.yaml"))
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
    channels = tuple(data_cfg.get("channels", ("V", "I", "T")))
    if int(model_cfg["input_dim"]) != len(channels):
        raise ValueError(
            f"model.input_dim={model_cfg['input_dim']} must match number of data.channels={len(channels)}."
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
    autoencoder = CycleAutoencoder(encoder, decoder)
    autoencoder.to(device)
    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"CycleAutoencoder parameters: {n_params:,}")
    print(f"Device: {device}")
    if not args.dry_run:
        train_loader, val_loader = build_dataloaders(cfg)
        if len(train_loader.dataset) == 0:
            raise SystemExit("No training cycles found. Run preprocessing first and check processed_dir.")
        optimizer = torch.optim.AdamW(
            autoencoder.parameters(),
            lr=float(train_cfg.get("lr", 1.0e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        scheduler = _build_scheduler(optimizer, train_cfg)
        output_dir = (
            args.resume.expanduser().resolve().parent
            if args.resume is not None
            else resolve_run_output_dir(cfg, "outputs/cycle_autoencoder")
        )
        print(f"Output dir: {output_dir}", flush=True)
        best_val = float("inf")
        epochs_without_improvement = 0
        start_epoch = 1
        if args.resume is not None:
            resume_path = args.resume.expanduser()
            payload = load_checkpoint(
                resume_path,
                autoencoder,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = int(payload.get("epoch") or 0) + 1
            training_state = payload.get("training_state", {})
            best_val = float(
                training_state.get(
                    "best_val",
                    payload.get("metrics", {}).get("val_recon", float("inf")),
                )
            )
            epochs_without_improvement = int(
                training_state.get("epochs_without_improvement", 0)
            )
            resume_lr = train_cfg.get("resume_lr")
            if resume_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = float(resume_lr)
            print(
                f"Resumed checkpoint: {resume_path}; start_epoch={start_epoch}; "
                f"best_val={best_val:.6f}; lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
        logger = build_metric_logger(cfg, output_dir)
        max_epochs = int(train_cfg.get("max_epochs", 1))
        early_stopping_cfg = train_cfg.get("early_stopping", {})
        early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
        early_stopping_patience = int(early_stopping_cfg.get("patience", 0))
        early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
        if start_epoch > max_epochs:
            raise ValueError(
                f"Resume checkpoint starts at epoch {start_epoch}, "
                f"but training.max_epochs={max_epochs}."
            )
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                train_result = train_one_epoch(
                    autoencoder, train_loader, optimizer, device, train_cfg, epoch, channels
                )
                val_result = (
                    evaluate(autoencoder, val_loader, device, channels)
                    if val_loader is not None
                    else train_result
                )
                train_loss = train_result["loss"]
                val_loss = val_result["loss"]
                previous_best = best_val
                is_best = val_loss < previous_best
                meaningful_improvement = (
                    val_loss < previous_best - early_stopping_min_delta
                )
                if is_best:
                    best_val = val_loss
                if meaningful_improvement:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if scheduler is not None:
                    scheduler.step(val_loss)
                current_lr = float(optimizer.param_groups[0]["lr"])
                metrics = {
                    "epoch": epoch,
                    "train/reconstruction_loss": train_loss,
                    "val/reconstruction_loss": val_loss,
                    **{
                        f"train/reconstruction_mse_{channel}": train_result[f"mse_{channel}"]
                        for channel in channels
                    },
                    **{
                        f"val/reconstruction_mse_{channel}": val_result[f"mse_{channel}"]
                        for channel in channels
                    },
                    "best/val_reconstruction_loss": best_val,
                    "train/num_cycles": len(train_loader.dataset),
                    "val/num_cycles": len(val_loader.dataset) if val_loader is not None else 0,
                    "model/parameters": n_params,
                    "optim/lr": current_lr,
                    "checkpoint/is_best": int(is_best),
                    "early_stopping/epochs_without_improvement": (
                        epochs_without_improvement
                    ),
                }
                logger.log(metrics)
                print(
                    f"epoch={epoch} train_recon={train_loss:.6f} "
                    f"val_recon={val_loss:.6f} lr={current_lr:.6g}"
                )
                checkpoint_state = {
                    "best_val": best_val,
                    "epochs_without_improvement": epochs_without_improvement,
                }
                save_checkpoint(
                    output_dir / "last.pt",
                    autoencoder,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"train_recon": train_loss, "val_recon": val_loss},
                    config=cfg,
                    scheduler=scheduler,
                    training_state=checkpoint_state,
                )
                if is_best:
                    save_checkpoint(
                        output_dir / "best.pt",
                        autoencoder,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={"train_recon": train_loss, "val_recon": val_loss},
                        config=cfg,
                        scheduler=scheduler,
                        training_state=checkpoint_state,
                    )
                if (
                    early_stopping_enabled
                    and early_stopping_patience > 0
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    print(
                        f"Early stopping at epoch={epoch}: no validation improvement "
                        f"for {epochs_without_improvement} epochs.",
                        flush=True,
                    )
                    break
        finally:
            logger.close()
        print(f"Wrote checkpoints to {output_dir}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_cfg = train_cfg.get("scheduler", {})
    if not bool(scheduler_cfg.get("enabled", False)):
        return None
    scheduler_name = str(scheduler_cfg.get("name", "reduce_on_plateau"))
    if scheduler_name != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler: {scheduler_name!r}.")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 4)),
        threshold=float(scheduler_cfg.get("threshold", 1.0e-4)),
        threshold_mode="abs",
        min_lr=float(scheduler_cfg.get("min_lr", 1.0e-6)),
    )


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    store = CanonicalCycleStore(data_cfg["processed_dir"])
    split_file = data_cfg.get("split_file")
    train_ids = resolve_cell_ids(store, split_file, "train") if split_file else resolve_cell_ids(store)
    val_ids = resolve_cell_ids(store, split_file, "val") if split_file else []
    channels = tuple(data_cfg.get("channels", ("V", "I", "T")))
    sequence_length = int(data_cfg.get("sequence_length", 1000))
    normalizer = load_channel_normalizer(data_cfg.get("normalizer"))
    if normalizer is not None:
        print(
            f"Channel normalizer: {Path(data_cfg['normalizer']).expanduser()} "
            f"mean={normalizer.mean.tolist()} std={normalizer.std.tolist()}",
            flush=True,
        )
    train_dataset = CycleTensorDataset(
        store, train_ids, channels, sequence_length, normalizer=normalizer
    )
    val_dataset = (
        CycleTensorDataset(store, val_ids, channels, sequence_length, normalizer=normalizer)
        if val_ids
        else None
    )
    batch_size = int(train_cfg.get("batch_size", 128))
    loader_kwargs = _loader_kwargs(train_cfg)
    print(
        f"DataLoader: train_cycles={len(train_dataset)}, "
        f"val_cycles={len(val_dataset) if val_dataset is not None else 0}, "
        f"batch_size={batch_size}, num_workers={loader_kwargs['num_workers']}, "
        f"pin_memory={loader_kwargs['pin_memory']}",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cycle_batch,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_cycle_batch,
            **loader_kwargs,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    return train_loader, val_loader


def _loader_kwargs(train_cfg: dict) -> dict:
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return kwargs


def train_one_epoch(
    model: CycleAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_cfg: dict,
    epoch: int,
    channels: tuple[str, ...],
) -> dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    log_every = int(train_cfg.get("log_every_n_steps", 0))
    running_total = 0.0
    running_count = 0
    channel_totals = np.zeros((len(channels),), dtype=float)
    for step, batch in enumerate(dataloader, start=1):
        x_cycle = batch["x_cycle"].to(device, non_blocking=True)
        outputs = model(x_cycle)
        loss = reconstruction_loss(outputs["x_recon"], x_cycle)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        channel_mse = (
            (outputs["x_recon"] - x_cycle).pow(2).mean(dim=(0, 1)).detach().cpu().numpy()
        )
        total += loss_value
        count += 1
        running_total += loss_value
        running_count += 1
        channel_totals += channel_mse
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == len(dataloader)):
            running_loss = running_total / max(running_count, 1)
            epoch_loss = total / max(count, 1)
            print(
                f"epoch={epoch} step={step}/{len(dataloader)} "
                f"train_recon_running={running_loss:.6f} train_recon_epoch={epoch_loss:.6f}",
                flush=True,
            )
            running_total = 0.0
            running_count = 0
    result = {"loss": total / max(count, 1)}
    result.update(
        {f"mse_{channel}": float(channel_totals[index] / max(count, 1)) for index, channel in enumerate(channels)}
    )
    return result


@torch.no_grad()
def evaluate(
    model: CycleAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
    channels: tuple[str, ...],
) -> dict[str, float]:
    model.eval()
    total = 0.0
    count = 0
    channel_totals = np.zeros((len(channels),), dtype=float)
    for batch in dataloader:
        x_cycle = batch["x_cycle"].to(device, non_blocking=True)
        outputs = model(x_cycle)
        loss = reconstruction_loss(outputs["x_recon"], x_cycle)
        total += float(loss.detach().cpu())
        channel_totals += (
            (outputs["x_recon"] - x_cycle).pow(2).mean(dim=(0, 1)).detach().cpu().numpy()
        )
        count += 1
    result = {"loss": total / max(count, 1)}
    result.update(
        {f"mse_{channel}": float(channel_totals[index] / max(count, 1)) for index, channel in enumerate(channels)}
    )
    return result


if __name__ == "__main__":
    main()
