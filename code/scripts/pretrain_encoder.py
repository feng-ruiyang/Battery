from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.torch_datasets import CycleTensorDataset, collate_cycle_batch, resolve_cell_ids
from battery_soh.models.encoder_decoder import CycleAutoencoder, CycleDecoder, CycleEncoder
from battery_soh.training.checkpointing import save_checkpoint
from battery_soh.training.losses import reconstruction_loss
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device
from battery_soh.utils.logging import build_metric_logger
from battery_soh.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain cycle encoder-decoder.")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_encoder.yaml"))
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
        output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/cycle_autoencoder"))
        logger = build_metric_logger(cfg, output_dir)
        best_val = float("inf")
        max_epochs = int(train_cfg.get("max_epochs", 1))
        try:
            for epoch in range(1, max_epochs + 1):
                train_loss = train_one_epoch(autoencoder, train_loader, optimizer, device)
                val_loss = evaluate(autoencoder, val_loader, device) if val_loader is not None else train_loss
                is_best = val_loss <= best_val
                if is_best:
                    best_val = val_loss
                metrics = {
                    "epoch": epoch,
                    "train/reconstruction_loss": train_loss,
                    "val/reconstruction_loss": val_loss,
                    "best/val_reconstruction_loss": best_val,
                    "train/num_cycles": len(train_loader.dataset),
                    "val/num_cycles": len(val_loader.dataset) if val_loader is not None else 0,
                    "model/parameters": n_params,
                    "optim/lr": float(train_cfg.get("lr", 1.0e-4)),
                    "checkpoint/is_best": int(is_best),
                }
                logger.log(metrics)
                print(f"epoch={epoch} train_recon={train_loss:.6f} val_recon={val_loss:.6f}")
                save_checkpoint(
                    output_dir / "last.pt",
                    autoencoder,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"train_recon": train_loss, "val_recon": val_loss},
                    config=cfg,
                )
                if is_best:
                    save_checkpoint(
                        output_dir / "best.pt",
                        autoencoder,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={"train_recon": train_loss, "val_recon": val_loss},
                        config=cfg,
                    )
        finally:
            logger.close()
        print(f"Wrote checkpoints to {output_dir}")


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    store = CanonicalCycleStore(data_cfg["processed_dir"])
    split_file = data_cfg.get("split_file")
    train_ids = resolve_cell_ids(store, split_file, "train") if split_file else resolve_cell_ids(store)
    val_ids = resolve_cell_ids(store, split_file, "val") if split_file else []
    channels = tuple(data_cfg.get("channels", ("V", "I", "T")))
    sequence_length = int(data_cfg.get("sequence_length", 1000))
    train_dataset = CycleTensorDataset(store, train_ids, channels, sequence_length)
    val_dataset = CycleTensorDataset(store, val_ids, channels, sequence_length) if val_ids else None
    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cycle_batch,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_cycle_batch)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    return train_loader, val_loader


def train_one_epoch(
    model: CycleAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in dataloader:
        x_cycle = batch["x_cycle"].to(device)
        outputs = model(x_cycle)
        loss = reconstruction_loss(outputs["x_recon"], x_cycle)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: CycleAutoencoder, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in dataloader:
        x_cycle = batch["x_cycle"].to(device)
        outputs = model(x_cycle)
        loss = reconstruction_loss(outputs["x_recon"], x_cycle)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


if __name__ == "__main__":
    main()
