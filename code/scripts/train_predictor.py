from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from battery_soh.data.torch_datasets import LatentCacheStore, LatentWindowDataset, load_split_ids
from battery_soh.models.soh_predictor import SOHProbabilisticPredictor
from battery_soh.training.checkpointing import save_checkpoint
from battery_soh.training.trainer import evaluate_one_epoch, train_one_epoch
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device
from battery_soh.utils.logging import build_metric_logger
from battery_soh.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SoH trajectory predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_latent_tf_vbll.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_model(cfg: dict) -> SOHProbabilisticPredictor:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    kind = model_cfg.get("kind", "latent_tf_vbll")
    summary_channels = tuple(data_cfg.get("summary_channels", ()))
    kwargs = {
        "latent_dim": int(data_cfg["latent_dim"]),
        "aux_dim": len(summary_channels) if summary_channels else int(data_cfg.get("aux_dim", 0)),
        "horizon": int(data_cfg["horizon"]),
        "history_dim": int(model_cfg.get("history_dim", 256)),
        "decoder_dim": int(model_cfg["decoder_dim"]) if "decoder_dim" in model_cfg else None,
    }
    if kind == "latent_tf_vbll":
        return SOHProbabilisticPredictor.vbll(**kwargs)
    if kind == "vbll_dynamics":
        return SOHProbabilisticPredictor.vbll(**kwargs)
    raise ValueError(f"Unsupported model kind: {kind}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model'].get('kind', 'latent_tf_vbll')}")
    print(f"Device: {device}")
    print(f"Parameters: {n_params:,}")
    if args.dry_run:
        batch_size = 2
        window = int(cfg["data"]["window"])
        latent_dim = int(cfg["data"]["latent_dim"])
        summary_channels = tuple(cfg["data"].get("summary_channels", ()))
        aux_dim = len(summary_channels) if summary_channels else int(cfg["data"].get("aux_dim", 0))
        z_window = torch.randn(batch_size, window, latent_dim, device=device)
        u_window = torch.randn(batch_size, window, aux_dim, device=device) if aux_dim else None
        soh_history = torch.linspace(1.0, 0.95, steps=window, device=device).repeat(batch_size, 1)
        soh_future = torch.linspace(0.94, 0.85, steps=int(cfg["data"]["horizon"]), device=device).repeat(
            batch_size, 1
        )
        outputs = model(
            z_window=z_window,
            soh_history=soh_history,
            soh_future=soh_future,
            u_window=u_window,
            num_samples=3,
        )
        print("Dry run outputs:")
        for key in ("next_z_pred", "state", "y_mean", "y_var", "y_cov", "trajectory_samples"):
            value = outputs[key]
            print(f"  {key}: {tuple(value.shape)}")
        return
    train_loader, val_loader = build_dataloaders(cfg)
    if len(train_loader.dataset) == 0:
        raise SystemExit("No latent training windows found. Build latent_cache and check window/horizon.")
    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/soh_predictor"))
    logger = build_metric_logger(cfg, output_dir)
    best_val = float("inf")
    max_epochs = int(train_cfg.get("max_epochs", 1))
    try:
        for epoch in range(1, max_epochs + 1):
            train_result = train_one_epoch(model, train_loader, optimizer, cfg, device)
            val_result = evaluate_one_epoch(model, val_loader, cfg, device) if val_loader else train_result
            is_best = val_result.loss <= best_val
            if is_best:
                best_val = val_result.loss
            metrics = {
                "epoch": epoch,
                **_prefix_metrics("train", train_result.metrics),
                **_prefix_metrics("val", val_result.metrics),
                "best/val_loss": best_val,
                "train/num_windows": len(train_loader.dataset),
                "val/num_windows": len(val_loader.dataset) if val_loader is not None else 0,
                "model/parameters": n_params,
                "optim/lr": float(train_cfg.get("lr", 1.0e-4)),
                "checkpoint/is_best": int(is_best),
            }
            logger.log(metrics)
            print(
                f"epoch={epoch} train_loss={train_result.loss:.6f} "
                f"val_loss={val_result.loss:.6f}"
            )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": train_result.loss, "val_loss": val_result.loss},
                config=cfg,
            )
            if is_best:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"train_loss": train_result.loss, "val_loss": val_result.loss},
                    config=cfg,
                )
    finally:
        logger.close()
    print(f"Wrote checkpoints to {output_dir}")


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    latent_cache = Path(data_cfg.get("latent_cache", Path(data_cfg["processed_dir"]) / "latents.npz"))
    cache = LatentCacheStore(latent_cache)
    if cache.z.shape[1] != int(data_cfg["latent_dim"]):
        raise ValueError(
            f"latent_cache has latent_dim={cache.z.shape[1]}, "
            f"but config data.latent_dim={data_cfg['latent_dim']}."
        )
    split_file = data_cfg.get("split_file")
    if split_file:
        train_ids = _filter_cached_ids(load_split_ids(split_file, "train") or [], cache)
        val_ids = _filter_cached_ids(load_split_ids(split_file, "val") or [], cache)
    else:
        train_ids = cache.cell_ids_unique()
        val_ids = []
    summary_channels = tuple(data_cfg.get("summary_channels", ()))
    aux_dim = len(summary_channels) if summary_channels else int(data_cfg.get("aux_dim", 0))
    if not summary_channels and aux_dim:
        summary_channels = cache.summary_names[:aux_dim]
    if len(summary_channels) != aux_dim:
        raise ValueError(
            f"aux_dim={aux_dim} but summary_channels has {len(summary_channels)} entries: "
            f"{summary_channels}"
        )
    train_dataset = LatentWindowDataset(
        cache=cache,
        cell_ids=train_ids,
        window=int(data_cfg["window"]),
        horizon=int(data_cfg["horizon"]),
        summary_channels=summary_channels,
    )
    val_dataset = (
        LatentWindowDataset(
            cache=cache,
            cell_ids=val_ids,
            window=int(data_cfg["window"]),
            horizon=int(data_cfg["horizon"]),
            summary_channels=summary_channels,
        )
        if val_ids
        else None
    )
    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    return train_loader, val_loader


def _filter_cached_ids(cell_ids: list[str], cache: LatentCacheStore) -> list[str]:
    cached = set(cache.cell_ids_unique())
    return [cell_id for cell_id in cell_ids if cell_id in cached]


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


if __name__ == "__main__":
    main()
