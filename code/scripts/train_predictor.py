from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from battery_soh.data.torch_datasets import LatentCacheStore, LatentWindowDataset, load_split_ids
from battery_soh.models.soh_predictor import SOHProbabilisticPredictor
from battery_soh.training.checkpointing import load_checkpoint, save_checkpoint
from battery_soh.training.trainer import (
    build_predictor_optimizer,
    evaluate_one_epoch,
    optimizer_learning_rates,
    stochastic_rollout_prefix_mode,
    train_one_epoch,
    vbll_posterior_metrics,
)
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device
from battery_soh.utils.logging import build_metric_logger
from battery_soh.utils.runs import resolve_run_output_dir
from battery_soh.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SoH trajectory predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_latent_tf_vbll.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--latent-cache", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
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
    if kind in {"latent_tf_vbll", "vbll_dynamics"}:
        prior_variance = float(model_cfg.get("vbll_prior_variance", 1.0))
        if prior_variance <= 0.0:
            raise ValueError("model.vbll_prior_variance must be positive.")
        return SOHProbabilisticPredictor.vbll(
            **kwargs,
            vbll_prior_std=math.sqrt(prior_variance),
            vbll_init_rho=float(model_cfg.get("vbll_init_rho", -5.0)),
        )
    if kind == "latent_tf_deterministic":
        return SOHProbabilisticPredictor.deterministic(**kwargs)
    raise ValueError(f"Unsupported model kind: {kind}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.processed_dir is not None:
        cfg.setdefault("data", {})["processed_dir"] = str(args.processed_dir)
    if args.split_file is not None:
        cfg.setdefault("data", {})["split_file"] = str(args.split_file)
    if args.latent_cache is not None:
        cfg.setdefault("data", {})["latent_cache"] = str(args.latent_cache)
    if args.run_name is not None:
        cfg.setdefault("experiment", {})["run_name"] = args.run_name
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
    optimizer = build_predictor_optimizer(model, cfg)
    learning_rates = optimizer_learning_rates(optimizer)
    print(
        f"Optimizer learning rates: base={learning_rates['base']:.6g}, "
        f"vbll_rho={learning_rates.get('vbll_rho', learning_rates['base']):.6g}",
        flush=True,
    )
    output_dir = (
        args.resume.expanduser().resolve().parent
        if args.resume is not None
        else resolve_run_output_dir(cfg, "outputs/soh_predictor")
    )
    print(f"Output dir: {output_dir}", flush=True)
    logger = build_metric_logger(cfg, output_dir)
    best_val = float("inf")
    start_epoch = 1
    max_epochs = int(train_cfg.get("max_epochs", 1))
    schedule_cfg = train_cfg.get("scheduled_sampling", {})
    val_sampling_probability = (
        float(schedule_cfg.get("validation_probability", 1.0))
        if bool(schedule_cfg.get("enabled", False))
        else 0.0
    )
    rollout_cfg = train_cfg.get("stochastic_rollout", {})
    validation_rollout_samples = (
        int(rollout_cfg.get("validation_num_samples", 0))
        if bool(rollout_cfg.get("enabled", False))
        else 0
    )
    validation_rollout_horizon = (
        int(rollout_cfg.get("validation_horizon", rollout_cfg.get("horizon", 0)))
        if validation_rollout_samples > 0
        else 0
    )
    validation_seed = int(rollout_cfg.get("validation_seed", 42))
    validation_prefix_mode = stochastic_rollout_prefix_mode(cfg)
    rollout_negative_weight = float(rollout_cfg.get("negative_weight", 0.0))
    if args.resume is not None:
        resume_path = args.resume.expanduser()
        payload = load_checkpoint(resume_path, model, optimizer=optimizer)
        start_epoch = int(payload.get("epoch") or 0) + 1
        best_val = _infer_resume_best_value(
            output_dir=output_dir,
            fallback_payload=payload,
            use_stochastic_selection=validation_rollout_samples > 0,
            negative_weight=rollout_negative_weight,
        )
        print(
            f"Resumed checkpoint: {resume_path}; start_epoch={start_epoch}; "
            f"best_selection={best_val:.6f}",
            flush=True,
        )
    if start_epoch > max_epochs:
        raise ValueError(
            f"Resume checkpoint starts at epoch {start_epoch}, "
            f"but training.max_epochs={max_epochs}."
        )
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            train_result = train_one_epoch(model, train_loader, optimizer, cfg, device, epoch=epoch)
            val_result = (
                evaluate_one_epoch(model, val_loader, cfg, device)
                if val_loader
                else train_result
            )
            val_rollout_result = (
                _evaluate_rollout_validation(
                    model,
                    val_loader,
                    cfg,
                    device,
                    probability=val_sampling_probability,
                    stochastic_num_samples=validation_rollout_samples,
                    stochastic_horizon=validation_rollout_horizon,
                    stochastic_prefix_mode=validation_prefix_mode,
                    seed=validation_seed,
                )
                if val_loader
                and (
                    val_sampling_probability > 0.0
                    or validation_rollout_samples > 0
                )
                else val_result
            )
            selection_value = (
                val_rollout_result.metrics["stochastic_rollout"]
                + rollout_negative_weight
                * val_rollout_result.metrics["stochastic_rollout_negative"]
                if validation_rollout_samples > 0
                else val_rollout_result.loss
            )
            is_best = selection_value <= best_val
            if is_best:
                best_val = selection_value
            posterior_metrics = vbll_posterior_metrics(model)
            learning_rates = optimizer_learning_rates(optimizer)
            metrics = {
                "epoch": epoch,
                **_prefix_metrics("train", train_result.metrics),
                **_prefix_metrics("val", val_result.metrics),
                **_prefix_metrics("val_rollout", val_rollout_result.metrics),
                **_prefix_metrics("vbll", posterior_metrics),
                "vbll/rho_grad_norm": train_result.metrics.get(
                    "vbll_rho_grad_norm",
                    0.0,
                ),
                "best/selection_metric": best_val,
                "train/num_windows": len(train_loader.dataset),
                "val/num_windows": len(val_loader.dataset) if val_loader is not None else 0,
                "model/parameters": n_params,
                "optim/lr": learning_rates["base"],
                "optim/vbll_rho_lr": learning_rates.get(
                    "vbll_rho",
                    learning_rates["base"],
                ),
                "checkpoint/is_best": int(is_best),
            }
            logger.log(metrics)
            print(
                f"epoch={epoch} train_loss={train_result.loss:.6f} "
                f"val_loss={val_result.loss:.6f} "
                f"val_rollout_loss={val_rollout_result.loss:.6f} "
                f"val_rollout_crps="
                f"{val_rollout_result.metrics.get('stochastic_rollout', 0.0):.6f} "
                f"val_rollout_negative="
                f"{val_rollout_result.metrics.get('stochastic_rollout_negative', 0.0):.6f} "
                f"rho_grad={train_result.metrics.get('vbll_rho_grad_norm', 0.0):.6f} "
                f"weight_std={posterior_metrics.get('weight_std_mean', 0.0):.6f}"
            )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train_loss": train_result.loss,
                    "val_loss": val_result.loss,
                    "val_rollout_loss": val_rollout_result.loss,
                    "val_stochastic_crps": val_rollout_result.metrics.get(
                        "stochastic_rollout",
                        0.0,
                    ),
                    "val_stochastic_negative": val_rollout_result.metrics.get(
                        "stochastic_rollout_negative",
                        0.0,
                    ),
                },
                config=cfg,
            )
            if is_best:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={
                        "train_loss": train_result.loss,
                        "val_loss": val_result.loss,
                        "val_rollout_loss": val_rollout_result.loss,
                        "val_stochastic_crps": val_rollout_result.metrics.get(
                            "stochastic_rollout",
                            0.0,
                        ),
                        "val_stochastic_negative": val_rollout_result.metrics.get(
                            "stochastic_rollout_negative",
                            0.0,
                        ),
                    },
                    config=cfg,
                )
    finally:
        logger.close()
    print(f"Wrote checkpoints to {output_dir}")


def _infer_resume_best_value(
    output_dir: Path,
    fallback_payload: dict,
    use_stochastic_selection: bool,
    negative_weight: float,
) -> float:
    best_path = output_dir / "best.pt"
    payload = fallback_payload
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu")
    metrics = payload.get("metrics", {})
    if use_stochastic_selection and "val_stochastic_crps" in metrics:
        return float(metrics["val_stochastic_crps"]) + negative_weight * float(
            metrics.get("val_stochastic_negative", 0.0)
        )
    if "val_loss" in metrics:
        return float(metrics["val_loss"])
    if "val_rollout_loss" in metrics:
        return float(metrics["val_rollout_loss"])
    return float("inf")


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
    loader_kwargs = _loader_kwargs(train_cfg)
    print(
        f"DataLoader: train_windows={len(train_dataset)}, "
        f"val_windows={len(val_dataset) if val_dataset is not None else 0}, "
        f"batch_size={batch_size}, num_workers={loader_kwargs['num_workers']}, "
        f"pin_memory={loader_kwargs['pin_memory']}",
        flush=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    return train_loader, val_loader


def _forward_with_sampling(
    model: SOHProbabilisticPredictor,
    batch: dict,
    probability: float,
    stochastic_num_samples: int = 0,
    stochastic_horizon: int = 0,
    stochastic_prefix_mode: str = "generated",
) -> dict:
    return model(
        z_window=batch["z_window"],
        soh_history=batch["soh_history"],
        soh_future=batch.get("soh_future"),
        u_window=batch.get("u_window"),
        history_mask=batch.get("history_mask"),
        scheduled_sampling_probability=probability,
        stochastic_rollout_num_samples=stochastic_num_samples,
        stochastic_rollout_horizon=stochastic_horizon,
        stochastic_rollout_prefix_mode=stochastic_prefix_mode,
    )


def _evaluate_rollout_validation(
    model: SOHProbabilisticPredictor,
    dataloader: DataLoader,
    cfg: dict,
    device: torch.device,
    probability: float,
    stochastic_num_samples: int,
    stochastic_horizon: int,
    stochastic_prefix_mode: str,
    seed: int,
):
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        return evaluate_one_epoch(
            model,
            dataloader,
            cfg,
            device,
            forward_fn=lambda current_model, batch: _forward_with_sampling(
                current_model,
                batch,
                probability=probability,
                stochastic_num_samples=stochastic_num_samples,
                stochastic_horizon=stochastic_horizon,
                stochastic_prefix_mode=stochastic_prefix_mode,
            ),
        )


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


def _filter_cached_ids(cell_ids: list[str], cache: LatentCacheStore) -> list[str]:
    cached = set(cache.cell_ids_unique())
    return [cell_id for cell_id in cell_ids if cell_id in cached]


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


if __name__ == "__main__":
    main()
