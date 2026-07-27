from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from battery_soh.data.torch_datasets import LatentCacheStore, LatentWindowDataset, load_split_ids
from battery_soh.evaluation.metrics import interval_coverage, point_metrics
from battery_soh.utils.device import resolve_device
from train_predictor import build_model


DEFAULT_HORIZONS = (1, 5, 10, 20, 50, 80)
DEFAULT_LEVELS = (0.5, 0.8, 0.9, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate free-running SoH trajectory rollouts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--latent-cache", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload.get("config")
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Checkpoint does not contain a training config: {args.checkpoint}")
    data_cfg = cfg["data"]
    latent_cache_path = args.latent_cache or Path(data_cfg["latent_cache"])
    split_file = args.split_file or Path(data_cfg["split_file"])
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))

    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    model_kind = str(cfg.get("model", {}).get("kind", "latent_tf_vbll"))
    probabilistic = model.decoder.head_kind == "vbll"
    effective_samples = args.num_samples if probabilistic else 1

    cache = LatentCacheStore(latent_cache_path)
    split_ids = load_split_ids(split_file, args.split) or []
    cached_ids = set(cache.cell_ids_unique())
    cell_ids = [cell_id for cell_id in split_ids if cell_id in cached_ids]
    summary_channels = tuple(data_cfg.get("summary_channels", ()))
    aux_dim = len(summary_channels) if summary_channels else int(data_cfg.get("aux_dim", 0))
    if not summary_channels and aux_dim:
        summary_channels = cache.summary_names[:aux_dim]
    dataset: Any = LatentWindowDataset(
        cache=cache,
        cell_ids=cell_ids,
        window=int(data_cfg["window"]),
        horizon=int(data_cfg["horizon"]),
        summary_channels=summary_channels,
        target="soh",
    )
    if args.max_windows is not None and args.max_windows < len(dataset):
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(len(dataset), size=args.max_windows, replace=False))
        dataset = Subset(dataset, indices.tolist())
    if len(dataset) == 0:
        raise SystemExit(f"No rollout windows found for split={args.split!r}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Checkpoint: {args.checkpoint}\n"
        f"Model: {model_kind}\n"
        f"Device: {device}\n"
        f"Split: {args.split}; windows={len(dataset)}; cells={len(cell_ids)}\n"
        f"Rollout samples: requested={args.num_samples}; effective={effective_samples}",
        flush=True,
    )

    collected = _collect_rollouts(
        model=model,
        loader=loader,
        device=device,
        num_samples=effective_samples,
    )
    metrics, summaries = _summarize_rollouts(
        samples=collected["samples"],
        targets=collected["targets"],
        mask=collected["mask"],
        cell_ids=collected["cell_ids"],
        probabilistic=probabilistic,
    )

    output_dir = args.output_dir or args.checkpoint.parent / (
        f"rollout_{args.split}_samples_{effective_samples}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "model_kind": model_kind,
        "split": args.split,
        "num_windows": len(dataset),
        "num_cells": len(set(collected["cell_ids"].tolist())),
        "num_samples": effective_samples,
        "metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)

    arrays = {
        "cell_id": collected["cell_ids"],
        "forecast_cycle_index": collected["forecast_cycle_index"],
        "target": collected["targets"],
        "mask": collected["mask"],
        **summaries,
    }
    if args.save_samples:
        arrays["samples"] = collected["samples"]
    np.savez_compressed(output_dir / "predictions.npz", **arrays)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"Wrote rollout evaluation to {output_dir}", flush=True)


@torch.inference_mode()
def _collect_rollouts(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
) -> dict[str, np.ndarray]:
    sample_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    mask_batches: list[np.ndarray] = []
    cell_ids: list[str] = []
    forecast_cycle_indices: list[int] = []
    log_every = min(25, max(1, len(loader) // 20))
    for step, batch in enumerate(loader, start=1):
        z_window = batch["z_window"].to(device, non_blocking=True)
        soh_history = batch["soh_history"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        u_window = batch.get("u_window")
        if u_window is not None:
            u_window = u_window.to(device, non_blocking=True)
        state_outputs = model.state_model(
            z_window=z_window,
            u_window=u_window,
            mask=history_mask,
        )
        samples = model.decoder.sample(
            state=state_outputs["state"],
            start_soh=soh_history[:, -1],
            num_samples=num_samples,
            use_mean=model.decoder.head_kind == "deterministic",
        )
        sample_batches.append(samples.cpu().numpy().astype(np.float32))
        target_batches.append(batch["soh_future"].numpy().astype(np.float32))
        mask_batches.append(batch["mask"].numpy().astype(bool))
        cell_ids.extend(str(value) for value in batch["cell_id"])
        forecast_cycle_indices.extend(
            int(value) for value in batch["forecast_cycle_index"].numpy().tolist()
        )
        if step == 1 or step % log_every == 0 or step == len(loader):
            print(f"rollout batch={step}/{len(loader)}", flush=True)

    return {
        "samples": np.concatenate(sample_batches, axis=1),
        "targets": np.concatenate(target_batches, axis=0),
        "mask": np.concatenate(mask_batches, axis=0),
        "cell_ids": np.asarray(cell_ids, dtype=str),
        "forecast_cycle_index": np.asarray(forecast_cycle_indices, dtype=np.int64),
    }


def _summarize_rollouts(
    samples: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    cell_ids: np.ndarray,
    probabilistic: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prediction_mean = samples.mean(axis=0)
    prediction_std = samples.std(axis=0)
    valid_targets = np.asarray(mask, dtype=bool) & np.isfinite(targets)
    metrics: dict[str, Any] = {
        "overall": point_metrics(targets, prediction_mean, mask),
        "macro_cell": _macro_cell_metrics(targets, prediction_mean, mask, cell_ids),
        "by_horizon": {},
        "stability": {
            "prediction_nonfinite_fraction": _masked_fraction(
                ~np.isfinite(prediction_mean),
                valid_targets,
            ),
            "sample_nonfinite_fraction": _sample_masked_fraction(
                ~np.isfinite(samples),
                valid_targets,
            ),
            "prediction_out_of_range_fraction": _masked_fraction(
                (prediction_mean < 0.0) | (prediction_mean > 1.2),
                valid_targets,
            ),
            "monotonic_violation_rate": _monotonic_violation_rate(
                prediction_mean,
                valid_targets,
            ),
        },
    }
    horizons = [value for value in DEFAULT_HORIZONS if value <= targets.shape[1]]
    for horizon in horizons:
        index = horizon - 1
        metrics["by_horizon"][str(horizon)] = point_metrics(
            targets[:, index],
            prediction_mean[:, index],
            mask[:, index],
        )

    summaries: dict[str, np.ndarray] = {
        "prediction_mean": prediction_mean.astype(np.float32),
        "prediction_std": prediction_std.astype(np.float32),
    }
    if probabilistic and samples.shape[0] > 1:
        crps = _empirical_crps(samples, targets)
        metrics["crps"] = _masked_mean(crps, mask)
        metrics["intervals"] = {}
        for level in DEFAULT_LEVELS:
            tail = (1.0 - level) / 2.0
            lower = np.quantile(samples, tail, axis=0)
            upper = np.quantile(samples, 1.0 - tail, axis=0)
            key = str(int(level * 100))
            summaries[f"lower_{key}"] = lower.astype(np.float32)
            summaries[f"upper_{key}"] = upper.astype(np.float32)
            metrics["intervals"][key] = {
                "coverage": interval_coverage(targets, lower, upper, mask),
                "mean_width": _masked_mean(upper - lower, mask),
            }
            for horizon in horizons:
                index = horizon - 1
                metrics["by_horizon"][str(horizon)].setdefault("intervals", {})[key] = {
                    "coverage": interval_coverage(
                        targets[:, index],
                        lower[:, index],
                        upper[:, index],
                        mask[:, index],
                    ),
                    "mean_width": _masked_mean(
                        upper[:, index] - lower[:, index],
                        mask[:, index],
                    ),
                }
        for horizon in horizons:
            index = horizon - 1
            metrics["by_horizon"][str(horizon)]["crps"] = _masked_mean(
                crps[:, index],
                mask[:, index],
            )
            metrics["by_horizon"][str(horizon)]["prediction_std_mean"] = _masked_mean(
                prediction_std[:, index],
                mask[:, index],
            )
    return metrics, summaries


def _macro_cell_metrics(
    targets: np.ndarray,
    prediction_mean: np.ndarray,
    mask: np.ndarray,
    cell_ids: np.ndarray,
) -> dict[str, float]:
    per_cell = []
    for cell_id in sorted(set(cell_ids.tolist())):
        selected = cell_ids == cell_id
        per_cell.append(point_metrics(targets[selected], prediction_mean[selected], mask[selected]))
    return {
        "mae": float(np.mean([row["mae"] for row in per_cell])),
        "rmse": float(np.mean([row["rmse"] for row in per_cell])),
    }


def _empirical_crps(samples: np.ndarray, targets: np.ndarray) -> np.ndarray:
    sample_count = samples.shape[0]
    first_term = np.mean(np.abs(samples - targets[None, ...]), axis=0)
    ordered = np.sort(samples, axis=0)
    coefficients = 2 * np.arange(1, sample_count + 1) - sample_count - 1
    second_term = (
        2.0
        * np.sum(
            ordered * coefficients.reshape(sample_count, 1, 1),
            axis=0,
        )
        / float(sample_count**2)
    )
    return first_term - 0.5 * second_term


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(values[valid]))


def _masked_fraction(events: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.asarray(events, dtype=bool)[valid]))


def _sample_masked_fraction(events: np.ndarray, mask: np.ndarray) -> float:
    expanded_mask = np.broadcast_to(mask[None, ...], events.shape)
    return _masked_fraction(events, expanded_mask)


def _monotonic_violation_rate(prediction: np.ndarray, mask: np.ndarray) -> float:
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    violations = prediction[:, 1:] > prediction[:, :-1]
    return _masked_fraction(violations, valid_pairs)


if __name__ == "__main__":
    main()
