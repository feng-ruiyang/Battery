from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.torch_datasets import (
    CycleTensorDataset,
    collate_cycle_batch,
    resolve_cell_ids,
    save_latent_cache,
)
from battery_soh.models.encoder_decoder import CycleAutoencoder, CycleDecoder, CycleEncoder
from battery_soh.training.checkpointing import load_checkpoint
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache cycle latent vectors from a trained encoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override config data.processed_dir for zero-shot external data.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--channel-fill",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fill a wholly missing cycle channel before encoding; repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    device = resolve_device(train_cfg.get("device", "auto"))
    model = build_autoencoder(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)
    print(f"Device: {device}", flush=True)

    processed_dir = args.processed_dir or Path(data_cfg["processed_dir"])
    store = CanonicalCycleStore(processed_dir)
    cell_ids = resolve_cell_ids(store)
    channel_fill_values = {
        str(name): float(value)
        for name, value in data_cfg.get("channel_fill_values", {}).items()
    }
    channel_fill_values.update(_parse_channel_fills(args.channel_fill))
    dataset = CycleTensorDataset(
        store=store,
        cell_ids=cell_ids,
        channels=tuple(data_cfg.get("channels", ("V", "I", "T"))),
        sequence_length=int(data_cfg.get("sequence_length", 1000)),
        channel_fill_values=channel_fill_values,
    )
    if len(dataset) == 0:
        raise SystemExit("No cycles found to encode. Run preprocessing first and check processed_dir.")
    loader_kwargs = _loader_kwargs(train_cfg)
    batch_size = int(train_cfg.get("batch_size", 128))
    print(
        f"Latent cache DataLoader: cycles={len(dataset)}, batch_size={batch_size}, "
        f"num_workers={loader_kwargs['num_workers']}, pin_memory={loader_kwargs['pin_memory']}, "
        f"channel_fill_values={channel_fill_values}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cycle_batch,
        **loader_kwargs,
    )

    z_batches: list[np.ndarray] = []
    metadata: dict[str, list] = {
        "cell_id": [],
        "dataset_id": [],
        "batch_id": [],
        "cycle_index": [],
        "soh": [],
    }
    summaries: list[np.ndarray] = []
    summary_names = ("q_discharge", "internal_resistance", "t_avg")
    log_every = int(train_cfg.get("log_every_n_steps", 0))
    encoded = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            x_cycle = batch["x_cycle"].to(device, non_blocking=True)
            z = model.encoder(x_cycle)
            z_batches.append(z.cpu().numpy())
            metadata["cell_id"].extend(batch["cell_id"])
            metadata["dataset_id"].extend(batch["dataset_id"])
            metadata["batch_id"].extend([value or "" for value in batch["batch_id"]])
            metadata["cycle_index"].extend(batch["cycle_index"].cpu().numpy().tolist())
            metadata["soh"].extend(batch["soh"].cpu().numpy().tolist())
            summary_values = [batch["summaries"][name].cpu().numpy() for name in summary_names]
            summaries.append(np.stack(summary_values, axis=1))
            encoded += int(x_cycle.shape[0])
            if log_every > 0 and (step == 1 or step % log_every == 0 or step == len(loader)):
                print(
                    f"cache step={step}/{len(loader)} encoded={encoded}/{len(dataset)} cycles",
                    flush=True,
                )

    if args.output is not None:
        output = args.output
    elif args.processed_dir is not None:
        output = processed_dir / "latents.npz"
    else:
        output = Path(data_cfg.get("latent_cache", processed_dir / "latents.npz"))
    save_latent_cache(
        output,
        z=np.concatenate(z_batches, axis=0),
        metadata={key: np.asarray(value) for key, value in metadata.items()},
        summary_names=summary_names,
        summaries=np.concatenate(summaries, axis=0),
    )
    print(f"Wrote latent cache to {output}")


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


def _parse_channel_fills(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected --channel-fill NAME=VALUE, got {item!r}.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Channel name must not be empty: {item!r}.")
        result[name] = float(raw_value)
    return result


def build_autoencoder(cfg: dict) -> CycleAutoencoder:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
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
    return CycleAutoencoder(encoder, decoder)


if __name__ == "__main__":
    main()
