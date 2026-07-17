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
    parser.add_argument("--output", type=Path, default=None)
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

    store = CanonicalCycleStore(data_cfg["processed_dir"])
    cell_ids = resolve_cell_ids(store)
    dataset = CycleTensorDataset(
        store=store,
        cell_ids=cell_ids,
        channels=tuple(data_cfg.get("channels", ("V", "I", "T"))),
        sequence_length=int(data_cfg.get("sequence_length", 1000)),
    )
    if len(dataset) == 0:
        raise SystemExit("No cycles found to encode. Run preprocessing first and check processed_dir.")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=False,
        collate_fn=collate_cycle_batch,
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
    with torch.no_grad():
        for batch in loader:
            x_cycle = batch["x_cycle"].to(device)
            z = model.encoder(x_cycle)
            z_batches.append(z.cpu().numpy())
            metadata["cell_id"].extend(batch["cell_id"])
            metadata["dataset_id"].extend(batch["dataset_id"])
            metadata["batch_id"].extend([value or "" for value in batch["batch_id"]])
            metadata["cycle_index"].extend(batch["cycle_index"].cpu().numpy().tolist())
            metadata["soh"].extend(batch["soh"].cpu().numpy().tolist())
            summary_values = [batch["summaries"][name].cpu().numpy() for name in summary_names]
            summaries.append(np.stack(summary_values, axis=1))

    output = args.output or Path(data_cfg.get("latent_cache", Path(data_cfg["processed_dir"]) / "latents.npz"))
    save_latent_cache(
        output,
        z=np.concatenate(z_batches, axis=0),
        metadata={key: np.asarray(value) for key, value in metadata.items()},
        summary_names=summary_names,
        summaries=np.concatenate(summaries, axis=0),
    )
    print(f"Wrote latent cache to {output}")


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
