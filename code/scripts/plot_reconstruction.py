from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.torch_datasets import CycleTensorDataset, resolve_cell_ids
from battery_soh.training.checkpointing import load_checkpoint
from battery_soh.utils.config import load_yaml
from battery_soh.utils.device import resolve_device
from cache_latents import build_autoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cycle autoencoder reconstruction examples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstructions"))
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
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
    split_file = data_cfg.get("split_file")
    cell_ids = resolve_cell_ids(store, split_file, args.split) if args.split != "all" else resolve_cell_ids(store)
    dataset = CycleTensorDataset(
        store=store,
        cell_ids=cell_ids,
        channels=tuple(data_cfg.get("channels", ("V", "I", "T"))),
        sequence_length=int(data_cfg.get("sequence_length", 1000)),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No cycles found for split={args.split!r}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    end = min(args.start_index + args.num_examples, len(dataset))
    print(
        f"Plotting reconstruction examples {args.start_index}:{end} "
        f"from {len(dataset)} cycles on split={args.split}.",
        flush=True,
    )
    for index in range(args.start_index, end):
        item = dataset[index]
        x = item["x_cycle"].unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(x)["x_recon"].squeeze(0).cpu().numpy()
        original = x.squeeze(0).cpu().numpy()
        out_path = args.output_dir / (
            f"{item['cell_id']}_cycle_{int(item['cycle_index']):05d}_reconstruction.png"
        )
        plot_cycle_reconstruction(
            original=original,
            reconstruction=recon,
            channels=tuple(data_cfg.get("channels", ("V", "I", "T"))),
            title=f"{item['cell_id']} cycle {int(item['cycle_index'])}",
            output_path=out_path,
        )
        print(out_path, flush=True)


def plot_cycle_reconstruction(
    original: np.ndarray,
    reconstruction: np.ndarray,
    channels: tuple[str, ...],
    title: str,
    output_path: Path,
) -> None:
    x_axis = np.arange(original.shape[0])
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.6 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for channel_index, (axis, channel) in enumerate(zip(axes, channels)):
        axis.plot(x_axis, original[:, channel_index], label="input", linewidth=1.2)
        axis.plot(x_axis, reconstruction[:, channel_index], label="reconstruction", linewidth=1.2)
        axis.set_ylabel(channel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Resampled point index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
