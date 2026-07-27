from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from battery_soh.data.datasets import CanonicalCycleStore
from battery_soh.data.normalization import ChannelNormalizer, save_channel_normalizer
from battery_soh.data.torch_datasets import CycleTensorDataset, resolve_cell_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit train-only cycle-channel normalization.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--channels", nargs="+", default=["V", "I", "T"])
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.expanduser()
    split_file = args.split_file.expanduser()
    channels = tuple(args.channels)
    store = CanonicalCycleStore(processed_dir)
    cell_ids = resolve_cell_ids(store, split_file, args.split)
    dataset = CycleTensorDataset(
        store=store,
        cell_ids=cell_ids,
        channels=channels,
        sequence_length=args.sequence_length,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No cycles found for split={args.split!r}.")

    channel_sum = np.zeros((len(channels),), dtype=np.float64)
    channel_sum_sq = np.zeros((len(channels),), dtype=np.float64)
    channel_count = np.zeros((len(channels),), dtype=np.int64)
    for index in range(len(dataset)):
        values = dataset[index]["x_cycle"].numpy().astype(np.float64)
        finite = np.isfinite(values)
        channel_sum += np.where(finite, values, 0.0).sum(axis=0)
        channel_sum_sq += np.where(finite, values * values, 0.0).sum(axis=0)
        channel_count += finite.sum(axis=0)
        if index == 0 or (index + 1) % 250 == 0 or index + 1 == len(dataset):
            print(f"normalizer cycles={index + 1}/{len(dataset)}", flush=True)

    if np.any(channel_count == 0):
        missing = [channels[index] for index in np.nonzero(channel_count == 0)[0]]
        raise ValueError(f"No finite training values for channels: {missing}")
    mean = channel_sum / channel_count
    variance = channel_sum_sq / channel_count - mean * mean
    std = np.sqrt(np.maximum(variance, 1.0e-12))
    normalizer = ChannelNormalizer(
        channels=channels,
        mean=mean,
        std=std,
        count=channel_count,
        metadata={
            "processed_dir": str(processed_dir),
            "split_file": str(split_file),
            "split": args.split,
            "num_cells": len(cell_ids),
            "num_cycles": len(dataset),
            "sequence_length": args.sequence_length,
            "fit_scope": "train_cells_only",
        },
    )
    save_channel_normalizer(args.output, normalizer)
    print(json.dumps(normalizer.to_json(), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote channel normalizer to {args.output.expanduser()}")


if __name__ == "__main__":
    main()
