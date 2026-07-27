from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_soh.data.io import read_jsonl, write_jsonl
from battery_soh.data.splits import assert_no_cell_leakage, make_cell_level_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible cell-level data split.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.expanduser()
    cells = read_jsonl(processed_dir / "cells.jsonl")
    cell_ids = [str(row["cell_id"]) for row in cells]
    split = make_cell_level_split(
        cell_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    ).to_dict()
    assert_no_cell_leakage(split)
    output = (
        args.output.expanduser()
        if args.output is not None
        else processed_dir / "splits" / f"cell_random_seed_{args.seed}.jsonl"
    )
    write_jsonl(output, [split])
    result = {
        "output": str(output),
        "seed": args.seed,
        "counts": {name: len(ids) for name, ids in split.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
