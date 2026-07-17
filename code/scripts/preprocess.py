from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_soh.data.severson import DEFAULT_CYCLE_ARRAYS, SeversonParser
from battery_soh.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess battery datasets.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--scan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    dataset = args.dataset or cfg.get("dataset")
    raw_dir = args.raw_dir or Path(cfg.get("raw_dir", "data/raw/severson"))
    output_dir = args.output_dir or Path(cfg.get("output_dir", "data/processed/severson"))

    if dataset != "severson":
        raise ValueError(f"Unsupported dataset {dataset!r}. Currently supported: 'severson'.")

    include_arrays = tuple(cfg.get("include_arrays", DEFAULT_CYCLE_ARRAYS))
    split_seed = int(cfg.get("splits", {}).get("seed", 42))
    max_cells = args.max_cells if args.max_cells is not None else cfg.get("max_cells")
    parser = SeversonParser(include_arrays=include_arrays, split_seed=split_seed, max_cells=max_cells)

    scan = parser.scan(raw_dir)
    print(json.dumps(scan.__dict__, ensure_ascii=False, indent=2))
    if args.scan_only:
        return
    parser.parse(raw_dir, output_dir)
    print(f"Wrote processed dataset to {output_dir}")


if __name__ == "__main__":
    main()
