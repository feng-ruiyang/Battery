from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_soh.data.calce import CalceFilterConfig, CalceParser
from battery_soh.data.hust import HustParser
from battery_soh.data.nasa import NasaParser
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
    raw_dir = (args.raw_dir or Path(cfg.get("raw_dir", f"data/raw/{dataset}"))).expanduser()
    output_dir = (
        args.output_dir or Path(cfg.get("output_dir", f"data/processed/{dataset}"))
    ).expanduser()
    max_cells = args.max_cells if args.max_cells is not None else cfg.get("max_cells")
    if dataset == "severson":
        include_arrays = tuple(cfg.get("include_arrays", DEFAULT_CYCLE_ARRAYS))
        split_seed = int(cfg.get("splits", {}).get("seed", 42))
        parser = SeversonParser(
            include_arrays=include_arrays,
            split_seed=split_seed,
            max_cells=max_cells,
        )
    elif dataset == "hust":
        parser = HustParser(max_cells=max_cells)
    elif dataset == "nasa":
        archive_dir = cfg.get("archive_dir")
        split_cfg = cfg.get("splits", {})
        parser = NasaParser(
            max_cells=max_cells,
            archive_dir=Path(archive_dir) if archive_dir else None,
            split_seed=int(split_cfg.get("seed", 42)),
            train_ratio=float(split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        )
    elif dataset == "calce":
        split_cfg = cfg.get("splits", {})
        filter_cfg = cfg.get("filters", {})
        parser = CalceParser(
            max_cells=max_cells,
            split_seed=int(split_cfg.get("seed", 42)),
            train_ratio=float(split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            nominal_capacity_ah=float(cfg.get("nominal_capacity_ah", 1.1)),
            filters=CalceFilterConfig(
                min_q_ah=float(filter_cfg.get("min_q_ah", 0.03)),
                max_charge_discharge_mismatch=float(
                    filter_cfg.get("max_charge_discharge_mismatch", 0.25)
                ),
                min_voltage_max_v=float(filter_cfg.get("min_voltage_max_v", 4.0)),
                max_voltage_min_v=float(filter_cfg.get("max_voltage_min_v", 2.9)),
                min_points=int(filter_cfg.get("min_points", 50)),
                local_outlier_window=int(filter_cfg.get("local_outlier_window", 3)),
                local_outlier_drop=float(filter_cfg.get("local_outlier_drop", 0.08)),
            ),
            cell_pattern=str(cfg.get("cell_pattern", r"CS2[_-]\d+")),
        )
    else:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. "
            "Currently supported: 'severson', 'hust', 'nasa', 'calce'."
        )

    scan = parser.scan(raw_dir)
    print(json.dumps(scan.__dict__, ensure_ascii=False, indent=2))
    if args.scan_only:
        return
    parser.parse(raw_dir, output_dir)
    print(f"Wrote processed dataset to {output_dir}")


if __name__ == "__main__":
    main()
