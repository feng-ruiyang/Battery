from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_soh.data.compatibility_audit import (
    audit_hust_pickle,
    audit_severson_training,
    compare_compatibility,
)
from battery_soh.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit HUST input compatibility with a processed Severson training split."
    )
    parser.add_argument("--hust-pickle", type=Path, required=True)
    parser.add_argument(
        "--severson-config",
        type=Path,
        default=Path("configs/pretrain_encoder.yaml"),
    )
    parser.add_argument("--severson-processed-dir", type=Path, default=None)
    parser.add_argument("--severson-split-file", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.severson_config)
    data_cfg = cfg["data"]
    processed_dir = args.severson_processed_dir or Path(data_cfg["processed_dir"])
    split_file = args.severson_split_file or Path(data_cfg["split_file"])

    print("Auditing HUST pickle...", flush=True)
    hust = audit_hust_pickle(args.hust_pickle)
    print("Auditing Severson training split...", flush=True)
    severson = audit_severson_training(
        processed_dir=processed_dir,
        split_file=split_file,
        split_name=args.split,
        log_every=args.log_every,
    )
    result = {
        "hust": hust,
        "severson": severson,
        "comparison": compare_compatibility(hust, severson),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
