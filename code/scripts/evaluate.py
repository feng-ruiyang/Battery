from __future__ import annotations

import argparse
from pathlib import Path

from battery_soh.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SoH trajectory predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    print(f"Evaluation config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Processed data: {cfg.get('data', {}).get('processed_dir')}")
    raise SystemExit("Evaluation wiring will be completed after training output format is frozen.")


if __name__ == "__main__":
    main()

