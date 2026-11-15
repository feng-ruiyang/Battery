from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample SoH trajectories from saved predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Predictions: {args.predictions}")
    print(f"Samples: {args.num_samples}")
    raise SystemExit("Trajectory sampling CLI will be completed after prediction schema is frozen.")


if __name__ == "__main__":
    main()

