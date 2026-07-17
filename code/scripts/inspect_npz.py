from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect processed cycle .npz files.")
    parser.add_argument(
        "path",
        type=Path,
        help="A .npz file, an arrays directory, or a processed dataset directory.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of .npz files to print.")
    parser.add_argument("--preview", type=int, default=0, help="Print the first N values per array.")
    parser.add_argument(
        "--check-cycles",
        action="store_true",
        help="If path is a processed dataset directory, compare cycles.jsonl array_path entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path
    npz_files = resolve_npz_files(path)
    print(f"npz files: {len(npz_files)}")

    if args.check_cycles:
        check_cycles(path, npz_files)

    for npz_path in npz_files[: args.limit]:
        inspect_npz(npz_path, preview=args.preview)

    if len(npz_files) > args.limit:
        print(f"... skipped {len(npz_files) - args.limit} files; increase --limit to print more.")


def resolve_npz_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)

    arrays_dir = path / "arrays" if (path / "arrays").is_dir() else path
    return sorted(arrays_dir.rglob("*.npz"))


def inspect_npz(path: Path, preview: int = 0) -> None:
    print(f"\n{path}")
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            values = data[name]
            line = f"  {name}: shape={values.shape}, dtype={values.dtype}"
            if preview > 0:
                line += f", preview={preview_values(values, preview)}"
            print(line)


def preview_values(values: np.ndarray, limit: int) -> list[Any]:
    flat = np.asarray(values).reshape(-1)
    out: list[Any] = []
    for value in flat[:limit]:
        if isinstance(value, np.generic):
            out.append(value.item())
        else:
            out.append(value)
    return out


def check_cycles(processed_dir: Path, npz_files: list[Path]) -> None:
    cycles_path = processed_dir / "cycles.jsonl"
    if not cycles_path.exists():
        print("cycles.jsonl: not found; skipping cycle check.")
        return

    npz_set = {path.resolve() for path in npz_files}
    total = 0
    with_array = 0
    missing = 0
    missing_examples: list[str] = []
    with cycles_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            array_path = row.get("array_path")
            if not array_path:
                continue
            with_array += 1
            resolved = (processed_dir / array_path).resolve()
            if resolved not in npz_set:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(array_path)

    print(f"cycles.jsonl rows: {total}")
    print(f"cycles with array_path: {with_array}")
    print(f"missing array files: {missing}")
    if missing_examples:
        print("missing examples:")
        for example in missing_examples:
            print(f"  {example}")


if __name__ == "__main__":
    main()
