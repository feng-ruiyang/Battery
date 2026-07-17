from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from battery_soh.data.schema import CanonicalCell, CanonicalCycle


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def write_cells(path: Path, cells: Iterable[CanonicalCell]) -> None:
    write_jsonl(path, (cell.to_json() for cell in cells))


def write_cycles(path: Path, cycles: Iterable[CanonicalCycle]) -> None:
    write_jsonl(path, (cycle.to_json() for cycle in cycles))


def write_cycle_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    clean = {key: np.asarray(value) for key, value in arrays.items() if value is not None}
    np.savez_compressed(path, **clean)

