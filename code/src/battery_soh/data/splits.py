from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CellSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def make_cell_level_split(
    cell_ids: list[str],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> CellSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")
    unique_ids = sorted(set(cell_ids))
    if len(unique_ids) != len(cell_ids):
        raise ValueError("cell_ids contains duplicates.")

    rng = random.Random(seed)
    shuffled = unique_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])
    return CellSplit(train=train, val=val, test=test)


def assert_no_cell_leakage(split: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in split.items():
        for cell_id in ids:
            if cell_id in seen:
                raise AssertionError(
                    f"Cell {cell_id!r} appears in both {seen[cell_id]!r} and {split_name!r}."
                )
            seen[cell_id] = split_name

