from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from battery_soh.data.io import write_jsonl
from battery_soh.data.splits import make_cell_level_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair processed JSON-derived cell ids without re-parsing raw data."
    )
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="Write repaired files and rename arrays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir
    cells_path = processed_dir / "cells.jsonl"
    cycles_path = processed_dir / "cycles.jsonl"
    arrays_dir = processed_dir / "arrays"

    cells = read_jsonl(cells_path)
    cycles = read_jsonl(cycles_path)
    repairs = build_repairs(cells, cycles)
    old_counts = Counter(repair.old_id for repair in repairs)

    print(f"cells: {len(cells)}")
    print(f"cycles: {len(cycles)}")
    print(f"unique old cell_ids: {len(old_counts)}")
    print(f"duplicate old cell_ids: {sum(1 for count in old_counts.values() if count > 1)}")

    for repair in repairs:
        duplicate_note = " duplicate" if old_counts[repair.old_id] > 1 else ""
        arrays_note = "keep arrays" if repair.keep_arrays else "drop array_path"
        print(
            f"{repair.old_id} -> {repair.new_id}"
            f"{duplicate_note}; cycles={len(repair.cycle_rows)}; {arrays_note}"
        )

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write repaired files.")
        return

    backup(cells_path)
    backup(cycles_path)

    renamed_dirs = rename_array_dirs(arrays_dir, repairs)
    repaired_cells, repaired_cycles = rewrite_rows(processed_dir, cells, repairs)
    write_jsonl(cells_path, repaired_cells)
    write_jsonl(cycles_path, repaired_cycles)
    split = make_cell_level_split([cell["cell_id"] for cell in repaired_cells], seed=args.split_seed)
    write_jsonl(processed_dir / "splits" / f"cell_random_seed_{args.split_seed}.jsonl", [split.to_dict()])

    print(f"\nRenamed array directories: {renamed_dirs}")
    print(f"Wrote repaired cells: {cells_path}")
    print(f"Wrote repaired cycles: {cycles_path}")
    print(f"Wrote split seed {args.split_seed}.")


class Repair:
    def __init__(self, old_id: str, new_id: str, cell_row: dict[str, Any], cycle_rows: list[dict[str, Any]]):
        self.old_id = old_id
        self.new_id = new_id
        self.cell_row = cell_row
        self.cycle_rows = cycle_rows
        self.keep_arrays = True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_repairs(cells: list[dict[str, Any]], cycles: list[dict[str, Any]]) -> list[Repair]:
    repairs: list[Repair] = []
    cursor = 0
    for cell in cells:
        old_id = str(cell["cell_id"])
        num_cycles = int(cell.get("num_cycles") or 0)
        cycle_rows = cycles[cursor : cursor + num_cycles]
        if len(cycle_rows) != num_cycles:
            raise ValueError(f"Not enough cycle rows for {old_id}: expected {num_cycles}.")
        if any(row.get("cell_id") != old_id for row in cycle_rows):
            raise ValueError(
                f"Cycle rows are not contiguous for {old_id}. Re-run full preprocessing instead."
            )
        repairs.append(Repair(old_id, _cell_id_from_source(cell), cell, cycle_rows))
        cursor += num_cycles

    if cursor != len(cycles):
        raise ValueError(f"Unassigned cycle rows: consumed {cursor}, total {len(cycles)}.")

    by_old: dict[str, list[Repair]] = defaultdict(list)
    for repair in repairs:
        by_old[repair.old_id].append(repair)
    for group in by_old.values():
        if len(group) <= 1:
            continue
        for repair in group[:-1]:
            repair.keep_arrays = False
        group[-1].keep_arrays = True

    new_ids = [repair.new_id for repair in repairs]
    duplicates = [cell_id for cell_id, count in Counter(new_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Repaired cell ids are still duplicated: {duplicates[:5]}")
    return repairs


def _cell_id_from_source(cell: dict[str, Any]) -> str:
    source = str(cell.get("source_file") or cell["cell_id"])
    member = source.split("::")[-1]
    stem = Path(member).stem
    if stem.endswith("_structure"):
        stem = stem[: -len("_structure")]
    return stem


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")
    shutil.copy2(path, backup_path)


def rename_array_dirs(arrays_dir: Path, repairs: list[Repair]) -> int:
    moved = 0
    for repair in repairs:
        if not repair.keep_arrays:
            continue
        old_dir = arrays_dir / repair.old_id
        new_dir = arrays_dir / repair.new_id
        if not old_dir.exists():
            continue
        if new_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing array dir: {new_dir}")
        old_dir.rename(new_dir)
        moved += 1
    return moved


def rewrite_rows(
    processed_dir: Path,
    cells: list[dict[str, Any]],
    repairs: list[Repair],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del cells
    repaired_cells: list[dict[str, Any]] = []
    repaired_cycles: list[dict[str, Any]] = []
    for repair in repairs:
        cell = dict(repair.cell_row)
        cell["cell_id"] = repair.new_id
        repaired_cells.append(cell)

        for old_cycle in repair.cycle_rows:
            cycle = dict(old_cycle)
            cycle["cell_id"] = repair.new_id
            array_path = cycle.get("array_path")
            if repair.keep_arrays and array_path:
                filename = Path(array_path).name
                cycle["array_path"] = (Path("arrays") / repair.new_id / filename).as_posix()
                if not (processed_dir / cycle["array_path"]).exists():
                    cycle["array_path"] = None
            else:
                cycle["array_path"] = None
            repaired_cycles.append(cycle)
    return repaired_cells, repaired_cycles


if __name__ == "__main__":
    main()
