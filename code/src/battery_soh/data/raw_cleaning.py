from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_and_clean_raw_json(json_path: str | Path) -> dict[str, Any]:
    """Load a raw JSON file and run the dataset-specific cleaning hook."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return clean_raw_json(path, data)


def clean_raw_json(json_path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for raw JSON cleaning.

    The implementation is intentionally empty for now. Future cleaning should keep
    this function as the raw JSON boundary so parsers receive normalized data.
    """
    _ = json_path
    return data
