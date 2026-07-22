from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_and_clean_raw_json(json_path: str | Path) -> dict[str, Any]:
    """Load a raw JSON file and run the dataset-specific cleaning hook."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return clean_raw_json(path, data)


def clean_raw_json(json_path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw JSON values before dataset-specific parsing.

    Some BEEP/Severson exports contain non-standard JSON constants such as NaN
    and Infinity. Python's `json` module accepts them, but downstream JSONL
    writers and numpy array serialization should receive regular JSON values.
    """
    _ = json_path
    cleaned = _replace_nonfinite(data)
    if not isinstance(cleaned, dict):
        raise TypeError(f"Expected raw JSON object after cleaning, got {type(cleaned)!r}.")
    return cleaned


def _replace_nonfinite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_replace_nonfinite(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_nonfinite(item) for key, item in value.items()}
    return value
