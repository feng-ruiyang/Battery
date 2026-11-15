from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_run_output_dir(cfg: dict[str, Any], default_output_dir: str) -> Path:
    experiment_cfg = cfg.get("experiment", {})
    base_dir = Path(experiment_cfg.get("output_dir", default_output_dir))
    if not bool(experiment_cfg.get("timestamped_output", True)):
        return base_dir

    run_name = experiment_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / "runs" / str(run_name)
