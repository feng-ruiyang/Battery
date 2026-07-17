from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol


class MetricLogger(Protocol):
    def log(self, row: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({key: row.get(key) for key in self._fieldnames})

    def close(self) -> None:
        return None


class WandbLogger:
    def __init__(self, run: Any) -> None:
        self.run = run

    def log(self, row: dict[str, Any]) -> None:
        self.run.log(row, step=row.get("epoch"))

    def close(self) -> None:
        self.run.finish()


class NullLogger:
    def log(self, row: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


def build_metric_logger(cfg: dict[str, Any], output_dir: str | Path) -> MetricLogger:
    logging_cfg = cfg.get("logging", {})
    backend = logging_cfg.get("backend", "csv")
    if backend == "none":
        return NullLogger()
    if backend == "csv":
        metrics_path = logging_cfg.get("path") or Path(output_dir) / "metrics.csv"
        return CSVLogger(metrics_path)
    if backend == "wandb":
        run = maybe_init_wandb(cfg)
        if run is None:
            return NullLogger()
        return WandbLogger(run)
    raise ValueError(f"Unsupported logging backend: {backend!r}")


def maybe_init_wandb(cfg: dict[str, Any]):
    logging_cfg = cfg.get("logging", {})
    if logging_cfg.get("backend") != "wandb":
        return None
    import wandb

    mode = "offline" if logging_cfg.get("offline", False) else "online"
    return wandb.init(
        project=logging_cfg.get("project", "battery-soh-bnn"),
        entity=logging_cfg.get("entity"),
        group=logging_cfg.get("group"),
        tags=logging_cfg.get("tags"),
        name=cfg.get("experiment", {}).get("name"),
        config=cfg,
        mode=mode,
    )
