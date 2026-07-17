from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from battery_soh.data.schema import ParserScanResult


class BaseBatteryParser(ABC):
    dataset_id: str

    @abstractmethod
    def scan(self, raw_dir: Path) -> ParserScanResult:
        """Return raw file inventory and parser notes without loading all data."""

    @abstractmethod
    def parse(self, raw_dir: Path, output_dir: Path) -> None:
        """Convert raw dataset files into canonical metadata and cycle arrays."""

