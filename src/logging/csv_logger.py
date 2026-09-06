from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVMetricLogger:
    """Append-only CSV writer with a stable, union-of-seen-keys header.

    Rows may carry different subsets of columns (e.g. eval rows vs trajectory
    rows). Missing values are written as empty cells. The header is finalized
    on first write, so callers should pass a `fieldnames` superset up front.
    """

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._rows: list[dict[str, Any]] = []
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in self.fieldnames}
        self._rows.append(clean)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(clean)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)
