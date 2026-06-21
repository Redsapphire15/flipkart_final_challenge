from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Iterable


SIGNATURE_COLUMNS = ["event_type", "event_cause", "zone", "corridor", "junction", "priority", "hour"]


def _signature(row: dict, columns: list[str]) -> tuple:
    return tuple(row.get(column, "Unknown") for column in columns)


@dataclass
class HierarchicalRegressor:
    """Small dependency-free regressor for sparse categorical event data."""

    target: str
    global_value: float = 0.0
    levels: list[list[str]] = field(
        default_factory=lambda: [
            SIGNATURE_COLUMNS,
            ["event_type", "event_cause", "corridor", "priority"],
            ["event_cause", "corridor", "priority"],
            ["event_cause", "priority"],
            ["priority"],
        ]
    )
    tables: list[dict[tuple, float]] = field(default_factory=list)
    min_samples: int = 3

    def fit(self, rows: Iterable[dict]) -> "HierarchicalRegressor":
        usable = [row for row in rows if row.get(self.target) is not None]
        values = [float(row[self.target]) for row in usable]
        self.global_value = float(median(values)) if values else 0.0
        self.tables = []
        for columns in self.levels:
            buckets: dict[tuple, list[float]] = {}
            for row in usable:
                buckets.setdefault(_signature(row, columns), []).append(float(row[self.target]))
            self.tables.append(
                {key: float(mean(vals)) for key, vals in buckets.items() if len(vals) >= self.min_samples}
            )
        return self

    def predict_one(self, row: dict) -> float:
        for columns, table in zip(self.levels, self.tables):
            key = _signature(row, columns)
            if key in table:
                return table[key]
        return self.global_value

    def predict(self, rows: Iterable[dict]) -> list[float]:
        return [self.predict_one(row) for row in rows]


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    errors = [p - a for a, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
    actual_mean = sum(actual) / len(actual)
    ss_tot = sum((a - actual_mean) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3)}
