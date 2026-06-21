from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class OfficerAllocationEngine:
    risk_per_officer: float = 20.0

    def recommend_by_corridor(self, risk_engine: Any, limit: int = 20) -> list[dict[str, Any]]:
        recommendations = []
        for row in risk_engine.top_corridors(limit):
            demand = max(1, math.ceil(float(row["risk_score"]) / self.risk_per_officer))
            recommendations.append(
                {
                    "corridor": row["corridor"],
                    "risk_score": row["risk_score"],
                    "frequency": row["frequency"],
                    "avg_duration": row["avg_duration"],
                    "officer_demand": demand,
                    "demand_band": self._band(demand),
                }
            )
        return recommendations

    @staticmethod
    def _band(demand: int) -> str:
        if demand >= 5:
            return "Red"
        if demand >= 3:
            return "Orange"
        if demand >= 2:
            return "Yellow"
        return "Green"
