from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable


@dataclass
class RiskCorridorEngine:
    corridor_scores: list[dict] = field(default_factory=list)
    junction_scores: list[dict] = field(default_factory=list)

    def fit(self, rows: Iterable[dict]) -> "RiskCorridorEngine":
        rows = list(rows)
        self.corridor_scores = self._score_dimension(rows, "corridor")
        self.junction_scores = self._score_dimension(rows, "junction")
        return self

    def _score_dimension(self, rows: list[dict], dimension: str) -> list[dict]:
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            key = row.get(dimension) or "Unknown"
            buckets.setdefault(str(key), []).append(row)

        raw = []
        for key, items in buckets.items():
            durations = [float(r["duration_minutes"]) for r in items if r.get("duration_minutes") is not None]
            avg_duration = mean(durations) if durations else 0.0
            closure_probability = sum(1 for r in items if r.get("requires_road_closure")) / len(items)
            risk_raw = len(items) * avg_duration * max(closure_probability, 0.05)
            raw.append(
                {
                    dimension: key,
                    "risk_raw": risk_raw,
                    "frequency": len(items),
                    "avg_duration": round(avg_duration, 2),
                    "closure_probability": round(closure_probability, 3),
                }
            )

        max_raw = max((r["risk_raw"] for r in raw), default=1.0) or 1.0
        ranked = []
        for item in raw:
            scored = dict(item)
            scored["risk_score"] = round(100 * item["risk_raw"] / max_raw, 2)
            scored.pop("risk_raw", None)
            ranked.append(scored)
        return sorted(ranked, key=lambda r: r["risk_score"], reverse=True)

    def top_corridors(self, limit: int = 10) -> list[dict]:
        return self.corridor_scores[:limit]

    def top_junctions(self, limit: int = 10) -> list[dict]:
        return self.junction_scores[:limit]

    def get_corridor_risk(self, corridor: str | None) -> float:
        for row in self.corridor_scores:
            if row.get("corridor") == corridor:
                return float(row["risk_score"])
        return 0.0
