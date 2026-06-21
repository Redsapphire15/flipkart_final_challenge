from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

from event_intelligence.src.common.serialization import load_pickle, save_pickle
from event_intelligence.src.resource_planner import ResourcePlanner


SIMILARITY_COLUMNS = ["event_type", "event_cause", "zone", "junction", "hour", "priority"]


@dataclass
class SimilarIncidentSearch:
    rows: list[dict] = field(default_factory=list)
    vocabulary: dict[str, int] = field(default_factory=dict)
    vectors: list[dict[int, float]] = field(default_factory=list)

    def fit(self, rows: Iterable[dict]) -> "SimilarIncidentSearch":
        self.rows = list(rows)
        self.vocabulary = {}
        for row in self.rows:
            for token in self._tokens(row):
                self.vocabulary.setdefault(token, len(self.vocabulary))
        self.vectors = [self._vector(row) for row in self.rows]
        return self

    def query(self, event: dict, top_k: int = 5, planner: ResourcePlanner | None = None) -> dict:
        query_vector = self._vector(event)
        scored = []
        for row, vector in zip(self.rows, self.vectors):
            score = _cosine(query_vector, vector)
            scored.append((score, row))
        matches = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        match_rows = [row for _, row in matches]
        durations = [float(r["duration_minutes"]) for r in match_rows if r.get("duration_minutes") is not None]
        impacts = [float(r["impact_score"]) for r in match_rows if r.get("impact_score") is not None]
        avg_duration = mean(durations) if durations else 0.0
        avg_impact = mean(impacts) if impacts else 0.0
        planner = planner or ResourcePlanner.from_config()
        resources = planner.recommend(
            predicted_impact=avg_impact,
            predicted_duration=avg_duration,
            road_closure=bool(event.get("requires_road_closure")),
            event_type=str(event.get("event_type", "")),
            event_cause=str(event.get("event_cause", "")),
            veh_type=str(event.get("veh_type", "")),
        )
        return {
            "matches": [
                {
                    "similarity_score": round(score, 3),
                    "id": row.get("id"),
                    "event_type": row.get("event_type"),
                    "event_cause": row.get("event_cause"),
                    "date": self._event_date(row),
                    "corridor": row.get("corridor"),
                    "junction": row.get("junction"),
                    "duration_minutes": row.get("duration_minutes"),
                    "impact_score": row.get("impact_score"),
                    "resources_used": self._resources_used(row, planner),
                    "outcome": self._outcome(row),
                }
                for score, row in matches
            ],
            "average_duration": round(avg_duration, 2),
            "average_impact": round(avg_impact, 2),
            "recommended_resources": resources,
        }

    def _tokens(self, row: dict) -> list[str]:
        tokens = []
        for column in SIMILARITY_COLUMNS:
            value = row.get(column, "Unknown")
            if column == "hour":
                try:
                    value = int(value) // 3
                    tokens.append(f"{column}=bucket_{value}")
                except Exception:
                    tokens.append(f"{column}=unknown")
            else:
                tokens.append(f"{column}={str(value).lower()}")
        return tokens

    def _vector(self, row: dict) -> dict[int, float]:
        vector: dict[int, float] = {}
        for token in self._tokens(row):
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] = vector.get(index, 0.0) + 1.0
        return vector

    @staticmethod
    def _event_date(row: dict) -> str:
        value = row.get("start_datetime_parsed") or row.get("start_datetime") or row.get("created_date")
        if hasattr(value, "date"):
            return value.date().isoformat()
        return str(value or "Unknown")[:10]

    @staticmethod
    def _resources_used(row: dict, planner: ResourcePlanner) -> dict:
        return planner.recommend(
            predicted_impact=float(row.get("impact_score") or 0),
            predicted_duration=float(row.get("duration_minutes") or 0),
            road_closure=bool(row.get("requires_road_closure")),
            event_type=str(row.get("event_type", "")),
            event_cause=str(row.get("event_cause", "")),
            veh_type=str(row.get("veh_type", "")),
        )

    @staticmethod
    def _outcome(row: dict) -> str:
        impact = float(row.get("impact_score") or 0)
        duration = float(row.get("duration_minutes") or 0)
        if impact >= 80 or duration >= 120:
            return "Escalated clearance"
        if duration <= 45 and impact < 60:
            return "Cleared within target"
        return "Resolved with field control"

    def save(self, path: str | Path) -> Path:
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SimilarIncidentSearch":
        return load_pickle(path)


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
