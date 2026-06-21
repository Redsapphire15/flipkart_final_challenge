from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExplainableAIEngine:
    """Operational explanations for predictions.

    Uses model importances when available, then enriches them with transparent
    traffic-domain rules so officers see actionable reasons.
    """

    def explain(self, event: dict[str, Any], impact_model: Any, duration_model: Any) -> list[dict[str, Any]]:
        contributions = self._domain_contributions(event)
        contributions.extend(self._model_importances(impact_model, prefix="Impact model"))
        contributions.extend(self._model_importances(duration_model, prefix="Duration model"))
        merged: dict[str, float] = {}
        for item in contributions:
            merged[item["feature"]] = max(merged.get(item["feature"], 0.0), float(item["contribution"]))
        return [
            {"feature": feature, "contribution": round(score, 2)}
            for feature, score in sorted(merged.items(), key=lambda item: item[1], reverse=True)[:8]
        ]

    @staticmethod
    def _domain_contributions(event: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        if event.get("requires_road_closure"):
            rows.append({"feature": "Road Closure", "contribution": 35})
        if event.get("is_peak_hour"):
            rows.append({"feature": "Peak Hour", "contribution": 22})
        if str(event.get("priority", "")).lower() == "high":
            rows.append({"feature": "Priority", "contribution": 18})
        if str(event.get("event_cause", "")).lower() in {"accident", "vehicle_breakdown"}:
            rows.append({"feature": "Incident Cause", "contribution": 16})
        if event.get("corridor") and event.get("corridor") != "Non-corridor":
            rows.append({"feature": "Corridor", "contribution": 12})
        return rows or [{"feature": "Historical Baseline", "contribution": 10}]

    @staticmethod
    def _model_importances(model: Any, prefix: str) -> list[dict[str, Any]]:
        backend = getattr(model, "backend_model", None)
        if backend is None:
            return []
        try:
            estimator = backend
            if hasattr(backend, "named_steps"):
                estimator = backend.named_steps.get("regressor", backend)
            values = estimator.get_feature_importance() if hasattr(estimator, "get_feature_importance") else estimator.feature_importances_
        except Exception:
            return []
        total = float(sum(abs(float(v)) for v in values)) or 1.0
        return [
            {"feature": f"{prefix} factor {index + 1}", "contribution": abs(float(value)) * 100 / total}
            for index, value in enumerate(values[:5])
        ]
