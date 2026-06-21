from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from event_intelligence.src.common.paths import CONFIG_DIR


DEFAULT_RULES = {
    "impact_bands": [
        {"min_impact": 80, "officers": 4, "barricades": 3, "escalation_level": "Command Center"},
        {"min_impact": 60, "officers": 3, "barricades": 2, "escalation_level": "Senior Inspector"},
        {"min_impact": 40, "officers": 2, "barricades": 1, "escalation_level": "Field Supervisor"},
        {"min_impact": 0, "officers": 1, "barricades": 0, "escalation_level": "Routine Patrol"},
    ],
    "tow_truck_causes": ["vehicle_breakdown", "accident"],
    "tow_truck_vehicle_types": ["truck", "heavy_vehicle", "lcv"],
    "long_duration_minutes": 90,
    "high_corridor_risk": 70,
}


@dataclass
class ResourcePlanner:
    rules: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RULES))

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "ResourcePlanner":
        path = Path(path) if path else CONFIG_DIR / "resource_rules.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return cls(json.load(handle))
        return cls()

    def recommend(
        self,
        *,
        predicted_impact: float,
        predicted_duration: float,
        road_closure: bool,
        corridor_risk: float = 0.0,
        event_type: str = "",
        event_cause: str = "",
        veh_type: str = "",
    ) -> dict[str, Any]:
        band = next(
            rule
            for rule in sorted(self.rules["impact_bands"], key=lambda r: r["min_impact"], reverse=True)
            if predicted_impact >= rule["min_impact"]
        )
        officers = int(band["officers"])
        barricades = int(band["barricades"])

        if road_closure:
            barricades += 1
            officers += 1
        if predicted_duration >= self.rules.get("long_duration_minutes", 90):
            officers += 1
        if corridor_risk >= self.rules.get("high_corridor_risk", 70):
            officers += 1
            barricades += 1

        cause = str(event_cause).lower()
        vehicle = str(veh_type).lower()
        tow_truck = cause in self.rules["tow_truck_causes"] or vehicle in self.rules["tow_truck_vehicle_types"]

        return {
            "officers": officers,
            "barricades": barricades,
            "tow_truck": tow_truck,
            "escalation_level": band["escalation_level"],
            "corridor_risk": round(corridor_risk, 2),
        }
