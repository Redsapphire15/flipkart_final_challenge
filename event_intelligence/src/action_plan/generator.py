from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActionPlanGenerator:
    review_interval_minutes: int = 30

    def generate(
        self,
        *,
        impact_score: float,
        duration_minutes: float,
        resources: dict[str, Any],
        dispatch: dict[str, Any],
        diversion_plan: dict[str, Any],
        event: dict[str, Any],
    ) -> list[str]:
        route = self._recommended_route(diversion_plan)
        junction = event.get("junction") or event.get("address") or "the incident location"
        escalation = resources.get("escalation_level", "Field Supervisor")
        steps = [
            (
                f"Dispatch {dispatch.get('assigned_officers', resources.get('officers', 0))} officers "
                f"from {dispatch.get('station_name', 'nearest available traffic police station')}."
            ),
            f"Deploy {resources.get('barricades', 0)} barricades at {junction}.",
        ]
        if route:
            steps.append(f"Activate {route['route_name']} for {route['suitability']} traffic.")
        if resources.get("tow_truck"):
            steps.append("Send a tow truck to the incident location.")
        steps.extend(
            [
                f"Notify {escalation} and keep the control room updated.",
                (
                    f"Review the situation after {self.review_interval_minutes} minutes; "
                    f"expected clearance window is {round(duration_minutes)} minutes."
                ),
            ]
        )
        if impact_score >= 80:
            steps.append("Prepare senior-command escalation if impact remains critical after the first review.")
        return steps

    @staticmethod
    def _recommended_route(diversion_plan: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("route_1", "route_2", "route_3"):
            route = diversion_plan.get(key)
            if route and route.get("path"):
                return route
        return None
