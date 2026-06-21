from __future__ import annotations

import datetime
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


def _json_serial(obj: Any) -> str:
    """Helper to convert datetime objects into ISO format strings for JSON."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


@dataclass
class LLMService:
    """OpenAI-compatible chat completion client with deterministic fallback."""

    api_base: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "LLMService":
        return cls(
            api_base=os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )

    def summarize(self, prompt: str, context: dict[str, Any]) -> str:
        if not self.api_base or not self.api_key:
            return self._fallback_summary(context, prompt)
        
        context_json_str = json.dumps(context, indent=2, default=_json_serial)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a traffic control room assistant. Do not invent predictions. "
                        "Only summarize the supplied module outputs and operational actions."
                    ),
                },
                {"role": "user", "content": f"Query: {prompt}\nModule outputs:\n{context_json_str}"},
            ],
            "temperature": 0.2,
        }
        
        request = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload, default=_json_serial).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            # Check for server side errors (5xx) or timeout network disruptions
            print(f"LLM API failure encountered: {e}. Executing deterministic fallback summary.")
            return self._fallback_summary(context, prompt)
        
    def _fallback_summary(self, context: dict[str, Any], prompt: str = "") -> str:
        if "active_incidents" in context:
            return self._fallback_command_center_summary(context, prompt)

        resources = context.get("resources", {})
        similar = context.get("similar_incidents", {})
        diversion = context.get("diversion", {})
        dispatch = context.get("dispatch", {})
        sufficiency = context.get("resource_sufficiency", {})
        action_plan = context.get("action_plan", [])
        timeline = context.get("timeline", [])
        route = self._recommended_route(diversion)
        action = action_plan[0] if action_plan else "Assess field conditions and deploy nearest available unit."
        review = timeline[-1]["minute"] if timeline else 30
        return "\n".join(
            [
                "Situation Summary",
                f"Impact is {context.get('predicted_impact', 0)}/100 with expected duration of {context.get('predicted_duration', 0)} minutes.",
                "",
                "Risk Assessment",
                f"Risk level is {context.get('risk_level', 'Unknown')}; {len(similar.get('matches', []))} similar historical incidents were found.",
                "",
                "Recommended Action",
                f"{action} Deploy {resources.get('officers', 0)} officers and {resources.get('barricades', 0)} barricades. Resource status: {sufficiency.get('status', 'UNKNOWN')}.",
                "",
                "Diversion Recommendation",
                (
                    f"{route.get('route_name')} for {route.get('suitability')} traffic; "
                    f"estimated delay {route.get('estimated_delay_min')} minutes."
                    if route
                    else "No diversion route is available from the current graph."
                ),
                "",
                "Dispatch Information",
                (
                    f"{dispatch.get('station_name', 'Nearest station unavailable')} is assigned, "
                    f"{dispatch.get('distance_km', 0)} km away, ETA {dispatch.get('eta_minutes', 0)} minutes."
                ),
                "",
                "Review Timeline",
                f"Review after 30 minutes and continue monitoring until the projected {review}-minute clearance window.",
            ]
        )

    def _fallback_command_center_summary(self, context: dict[str, Any], prompt: str) -> str:
        question = prompt.lower()
        active = context.get("active_incidents", [])
        stats = context.get("summary_stats", {})
        resources = context.get("resource_availability", [])
        diversions = context.get("diversions", [])

        if "immediate attention" in question or "requires immediate" in question:
            ranked = sorted(
                active,
                key=lambda row: (
                    row.get("Risk") == "Critical",
                    row.get("Risk") == "High",
                    float(row.get("Impact") or 0),
                    float(row.get("Duration") or 0),
                ),
                reverse=True,
            )
            if not ranked:
                return "No active incident currently requires attention because there are no active incidents in the supplied outputs."
            top = ranked[0]
            return (
                f"{top.get('Incident ID')} requires immediate attention: risk {top.get('Risk')}, "
                f"impact {top.get('Impact')}, duration {top.get('Duration')} minutes, "
                f"assigned to {top.get('Station')} with status {top.get('Status')}."
            )

        if "overloaded" in question or "shortage" in question:
            overloaded = [
                row for row in resources if row.get("Officer Deficit", 0) or row.get("Barricade Deficit", 0)
            ]
            if overloaded:
                return "\n".join(
                    [
                        "Overloaded stations from the Resource Availability Board:",
                        *[
                            (
                                f"- {row.get('Police Station')}: officer deficit {row.get('Officer Deficit')}, "
                                f"barricade deficit {row.get('Barricade Deficit')}."
                            )
                            for row in overloaded
                        ],
                    ]
                )
            lowest = sorted(resources, key=lambda row: row.get("Available Officers", 0))[:1]
            if lowest:
                row = lowest[0]
                return (
                    f"No formal deficit is reported. Lowest available officer capacity is at "
                    f"{row.get('Police Station')} with {row.get('Available Officers')} available officers."
                )
            return "No resource availability output is available."

        if "south zone" in question and "diversion" in question:
            south_ids = {row.get("Incident ID") for row in active if "south" in str(row.get("Zone", "")).lower()}
            lines = []
            for item in diversions:
                if item.get("incident_id") not in south_ids:
                    continue
                for route in item.get("routes", []):
                    if route:
                        lines.append(
                            f"- {item.get('incident_id')}: {route.get('route_name')} "
                            f"({route.get('distance_km')} km, {route.get('estimated_delay_min')} min delay)."
                        )
            return "\n".join(["Active diversions in South Zone:", *lines]) if lines else "No active South Zone diversions are present in the supplied outputs."

        if "why" in question and "route" in question:
            for item in diversions:
                for route in item.get("routes", []):
                    if route and route.get("why_selected"):
                        reasons = "\n".join(f"- {reason}" for reason in route["why_selected"])
                        return f"{item.get('incident_id')} {route.get('route_name')} was selected because:\n{reasons}"
            return "No route explanation is present in the supplied diversion outputs."

        if "officer" in question and ("deployed" in question or "currently" in question):
            return f"{stats.get('Officers Deployed', 0)} officers are currently deployed across active incidents."

        return "\n".join(
            [
                "Command Center Summary",
                f"Active incidents: {stats.get('Active Events', len(active))}",
                f"Critical incidents: {stats.get('Critical Events', 0)}",
                f"Active diversions: {stats.get('Diversions Active', 0)}",
                f"Officers deployed: {stats.get('Officers Deployed', 0)}",
                f"Barricades deployed: {stats.get('Barricades Deployed', 0)}",
            ]
        )

    @staticmethod
    def _recommended_route(diversion: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("route_1", "route_2", "route_3"):
            route = diversion.get(key)
            if route and route.get("path"):
                return route
        return None


@dataclass
class LLMCopilot:
    llm_service: LLMService

    def answer(self, query: str, module_outputs: dict[str, Any]) -> str:
        return self.llm_service.summarize(query, module_outputs)
