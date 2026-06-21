from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RouteVisualizer:
    colors: tuple[str, str, str] = ("#2563eb", "#f97316", "#16a34a")

    def build_map_figure(self, route_data: dict[str, Any], event_location: tuple[float, float] | None = None):
        import plotly.graph_objects as go

        fig = go.Figure()
        if event_location and event_location[0] is not None and event_location[1] is not None:
            lat, lon = event_location
            fig.add_trace(
                go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode="markers",
                    marker={"size": 15, "color": "#dc2626"},
                    name="Incident Location",
                )
            )
            fig.add_trace(
                go.Scattermapbox(
                    lat=[lat],
                    lon=[lon],
                    mode="markers",
                    marker={"size": 45, "color": "rgba(220,38,38,0.18)"},
                    name="Impact Zone",
                )
            )

        for index, key in enumerate(("route_1", "route_2", "route_3")):
            route = route_data.get(key)
            coordinates = route.get("coordinates", []) if route else []
            if not coordinates:
                continue
            fig.add_trace(
                go.Scattermapbox(
                    lat=[point[0] for point in coordinates],
                    lon=[point[1] for point in coordinates],
                    mode="lines+markers",
                    line={"width": 5, "color": self.colors[index]},
                    marker={"size": 7, "color": self.colors[index]},
                    name=route.get("route_name", key),
                )
            )

        center = self._center(route_data, event_location)
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox={"center": {"lat": center[0], "lon": center[1]}, "zoom": 11},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=520,
            legend={"orientation": "h"},
        )
        return fig

    @staticmethod
    def _center(route_data: dict[str, Any], event_location: tuple[float, float] | None) -> tuple[float, float]:
        points = []
        if event_location and event_location[0] is not None and event_location[1] is not None:
            points.append(event_location)
        for key in ("route_1", "route_2", "route_3"):
            route = route_data.get(key) or {}
            points.extend(route.get("coordinates", []))
        if not points:
            return (12.9716, 77.5946)
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
