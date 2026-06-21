from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx


@dataclass
class DiversionPlanner:
    """Junction graph for road-closure diversion planning.

    The planner can be populated from OSMnx when that optional dependency and
    network access are available. The persisted project artifact still uses the
    historical NetworkX junction graph, so all advanced route scoring also works
    on the existing model without retraining.
    """

    graph: nx.Graph = field(default_factory=nx.Graph)

    def fit(self, rows: Iterable[dict]) -> "DiversionPlanner":
        by_corridor: dict[str, list[dict]] = {}
        for row in rows:
            junction = row.get("junction")
            corridor = row.get("corridor")
            if not junction or junction == "Unknown" or not corridor or corridor == "Unknown":
                continue
            by_corridor.setdefault(str(corridor), []).append(row)

        for corridor, events in by_corridor.items():
            unique: dict[str, dict] = {}
            for event in events:
                unique.setdefault(str(event["junction"]), event)
            ordered = sorted(
                unique.values(),
                key=lambda r: (
                    r.get("latitude") if r.get("latitude") is not None else 0,
                    r.get("longitude") if r.get("longitude") is not None else 0,
                ),
            )
            for row in ordered:
                self.graph.add_node(
                    row["junction"],
                    corridor=corridor,
                    latitude=row.get("latitude"),
                    longitude=row.get("longitude"),
                )
            for left, right in zip(ordered, ordered[1:]):
                distance = _distance_weight(left, right)
                self.graph.add_edge(left["junction"], right["junction"], corridor=corridor, weight=distance)

        # Add transfer edges between nearby junctions from different corridors so
        # Dijkstra can still produce a usable prototype route without inventing
        # unrealistically cheap jumps across the city.
        nodes = list(self.graph.nodes)
        for idx, left in enumerate(nodes):
            for right in nodes[idx + 1 : idx + 6]:
                if not self.graph.has_edge(left, right):
                    distance = _node_distance_weight(self.graph.nodes[left], self.graph.nodes[right])
                    self.graph.add_edge(left, right, corridor="transfer", weight=round(distance * 1.15, 3))
        return self

    def plan(
        self,
        origin_junction: str,
        destination_junction: str,
        affected_corridor: str | None = None,
        road_closure: bool = True,
    ) -> dict:
        if origin_junction not in self.graph or destination_junction not in self.graph:
            return {
                "alternative_route": [],
                "estimated_extra_distance": 0.0,
                "message": "Origin or destination junction is not available in the simplified graph.",
            }

        graph = self.graph.copy()
        if road_closure and affected_corridor:
            for u, v, data in list(graph.edges(data=True)):
                if data.get("corridor") == affected_corridor:
                    graph.remove_edge(u, v)
        try:
            route = nx.shortest_path(graph, origin_junction, destination_junction, weight="weight")
            route_distance = nx.shortest_path_length(graph, origin_junction, destination_junction, weight="weight")
            baseline = nx.shortest_path_length(self.graph, origin_junction, destination_junction, weight="weight")
        except nx.NetworkXNoPath:
            return {"alternative_route": [], "estimated_extra_distance": 0.0, "message": "No diversion route found."}

        return {
            "alternative_route": route,
            "estimated_extra_distance": round(max(0.0, route_distance - baseline), 2),
            "message": "Diversion route generated.",
        }

    def plan_multiple(
        self,
        origin_junction: str,
        destination_junction: str,
        affected_corridor: str | None = None,
        road_closure: bool = True,
        corridor_risks: dict[str, float] | None = None,
        blocked_segments: list[tuple[str, str]] | None = None,
    ) -> dict:
        if origin_junction not in self.graph or destination_junction not in self.graph:
            return {
                "route_1": {},
                "route_2": {},
                "route_3": {},
                "message": "Origin or destination junction is not available in the simplified graph.",
            }

        graph = self.graph.copy()
        if road_closure and affected_corridor:
            for u, v, data in list(graph.edges(data=True)):
                if data.get("corridor") == affected_corridor:
                    graph.remove_edge(u, v)
        for left, right in blocked_segments or []:
            if graph.has_edge(left, right):
                graph.remove_edge(left, right)

        try:
            candidates = []
            for path in nx.shortest_simple_paths(graph, origin_junction, destination_junction, weight="weight"):
                distance = self._path_distance(graph, path)
                risk = self._path_risk(graph, path, corridor_risks or {})
                junction_count = max(0, len(path) - 2)
                delay = max(1, round(distance * 2.4 + risk / 18 + junction_count * 0.8))
                score = distance + delay * 0.35 + risk * 0.08
                candidates.append(
                    {
                        "path": path,
                        "distance_km": round(distance, 2),
                        "estimated_delay_min": delay,
                        "junction_count": junction_count,
                        "corridor_risk": round(risk, 2),
                        "risk_score": round(risk, 2),
                        "route_score": round(score, 2),
                    }
                )
                if len(candidates) >= 8:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"route_1": {}, "route_2": {}, "route_3": {}, "message": "No diversion route found."}

        ordered = sorted(candidates, key=lambda item: item["route_score"])[:3]
        labels = [
            ("Primary Route", "Cars"),
            ("Alternative Route", "Heavy Vehicles"),
            ("Emergency Route", "Emergency Vehicles"),
        ]
        result = {"message": "Diversion routes generated."}
        for index, route in enumerate(ordered):
            route_name, suitability = labels[index]
            route.update(
                {
                    "route_name": route_name,
                    "suitability": suitability if index < 2 else self._emergency_suitability(route),
                    "coordinates": self.route_coordinates(route["path"]),
                    "status": "Active" if index == 0 else "Standby",
                    "why_selected": self._route_explanation(route, ordered, index),
                }
            )
            result[f"route_{index + 1}"] = route
        for index in range(len(ordered), 3):
            result[f"route_{index + 1}"] = {}
        return result

    def load_osmnx_graph(
        self,
        place_name: str = "Bengaluru, Karnataka, India",
        network_type: str = "drive",
        simplify: bool = True,
    ) -> bool:
        """Populate the graph from OpenStreetMap via OSMnx when available.

        This method is intentionally opt-in because fetching a live city graph
        needs network access and can be slow. It returns False when OSMnx is not
        installed or the graph cannot be fetched.
        """
        try:
            import osmnx as ox  # type: ignore
        except Exception:
            return False

        try:
            osm_graph = ox.graph_from_place(place_name, network_type=network_type, simplify=simplify)
        except Exception:
            return False

        graph = nx.Graph()
        for node, data in osm_graph.nodes(data=True):
            graph.add_node(str(node), latitude=data.get("y"), longitude=data.get("x"), corridor="osm")
        for left, right, data in osm_graph.edges(data=True):
            distance_km = float(data.get("length", 1000.0)) / 1000.0
            graph.add_edge(str(left), str(right), corridor=str(data.get("name") or "osm"), weight=max(0.05, distance_km))
        if graph.number_of_nodes() < 2:
            return False
        self.graph = graph
        return True

    def route_edges(self, route: list[str]) -> list[dict]:
        edges = []
        for left, right in zip(route, route[1:]):
            data = self.graph.get_edge_data(left, right) or {}
            edges.append({"from": left, "to": right, "corridor": data.get("corridor"), "weight": data.get("weight")})
        return edges

    def route_coordinates(self, route: list[str]) -> list[tuple[float, float]]:
        coordinates = []
        for node in route:
            data = self.graph.nodes.get(node, {})
            lat = data.get("latitude")
            lon = data.get("longitude")
            if lat is not None and lon is not None:
                coordinates.append((float(lat), float(lon)))
        return coordinates

    @staticmethod
    def _path_distance(graph: nx.Graph, path: list[str]) -> float:
        return sum(float((graph.get_edge_data(left, right) or {}).get("weight", 1.0)) for left, right in zip(path, path[1:]))

    @staticmethod
    def _path_risk(graph: nx.Graph, path: list[str], corridor_risks: dict[str, float]) -> float:
        risks = []
        for left, right in zip(path, path[1:]):
            corridor = (graph.get_edge_data(left, right) or {}).get("corridor")
            if corridor:
                risks.append(float(corridor_risks.get(corridor, 0.0)))
        return sum(risks) / len(risks) if risks else 0.0

    @staticmethod
    def _emergency_suitability(route: dict) -> str:
        if route.get("estimated_delay_min", 99) <= 5 and route.get("risk_score", 100) <= 40:
            return "Emergency Vehicles"
        return "Backup Route"

    @staticmethod
    def _route_explanation(route: dict, ordered: list[dict], index: int) -> list[str]:
        if not ordered:
            return ["Route selected from available connected road graph."]
        lowest_risk = min(item.get("corridor_risk", 0) for item in ordered)
        lowest_delay = min(item.get("estimated_delay_min", 0) for item in ordered)
        fewest_junctions = min(item.get("junction_count", 0) for item in ordered)
        reasons = []
        if route.get("corridor_risk", 0) <= lowest_risk:
            reasons.append("Lowest corridor risk among candidate routes")
        if route.get("estimated_delay_min", 0) <= lowest_delay:
            reasons.append("Lowest predicted delay")
        if route.get("junction_count", 0) <= fewest_junctions:
            reasons.append("Fewest junction transitions")
        if index == 2:
            reasons.append("Reserved as emergency or backup movement path")
        if not reasons:
            reasons.append("Balanced distance, delay, junction count, and corridor risk")
        return reasons


def _distance_weight(left: dict, right: dict) -> float:
    if left.get("latitude") is None or right.get("latitude") is None:
        return 1.0
    lat_delta = float(left["latitude"]) - float(right["latitude"])
    lon_delta = float(left["longitude"]) - float(right["longitude"])
    return max(0.5, ((lat_delta * lat_delta + lon_delta * lon_delta) ** 0.5) * 111)


def _node_distance_weight(left: dict, right: dict) -> float:
    if left.get("latitude") is None or right.get("latitude") is None:
        return 3.0
    return _distance_weight(left, right)
