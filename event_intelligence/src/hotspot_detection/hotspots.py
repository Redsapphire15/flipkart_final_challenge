from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable


EARTH_RADIUS_KM = 6371.0088


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [a_lat, a_lon, b_lat, b_lon])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


@dataclass
class HotspotDiscoveryEngine:
    """Identify recurring incident hotspots with DBSCAN-style clustering."""

    eps_km: float = 0.45
    min_samples: int = 8
    clustered_events: list[dict] = field(default_factory=list)
    hotspot_stats: list[dict] = field(default_factory=list)
    _grid: dict[tuple[int, int], list[int]] = field(default_factory=dict, init=False, repr=False)
    _cell_degrees: float = field(default=0.005, init=False, repr=False)

    def fit(self, rows: Iterable[dict]) -> "HotspotDiscoveryEngine":
        events = [
            dict(row)
            for row in rows
            if isinstance(row.get("latitude"), (float, int)) and isinstance(row.get("longitude"), (float, int))
        ]
        self._build_grid(events)
        labels = self._dbscan(events)
        self.clustered_events = []
        for row, label in zip(events, labels):
            enriched = dict(row)
            enriched["hotspot_cluster"] = label
            self.clustered_events.append(enriched)
        self.hotspot_stats = self._cluster_stats(self.clustered_events)
        return self

    def _dbscan(self, events: list[dict]) -> list[int]:
        labels = [-99] * len(events)
        cluster_id = 0
        for index in range(len(events)):
            if labels[index] != -99:
                continue
            neighbors = self._neighbors(events, index)
            if len(neighbors) < self.min_samples:
                labels[index] = -1
                continue
            labels[index] = cluster_id
            seeds = [n for n in neighbors if n != index]
            while seeds:
                current = seeds.pop()
                if labels[current] == -1:
                    labels[current] = cluster_id
                if labels[current] != -99:
                    continue
                labels[current] = cluster_id
                current_neighbors = self._neighbors(events, current)
                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if labels[neighbor] in {-99, -1} and neighbor not in seeds:
                            seeds.append(neighbor)
            cluster_id += 1
        return labels

    def _neighbors(self, events: list[dict], index: int) -> list[int]:
        row = events[index]
        cell = self._cell(row["latitude"], row["longitude"])
        candidates: list[int] = []
        for lat_offset in (-1, 0, 1):
            for lon_offset in (-1, 0, 1):
                candidates.extend(self._grid.get((cell[0] + lat_offset, cell[1] + lon_offset), []))
        return [
            idx
            for idx in candidates
            for other in [events[idx]]
            if haversine_km(row["latitude"], row["longitude"], other["latitude"], other["longitude"]) <= self.eps_km
        ]

    def _build_grid(self, events: list[dict]) -> None:
        self._cell_degrees = max(self.eps_km / 111.0, 0.001)
        self._grid = {}
        for index, row in enumerate(events):
            self._grid.setdefault(self._cell(row["latitude"], row["longitude"]), []).append(index)

    def _cell(self, latitude: float, longitude: float) -> tuple[int, int]:
        return (int(float(latitude) / self._cell_degrees), int(float(longitude) / self._cell_degrees))

    def _cluster_stats(self, events: list[dict]) -> list[dict]:
        buckets: dict[int, list[dict]] = {}
        for row in events:
            label = row.get("hotspot_cluster", -1)
            if label == -1:
                continue
            buckets.setdefault(label, []).append(row)

        stats = []
        for label, rows in buckets.items():
            durations = [float(r["duration_minutes"]) for r in rows if r.get("duration_minutes") is not None]
            stats.append(
                {
                    "cluster": label,
                    "event_count": len(rows),
                    "center_latitude": round(mean(float(r["latitude"]) for r in rows), 6),
                    "center_longitude": round(mean(float(r["longitude"]) for r in rows), 6),
                    "avg_duration": round(mean(durations), 2) if durations else 0.0,
                    "top_corridor": _mode(rows, "corridor"),
                    "top_cause": _mode(rows, "event_cause"),
                }
            )
        return sorted(stats, key=lambda item: item["event_count"], reverse=True)

    def get_hotspots(self, limit: int | None = None) -> list[dict]:
        return self.hotspot_stats[:limit] if limit else list(self.hotspot_stats)

    def save_geojson(self, path: str | Path) -> Path:
        features = []
        for row in self.clustered_events:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                    "properties": {
                        "cluster": row.get("hotspot_cluster", -1),
                        "event_cause": row.get("event_cause"),
                        "corridor": row.get("corridor"),
                    },
                }
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
        return path

    def create_folium_map(self, path: str | Path | None = None):
        try:
            import folium  # type: ignore
            from folium.plugins import HeatMap  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            if path:
                fallback = self._html_fallback()
                Path(path).write_text(fallback, encoding="utf-8")
                return path
            raise RuntimeError("Install folium to render interactive hotspot maps.") from exc

        center = self._map_center()
        fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
        heat_points = [[r["latitude"], r["longitude"]] for r in self.clustered_events]
        if heat_points:
            HeatMap(heat_points, radius=14, blur=18).add_to(fmap)
        for stat in self.hotspot_stats[:50]:
            folium.CircleMarker(
                location=[stat["center_latitude"], stat["center_longitude"]],
                radius=max(5, min(20, stat["event_count"] / 8)),
                popup=f"Cluster {stat['cluster']}: {stat['event_count']} events",
                color="#c2410c",
                fill=True,
                fill_opacity=0.7,
            ).add_to(fmap)
        if path:
            fmap.save(str(path))
            return path
        return fmap

    def _map_center(self) -> list[float]:
        if not self.clustered_events:
            return [12.9716, 77.5946]
        return [
            mean(float(r["latitude"]) for r in self.clustered_events),
            mean(float(r["longitude"]) for r in self.clustered_events),
        ]

    def _html_fallback(self) -> str:
        rows = "\n".join(
            f"<tr><td>{s['cluster']}</td><td>{s['event_count']}</td><td>{s['center_latitude']}</td>"
            f"<td>{s['center_longitude']}</td><td>{s['top_corridor']}</td></tr>"
            for s in self.hotspot_stats
        )
        return (
            "<html><body><h1>Hotspot Summary</h1><table border='1'>"
            "<tr><th>Cluster</th><th>Events</th><th>Lat</th><th>Lon</th><th>Corridor</th></tr>"
            f"{rows}</table></body></html>"
        )


def _mode(rows: list[dict], column: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column) or "Unknown")
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "Unknown"
