from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from event_intelligence.src.common.paths import DATA_DIR


EARTH_RADIUS_KM = 6371.0088


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [a_lat, a_lon, b_lat, b_lon])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


@dataclass
class DispatchEngine:
    stations: list[dict[str, Any]] = field(default_factory=list)
    average_speed_kmph: float = 28.0

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> "DispatchEngine":
        path = Path(path) if path else DATA_DIR / "police_stations.csv"
        stations = []
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    try:
                        stations.append(
                            {
                                "station_name": row["station_name"],
                                "latitude": float(row["latitude"]),
                                "longitude": float(row["longitude"]),
                                "available_officers": int(float(row["available_officers"])),
                            }
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        return cls(stations=stations)

    def assign(
        self,
        event_latitude: float | None,
        event_longitude: float | None,
        required_officers: int,
    ) -> dict[str, Any]:
        if event_latitude is None or event_longitude is None:
            return {
                "station_name": "Location unavailable",
                "distance_km": 0.0,
                "eta_minutes": 0,
                "assigned_officers": 0,
                "available_officers": 0,
                "status": "UNASSIGNED",
            }
        if not self.stations:
            return {
                "station_name": "No station data",
                "distance_km": 0.0,
                "eta_minutes": 0,
                "assigned_officers": 0,
                "available_officers": 0,
                "status": "UNASSIGNED",
            }

        ranked = sorted(
            (
                (
                    haversine_km(event_latitude, event_longitude, station["latitude"], station["longitude"]),
                    station,
                )
                for station in self.stations
            ),
            key=lambda item: (item[0], -item[1]["available_officers"]),
        )
        distance, station = ranked[0]
        assigned = min(required_officers, station["available_officers"])
        eta = math.ceil((distance / self.average_speed_kmph) * 60)
        return {
            "station_name": station["station_name"],
            "distance_km": round(distance, 2),
            "eta_minutes": eta,
            "assigned_officers": assigned,
            "available_officers": station["available_officers"],
            "status": "ASSIGNED" if assigned else "UNASSIGNED",
        }
