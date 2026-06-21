from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable


MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a"}
PEAK_HOURS = set(range(7, 11)) | set(range(17, 22))

FEATURE_COLUMNS = [
    "event_type",
    "event_cause",
    "zone",
    "corridor",
    "junction",
    "hour",
    "priority",
    "requires_road_closure",
]

DURATION_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "latitude",
    "longitude",
    "veh_type",
    "direction",
    "police_station",
    # ── engineered features (NO duration-derived values here) ────────────
    # impact_score is intentionally excluded: it encodes duration_minutes,
    # which is the prediction target — including it causes data leakage.
    "priority_closure_score",  # leakage-free proxy: priority + road closure only
    "hour_sin",                # cyclic encoding of hour
    "hour_cos",
    "dow_sin",                 # cyclic encoding of day-of-week
    "dow_cos",
    "month_sin",               # cyclic encoding of month
    "month_cos",
    "event_type_cause",        # interaction: type × cause
    "zone_event_type",         # interaction: zone × type
    "corridor_event_type",     # interaction: corridor × type
]


def _clean_text(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return default
    return text


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value, "false").lower()
    return text in {"true", "1", "yes", "y", "road_closure", "closure"}


def _parse_float(value: Any) -> float | None:
    text = _clean_text(value, "")
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or parsed == 0:
        return None
    return parsed


def _parse_datetime(value: Any) -> datetime | None:
    text = _clean_text(value, "")
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    if text.endswith("+00"):
        text = f"{text}:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _percentile(values: list[float], pct: float, default: float = 60.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return float(ordered[index]) or default


def _cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a periodic value as (sin, cos) so the model sees the wraparound."""
    angle = 2 * math.pi * value / period
    return round(math.sin(angle), 6), round(math.cos(angle), 6)


@dataclass
class EventPreprocessor:
    """Reusable data cleaning and feature engineering pipeline.

    The project can run in minimal environments, so the canonical internal
    representation is a list of dictionaries. `to_dataframe` is available when
    pandas is installed.
    """

    duration_cap_minutes: float | None = None
    min_duration_minutes: float = 5.0
    max_operational_duration_minutes: float = 360.0
    columns_seen: list[str] = field(default_factory=list)

    def read_csv(self, path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            self.columns_seen = list(reader.fieldnames or [])
            return [dict(row) for row in reader]

    def fit(self, rows: Iterable[dict[str, Any]]) -> "EventPreprocessor":
        durations = []
        for row in rows:
            duration = self._raw_duration(row)
            if duration is not None:
                durations.append(duration)
        p90_duration = _percentile(durations, 0.90, default=120.0)
        self.duration_cap_minutes = min(
            self.max_operational_duration_minutes,
            max(90.0, p90_duration * 1.4),
        )
        return self

    def transform(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.duration_cap_minutes is None:
            rows = list(rows)
            self.fit(rows)
        return [self._clean_row(row) for row in rows]

    def fit_transform(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = list(rows)
        self.fit(rows)
        return self.transform(rows)

    def load_and_transform(self, path: str | Path) -> list[dict[str, Any]]:
        return self.fit_transform(self.read_csv(path))

    def to_dataframe(self, rows: list[dict[str, Any]]):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install pandas and numpy to use dataframe output.") from exc
        return pd.DataFrame(rows)

    def _clean_row(self, row: dict[str, Any]) -> dict[str, Any]:
        start_dt = _parse_datetime(row.get("start_datetime")) or _parse_datetime(row.get("created_date"))
        raw_duration = self._raw_duration(row, start_dt=start_dt)
        duration = self._compute_duration(row, start_dt=start_dt, raw_duration=raw_duration)
        priority = _clean_text(row.get("priority"), "Low").title()
        closure = _parse_bool(row.get("requires_road_closure"))
        latitude = _parse_float(row.get("latitude"))
        longitude = _parse_float(row.get("longitude"))

        event_type = _clean_text(row.get("event_type"), "Unknown").lower()
        event_cause = _clean_text(row.get("event_cause"), "Unknown").lower()
        zone = _clean_text(row.get("zone"), "Unknown")
        corridor = _clean_text(row.get("corridor"), "Unknown")

        # ── cyclic temporal encodings ────────────────────────────────────
        hour = start_dt.hour if start_dt else 12
        dow = start_dt.weekday() if start_dt else 0
        month = start_dt.month if start_dt else 6
        hour_sin, hour_cos = _cyclic_encode(hour, 24)
        dow_sin, dow_cos = _cyclic_encode(dow, 7)
        month_sin, month_cos = _cyclic_encode(month - 1, 12)   # month 1-12 → 0-11

        cleaned = dict(row)
        cleaned.update(
            {
                "event_type": event_type,
                "event_cause": event_cause,
                "zone": zone,
                "junction": _clean_text(row.get("junction"), "Unknown"),
                "corridor": corridor,
                "priority": priority if priority in {"High", "Low"} else priority,
                "requires_road_closure": closure,
                "latitude": latitude,
                "longitude": longitude,
                "start_datetime_parsed": start_dt,
                "raw_duration_minutes": raw_duration,
                "duration_minutes": duration,
                "duration_was_capped": bool(
                    raw_duration is not None and duration is not None and raw_duration > duration
                ),
                "hour": hour,
                "day_of_week": dow,
                "month": month,
                "is_weekend": start_dt.weekday() >= 5 if start_dt else False,
                "is_peak_hour": start_dt.hour in PEAK_HOURS if start_dt else False,
                # ── cyclic features ──────────────────────────────────────
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                # ── interaction features ─────────────────────────────────
                "event_type_cause": f"{event_type}|{event_cause}",
                "zone_event_type": f"{zone}|{event_type}",
                "corridor_event_type": f"{corridor}|{event_type}",
            }
        )
        cleaned["impact_score"] = self.compute_impact_score(cleaned)
        # Leakage-free score: only uses information available at event creation time.
        # Used as a model feature; impact_score (which encodes duration) is kept for
        # downstream reporting but must NOT be passed to any duration model.
        cleaned["priority_closure_score"] = self.compute_priority_closure_score(cleaned)
        return cleaned

    def _compute_duration(
        self,
        row: dict[str, Any],
        start_dt: datetime | None = None,
        raw_duration: float | None = None,
    ) -> float | None:
        raw = self._raw_duration(row, start_dt=start_dt) if raw_duration is None else raw_duration
        if raw is None:
            return None
        cap = self.duration_cap_minutes or self.max_operational_duration_minutes
        return round(max(self.min_duration_minutes, min(raw, cap)), 2)

    def _raw_duration(self, row: dict[str, Any], start_dt: datetime | None = None) -> float | None:
        start = start_dt or _parse_datetime(row.get("start_datetime")) or _parse_datetime(row.get("created_date"))
        if not start:
            return None
        for column in ("resolved_datetime", "closed_datetime", "end_datetime", "modified_datetime"):
            end = _parse_datetime(row.get(column))
            if not end:
                continue
            minutes = (end - start).total_seconds() / 60
            if 0 <= minutes <= 7 * 24 * 60:
                return round(minutes, 2)
        return None

    def compute_impact_score(self, row: dict[str, Any]) -> float:
        cap = self.duration_cap_minutes or 120.0
        duration = row.get("duration_minutes")
        if duration is None:
            duration_component = 0.5
        else:
            duration_component = min(float(duration), cap) / cap

        road_closure_weight = 1.0 if row.get("requires_road_closure") else 0.0
        priority = _clean_text(row.get("priority"), "Low").lower()
        priority_weight = 1.0 if priority == "high" else 0.25
        score = 100 * (
            0.4 * duration_component
            + 0.3 * road_closure_weight
            + 0.3 * priority_weight
        )
        return round(max(0.0, min(100.0, score)), 2)

    def compute_priority_closure_score(self, row: dict[str, Any]) -> float:
        """
        Leakage-free feature for duration models.

        Uses only information available at event creation time:
        priority and whether a road closure is required.
        Does NOT touch duration_minutes or any resolved/closed timestamps.
        """
        road_closure_weight = 1.0 if row.get("requires_road_closure") else 0.0
        priority = _clean_text(row.get("priority"), "Low").lower()
        priority_weight = 1.0 if priority == "high" else 0.25
        score = 100 * (0.5 * road_closure_weight + 0.5 * priority_weight)
        return round(max(0.0, min(100.0, score)), 2)

    @staticmethod
    def feature_columns() -> list[str]:
        return list(FEATURE_COLUMNS)

    @staticmethod
    def median_duration(rows: Iterable[dict[str, Any]], default: float = 60.0) -> float:
        values = [float(r["duration_minutes"]) for r in rows if r.get("duration_minutes") is not None]
        return float(median(values)) if values else default