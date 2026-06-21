from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventTimelineSimulator:
    interval_minutes: int = 15

    def simulate(self, impact_score: float, duration_minutes: float) -> list[dict[str, float]]:
        duration = max(float(duration_minutes), float(self.interval_minutes))
        checkpoints = sorted({0, self.interval_minutes, int(duration * 0.35), int(duration * 0.65), int(duration)})
        timeline = []
        for minute in checkpoints:
            progress = minute / duration if duration else 1.0
            if progress <= 0.2:
                impact = impact_score * (0.35 + 2.5 * progress)
            elif progress <= 0.7:
                impact = impact_score * (0.9 + 0.1 * (1 - abs(progress - 0.45)))
            else:
                impact = impact_score * max(0.15, 1 - ((progress - 0.7) / 0.3) * 0.85)
            timeline.append({"minute": int(minute), "impact": round(min(100.0, max(0.0, impact)), 2)})
        return timeline
