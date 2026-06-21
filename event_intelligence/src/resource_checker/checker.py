from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceSufficiencyChecker:
    def check(
        self,
        *,
        required_officers: int,
        required_barricades: int,
        available_officers: int,
        available_barricades: int,
    ) -> dict:
        officer_gap = max(0, required_officers - available_officers)
        barricade_gap = max(0, required_barricades - available_barricades)
        status = "SUFFICIENT" if officer_gap == 0 and barricade_gap == 0 else "INSUFFICIENT"
        return {
            "officer_gap": officer_gap,
            "barricade_gap": barricade_gap,
            "status": status,
            "available_officers": available_officers,
            "available_barricades": available_barricades,
        }
