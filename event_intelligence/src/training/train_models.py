from __future__ import annotations

import argparse
import json
from pathlib import Path

from event_intelligence.src.common.paths import MODELS_DIR, discover_dataset
from event_intelligence.src.common.serialization import save_pickle
from event_intelligence.src.diversion_planner import DiversionPlanner
from event_intelligence.src.hotspot_detection import HotspotDiscoveryEngine
from event_intelligence.src.prediction import DurationPredictionEngine, ImpactPredictionEngine
from event_intelligence.src.preprocessing import EventPreprocessor
from event_intelligence.src.risk import RiskCorridorEngine
from event_intelligence.src.similarity_search import SimilarIncidentSearch


def train_all(
    dataset_path: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    require_ml_backends: bool = False,
) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dataset = discover_dataset(dataset_path)

    preprocessor = EventPreprocessor()
    rows = preprocessor.load_and_transform(dataset)
    save_pickle(preprocessor, models_dir / "preprocessor.pkl")

    duration = DurationPredictionEngine().fit(rows)
    if require_ml_backends and duration.backend_model is None:
        raise RuntimeError(
            f"LightGBM backend unavailable: {duration.backend_error or 'LightGBM training failed or was unavailable.'}"
        )
    duration.save(models_dir / "duration_model.pkl")

    impact = ImpactPredictionEngine().fit(rows)
    if require_ml_backends and impact.backend_model is None:
        raise RuntimeError(
            f"CatBoost backend unavailable: {impact.backend_error or 'CatBoost training failed or was unavailable.'}"
        )
    impact.save(models_dir / "impact_model.pkl")

    hotspots = HotspotDiscoveryEngine().fit(rows)
    save_pickle(hotspots, models_dir / "hotspots.pkl")
    hotspots.save_geojson(models_dir / "hotspots.geojson")
    hotspots.create_folium_map(models_dir / "hotspot_map.html")

    risk = RiskCorridorEngine().fit(rows)
    save_pickle(risk, models_dir / "risk_corridors.pkl")

    similar = SimilarIncidentSearch().fit(rows)
    similar.save(models_dir / "similar_incidents.pkl")

    diversion = DiversionPlanner().fit(rows)
    save_pickle(diversion, models_dir / "diversion_graph.pkl")

    report = {
        "dataset": str(dataset),
        "row_count": len(rows),
        "duration_metrics": duration.metrics_,
        "duration_backend": duration.backend_name,
        "duration_backend_error": duration.backend_error,
        "impact_metrics": impact.metrics_,
        "impact_backend": impact.backend_name,
        "impact_backend_error": impact.backend_error,
        "hotspot_count": len(hotspots.hotspot_stats),
        "top_hotspots": hotspots.get_hotspots(10),
        "top_corridors": risk.top_corridors(10),
    }
    (models_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EIIP prototype models and indexes.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the raw event CSV.")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Directory for trained artifacts.")
    parser.add_argument(
        "--require-ml-backends",
        action="store_true",
        help="Fail instead of falling back unless LightGBM and CatBoost train successfully.",
    )
    args = parser.parse_args()
    report = train_all(args.dataset, args.models_dir, require_ml_backends=args.require_ml_backends)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
