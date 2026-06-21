from __future__ import annotations

from pathlib import Path

from event_intelligence.src.common.paths import MODELS_DIR, discover_dataset
from event_intelligence.src.common.serialization import load_pickle
from event_intelligence.src.action_plan import ActionPlanGenerator
from event_intelligence.src.dispatch import DispatchEngine
from event_intelligence.src.diversion_planner import DiversionPlanner
from event_intelligence.src.explainability import ExplainableAIEngine
from event_intelligence.src.hotspot_detection import HotspotDiscoveryEngine
from event_intelligence.src.officer_allocation import OfficerAllocationEngine
from event_intelligence.src.prediction import DurationPredictionEngine, ImpactPredictionEngine
from event_intelligence.src.preprocessing import EventPreprocessor
from event_intelligence.src.resource_checker import ResourceSufficiencyChecker
from event_intelligence.src.resource_planner import ResourcePlanner
from event_intelligence.src.risk import RiskCorridorEngine
from event_intelligence.src.route_visualization import RouteVisualizer
from event_intelligence.src.similarity_search import SimilarIncidentSearch
from event_intelligence.src.timeline_simulator import EventTimelineSimulator


def load_or_train(models_dir: str | Path = MODELS_DIR, dataset_path: str | Path | None = None) -> dict:
    models_dir = Path(models_dir)
    required = [
        "preprocessor.pkl",
        "duration_model.pkl",
        "impact_model.pkl",
        "hotspots.pkl",
        "risk_corridors.pkl",
        "similar_incidents.pkl",
        "diversion_graph.pkl",
    ]
    if not all((models_dir / name).exists() for name in required):
        from event_intelligence.src.training.train_models import train_all

        train_all(dataset_path=dataset_path, models_dir=models_dir)

    return {
        "preprocessor": load_pickle(models_dir / "preprocessor.pkl"),
        "duration": DurationPredictionEngine.load(models_dir / "duration_model.pkl"),
        "impact": ImpactPredictionEngine.load(models_dir / "impact_model.pkl"),
        "hotspots": load_pickle(models_dir / "hotspots.pkl"),
        "risk": load_pickle(models_dir / "risk_corridors.pkl"),
        "similar": SimilarIncidentSearch.load(models_dir / "similar_incidents.pkl"),
        "diversion": load_pickle(models_dir / "diversion_graph.pkl"),
        "resources": ResourcePlanner.from_config(),
        "dispatch": DispatchEngine.from_csv(),
        "officer_allocation": OfficerAllocationEngine(),
        "resource_checker": ResourceSufficiencyChecker(),
        "timeline": EventTimelineSimulator(),
        "action_plan": ActionPlanGenerator(),
        "explainability": ExplainableAIEngine(),
        "route_visualizer": RouteVisualizer(),
        "dataset": discover_dataset(dataset_path),
    }


def build_event(preprocessor: EventPreprocessor, **values) -> dict:
    return preprocessor.transform([values])[0]
