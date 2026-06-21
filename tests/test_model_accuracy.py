from pathlib import Path

from event_intelligence.src.common.paths import discover_dataset, MODELS_DIR
from event_intelligence.src.common.serialization import load_pickle
from event_intelligence.src.prediction import DurationPredictionEngine, ImpactPredictionEngine
from event_intelligence.src.preprocessing import EventPreprocessor


def test_model_accuracy_report():
    dataset = discover_dataset()
    preprocessor = EventPreprocessor()
    rows = preprocessor.load_and_transform(dataset)

    duration_rows = [row for row in rows if row.get("duration_minutes") is not None]
    impact_rows = [row for row in rows if row.get("impact_score") is not None]

    assert duration_rows, "No duration rows available for evaluation"
    assert impact_rows, "No impact rows available for evaluation"

    models_dir = Path(MODELS_DIR)
    duration_engine = DurationPredictionEngine.load(models_dir / "duration_model.pkl")
    impact_engine = ImpactPredictionEngine.load(models_dir / "impact_model.pkl")

    duration_metrics = duration_engine.evaluate(duration_rows)
    impact_metrics = impact_engine.evaluate(impact_rows)

    print("Duration model backend:", duration_engine.backend_name)
    print("Duration metrics:", duration_metrics)
    print("Impact model backend:", impact_engine.backend_name)
    print("Impact metrics:", impact_metrics)

    assert "mae" in duration_metrics and "rmse" in duration_metrics and "r2" in duration_metrics
    assert "mae" in impact_metrics and "rmse" in impact_metrics and "r2" in impact_metrics


if __name__ == "__main__":
    test_model_accuracy_report()
