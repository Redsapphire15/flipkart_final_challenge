from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from event_intelligence.src.common.serialization import load_pickle, save_pickle
from event_intelligence.src.prediction.baseline import HierarchicalRegressor, regression_metrics
from event_intelligence.src.preprocessing.pipeline import FEATURE_COLUMNS

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


@dataclass
class ImpactPredictionEngine:
    """Predict operational impact score on a 0-100 scale."""

    model: HierarchicalRegressor = field(default_factory=lambda: HierarchicalRegressor("impact_score"))
    backend_model: Any | None = None
    backend_name: str = "hierarchical_baseline"
    backend_error: str | None = None
    cat_features: list[str] = field(default_factory=list)
    metrics_: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: Iterable[dict]) -> "ImpactPredictionEngine":
        rows = [r for r in rows if r.get("impact_score") is not None]
        if self._fit_catboost(rows):
            return self
        self.model.fit(rows)
        predicted = self.model.predict(rows)
        actual = [float(r["impact_score"]) for r in rows]
        self.metrics_ = regression_metrics(actual, predicted)
        return self

    def predict_one(self, event: dict) -> float:
        if self.backend_model is not None:
            try:
                import pandas as pd  # type: ignore

                frame = pd.DataFrame([event])
                prediction = float(self.backend_model.predict(frame[FEATURE_COLUMNS])[0])
                return round(max(0.0, min(100.0, prediction)), 2)
            except Exception:
                pass
        return round(max(0.0, min(100.0, self.model.predict_one(event))), 2)

    def predict(self, events: Iterable[dict]) -> list[float]:
        return [self.predict_one(event) for event in events]

    @staticmethod
    def risk_level(score: float) -> str:
        if score >= 80:
            return "Critical"
        if score >= 60:
            return "High"
        if score >= 40:
            return "Moderate"
        return "Low"

    def evaluate(self, rows: Iterable[dict]) -> dict[str, float]:
        rows = [r for r in rows if r.get("impact_score") is not None]
        return regression_metrics(
            [float(r["impact_score"]) for r in rows],
            self.predict(rows),
        )

    def save(self, path: str | Path) -> Path:
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ImpactPredictionEngine":
        return load_pickle(path)

    def _fit_catboost(self, rows: list[dict]) -> bool:
        try:
            import pandas as pd  # type: ignore
            from catboost import CatBoostRegressor  # type: ignore
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            return False

        if len(rows) < 50:
            self.backend_error = "Not enough rows to train CatBoost."
            return False

        df = pd.DataFrame(rows)
        feature_columns = [column for column in FEATURE_COLUMNS if column in df.columns]
        x = df[feature_columns].copy()
        for column in feature_columns:
            if column not in {"hour", "requires_road_closure"}:
                x[column] = x[column].fillna("Unknown").astype(str)
        y = df["impact_score"].astype(float)
        self.cat_features = [c for c in feature_columns if c not in {"hour", "requires_road_closure"}]
        model = CatBoostRegressor(
            iterations=350,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        model.fit(x, y, cat_features=self.cat_features)
        predicted = model.predict(x)
        self.backend_model = model
        self.backend_name = "catboost"
        self.metrics_ = {
            "mae": round(float(mean_absolute_error(y, predicted)), 3),
            "rmse": round(float(mean_squared_error(y, predicted) ** 0.5), 3),
            "r2": round(float(r2_score(y, predicted)), 3),
        }
        return True
