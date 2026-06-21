from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from event_intelligence.src.common.serialization import load_pickle, save_pickle
from event_intelligence.src.prediction.baseline import HierarchicalRegressor, regression_metrics
from event_intelligence.src.preprocessing.pipeline import DURATION_FEATURE_COLUMNS


# ── Advanced OOF Cluster & Target Space Preprocessor ────────────────────────

def _compute_advanced_oof_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    target_col: str,
    splits: list[tuple[list[int], list[int]]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Computes out-of-fold geographical spatial cluster definitions, computes cluster target profiles,
    and applies smooth categorical target encoding to isolate core variance vectors.
    """
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    
    train_encoded = train_df.copy()
    val_encoded = val_df.copy()
    
    global_mean = train_df[target_col].mean()
    m_smoothing = 15.0

    # 1. Standard Smooth Target Encoding
    for col in categorical_cols:
        train_encoded[f"{col}_te_mean"] = global_mean
        val_encoded[f"{col}_te_mean"] = global_mean

    train_indices = train_df.index.tolist()
    for fold_train_pos, fold_val_pos in splits:
        f_train_idx = [train_indices[p] for p in fold_train_pos]
        f_val_idx = [train_indices[p] for p in fold_val_pos]
        f_train = train_df.loc[f_train_idx]

        for col in categorical_cols:
            stats = f_train.groupby(col)[target_col].agg(["count", "mean"])
            smooth_mean = (stats["count"] * stats["mean"] + m_smoothing * global_mean) / (stats["count"] + m_smoothing)
            train_encoded.loc[f_val_idx, f"{col}_te_mean"] = train_df.loc[f_val_idx, col].map(smooth_mean).fillna(global_mean)

    for col in categorical_cols:
        stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
        smooth_mean = (stats["count"] * stats["mean"] + m_smoothing * global_mean) / (stats["count"] + m_smoothing)
        val_encoded[f"{col}_te_mean"] = val_df[col].map(smooth_mean).fillna(global_mean)

    # 2. Geospatial Clustering Engine
    geo_cols = [c for c in ["latitude", "longitude"] if c in numeric_cols]
    n_clusters = 12
    
    if len(geo_cols) == 2:
        # Initialize columns
        train_encoded["geo_cluster_te_mean"] = global_mean
        val_encoded["geo_cluster_te_mean"] = global_mean
        
        # Fit clustering on complete training pool
        train_geo = train_df[geo_cols].fillna(train_df[geo_cols].median())
        val_geo = val_df[geo_cols].fillna(train_df[geo_cols].median())
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512, random_state=42)
        train_clusters = kmeans.fit_predict(train_geo)
        val_clusters = kmeans.predict(val_geo)
        
        train_encoded["__cluster_id"] = train_clusters
        val_encoded["__cluster_id"] = val_clusters
        
        # Compute out-of-fold target values per geographic neighborhood cluster
        for fold_train_pos, fold_val_pos in splits:
            f_train_idx = [train_indices[p] for p in fold_train_pos]
            f_val_idx = [train_indices[p] for p in fold_val_pos]
            f_train_c = train_encoded.loc[f_train_idx]
            
            c_stats = f_train_c.groupby("__cluster_id")[target_col].agg(["count", "mean"])
            c_smooth = (c_stats["count"] * c_stats["mean"] + m_smoothing * global_mean) / (c_stats["count"] + m_smoothing)
            train_encoded.loc[f_val_idx, "geo_cluster_te_mean"] = train_encoded.loc[f_val_idx, "__cluster_id"].map(c_smooth).fillna(global_mean)
            
        c_stats = train_encoded.groupby("__cluster_id")[target_col].agg(["count", "mean"])
        c_smooth = (c_stats["count"] * c_stats["mean"] + m_smoothing * global_mean) / (c_stats["count"] + m_smoothing)
        val_encoded["geo_cluster_te_mean"] = val_encoded["__cluster_id"].map(c_smooth).fillna(global_mean)
        
        train_encoded.drop(columns=["__cluster_id"], inplace=True)
        val_encoded.drop(columns=["__cluster_id"], inplace=True)

    return train_encoded, val_encoded, n_clusters


# ── Shuffling & Validation Framework ─────────────────────────────────────────

def _shuffled_kfold_indices(n: int, k: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_size = n // k
    splits = []
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n
        val_global = indices[val_start:val_end]
        train_global = indices[:val_start] + indices[val_end:]
        splits.append((train_global, val_global))
    return splits


def _oof_predict(
    engine: DurationPredictionEngine,
    rows: list[dict],
    categorical_columns: list[str],
    numeric_columns: list[str],
    k: int = 5,
) -> tuple[list[float], list[float], dict[str, float], dict[str, float]]:
    n = len(rows)
    lgbm_oof = [float("nan")] * n
    hier_oof = [float("nan")] * n
    df = pd.DataFrame(rows)
    
    y_raw = df["duration_minutes"].astype(float).values
    
    if engine.winsorize_quantile is not None:
        upper_bound = np.quantile(y_raw, engine.winsorize_quantile)
        y_eval = np.clip(y_raw, engine.min_duration_minutes, upper_bound)
    else:
        y_eval = y_raw

    if engine.target_transform == "log1p":
        y_transformed = np.log1p(y_eval)
    elif engine.target_transform == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=min(1500, len(rows)), output_distribution="normal", random_state=42)
        y_transformed = qt.fit_transform(y_eval.reshape(-1, 1)).flatten()
    else:
        y_transformed = y_eval

    splits = _shuffled_kfold_indices(n, k)

    for train_idx, val_idx in splits:
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        train_df["__target_temp"] = y_eval[train_idx]

        internal_splits = _shuffled_kfold_indices(len(train_idx), k=k)
        train_te, val_te, _ = _compute_advanced_oof_features(
            train_df, val_df, categorical_columns, numeric_columns, "__target_temp", internal_splits
        )

        active_features = [c for c in train_te.columns if c.endswith("_te_mean")] + numeric_columns

        fold_lgbm = engine._make_lgbm_pipeline(numeric=active_features)
        fold_y_train = y_transformed[train_idx]
        
        fold_lgbm.fit(train_te[active_features], fold_y_train)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            fold_preds = fold_lgbm.predict(val_te[active_features])

        if engine.target_transform == "log1p":
            fold_preds = np.expm1(fold_preds)
        elif engine.target_transform == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            local_qt = QuantileTransformer(n_quantiles=min(1500, len(train_idx)), output_distribution="normal", random_state=42)
            local_qt.fit(y_eval[train_idx].reshape(-1, 1))
            fold_preds = local_qt.inverse_transform(fold_preds.reshape(-1, 1)).flatten()

        for local_i, global_i in enumerate(val_idx):
            lgbm_oof[global_i] = float(fold_preds[local_i])

        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        h_fold = HierarchicalRegressor("duration_minutes", min_samples=engine.model.min_samples)
        h_fold.fit(train_rows)
        h_preds = h_fold.predict(val_rows)
        for local_i, global_i in enumerate(val_idx):
            hier_oof[global_i] = h_preds[local_i]

    def _metrics(preds: list[float]) -> dict[str, float]:
        pairs = [(float(y_eval[i]), preds[i]) for i in range(n) if not math.isnan(preds[i])]
        return regression_metrics([a for a, _ in pairs], [p for _, p in pairs])

    return lgbm_oof, hier_oof, _metrics(lgbm_oof), _metrics(hier_oof)


# ── Core Adaptive Ensemble Engine ───────────────────────────────────────────

@dataclass
class DurationPredictionEngine:
    """Production ML Engine extracting cluster target maps and using spatial-temporal splines."""

    model: HierarchicalRegressor = field(
        default_factory=lambda: HierarchicalRegressor("duration_minutes", min_samples=2)
    )
    backend_model: Any | None = None
    backend_name: str = "ensemble"
    backend_error: str | None = None
    target_transform: str = "quantile"  
    winsorize_quantile: float | None = 0.95  
    min_duration_minutes: float = 5.0
    max_duration_minutes: float = 360.0
    metrics_: dict[str, float] = field(default_factory=dict)
    candidate_metrics_: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_columns_: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)
    numeric_columns_: list[str] = field(default_factory=list)
    _oof_lgbm: list[float] = field(default_factory=list)
    _oof_hier: list[float] = field(default_factory=list)
    _quantile_transformer: Any | None = None
    _global_te_stats: dict[str, Any] = field(default_factory=dict)
    _global_te_priors: dict[str, float] = field(default_factory=dict)
    _kmeans_model: Any | None = None

    def fit(self, rows: Iterable[dict]) -> "DurationPredictionEngine":
        rows = [r for r in rows if r.get("duration_minutes") is not None]
        if not rows:
            return self
            
        actual = np.array([float(r["duration_minutes"]) for r in rows])
        
        if self.winsorize_quantile is not None:
            self.max_duration_minutes = float(np.quantile(actual, self.winsorize_quantile))
        else:
            self.max_duration_minutes = float(max(actual))

        y_eval = np.clip(actual, self.min_duration_minutes, self.max_duration_minutes)
        self.model.fit(rows)
        
        df = pd.DataFrame(rows)
        feature_columns = [c for c in DURATION_FEATURE_COLUMNS if c in df.columns]
        numeric_names = {
            "hour", "requires_road_closure", "day_of_week", "month",
            "is_weekend", "is_peak_hour", "latitude", "longitude",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "priority_closure_score",
        }
        self.categorical_columns_ = [c for c in feature_columns if c not in numeric_names]
        self.numeric_columns_ = [c for c in feature_columns if c in numeric_names]

        lgbm_ok = self._fit_lightgbm_pipeline(df, y_eval, rows)

        if not lgbm_ok:
            insample_preds = self.model.predict(rows)
            self.metrics_ = regression_metrics(list(y_eval), insample_preds)
            self.backend_name = "hierarchical"
            return self

        k = min(5, max(2, len(rows) // 100))
        lgbm_oof, hier_oof, lgbm_cv, hier_cv = _oof_predict(
            self, rows, self.categorical_columns_, self.numeric_columns_, k=k
        )
        self._oof_lgbm = lgbm_oof
        self._oof_hier = hier_oof

        self.candidate_metrics_["lightgbm_cv"] = {k_: round(v, 4) for k_, v in lgbm_cv.items()}
        self.candidate_metrics_["hierarchical_cv"] = {k_: round(v, 4) for k_, v in hier_cv.items()}

        ensemble_oof = [0.85 * a + 0.15 * b for a, b in zip(lgbm_oof, hier_oof)]
        ensemble_cv = {k_: round(v, 4) for k_, v in regression_metrics(list(y_eval), ensemble_oof).items()}
        self.candidate_metrics_["ensemble_cv"] = ensemble_cv

        best_name, best_oof = "hierarchical", hier_cv
        if self._is_better(lgbm_cv, best_oof):
            best_name, best_oof = "lightgbm", lgbm_cv
        if self._is_better(ensemble_cv, best_oof):
            best_name = "ensemble"

        self.backend_name = best_name
        self.metrics_ = {k_: round(v, 4) for k_, v in self.evaluate(rows).items()}
        return self

    def predict_one(self, event: dict) -> float:
        hierarchical = round(self._clip_prediction(self.model.predict_one(event)), 2)

        if self.backend_name == "hierarchical" or self.backend_model is None:
            return hierarchical

        lgbm_pred = self._lgbm_predict_one(event)
        if lgbm_pred is None:
            return hierarchical

        if self.backend_name == "ensemble":
            blended = 0.85 * lgbm_pred + 0.15 * hierarchical
            return round(self._clip_prediction(blended), 2)

        return round(self._clip_prediction(lgbm_pred), 2)

    def predict(self, events: Iterable[dict]) -> list[float]:
        return [self.predict_one(event) for event in events]

    def evaluate(self, rows: Iterable[dict]) -> dict[str, float]:
        rows = [r for r in rows if r.get("duration_minutes") is not None]
        actual = np.array([float(r["duration_minutes"]) for r in rows])
        if self.winsorize_quantile is not None:
            upper_bound = np.quantile(actual, self.winsorize_quantile)
            y_eval = np.clip(actual, self.min_duration_minutes, upper_bound)
        else:
            y_eval = actual
        return regression_metrics(list(y_eval), self.predict(rows))

    def save(self, path: str | Path) -> Path:
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DurationPredictionEngine":
        return load_pickle(path)

    # ── Pipeline Processing Core ──────────────────────────────────────────────

    def _lgbm_predict_one(self, event: dict) -> float | None:
        if self.backend_model is None:
            return None
        try:
            df_one = pd.DataFrame([event])
            g_mean = self._global_te_priors["mean"]

            for col in self.categorical_columns_:
                val = event.get(col)
                if col in self._global_te_stats and val in self._global_te_stats[col]:
                    df_one[f"{col}_te_mean"] = self._global_te_stats[col][val]
                else:
                    df_one[f"{col}_te_mean"] = g_mean

            df_one["geo_cluster_te_mean"] = g_mean
            geo_cols = [c for c in ["latitude", "longitude"] if c in self.numeric_columns_]
            if self._kmeans_model is not None and len(geo_cols) == 2:
                try:
                    pt = df_one[geo_cols].fillna(g_mean)
                    cluster_id = int(self._kmeans_model.predict(pt)[0])
                    if cluster_id in self._global_te_stats["geo_cluster"]:
                        df_one["geo_cluster_te_mean"] = self._global_te_stats["geo_cluster"][cluster_id]
                except Exception:
                    pass

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                raw = float(self.backend_model.predict(df_one[self.feature_columns_])[0])
            
            if self.target_transform == "log1p":
                pred = math.expm1(raw)
            elif self.target_transform == "quantile" and self._quantile_transformer is not None:
                pred = float(self._quantile_transformer.inverse_transform(np.array([[raw]]))[0][0])
            else:
                pred = raw
            return self._clip_prediction(pred)
        except Exception:
            return None

    def _fit_lightgbm_pipeline(self, df: pd.DataFrame, y_eval: np.ndarray, rows: list[dict]) -> bool:
        try:
            from lightgbm import LGBMRegressor
            from sklearn.cluster import MiniBatchKMeans
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            return False

        m_smoothing = 15.0
        g_mean = float(np.mean(y_eval))
        self._global_te_priors = {"mean": g_mean}

        df_encoded = df.copy()
        df_encoded["__target_temp"] = y_eval

        for col in self.categorical_columns_:
            stats = df_encoded.groupby(col)["__target_temp"].agg(["count", "mean"])
            smooth_mean = (stats["count"] * stats["mean"] + m_smoothing * g_mean) / (stats["count"] + m_smoothing)
            self._global_te_stats[col] = smooth_mean.to_dict()
            df_encoded[f"{col}_te_mean"] = df_encoded[col].map(smooth_mean).fillna(g_mean)

        geo_cols = [c for c in ["latitude", "longitude"] if c in self.numeric_columns_]
        df_encoded["geo_cluster_te_mean"] = g_mean
        if len(geo_cols) == 2:
            try:
                geo_data = df_encoded[geo_cols].fillna(df_encoded[geo_cols].median())
                self._kmeans_model = MiniBatchKMeans(n_clusters=12, batch_size=512, random_state=42)
                clusters = self._kmeans_model.fit_predict(geo_data)
                df_encoded["__cluster_id"] = clusters
                
                c_stats = df_encoded.groupby("__cluster_id")["__target_temp"].agg(["count", "mean"])
                c_smooth = (c_stats["count"] * c_stats["mean"] + m_smoothing * g_mean) / (c_stats["count"] + m_smoothing)
                self._global_te_stats["geo_cluster"] = c_smooth.to_dict()
                df_encoded["geo_cluster_te_mean"] = df_encoded["__cluster_id"].map(c_smooth).fillna(g_mean)
                df_encoded.drop(columns=["__cluster_id"], inplace=True)
            except Exception:
                self._kmeans_model = None

        te_cols = [f"{c}_te_mean" for c in self.categorical_columns_] + ["geo_cluster_te_mean"]
        self.feature_columns_ = self.numeric_columns_ + te_cols

        if self.target_transform == "log1p":
            y_transformed = np.log1p(y_eval)
        elif self.target_transform == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            self._quantile_transformer = QuantileTransformer(n_quantiles=min(1500, len(rows)), output_distribution="normal", random_state=42)
            y_transformed = self._quantile_transformer.fit_transform(y_eval.reshape(-1, 1)).flatten()
        else:
            y_transformed = y_eval

        pipeline = self._make_lgbm_pipeline(numeric=self.feature_columns_)
        pipeline.fit(df_encoded[self.feature_columns_], y_transformed)
        self.backend_model = pipeline
        return True

    def _make_lgbm_pipeline(self, numeric: list[str]):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer
        from lightgbm import LGBMRegressor

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=5, degree=3))]), numeric),
            ],
            remainder="drop"
        )
        
        lgbm = LGBMRegressor(
            objective="fair",            
            n_estimators=1100,
            learning_rate=0.02,
            num_leaves=42,               
            min_child_samples=22,        
            subsample=0.82,
            colsample_bytree=0.78,        
            reg_alpha=2.5,               
            reg_lambda=4.5,              
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        return Pipeline([("features", preprocessor), ("regressor", lgbm)])

    def _clip_prediction(self, value: float) -> float:
        return max(self.min_duration_minutes, min(float(value), self.max_duration_minutes))

    def _is_better(self, candidate: dict[str, float], baseline: dict[str, float]) -> bool:
        rmse_better = candidate.get("rmse", float("inf")) < baseline.get("rmse", float("inf")) * 0.98
        mae_better  = candidate.get("mae",  float("inf")) < baseline.get("mae",  float("inf")) * 0.98
        r2_better   = candidate.get("r2",   float("-inf")) > baseline.get("r2",   float("-inf")) + 0.01
        return rmse_better or mae_better or r2_better