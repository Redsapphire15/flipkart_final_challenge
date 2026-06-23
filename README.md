# Event Impact Intelligence Platform

EIIP is a modular traffic-control-room prototype for forecasting event impact,
estimating clearance duration, finding hotspots, recommending resources,
retrieving similar incidents, summarizing module outputs with an LLM-compatible
copilot, and planning simplified diversions.

## Quick Start

```bash
python -m event_intelligence.src.training.train_models
streamlit run event_intelligence/dashboard/app.py
```

To force the trainer to use LightGBM and CatBoost, and fail loudly if either
backend is unavailable:

```bash
python -m event_intelligence.src.training.train_models --require-ml-backends
python -c "from event_intelligence.src.prediction import DurationPredictionEngine, ImpactPredictionEngine; print(DurationPredictionEngine.load('event_intelligence/models/duration_model.pkl').backend_name); print(ImpactPredictionEngine.load('event_intelligence/models/impact_model.pkl').backend_name)"
```

The verification command should print `ensemble` and `catboost`.

The trainer auto-discovers CSV files in `event_intelligence/data/` or the
repository root. The current workspace includes the anonymized Astram event CSV.

Install the full dashboard stack when network/package access is available:

```bash
pip install -r requirements.txt
```

The core pipeline is dependency-aware and can train baseline artifacts with only
the Python standard library plus `networkx`.

## Architecture

- `src/preprocessing`: column detection, missing-value handling, temporal
  features, spatial/event features, `duration_minutes`, and derived impact score.
- `src/prediction`: duration and impact prediction engines with saved pickle
  artifacts at `event_intelligence/models/duration_model.pkl` and
  `event_intelligence/models/impact_model.pkl`.
- `src/hotspot_detection`: DBSCAN-style hotspot discovery, cluster statistics,
  GeoJSON output, and Folium heatmap generation when Folium is installed.
- `src/risk`: corridor and junction risk ranking.
- `src/resource_planner`: configurable rule engine for officers, barricades,
  tow truck need, and escalation.
- `src/similarity_search`: historical incident retrieval and aggregate outcomes.
- `src/llm_assistant`: OpenAI-compatible LLM service that summarizes module
  outputs only, with a deterministic fallback.
- `src/diversion_planner`: simplified NetworkX graph and Dijkstra diversion
  planner for closure scenarios.
- `dashboard/app.py`: Streamlit pages for live assessment, hotspots, resources,
  similar incidents, executive analytics, copilot, and diversion planning.

## Training Outputs

Running the trainer writes:

- `preprocessor.pkl`
- `duration_model.pkl`
- `impact_model.pkl`
- `hotspots.pkl`
- `hotspots.geojson`
- `hotspot_map.html`
- `risk_corridors.pkl`
- `similar_incidents.pkl`
- `diversion_graph.pkl`
- `evaluation_report.json`

## LLM Configuration

The copilot uses these optional environment variables:

```bash
export LLM_API_BASE=https://api.openai.com/v1
export LLM_API_KEY=...
export LLM_MODEL=gpt-4o-mini
```

Without those variables, it returns a deterministic operational summary from
the prediction, resource, diversion, and similarity modules.
