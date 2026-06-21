from event_intelligence.src.common.paths import discover_dataset
from event_intelligence.src.dispatch import DispatchEngine
from event_intelligence.src.diversion_planner import DiversionPlanner
from event_intelligence.src.officer_allocation import OfficerAllocationEngine
from event_intelligence.src.prediction import DurationPredictionEngine, ImpactPredictionEngine
from event_intelligence.src.preprocessing import EventPreprocessor
from event_intelligence.src.resource_checker import ResourceSufficiencyChecker
from event_intelligence.src.resource_planner import ResourcePlanner
from event_intelligence.src.risk import RiskCorridorEngine
from event_intelligence.src.timeline_simulator import EventTimelineSimulator


def test_pipeline_smoke():
    dataset = discover_dataset()
    preprocessor = EventPreprocessor()
    rows = preprocessor.load_and_transform(dataset)[:200]
    assert rows
    assert {"hour", "duration_minutes", "impact_score"}.issubset(rows[0])

    duration = DurationPredictionEngine().fit(rows)
    impact = ImpactPredictionEngine().fit(rows)
    event = rows[0]
    prediction = duration.predict_one(event)
    score = impact.predict_one(event)
    resources = ResourcePlanner.from_config().recommend(
        predicted_impact=score,
        predicted_duration=prediction,
        road_closure=event["requires_road_closure"],
        event_cause=event["event_cause"],
    )
    assert prediction >= 1
    assert 0 <= score <= 100
    assert resources["officers"] >= 1


def test_operational_layer_smoke():
    dataset = discover_dataset()
    preprocessor = EventPreprocessor()
    rows = preprocessor.load_and_transform(dataset)[:800]
    risk = RiskCorridorEngine().fit(rows)
    allocation = OfficerAllocationEngine().recommend_by_corridor(risk, 5)
    assert allocation
    assert allocation[0]["officer_demand"] >= 1

    resources = {"officers": 4, "barricades": 2}
    dispatch = DispatchEngine.from_csv().assign(rows[0]["latitude"], rows[0]["longitude"], resources["officers"])
    assert dispatch["station_name"]

    sufficiency = ResourceSufficiencyChecker().check(
        required_officers=resources["officers"],
        required_barricades=resources["barricades"],
        available_officers=dispatch["available_officers"],
        available_barricades=1,
    )
    assert sufficiency["status"] in {"SUFFICIENT", "INSUFFICIENT"}

    timeline = EventTimelineSimulator().simulate(impact_score=75, duration_minutes=90)
    assert timeline[0]["minute"] == 0
    assert timeline[-1]["minute"] == 90

    diversion = DiversionPlanner().fit(rows)
    nodes = list(diversion.graph.nodes)
    if len(nodes) >= 2:
        routes = diversion.plan_multiple(nodes[0], nodes[1], corridor_risks={})
        assert {"route_1", "route_2", "route_3"}.issubset(routes)
