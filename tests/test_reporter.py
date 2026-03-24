"""Tests for the reporter module."""

import json
from pathlib import Path

import pytest

from claweval.reporter import (
    aggregate_results,
    save_json_results,
    generate_dashboard,
    ModelSummary,
)
from claweval.runner import TaskResult, TimingInfo
from claweval.scorer import ScoreResult


def _make_result(task_id, model_id, score=0.8, tok_s=50.0, ttft=500.0):
    return TaskResult(
        task_id=task_id,
        model_id=model_id,
        score=ScoreResult(task_id=task_id, total_score=score),
        timing=TimingInfo(
            wall_clock_ms=1000,
            ttft_ms=ttft,
            total_tokens=100,
            completion_tokens=50,
            tokens_per_second=tok_s,
        ),
        response_text="Test response",
    )


@pytest.fixture
def sample_results():
    return [
        _make_result("tool_calling_001", "model-a", 0.9, 60.0),
        _make_result("tool_calling_002", "model-a", 0.8, 55.0),
        _make_result("coding_001", "model-a", 0.7, 50.0),
        _make_result("tool_calling_001", "model-b", 0.6, 40.0),
        _make_result("tool_calling_002", "model-b", 0.5, 35.0),
        _make_result("coding_001", "model-b", 0.9, 45.0),
    ]


def test_aggregate_results(sample_results):
    summaries = aggregate_results(sample_results)
    assert "model-a" in summaries
    assert "model-b" in summaries

    model_a = summaries["model-a"]
    assert isinstance(model_a, ModelSummary)
    assert model_a.categories["tool_calling"] == pytest.approx(0.85, abs=0.01)
    assert model_a.categories["coding"] == pytest.approx(0.7, abs=0.01)


def test_aggregate_results_speed(sample_results):
    summaries = aggregate_results(sample_results)
    model_a = summaries["model-a"]
    assert model_a.speed["avg_tok_s"] > 0


def test_aggregate_results_overall(sample_results):
    summaries = aggregate_results(sample_results)
    model_a = summaries["model-a"]
    # Overall should be average of category scores
    expected = (0.85 + 0.7) / 2
    assert model_a.overall == pytest.approx(expected, abs=0.01)


def test_aggregate_with_model_names(sample_results):
    names = {"model-a": "Model Alpha", "model-b": "Model Beta"}
    summaries = aggregate_results(sample_results, names)
    assert summaries["model-a"].name == "Model Alpha"


def test_save_json_results(sample_results, tmp_path):
    path = save_json_results(
        sample_results,
        output_dir=tmp_path,
        run_id="test-run-001",
    )
    assert path.exists()
    assert path.suffix == ".json"

    with open(path) as f:
        data = json.load(f)

    assert data["run_id"] == "test-run-001"
    assert "model-a" in data["models"]
    assert "overall" in data["models"]["model-a"]


def test_generate_dashboard(sample_results, tmp_path):
    path = generate_dashboard(
        sample_results,
        model_names={"model-a": "Model A", "model-b": "Model B"},
        output_dir=tmp_path,
        run_id="test-run-001",
    )
    assert path.exists()
    assert path.suffix == ".html"

    content = path.read_text()
    assert "ClawEval" in content
    assert "chart.js" in content
    assert "radarChart" in content
    assert "speedChart" in content
    assert "Model A" in content
    assert "Model B" in content


def test_dashboard_contains_scores(sample_results, tmp_path):
    path = generate_dashboard(
        sample_results,
        output_dir=tmp_path,
        run_id="test-run-002",
    )
    content = path.read_text()
    # Should contain task IDs in detail table
    assert "tool_calling_001" in content
    assert "coding_001" in content


def test_speed_category_uses_relative_wall_clock():
    """Speed category score should reflect relative wall-clock time, not task pass rate."""
    # Model A: fast (500ms avg), Model B: slow (1000ms avg, 2x slower)
    results = [
        # speed category tasks
        _make_result("speed_001", "model-a", 0.9, 60.0),
        _make_result("speed_002", "model-a", 0.8, 55.0),
        _make_result("speed_001", "model-b", 0.9, 40.0),
        _make_result("speed_002", "model-b", 0.8, 35.0),
        # other category
        _make_result("coding_001", "model-a", 0.7, 50.0),
        _make_result("coding_001", "model-b", 0.6, 45.0),
    ]
    # Override wall_clock_ms: model-a is fast (500ms), model-b is slow (1000ms)
    for r in results:
        if r.model_id == "model-a":
            r.timing.wall_clock_ms = 500
        else:
            r.timing.wall_clock_ms = 1000

    summaries = aggregate_results(results)

    # Fastest model (model-a, 500ms) should get speed score of 1.0
    assert summaries["model-a"].categories["speed"] == pytest.approx(1.0, abs=0.01)
    # Model-b is 2x slower, should get 0.5
    assert summaries["model-b"].categories["speed"] == pytest.approx(0.5, abs=0.01)

    # Overall should incorporate the new speed score
    model_a = summaries["model-a"]
    expected_overall_a = (model_a.categories["coding"] + 1.0) / 2
    assert model_a.overall == pytest.approx(expected_overall_a, abs=0.01)

    model_b = summaries["model-b"]
    expected_overall_b = (model_b.categories["coding"] + 0.5) / 2
    assert model_b.overall == pytest.approx(expected_overall_b, abs=0.01)


def test_save_empty_results(tmp_path):
    path = save_json_results([], output_dir=tmp_path, run_id="empty")
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["models"] == {}
