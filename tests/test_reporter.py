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


def test_save_empty_results(tmp_path):
    path = save_json_results([], output_dir=tmp_path, run_id="empty")
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["models"] == {}
