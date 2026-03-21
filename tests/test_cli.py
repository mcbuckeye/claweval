"""Tests for the CLI module."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from claweval.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ClawEval" in result.output


def test_tasks_command(runner):
    result = runner.invoke(cli, ["tasks"])
    assert result.exit_code == 0
    assert "tool_calling" in result.output
    assert "coding" in result.output
    assert "Total:" in result.output


def test_tasks_command_filter(runner):
    result = runner.invoke(cli, ["tasks", "--category", "coding"])
    assert result.exit_code == 0
    assert "coding" in result.output


def test_run_missing_config(runner):
    result = runner.invoke(cli, ["run", "--config", "/nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_report_missing_results(runner, tmp_path):
    result = runner.invoke(cli, ["report", "--results-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "No results" in result.output


def test_compare_missing_results(runner, tmp_path):
    result = runner.invoke(cli, ["compare", "model-a", "model-b", "--results-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_report_from_json(runner, tmp_path):
    """Test report generation from a JSON results file."""
    data = {
        "run_id": "test-001",
        "models": {
            "model-a": {
                "name": "Model A",
                "overall": 0.85,
                "categories": {"coding": 0.9, "reasoning": 0.8},
                "speed": {"avg_tok_s": 50, "avg_ttft_ms": 300},
                "tasks": [
                    {
                        "task_id": "coding_001",
                        "model_id": "model-a",
                        "score": {"task_id": "coding_001", "total_score": 0.9, "breakdown": {}, "details": {}},
                        "timing": {"wall_clock_ms": 1000, "ttft_ms": 200, "total_tokens": 100,
                                   "prompt_tokens": 50, "completion_tokens": 50, "tokens_per_second": 50},
                        "response_text": "test",
                        "tool_calls_made": [],
                        "error": "",
                    },
                ],
            },
        },
    }

    json_file = tmp_path / "results_test-001.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    result = runner.invoke(cli, ["report", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Dashboard generated" in result.output

    # Check HTML was created
    html_files = list(tmp_path.glob("dashboard_*.html"))
    assert len(html_files) == 1


def test_compare_from_results(runner, tmp_path):
    """Test compare command with existing results."""
    data = {
        "run_id": "test-002",
        "models": {
            "model-a": {
                "name": "Model A",
                "overall": 0.85,
                "categories": {"coding": 0.9, "reasoning": 0.8},
                "speed": {"avg_tok_s": 50},
                "tasks": [],
            },
            "model-b": {
                "name": "Model B",
                "overall": 0.75,
                "categories": {"coding": 0.7, "reasoning": 0.8},
                "speed": {"avg_tok_s": 80},
                "tasks": [],
            },
        },
    }

    json_file = tmp_path / "results_test-002.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    result = runner.invoke(cli, ["compare", "model-a", "model-b", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "coding" in result.output
    assert "Overall" in result.output
