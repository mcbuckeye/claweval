"""Tests for the scoring engine."""

import pytest

from claweval.scorer import (
    score_tool_calls,
    score_response_contains,
    score_exact_match,
    score_task,
    ScoreResult,
)
from claweval.mock_tools import ToolCall
from claweval.task_loader import Task


def test_score_tool_calls_perfect():
    actual = [ToolCall(name="read_file", arguments={"path": "/test.txt"})]
    expected = [{"name": "read_file", "args": {"path": "/test.txt"}}]
    name_score, param_score, details = score_tool_calls(actual, expected)
    assert name_score == 1.0
    assert param_score == 1.0


def test_score_tool_calls_wrong_name():
    actual = [ToolCall(name="write_file", arguments={"path": "/test.txt"})]
    expected = [{"name": "read_file", "args": {"path": "/test.txt"}}]
    name_score, param_score, _ = score_tool_calls(actual, expected)
    assert name_score == 0.0
    assert param_score == 0.0


def test_score_tool_calls_right_name_wrong_params():
    actual = [ToolCall(name="read_file", arguments={"path": "/wrong.txt"})]
    expected = [{"name": "read_file", "args": {"path": "/test.txt"}}]
    name_score, param_score, _ = score_tool_calls(actual, expected)
    assert name_score == 1.0
    assert param_score == 0.0


def test_score_tool_calls_no_expected():
    name_score, param_score, _ = score_tool_calls([], [])
    assert name_score == 1.0
    assert param_score == 1.0


def test_score_tool_calls_unexpected_calls():
    actual = [ToolCall(name="read_file", arguments={})]
    name_score, param_score, _ = score_tool_calls(actual, [])
    assert name_score == 0.0


def test_score_tool_calls_missing_calls():
    expected = [{"name": "read_file", "args": {}}]
    name_score, param_score, _ = score_tool_calls([], expected)
    assert name_score == 0.0


def test_score_response_contains_all_found():
    score, details = score_response_contains(
        "Python and FastAPI are great for building APIs",
        ["Python", "FastAPI", "API"],
    )
    assert score == 1.0
    assert len(details["missing"]) == 0


def test_score_response_contains_partial():
    score, details = score_response_contains(
        "Python is great",
        ["Python", "FastAPI", "Django"],
    )
    assert abs(score - 1/3) < 0.01
    assert "FastAPI" in details["missing"]


def test_score_response_contains_case_insensitive():
    score, details = score_response_contains(
        "python is awesome",
        ["Python"],
    )
    assert score == 1.0


def test_score_response_contains_empty_keywords():
    score, _ = score_response_contains("anything", [])
    assert score == 1.0


def test_score_exact_match_perfect():
    score, details = score_exact_match("  hello world  ", "hello world")
    assert score == 1.0


def test_score_exact_match_case_insensitive():
    score, details = score_exact_match("Hello World", "hello world")
    assert score == 0.8


def test_score_exact_match_no_match():
    score, _ = score_exact_match("foo", "bar")
    assert score == 0.0


def test_score_exact_match_empty_expected():
    score, _ = score_exact_match("anything", "")
    assert score == 1.0


def test_score_task_tool_calling():
    task = Task.from_dict({
        "id": "tc_001",
        "name": "Test",
        "category": "tool_calling",
        "expected": {
            "tool_calls": [{"name": "read_file", "args": {"path": "/test"}}],
            "response_contains": ["hello"],
        },
        "scoring": {
            "method": "deterministic",
            "weights": {"correct_tool": 0.4, "correct_params": 0.3, "response_quality": 0.3},
        },
    })

    tool_calls = [ToolCall(name="read_file", arguments={"path": "/test"})]
    result = score_task(task, tool_calls, "hello world")

    assert isinstance(result, ScoreResult)
    assert result.total_score == 1.0
    assert result.breakdown["correct_tool"] == 1.0


def test_score_task_no_tools():
    task = Task.from_dict({
        "id": "write_001",
        "name": "Writing test",
        "category": "writing",
        "expected": {"response_contains": ["important", "key"]},
    })

    result = score_task(task, [], "This is important and a key insight")
    assert result.total_score == 1.0


def test_score_task_exact_match():
    task = Task.from_dict({
        "id": "exact_001",
        "name": "Exact match test",
        "category": "reasoning",
        "expected": {"exact_match": "42"},
    })

    result = score_task(task, [], "42")
    assert result.total_score == 1.0


def test_score_result_to_dict():
    sr = ScoreResult(
        task_id="test_001",
        total_score=0.85,
        breakdown={"correct_tool": 1.0, "response_quality": 0.7},
    )
    d = sr.to_dict()
    assert d["task_id"] == "test_001"
    assert d["total_score"] == 0.85
    assert "correct_tool" in d["breakdown"]


def test_score_multiple_tool_calls():
    actual = [
        ToolCall(name="search", arguments={"q": "test"}),
        ToolCall(name="read_file", arguments={"path": "/a.txt"}),
    ]
    expected = [
        {"name": "search", "args": {"q": "test"}},
        {"name": "read_file", "args": {"path": "/a.txt"}},
    ]
    name_score, param_score, _ = score_tool_calls(actual, expected)
    assert name_score == 1.0
    assert param_score == 1.0
