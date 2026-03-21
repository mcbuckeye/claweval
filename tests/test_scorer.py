"""Tests for the scoring engine."""

import pytest

from claweval.scorer import (
    score_tool_calls,
    score_response_contains,
    score_exact_match,
    score_task,
    score_task_hybrid,
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


def test_score_result_has_judge_score_field():
    sr = ScoreResult(task_id="test", total_score=0.5, judge_score={"overall": 0.8})
    d = sr.to_dict()
    assert d["judge_score"]["overall"] == 0.8


def test_score_result_no_judge_score():
    sr = ScoreResult(task_id="test", total_score=0.5)
    d = sr.to_dict()
    assert "judge_score" not in d


def test_score_task_hybrid_deterministic_mode():
    task = Task.from_dict({
        "id": "write_001",
        "name": "Test",
        "category": "writing",
        "expected": {"response_contains": ["hello"]},
    })
    result = score_task_hybrid(task, [], "hello world", judge_scorer=None, scoring_mode="deterministic")
    assert result.total_score == 1.0
    assert result.judge_score is None


def test_score_task_hybrid_tool_calling_stays_deterministic():
    """Tool calling category should stay deterministic even in hybrid mode."""
    from unittest.mock import MagicMock
    task = Task.from_dict({
        "id": "tc_001",
        "name": "Test",
        "category": "tool_calling",
        "expected": {"response_contains": ["hello"]},
    })
    mock_judge = MagicMock()
    result = score_task_hybrid(task, [], "hello", judge_scorer=mock_judge, scoring_mode="hybrid")
    assert result.total_score == 1.0
    mock_judge.score_response.assert_not_called()


def test_score_task_hybrid_mode():
    from unittest.mock import MagicMock
    from claweval.judge import JudgeScore

    task = Task.from_dict({
        "id": "coding_001",
        "name": "Test",
        "category": "coding",
        "expected": {"response_contains": ["def"]},
    })

    mock_judge = MagicMock()
    mock_judge.score_response.return_value = JudgeScore(
        task_id="coding_001",
        criteria_scores={"correctness": 8},
        overall=0.8,
        feedback="Good.",
    )

    result = score_task_hybrid(task, [], "def foo(): pass", judge_scorer=mock_judge, scoring_mode="hybrid")
    # 0.4 * 1.0 (deterministic) + 0.6 * 0.8 (judge) = 0.88
    assert result.total_score == pytest.approx(0.88, abs=0.01)
    assert result.judge_score is not None
    assert result.judge_score["overall"] == 0.8


def test_score_task_judge_only_mode():
    from unittest.mock import MagicMock
    from claweval.judge import JudgeScore

    task = Task.from_dict({
        "id": "coding_001",
        "name": "Test",
        "category": "coding",
        "expected": {"response_contains": ["def"]},
    })

    mock_judge = MagicMock()
    mock_judge.score_response.return_value = JudgeScore(
        task_id="coding_001",
        overall=0.7,
        feedback="OK.",
    )

    result = score_task_hybrid(task, [], "def foo()", judge_scorer=mock_judge, scoring_mode="judge")
    assert result.total_score == 0.7
