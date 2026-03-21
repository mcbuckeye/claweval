"""Tests for the LLM judge scorer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from claweval.judge import (
    JudgeScorer,
    JudgeScore,
    _build_judge_prompt,
    _parse_judge_response,
    RUBRICS,
)


def test_rubrics_have_expected_categories():
    assert "coding" in RUBRICS
    assert "writing" in RUBRICS
    assert "research" in RUBRICS
    assert "reasoning" in RUBRICS
    assert "memory" in RUBRICS
    # tool_calling and speed should NOT have rubrics (deterministic)
    assert "tool_calling" not in RUBRICS
    assert "speed" not in RUBRICS


def test_build_judge_prompt_coding():
    prompt = _build_judge_prompt("coding", "Write hello world", "print('hello')")
    assert "correctness" in prompt
    assert "readability" in prompt
    assert "edge_cases" in prompt
    assert "best_practices" in prompt
    assert "Write hello world" in prompt
    assert "print('hello')" in prompt


def test_build_judge_prompt_writing():
    prompt = _build_judge_prompt("writing", "Write an email", "Dear Sir")
    assert "clarity" in prompt
    assert "tone" in prompt
    assert "completeness" in prompt
    assert "conciseness" in prompt


def test_build_judge_prompt_unknown_category():
    """Unknown categories should fall back to writing rubric."""
    prompt = _build_judge_prompt("unknown_category", "task", "response")
    assert "clarity" in prompt


def test_parse_judge_response_valid():
    raw = json.dumps({
        "scores": {"correctness": 8, "readability": 7, "edge_cases": 6, "best_practices": 9},
        "feedback": "Good work overall.",
    })
    criteria = ["correctness", "readability", "edge_cases", "best_practices"]
    scores, feedback = _parse_judge_response(raw, criteria)
    assert scores["correctness"] == 8.0
    assert scores["readability"] == 7.0
    assert feedback == "Good work overall."


def test_parse_judge_response_markdown_wrapped():
    raw = """Here's my evaluation:
```json
{"scores": {"clarity": 9, "tone": 8}, "feedback": "Well written."}
```
"""
    scores, feedback = _parse_judge_response(raw, ["clarity", "tone"])
    assert scores["clarity"] == 9.0
    assert scores["tone"] == 8.0


def test_parse_judge_response_missing_criteria():
    raw = json.dumps({"scores": {"clarity": 7}, "feedback": "ok"})
    scores, feedback = _parse_judge_response(raw, ["clarity", "tone"])
    assert scores["clarity"] == 7.0
    assert scores["tone"] == 5.0  # default


def test_parse_judge_response_invalid_json():
    scores, feedback = _parse_judge_response("not json at all", ["clarity"])
    assert scores["clarity"] == 5.0  # fallback
    assert "Failed to parse" in feedback


def test_parse_judge_response_clamps_scores():
    raw = json.dumps({"scores": {"clarity": 15, "tone": -3}, "feedback": "clamped"})
    scores, _ = _parse_judge_response(raw, ["clarity", "tone"])
    assert scores["clarity"] == 10.0
    assert scores["tone"] == 0.0


def test_judge_score_to_dict():
    js = JudgeScore(
        task_id="test_001",
        criteria_scores={"clarity": 8.0, "tone": 7.5},
        overall=0.775,
        feedback="Good work.",
    )
    d = js.to_dict()
    assert d["task_id"] == "test_001"
    assert d["criteria_scores"]["clarity"] == 8.0
    assert d["overall"] == 0.775
    assert d["feedback"] == "Good work."


def test_judge_scorer_deterministic_categories():
    """Tool calling and speed should return empty scores."""
    scorer = JudgeScorer(api_key="test-key")
    result = scorer.score_response("tc_001", "tool_calling", "task", "response")
    assert result.overall == 0.0
    assert "deterministic" in result.feedback

    result = scorer.score_response("speed_001", "speed", "task", "response")
    assert result.overall == 0.0


def test_judge_scorer_calls_api():
    """Test that judge scorer correctly calls the API and parses response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "scores": {"correctness": 8, "readability": 9, "edge_cases": 7, "best_practices": 8},
        "feedback": "Solid implementation.",
    })

    scorer = JudgeScorer(api_key="test-key")
    scorer._client = MagicMock()
    scorer._client.chat.completions.create.return_value = mock_response

    result = scorer.score_response("coding_001", "coding", "Write code", "def foo(): pass")

    assert result.task_id == "coding_001"
    assert result.criteria_scores["correctness"] == 8.0
    assert result.overall == pytest.approx(0.8, abs=0.01)
    assert result.feedback == "Solid implementation."


def test_judge_scorer_handles_api_error():
    scorer = JudgeScorer(api_key="test-key")
    scorer._client = MagicMock()
    scorer._client.chat.completions.create.side_effect = Exception("API error")

    result = scorer.score_response("test_001", "coding", "task", "response")
    assert result.overall == 0.0
    assert "Judge error" in result.feedback
