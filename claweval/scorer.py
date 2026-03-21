"""Deterministic scoring engine for ClawEval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claweval.task_loader import Task, ExpectedResult
from claweval.mock_tools import ToolCall


@dataclass
class ScoreResult:
    """Result of scoring a single task."""

    task_id: str
    total_score: float  # 0.0 - 1.0
    breakdown: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_score": round(self.total_score, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "details": self.details,
        }


def score_tool_calls(
    actual: list[ToolCall],
    expected: list[dict[str, Any]],
) -> tuple[float, float, dict[str, Any]]:
    """Score tool call accuracy.

    Returns (tool_name_score, params_score, details).
    """
    if not expected:
        # No tool calls expected — if none made, perfect score
        return (1.0, 1.0, {"note": "no tool calls expected"}) if not actual else (0.0, 0.0, {"note": "unexpected tool calls"})

    if not actual:
        return (0.0, 0.0, {"note": "no tool calls made, expected some"})

    details: dict[str, Any] = {"expected_count": len(expected), "actual_count": len(actual)}
    name_matches = 0
    param_matches = 0

    for i, exp in enumerate(expected):
        exp_name = exp.get("name", "")
        exp_args = exp.get("args", exp.get("arguments", {}))

        if i < len(actual):
            act = actual[i]
            if act.name == exp_name:
                name_matches += 1
                # Check params
                if _args_match(act.arguments, exp_args):
                    param_matches += 1
        # else: missing call

    total = len(expected)
    name_score = name_matches / total
    param_score = param_matches / total

    details["name_matches"] = name_matches
    details["param_matches"] = param_matches

    return (name_score, param_score, details)


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check if expected args are a subset of actual args (flexible matching)."""
    for key, val in expected.items():
        if key not in actual:
            return False
        if str(actual[key]) != str(val):
            return False
    return True


def score_response_contains(
    response_text: str,
    keywords: list[str],
) -> tuple[float, dict[str, Any]]:
    """Score whether the response contains expected keywords."""
    if not keywords:
        return (1.0, {"note": "no keywords to check"})

    response_lower = response_text.lower()
    found = []
    missing = []

    for kw in keywords:
        if kw.lower() in response_lower:
            found.append(kw)
        else:
            missing.append(kw)

    score = len(found) / len(keywords) if keywords else 1.0
    return (score, {"found": found, "missing": missing})


def score_exact_match(response_text: str, expected: str) -> tuple[float, dict[str, Any]]:
    """Score exact match (after stripping whitespace)."""
    if not expected:
        return (1.0, {"note": "no exact match expected"})

    actual_clean = response_text.strip()
    expected_clean = expected.strip()

    if actual_clean == expected_clean:
        return (1.0, {"match": True})

    # Case-insensitive partial credit
    if actual_clean.lower() == expected_clean.lower():
        return (0.8, {"match": "case_insensitive"})

    return (0.0, {"match": False, "expected": expected_clean[:100]})


def score_task(
    task: Task,
    tool_calls: list[ToolCall],
    response_text: str,
) -> ScoreResult:
    """Score a completed task using deterministic methods."""
    weights = task.scoring.weights or _default_weights(task)
    breakdown: dict[str, float] = {}
    details: dict[str, Any] = {}

    # Tool call scoring
    if task.expected.tool_calls or "correct_tool" in weights:
        name_score, param_score, tc_details = score_tool_calls(
            tool_calls, task.expected.tool_calls,
        )
        breakdown["correct_tool"] = name_score
        breakdown["correct_params"] = param_score
        details["tool_calls"] = tc_details

    # Response contains scoring
    if task.expected.response_contains or "response_quality" in weights:
        kw_score, kw_details = score_response_contains(
            response_text, task.expected.response_contains,
        )
        breakdown["response_quality"] = kw_score
        details["response_contains"] = kw_details

    # Exact match scoring
    if task.expected.exact_match:
        em_score, em_details = score_exact_match(
            response_text, task.expected.exact_match,
        )
        breakdown["exact_match"] = em_score
        details["exact_match"] = em_details

    # Compute weighted total
    total = _weighted_total(breakdown, weights)

    return ScoreResult(
        task_id=task.id,
        total_score=total,
        breakdown=breakdown,
        details=details,
    )


def _default_weights(task: Task) -> dict[str, float]:
    """Generate default weights based on what's expected."""
    weights: dict[str, float] = {}

    if task.expected.tool_calls:
        weights["correct_tool"] = 0.4
        weights["correct_params"] = 0.3
        weights["response_quality"] = 0.3
    elif task.expected.exact_match:
        weights["exact_match"] = 1.0
    elif task.expected.response_contains:
        weights["response_quality"] = 1.0
    else:
        weights["response_quality"] = 1.0

    return weights


def _weighted_total(
    breakdown: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute weighted total score."""
    if not breakdown:
        return 0.0

    total = 0.0
    weight_sum = 0.0

    for key, score in breakdown.items():
        w = weights.get(key, 1.0)
        total += score * w
        weight_sum += w

    return total / weight_sum if weight_sum > 0 else 0.0
