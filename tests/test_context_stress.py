"""Tests for context stress test framework."""

import pytest

from claweval.context_stress import (
    _generate_filler,
    StressTestResult,
    CONTEXT_SIZES,
)
from claweval.task_loader import Task
from claweval.runner import TaskResult, TimingInfo


def test_generate_filler_produces_messages():
    filler = _generate_filler(2000)
    assert len(filler) > 0
    for msg in filler:
        assert "role" in msg
        assert "content" in msg
        assert msg["role"] in ("user", "assistant")


def test_generate_filler_scales_with_target():
    small = _generate_filler(1000)
    large = _generate_filler(10000)
    # Larger target should produce more messages
    small_chars = sum(len(m["content"]) for m in small)
    large_chars = sum(len(m["content"]) for m in large)
    assert large_chars > small_chars


def test_generate_filler_contains_realistic_content():
    filler = _generate_filler(4000)
    all_text = " ".join(m["content"] for m in filler)
    # Should contain code, documentation, and conversation content
    assert "def " in all_text or "process" in all_text
    assert "Architecture" in all_text or "service" in all_text


def test_context_sizes_defaults():
    assert len(CONTEXT_SIZES) == 5
    assert CONTEXT_SIZES[0] < CONTEXT_SIZES[-1]


def test_stress_test_result_to_dict():
    tr = TaskResult(
        task_id="test_001_ctx8k",
        model_id="model-a",
        timing=TimingInfo(wall_clock_ms=1000),
    )
    sr = StressTestResult(context_tokens=8000, task_result=tr, score_delta=-0.1)
    d = sr.to_dict()
    assert d["context_tokens"] == 8000
    assert d["score_delta"] == -0.1
    assert "task_result" in d
