"""Tests for the task runner."""

from unittest.mock import MagicMock, patch
import json

import pytest

from claweval.runner import (
    run_task,
    _build_messages,
    _build_tools,
    TaskResult,
    TimingInfo,
)
from claweval.config import ModelConfig, Settings
from claweval.task_loader import Task


@pytest.fixture
def simple_task():
    return Task.from_dict({
        "id": "test_001",
        "name": "Test task",
        "category": "reasoning",
        "system_prompt": "You are helpful.",
        "user_message": "What is 2+2?",
        "expected": {"response_contains": ["4"]},
    })


@pytest.fixture
def tool_task():
    return Task.from_dict({
        "id": "test_002",
        "name": "Tool task",
        "category": "tool_calling",
        "system_prompt": "You have tools.",
        "user_message": "Read the file.",
        "tools": [{
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }],
        "mock_tool_responses": {
            "read_file": {"response": "file content"},
        },
        "expected": {
            "tool_calls": [{"name": "read_file", "args": {"path": "/test"}}],
            "response_contains": ["file"],
        },
    })


@pytest.fixture
def model_config():
    return ModelConfig(
        id="test-model",
        name="Test Model",
        base_url="http://localhost:1234/v1",
        api_key="test-key",
    )


@pytest.fixture
def settings():
    return Settings(timeout_seconds=30)


def test_build_messages_simple(simple_task):
    messages = _build_messages(simple_task)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "2+2" in messages[1]["content"]


def test_build_messages_with_conversation():
    task = Task.from_dict({
        "id": "conv_001",
        "name": "Conv",
        "system_prompt": "Be helpful.",
        "conversation": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "user_message": "How are you?",
    })
    messages = _build_messages(task)
    assert len(messages) == 4  # system + 2 conv + user


def test_build_tools(tool_task):
    tools = _build_tools(tool_task)
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "read_file"


def test_build_tools_none(simple_task):
    tools = _build_tools(simple_task)
    assert tools is None


def test_run_task_with_mock_client(simple_task, model_config, settings):
    """Test run_task with a mocked OpenAI client."""
    # Create a mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "The answer is 4."
    mock_chunk.choices[0].delta.tool_calls = None
    mock_chunk.usage = MagicMock()
    mock_chunk.usage.prompt_tokens = 20
    mock_chunk.usage.completion_tokens = 5
    mock_chunk.usage.total_tokens = 25

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([mock_chunk])

    result = run_task(simple_task, model_config, settings, client=mock_client)

    assert isinstance(result, TaskResult)
    assert result.task_id == "test_001"
    assert result.model_id == "test-model"
    assert result.error == ""
    assert "4" in result.response_text
    assert result.score is not None
    assert result.score.total_score == 1.0


def test_run_task_error_handling(simple_task, model_config, settings):
    """Test run_task handles exceptions gracefully."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Connection failed")

    result = run_task(simple_task, model_config, settings, client=mock_client)
    assert result.error == "Connection failed"
    assert result.score is None


def test_task_result_to_dict():
    result = TaskResult(
        task_id="test_001",
        model_id="model-a",
        score=None,
        timing=TimingInfo(wall_clock_ms=1500, ttft_ms=200),
        response_text="test response",
        error="",
    )
    d = result.to_dict()
    assert d["task_id"] == "test_001"
    assert d["timing"]["wall_clock_ms"] == 1500
    assert d["timing"]["ttft_ms"] == 200


def test_timing_info_to_dict():
    t = TimingInfo(
        wall_clock_ms=1234.5,
        ttft_ms=567.8,
        total_tokens=100,
        prompt_tokens=50,
        completion_tokens=50,
        tokens_per_second=42.5,
    )
    d = t.to_dict()
    assert d["wall_clock_ms"] == 1234.5
    assert d["tokens_per_second"] == 42.5
