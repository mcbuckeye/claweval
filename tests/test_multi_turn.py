"""Tests for multi-turn conversation runner."""

import pytest
from unittest.mock import MagicMock

from claweval.multi_turn import (
    MultiTurnTask,
    TurnResult,
    ConversationResult,
    run_multi_turn,
)
from claweval.config import ModelConfig, Settings


def test_multi_turn_task_from_dict():
    task = MultiTurnTask.from_dict({
        "id": "conv_001",
        "name": "Test conversation",
        "system_prompt": "You are helpful.",
        "turns": [
            {"role": "user", "content": "Hello"},
            {"expected_behavior": "Should greet back"},
            {"role": "user", "content": "How are you?"},
        ],
    })
    assert task.id == "conv_001"
    assert len(task.turns) == 3
    assert task.system_prompt == "You are helpful."


def test_turn_result_to_dict():
    tr = TurnResult(
        turn_index=0,
        user_message="Hello",
        assistant_response="Hi there!",
        expected_behavior="Should greet",
    )
    d = tr.to_dict()
    assert d["turn_index"] == 0
    assert d["user_message"] == "Hello"
    assert d["assistant_response"] == "Hi there!"


def test_conversation_result_to_dict():
    cr = ConversationResult(
        task_id="conv_001",
        model_id="model-a",
        total_wall_clock_ms=5000,
    )
    d = cr.to_dict()
    assert d["task_id"] == "conv_001"
    assert d["model_id"] == "model-a"
    assert d["total_wall_clock_ms"] == 5000


def test_run_multi_turn_with_mock():
    task = MultiTurnTask.from_dict({
        "id": "conv_001",
        "name": "Test",
        "system_prompt": "Be helpful.",
        "turns": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ],
    })

    model = ModelConfig(
        id="test-model",
        name="Test",
        base_url="http://localhost:1234/v1",
        api_key="test",
    )
    settings = Settings(timeout_seconds=30)

    # Mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Hello! I'm here to help."
    mock_chunk.choices[0].delta.tool_calls = None
    mock_chunk.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([mock_chunk])

    result = run_multi_turn(task, model, settings, client=mock_client)

    assert result.task_id == "conv_001"
    assert result.model_id == "test-model"
    assert len(result.turns) == 2
    assert result.error == ""
    assert result.total_wall_clock_ms > 0


def test_run_multi_turn_error_handling():
    task = MultiTurnTask.from_dict({
        "id": "conv_err",
        "name": "Error test",
        "turns": [{"role": "user", "content": "Hello"}],
    })

    model = ModelConfig(id="test", name="Test", base_url="http://localhost/v1", api_key="x")
    settings = Settings(timeout_seconds=5)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Connection failed")

    result = run_multi_turn(task, model, settings, client=mock_client)
    assert result.error == "Connection failed"
