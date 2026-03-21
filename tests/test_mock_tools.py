"""Tests for mock tool execution framework."""

import json

import pytest

from claweval.mock_tools import MockToolExecutor, ToolCall


def test_execute_simple_response():
    mocks = {"read_file": {"response": "file contents here"}}
    executor = MockToolExecutor(mocks)

    call = ToolCall(name="read_file", arguments={"path": "/test.txt"}, call_id="tc_1")
    result = executor.execute(call)
    assert result == "file contents here"


def test_execute_json_response():
    mocks = {"get_data": {"response": {"key": "value", "count": 42}}}
    executor = MockToolExecutor(mocks)

    call = ToolCall(name="get_data", arguments={}, call_id="tc_1")
    result = executor.execute(call)
    parsed = json.loads(result)
    assert parsed["key"] == "value"
    assert parsed["count"] == 42


def test_execute_unknown_tool():
    executor = MockToolExecutor({})
    call = ToolCall(name="unknown_tool", arguments={}, call_id="tc_1")
    result = executor.execute(call)
    parsed = json.loads(result)
    assert "error" in parsed
    assert "unknown_tool" in parsed["error"].lower() or "Unknown" in parsed["error"]


def test_execute_all():
    mocks = {
        "tool_a": {"response": "result_a"},
        "tool_b": {"response": "result_b"},
    }
    executor = MockToolExecutor(mocks)

    calls = [
        ToolCall(name="tool_a", arguments={}, call_id="tc_1"),
        ToolCall(name="tool_b", arguments={}, call_id="tc_2"),
    ]
    results = executor.execute_all(calls)

    assert len(results) == 2
    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "tc_1"
    assert results[0]["content"] == "result_a"
    assert results[1]["tool_call_id"] == "tc_2"
    assert results[1]["content"] == "result_b"


def test_tool_call_from_openai():
    """Test parsing from OpenAI format (uses a mock object)."""
    class FakeFunction:
        name = "read_file"
        arguments = '{"path": "/test.txt"}'

    class FakeToolCall:
        id = "call_abc123"
        function = FakeFunction()

    tc = ToolCall.from_openai(FakeToolCall())
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "/test.txt"}
    assert tc.call_id == "call_abc123"


def test_tool_call_from_openai_invalid_json():
    class FakeFunction:
        name = "tool"
        arguments = "not valid json"

    class FakeToolCall:
        id = "call_1"
        function = FakeFunction()

    tc = ToolCall.from_openai(FakeToolCall())
    assert tc.arguments == {"_raw": "not valid json"}


def test_execute_dict_without_response_key():
    """When mock_tool_responses has extra keys beyond 'response'."""
    mocks = {"search": {"query": "test", "response": "search results"}}
    executor = MockToolExecutor(mocks)
    call = ToolCall(name="search", arguments={"query": "test"}, call_id="tc_1")
    result = executor.execute(call)
    assert result == "search results"
