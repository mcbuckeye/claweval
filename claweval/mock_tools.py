"""Mock tool execution framework for eval tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call made by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    @classmethod
    def from_openai(cls, tc: Any) -> ToolCall:
        """Parse from an OpenAI ChatCompletion tool_call object."""
        args = tc.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        return cls(
            name=tc.function.name,
            arguments=args,
            call_id=tc.id,
        )


class MockToolExecutor:
    """Execute tool calls using predefined mock responses.

    Mock responses are defined per-task in the YAML as:
        mock_tool_responses:
          tool_name:
            <match_args>:
              response: <value>
    
    Or simpler flat format:
        mock_tool_responses:
          tool_name:
            response: <value>
    """

    def __init__(self, mock_responses: dict[str, Any]) -> None:
        self._responses = mock_responses

    def execute(self, call: ToolCall) -> str:
        """Execute a mock tool call and return the response string."""
        tool_mocks = self._responses.get(call.name)
        if tool_mocks is None:
            return json.dumps({"error": f"Unknown tool: {call.name}"})

        # If it has a direct "response" key, return that
        if "response" in tool_mocks:
            return self._format_response(tool_mocks["response"])

        # Otherwise try to match by checking if mock args are subset of call args
        # For nested format: { tool_name: { arg: val, response: ... } }
        response_val = tool_mocks.get("response")
        if response_val is not None:
            return self._format_response(response_val)

        # Fallback: return the whole mock dict as the response
        return self._format_response(tool_mocks)

    def execute_all(self, calls: list[ToolCall]) -> list[dict[str, str]]:
        """Execute a list of tool calls, return messages for the API."""
        results = []
        for call in calls:
            response = self.execute(call)
            results.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": response,
            })
        return results

    @staticmethod
    def _format_response(val: Any) -> str:
        if isinstance(val, str):
            return val
        return json.dumps(val)
