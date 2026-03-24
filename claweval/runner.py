"""Task runner — sequential model execution with timing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from claweval.config import ModelConfig, Settings
from claweval.task_loader import Task
from claweval.mock_tools import MockToolExecutor, ToolCall
from claweval.scorer import score_task, score_task_hybrid, ScoreResult


@dataclass
class TimingInfo:
    """Timing metrics for a single task execution."""

    wall_clock_ms: float = 0.0
    ttft_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    chunk_count: int = 0
    estimated_gen_tok_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_clock_ms": round(self.wall_clock_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "chunk_count": self.chunk_count,
            "estimated_gen_tok_s": round(self.estimated_gen_tok_s, 2),
        }


@dataclass
class TaskResult:
    """Result of running a single task against a single model."""

    task_id: str
    model_id: str
    score: ScoreResult | None = None
    timing: TimingInfo = field(default_factory=TimingInfo)
    response_text: str = ""
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "score": self.score.to_dict() if self.score else None,
            "timing": self.timing.to_dict(),
            "response_text": self.response_text[:2000],
            "tool_calls_made": self.tool_calls_made,
            "error": self.error,
        }


def _build_messages(task: Task) -> list[dict[str, Any]]:
    """Build the initial message list for a task."""
    messages: list[dict[str, Any]] = []

    if task.system_prompt:
        messages.append({"role": "system", "content": task.system_prompt})

    if task.conversation:
        messages.extend(task.conversation)

    if task.user_message:
        messages.append({"role": "user", "content": task.user_message})

    return messages


def _build_tools(task: Task) -> list[dict[str, Any]] | None:
    """Build OpenAI-format tools list."""
    if not task.tools:
        return None
    return [t.to_openai() for t in task.tools]


def run_task(
    task: Task,
    model: ModelConfig,
    settings: Settings,
    client: OpenAI | None = None,
    scoring_mode: str = "deterministic",
    judge_scorer: Any = None,
) -> TaskResult:
    """Run a single task against a single model."""
    result = TaskResult(task_id=task.id, model_id=model.id)

    if client is None:
        client = OpenAI(**model.client_kwargs())

    messages = _build_messages(task)
    tools = _build_tools(task)
    mock_executor = MockToolExecutor(task.mock_tool_responses)
    all_tool_calls: list[ToolCall] = []

    try:
        # Main execution loop (handles multi-turn tool calling)
        max_turns = 5
        start_time = time.perf_counter()
        first_token_time: float | None = None

        for turn in range(max_turns):
            kwargs: dict[str, Any] = {
                "model": model.id,
                "messages": messages,
                "timeout": settings.timeout_seconds if settings.timeout_seconds > 0 else None,
                "max_tokens": 16384,
            }
            if tools:
                kwargs["tools"] = tools

            # Use streaming to measure TTFT
            stream = client.chat.completions.create(stream=True, **kwargs)

            response_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            turn_tool_calls: dict[int, dict[str, Any]] = {}
            usage_data: dict[str, int] = {}
            chunk_count = 0

            for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunk_count += 1

                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.delta:
                    # Text content — the actual answer
                    if choice.delta.content:
                        response_chunks.append(choice.delta.content)
                    # Reasoning/thinking content — track separately
                    reasoning = getattr(choice.delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_chunks.append(reasoning)

                    # Tool calls
                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index
                            if idx not in turn_tool_calls:
                                turn_tool_calls[idx] = {
                                    "id": tc.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc.id:
                                turn_tool_calls[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    turn_tool_calls[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    turn_tool_calls[idx]["arguments"] += tc.function.arguments

                # Usage (often in the last chunk)
                if chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            # Process tool calls from this turn
            if turn_tool_calls:
                parsed_calls = []
                assistant_tool_calls = []
                for _idx, tc_data in sorted(turn_tool_calls.items()):
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {"_raw": tc_data["arguments"]}

                    call = ToolCall(
                        name=tc_data["name"],
                        arguments=args,
                        call_id=tc_data["id"],
                    )
                    parsed_calls.append(call)
                    all_tool_calls.append(call)

                    assistant_tool_calls.append({
                        "id": tc_data["id"],
                        "type": "function",
                        "function": {
                            "name": tc_data["name"],
                            "arguments": tc_data["arguments"],
                        },
                    })

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "tool_calls": assistant_tool_calls,
                })

                # Execute mock tools and add responses
                tool_results = mock_executor.execute_all(parsed_calls)
                messages.extend(tool_results)

                # Store for result
                for tc in parsed_calls:
                    result.tool_calls_made.append({
                        "name": tc.name,
                        "arguments": tc.arguments,
                    })
            else:
                # No more tool calls — we're done
                # Use actual content; fall back to reasoning only if content is empty
                content = "".join(response_chunks)
                if content.strip():
                    result.response_text = content
                else:
                    result.response_text = "".join(reasoning_chunks)
                break

        end_time = time.perf_counter()

        # Compute timing
        wall_ms = (end_time - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else 0
        gen_time_ms = wall_ms - ttft_ms if ttft_ms > 0 else wall_ms
        comp_tokens = usage_data.get("completion_tokens", 0)

        # Estimate tok/s from chunks as fallback if no usage data
        estimated_gen_tok_s = 0.0
        if comp_tokens and gen_time_ms > 0:
            estimated_gen_tok_s = comp_tokens / (gen_time_ms / 1000)
        elif chunk_count > 0 and gen_time_ms > 0:
            # Rough estimate: ~1.3 tokens per chunk on average
            estimated_gen_tok_s = (chunk_count * 1.3) / (gen_time_ms / 1000)

        result.timing = TimingInfo(
            wall_clock_ms=wall_ms,
            ttft_ms=ttft_ms,
            total_tokens=usage_data.get("total_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=comp_tokens,
            tokens_per_second=(
                comp_tokens / (wall_ms / 1000)
                if wall_ms > 0 and comp_tokens
                else 0
            ),
            chunk_count=chunk_count,
            estimated_gen_tok_s=estimated_gen_tok_s,
        )

        # Score the result
        result.score = score_task_hybrid(
            task, all_tool_calls, result.response_text,
            judge_scorer=judge_scorer, scoring_mode=scoring_mode,
        )

    except Exception as e:
        result.error = str(e)

    return result


def run_tasks(
    tasks: list[Task],
    model: ModelConfig,
    settings: Settings,
    on_complete: Any = None,
    scoring_mode: str = "deterministic",
    judge_scorer: Any = None,
) -> list[TaskResult]:
    """Run multiple tasks sequentially against a model."""
    client = OpenAI(**model.client_kwargs())
    results: list[TaskResult] = []

    for task in tasks:
        try:
            task_result = run_task(
                task, model, settings, client,
                scoring_mode=scoring_mode, judge_scorer=judge_scorer,
            )
        except Exception as e:
            task_result = TaskResult(task_id=task.id, model_id=model.id, error=str(e))
        results.append(task_result)

        if on_complete:
            on_complete(task_result)

    return results
