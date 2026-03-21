"""Multi-turn conversation runner for ClawEval."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from claweval.config import ModelConfig, Settings
from claweval.task_loader import Task
from claweval.runner import TimingInfo


@dataclass
class TurnResult:
    """Result of a single conversation turn."""

    turn_index: int
    user_message: str
    assistant_response: str
    expected_behavior: str = ""
    timing: TimingInfo = field(default_factory=TimingInfo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "user_message": self.user_message[:200],
            "assistant_response": self.assistant_response[:500],
            "expected_behavior": self.expected_behavior,
            "timing": self.timing.to_dict(),
        }


@dataclass
class ConversationResult:
    """Result of running a multi-turn conversation."""

    task_id: str
    model_id: str
    turns: list[TurnResult] = field(default_factory=list)
    total_wall_clock_ms: float = 0.0
    coherence_score: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "turns": [t.to_dict() for t in self.turns],
            "total_wall_clock_ms": round(self.total_wall_clock_ms, 2),
            "coherence_score": round(self.coherence_score, 4),
            "error": self.error,
        }


@dataclass
class MultiTurnTask:
    """A multi-turn conversation evaluation task."""

    id: str
    name: str
    category: str = "conversation"
    description: str = ""
    difficulty: str = "hard"
    system_prompt: str = ""
    turns: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiTurnTask:
        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            category=d.get("category", "conversation"),
            description=d.get("description", ""),
            difficulty=d.get("difficulty", "hard"),
            system_prompt=d.get("system_prompt", ""),
            turns=d.get("turns", []),
        )


def run_multi_turn(
    task: MultiTurnTask,
    model: ModelConfig,
    settings: Settings,
    client: OpenAI | None = None,
) -> ConversationResult:
    """Run a multi-turn conversation task against a model."""
    if client is None:
        client = OpenAI(**model.client_kwargs())

    result = ConversationResult(task_id=task.id, model_id=model.id)
    messages: list[dict[str, str]] = []

    if task.system_prompt:
        messages.append({"role": "system", "content": task.system_prompt})

    overall_start = time.perf_counter()

    try:
        turn_index = 0
        for turn_spec in task.turns:
            role = turn_spec.get("role", "")
            content = turn_spec.get("content", "")
            expected = turn_spec.get("expected_behavior", "")

            if role == "user":
                messages.append({"role": "user", "content": content})

                # Get model response
                start = time.perf_counter()
                first_token_time: float | None = None
                response_chunks: list[str] = []
                chunk_count = 0

                stream = client.chat.completions.create(
                    model=model.id,
                    messages=messages,
                    stream=True,
                    timeout=settings.timeout_seconds,
                )

                for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    chunk_count += 1
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice and choice.delta and choice.delta.content:
                        response_chunks.append(choice.delta.content)

                end = time.perf_counter()
                assistant_text = "".join(response_chunks)
                messages.append({"role": "assistant", "content": assistant_text})

                wall_ms = (end - start) * 1000
                ttft_ms = ((first_token_time - start) * 1000) if first_token_time else 0

                turn_result = TurnResult(
                    turn_index=turn_index,
                    user_message=content,
                    assistant_response=assistant_text,
                    expected_behavior=expected,
                    timing=TimingInfo(
                        wall_clock_ms=wall_ms,
                        ttft_ms=ttft_ms,
                        chunk_count=chunk_count,
                    ),
                )
                result.turns.append(turn_result)
                turn_index += 1

            elif expected:
                # This is an expected_behavior annotation on the previous turn
                if result.turns:
                    result.turns[-1].expected_behavior = expected

        result.total_wall_clock_ms = (time.perf_counter() - overall_start) * 1000

    except Exception as e:
        result.error = str(e)

    return result
