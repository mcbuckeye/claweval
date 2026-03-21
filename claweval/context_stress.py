"""Context length stress test runner for ClawEval."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from claweval.config import ModelConfig, Settings
from claweval.task_loader import Task
from claweval.runner import TaskResult, TimingInfo, run_task


# Realistic filler content for padding context
_CODE_FILLER = '''
def process_batch(items: list[dict], config: dict) -> list[dict]:
    """Process a batch of items according to configuration rules."""
    results = []
    for item in items:
        if item.get("status") == "active":
            transformed = {
                "id": item["id"],
                "value": item.get("value", 0) * config.get("multiplier", 1),
                "tags": [t.lower() for t in item.get("tags", [])],
                "processed_at": "2025-01-15T10:30:00Z",
            }
            if config.get("validate", False):
                if transformed["value"] < 0:
                    transformed["error"] = "negative value"
                    continue
            results.append(transformed)
    return results
'''

_DOC_FILLER = """
## System Architecture Overview

The platform consists of three primary services: the API Gateway, the Processing Engine,
and the Storage Layer. Each service communicates via gRPC for internal calls and exposes
REST endpoints for external consumers. The API Gateway handles authentication, rate limiting,
and request routing. It maintains a connection pool of up to 500 concurrent connections per
downstream service. The Processing Engine implements a pipeline architecture with configurable
stages. Each stage can be independently scaled based on throughput requirements. The Storage
Layer provides both relational (PostgreSQL) and document (MongoDB) storage backends, with
automatic failover and read replicas for high-availability deployments.
"""

_CONV_FILLER_USER = "Can you explain how the {topic} component works in our system?"
_CONV_FILLER_ASSISTANT = (
    "The {topic} component is responsible for handling {desc}. It integrates with "
    "the main pipeline through event-driven messaging and maintains its own state store "
    "for consistency. Key configuration parameters include timeout thresholds, retry policies, "
    "and circuit breaker settings. The component was redesigned in Q3 2024 to support "
    "horizontal scaling, which required changes to the session affinity model."
)

_TOPICS = [
    ("authentication", "user identity verification and session management"),
    ("caching", "distributed cache invalidation and TTL-based eviction"),
    ("logging", "structured log aggregation and trace correlation"),
    ("monitoring", "metrics collection, alerting, and dashboard generation"),
    ("deployment", "CI/CD pipelines and blue-green deployment orchestration"),
    ("database", "connection pooling, query optimization, and migration management"),
    ("messaging", "async message queues, dead letter handling, and replay logic"),
    ("search", "full-text indexing, ranking algorithms, and query parsing"),
    ("notification", "multi-channel delivery, templating, and preference management"),
    ("billing", "usage metering, invoice generation, and payment processing"),
    ("analytics", "event ingestion, aggregation pipelines, and report generation"),
    ("security", "encryption at rest, key rotation, and audit logging"),
]


def _generate_filler(target_tokens: int) -> list[dict[str, str]]:
    """Generate realistic filler conversation to pad context to target size.

    Rough estimate: 1 token ~ 4 chars.
    """
    target_chars = target_tokens * 4
    messages: list[dict[str, str]] = []
    current_chars = 0
    idx = 0

    while current_chars < target_chars:
        topic, desc = _TOPICS[idx % len(_TOPICS)]

        # Alternate between code, docs, and conversation filler
        if idx % 3 == 0:
            content = _CODE_FILLER * 2
            messages.append({"role": "user", "content": f"Here's the code for {topic}:\n{content}"})
            current_chars += len(content) + 50
        elif idx % 3 == 1:
            content = _DOC_FILLER * 2
            messages.append({"role": "user", "content": f"Documentation for {topic}:\n{content}"})
            current_chars += len(content) + 50
        else:
            user_msg = _CONV_FILLER_USER.format(topic=topic)
            asst_msg = _CONV_FILLER_ASSISTANT.format(topic=topic, desc=desc)
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": asst_msg})
            current_chars += len(user_msg) + len(asst_msg)

        idx += 1

    return messages


@dataclass
class StressTestResult:
    """Result of a context stress test at a specific context size."""

    context_tokens: int
    task_result: TaskResult
    score_delta: float = 0.0  # Degradation from baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_tokens": self.context_tokens,
            "task_result": self.task_result.to_dict(),
            "score_delta": round(self.score_delta, 4),
        }


# Default context sizes to test
CONTEXT_SIZES = [2_000, 8_000, 32_000, 64_000, 128_000]


def run_context_stress(
    task: Task,
    model: ModelConfig,
    settings: Settings,
    context_sizes: list[int] | None = None,
    client: OpenAI | None = None,
) -> list[StressTestResult]:
    """Run a task at increasing context sizes and measure degradation."""
    sizes = context_sizes or CONTEXT_SIZES
    if client is None:
        client = OpenAI(**model.client_kwargs())

    results: list[StressTestResult] = []
    baseline_score: float | None = None

    for size in sizes:
        # Create a modified task with filler conversation prepended
        filler = _generate_filler(size)

        padded_task = Task(
            id=f"{task.id}_ctx{size // 1000}k",
            name=f"{task.name} (ctx={size // 1000}k)",
            category=task.category,
            description=task.description,
            difficulty=task.difficulty,
            system_prompt=task.system_prompt,
            user_message=task.user_message,
            tools=task.tools,
            mock_tool_responses=task.mock_tool_responses,
            expected=task.expected,
            scoring=task.scoring,
            conversation=filler + task.conversation,
        )

        task_result = run_task(padded_task, model, settings, client)

        if baseline_score is None and task_result.score:
            baseline_score = task_result.score.total_score

        score_delta = 0.0
        if baseline_score and task_result.score:
            score_delta = task_result.score.total_score - baseline_score

        results.append(StressTestResult(
            context_tokens=size,
            task_result=task_result,
            score_delta=score_delta,
        ))

    return results
