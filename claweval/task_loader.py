"""Load task definitions from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class ToolDef:
    """An OpenAI-compatible tool definition."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ExpectedResult:
    """What we expect the model to produce."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response_contains: list[str] = field(default_factory=list)
    exact_match: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExpectedResult:
        return cls(
            tool_calls=d.get("tool_calls", []),
            response_contains=d.get("response_contains", []),
            exact_match=d.get("exact_match", ""),
        )


@dataclass
class ScoringConfig:
    """How to score this task."""

    method: str = "deterministic"
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScoringConfig:
        return cls(
            method=d.get("method", "deterministic"),
            weights=d.get("weights", {}),
        )


@dataclass
class Task:
    """A single evaluation task."""

    id: str
    name: str
    category: str
    description: str = ""
    difficulty: str = "medium"
    system_prompt: str = ""
    user_message: str = ""
    tools: list[ToolDef] = field(default_factory=list)
    mock_tool_responses: dict[str, Any] = field(default_factory=dict)
    expected: ExpectedResult = field(default_factory=ExpectedResult)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    conversation: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        tools = []
        for t in d.get("tools", []):
            tools.append(ToolDef(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            ))

        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            category=d.get("category", ""),
            description=d.get("description", ""),
            difficulty=d.get("difficulty", "medium"),
            system_prompt=d.get("system_prompt", ""),
            user_message=d.get("user_message", ""),
            tools=tools,
            mock_tool_responses=d.get("mock_tool_responses", {}),
            expected=ExpectedResult.from_dict(d.get("expected", {})),
            scoring=ScoringConfig.from_dict(d.get("scoring", {})),
            conversation=d.get("conversation", []),
        )


def load_task(path: str | Path) -> Task:
    """Load a single task from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Task.from_dict(raw)


def load_tasks(
    categories: list[str] | None = None,
    tasks_dir: Path | None = None,
) -> list[Task]:
    """Load all tasks, optionally filtering by category."""
    base = tasks_dir or TASKS_DIR
    tasks: list[Task] = []

    if not base.exists():
        return tasks

    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        if categories and cat_dir.name not in categories:
            continue
        for yaml_file in sorted(cat_dir.glob("*.yaml")):
            tasks.append(load_task(yaml_file))

    return tasks


def list_tasks(
    categories: list[str] | None = None,
    tasks_dir: Path | None = None,
) -> dict[str, list[Task]]:
    """List tasks grouped by category."""
    all_tasks = load_tasks(categories, tasks_dir)
    grouped: dict[str, list[Task]] = {}
    for t in all_tasks:
        grouped.setdefault(t.category, []).append(t)
    return grouped
