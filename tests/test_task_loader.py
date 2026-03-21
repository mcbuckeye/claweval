"""Tests for task loader."""

from pathlib import Path

import pytest
import yaml

from claweval.task_loader import (
    Task, ToolDef, ExpectedResult, ScoringConfig,
    load_task, load_tasks, list_tasks,
)


@pytest.fixture
def sample_task_yaml(tmp_path):
    cat_dir = tmp_path / "tool_calling"
    cat_dir.mkdir()

    data = {
        "id": "tool_calling_001",
        "name": "Simple file read",
        "category": "tool_calling",
        "description": "Read a file",
        "difficulty": "easy",
        "system_prompt": "You are a helpful assistant.",
        "user_message": "Read /workspace/README.md",
        "tools": [{
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        }],
        "mock_tool_responses": {
            "read_file": {"response": "Hello world"},
        },
        "expected": {
            "tool_calls": [{"name": "read_file", "args": {"path": "/workspace/README.md"}}],
            "response_contains": ["Hello"],
        },
        "scoring": {
            "method": "deterministic",
            "weights": {"correct_tool": 0.5, "response_quality": 0.5},
        },
    }

    path = cat_dir / "001.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)

    return path


def test_load_single_task(sample_task_yaml):
    task = load_task(sample_task_yaml)
    assert task.id == "tool_calling_001"
    assert task.name == "Simple file read"
    assert task.category == "tool_calling"
    assert task.difficulty == "easy"
    assert len(task.tools) == 1
    assert task.tools[0].name == "read_file"


def test_task_expected(sample_task_yaml):
    task = load_task(sample_task_yaml)
    assert len(task.expected.tool_calls) == 1
    assert task.expected.tool_calls[0]["name"] == "read_file"
    assert "Hello" in task.expected.response_contains


def test_task_scoring(sample_task_yaml):
    task = load_task(sample_task_yaml)
    assert task.scoring.method == "deterministic"
    assert task.scoring.weights["correct_tool"] == 0.5


def test_tool_to_openai(sample_task_yaml):
    task = load_task(sample_task_yaml)
    openai_fmt = task.tools[0].to_openai()
    assert openai_fmt["type"] == "function"
    assert openai_fmt["function"]["name"] == "read_file"


def test_load_tasks_by_category(tmp_path):
    # Create two categories
    for cat in ["coding", "reasoning"]:
        d = tmp_path / cat
        d.mkdir()
        for i in range(3):
            data = {
                "id": f"{cat}_{i:03d}",
                "name": f"Task {i}",
                "category": cat,
            }
            with open(d / f"{i:03d}.yaml", "w") as f:
                yaml.dump(data, f)

    tasks = load_tasks(categories=["coding"], tasks_dir=tmp_path)
    assert len(tasks) == 3
    assert all(t.category == "coding" for t in tasks)


def test_load_all_tasks(tmp_path):
    for cat in ["coding", "reasoning"]:
        d = tmp_path / cat
        d.mkdir()
        data = {"id": f"{cat}_001", "name": "Task", "category": cat}
        with open(d / "001.yaml", "w") as f:
            yaml.dump(data, f)

    tasks = load_tasks(tasks_dir=tmp_path)
    assert len(tasks) == 2


def test_list_tasks_grouped(tmp_path):
    for cat in ["coding", "writing"]:
        d = tmp_path / cat
        d.mkdir()
        for i in range(2):
            data = {"id": f"{cat}_{i:03d}", "name": f"Task {i}", "category": cat}
            with open(d / f"{i:03d}.yaml", "w") as f:
                yaml.dump(data, f)

    grouped = list_tasks(tasks_dir=tmp_path)
    assert "coding" in grouped
    assert "writing" in grouped
    assert len(grouped["coding"]) == 2


def test_load_tasks_empty_dir(tmp_path):
    tasks = load_tasks(tasks_dir=tmp_path)
    assert tasks == []


def test_load_tasks_nonexistent_dir(tmp_path):
    tasks = load_tasks(tasks_dir=tmp_path / "nonexistent")
    assert tasks == []


def test_task_from_dict_minimal():
    task = Task.from_dict({"id": "test_001", "name": "Test"})
    assert task.id == "test_001"
    assert task.difficulty == "medium"  # default
    assert task.tools == []
    assert task.expected.tool_calls == []


def test_task_with_conversation():
    data = {
        "id": "mem_001",
        "name": "Memory test",
        "category": "memory",
        "conversation": [
            {"role": "user", "content": "Remember: my dog is named Rex"},
            {"role": "assistant", "content": "Got it!"},
        ],
        "user_message": "What's my dog's name?",
    }
    task = Task.from_dict(data)
    assert len(task.conversation) == 2
    assert task.conversation[0]["content"] == "Remember: my dog is named Rex"


def test_load_actual_tasks():
    """Load the real task files to verify they parse correctly."""
    from claweval.task_loader import TASKS_DIR
    tasks = load_tasks()
    assert len(tasks) == 35, f"Expected 35 tasks, got {len(tasks)}"

    categories = set(t.category for t in tasks)
    expected_cats = {"tool_calling", "coding", "reasoning", "writing", "research", "memory", "speed"}
    assert categories == expected_cats, f"Missing categories: {expected_cats - categories}"
