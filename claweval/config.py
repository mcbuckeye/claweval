"""Configuration loader for ClawEval."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """A single model to evaluate."""

    id: str
    name: str
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    ram_gb: float = 0.0

    def client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for openai.OpenAI() constructor."""
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
        }


@dataclass
class Settings:
    """Global run settings."""

    judge_model: str = ""
    judge_api_key_env: str = "ANTHROPIC_API_KEY"
    scoring_mode: str = "deterministic"  # "deterministic", "judge", or "hybrid"
    parallel_tasks: int = 1
    timeout_seconds: int = 300
    warmup_requests: int = 2
    runs_per_task: int = 1
    categories: list[str] = field(default_factory=lambda: [
        "tool_calling", "coding", "reasoning", "writing",
        "research", "memory", "speed",
    ])
    raw: dict = field(default_factory=dict)  # raw settings for extra fields


@dataclass
class EvalConfig:
    """Top-level configuration."""

    models: list[ModelConfig] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)

    def get_model(self, model_id: str) -> ModelConfig | None:
        """Look up a model by ID."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None


def _resolve_api_key(raw: dict) -> str:
    """Resolve api_key or api_key_env to an actual key string."""
    if "api_key" in raw:
        return str(raw["api_key"])
    env_var = raw.get("api_key_env", "")
    if env_var:
        return os.environ.get(env_var, "")
    return ""


def load_config(path: str | Path) -> EvalConfig:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    models: list[ModelConfig] = []
    for provider in raw.get("providers", []):
        base_url = provider.get("base_url", "")
        api_key = _resolve_api_key(provider)
        provider_name = provider.get("name", "")
        for m in provider.get("models", []):
            models.append(ModelConfig(
                id=m["id"],
                name=m.get("name", m["id"]),
                provider=provider_name,
                base_url=base_url,
                api_key=api_key,
                ram_gb=float(m.get("ram_gb", 0)),
            ))

    raw_settings = raw.get("settings", {})
    settings = Settings(
        judge_model=raw_settings.get("judge_model", ""),
        judge_api_key_env=raw_settings.get("judge_api_key_env", "ANTHROPIC_API_KEY"),
        scoring_mode=raw_settings.get("scoring_mode", "deterministic"),
        parallel_tasks=raw_settings.get("parallel_tasks", 1),
        timeout_seconds=raw_settings.get("timeout_seconds", 300),
        warmup_requests=raw_settings.get("warmup_requests", 2),
        runs_per_task=raw_settings.get("runs_per_task", 1),
        categories=raw_settings.get("categories", Settings().categories),
        raw=raw_settings,
    )

    return EvalConfig(models=models, settings=settings)
