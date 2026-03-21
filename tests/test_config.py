"""Tests for config loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from claweval.config import load_config, EvalConfig, ModelConfig, Settings


@pytest.fixture
def sample_config_yaml():
    return {
        "providers": [
            {
                "name": "lmstudio",
                "base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "models": [
                    {"id": "model-a", "name": "Model A"},
                    {"id": "model-b", "name": "Model B"},
                ],
            },
            {
                "name": "ollama",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "models": [
                    {"id": "model-c"},
                ],
            },
        ],
        "settings": {
            "judge_model": "gpt-4",
            "parallel_tasks": 1,
            "timeout_seconds": 120,
            "warmup_requests": 1,
            "runs_per_task": 2,
            "categories": ["coding", "reasoning"],
        },
    }


@pytest.fixture
def config_file(sample_config_yaml, tmp_path):
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(sample_config_yaml, f)
    return path


def test_load_config_basic(config_file):
    cfg = load_config(config_file)
    assert isinstance(cfg, EvalConfig)
    assert len(cfg.models) == 3


def test_load_config_models(config_file):
    cfg = load_config(config_file)
    model_a = cfg.get_model("model-a")
    assert model_a is not None
    assert model_a.name == "Model A"
    assert model_a.provider == "lmstudio"
    assert model_a.base_url == "http://localhost:1234/v1"
    assert model_a.api_key == "lm-studio"


def test_load_config_model_defaults(config_file):
    cfg = load_config(config_file)
    model_c = cfg.get_model("model-c")
    assert model_c is not None
    assert model_c.name == "model-c"  # Defaults to id


def test_load_config_settings(config_file):
    cfg = load_config(config_file)
    assert cfg.settings.judge_model == "gpt-4"
    assert cfg.settings.timeout_seconds == 120
    assert cfg.settings.runs_per_task == 2
    assert cfg.settings.categories == ["coding", "reasoning"]


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_config_api_key_env(tmp_path):
    os.environ["TEST_API_KEY_XYZ"] = "secret-key-123"
    try:
        data = {
            "providers": [{
                "name": "test",
                "base_url": "http://localhost/v1",
                "api_key_env": "TEST_API_KEY_XYZ",
                "models": [{"id": "m1"}],
            }],
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        cfg = load_config(path)
        assert cfg.models[0].api_key == "secret-key-123"
    finally:
        del os.environ["TEST_API_KEY_XYZ"]


def test_load_config_empty_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    cfg = load_config(path)
    assert cfg.models == []
    assert isinstance(cfg.settings, Settings)


def test_model_client_kwargs(config_file):
    cfg = load_config(config_file)
    model = cfg.get_model("model-a")
    kwargs = model.client_kwargs()
    assert kwargs["base_url"] == "http://localhost:1234/v1"
    assert kwargs["api_key"] == "lm-studio"


def test_get_model_not_found(config_file):
    cfg = load_config(config_file)
    assert cfg.get_model("nonexistent") is None


def test_default_settings():
    s = Settings()
    assert s.timeout_seconds == 300
    assert "tool_calling" in s.categories
    assert len(s.categories) == 7
    assert s.scoring_mode == "deterministic"
    assert s.judge_api_key_env == "ANTHROPIC_API_KEY"


def test_model_config_ram_gb(tmp_path):
    data = {
        "providers": [{
            "name": "test",
            "base_url": "http://localhost/v1",
            "api_key": "test",
            "models": [{"id": "m1", "name": "Model 1", "ram_gb": 33.6}],
        }],
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    cfg = load_config(path)
    assert cfg.models[0].ram_gb == 33.6


def test_settings_scoring_mode(tmp_path):
    data = {
        "providers": [],
        "settings": {
            "scoring_mode": "hybrid",
            "judge_api_key_env": "MY_KEY",
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    cfg = load_config(path)
    assert cfg.settings.scoring_mode == "hybrid"
    assert cfg.settings.judge_api_key_env == "MY_KEY"
