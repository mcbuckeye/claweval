# 🦞 ClawEval

OpenClaw Model Evaluation Suite — benchmark LLMs across the full spectrum of agent tasks.

## Features

- **7 categories**: tool calling, coding, reasoning, writing, research, memory, speed
- **35 tasks** (5 per category) with deterministic scoring
- **Any OpenAI-compatible API**: LM Studio, Ollama, vLLM, cloud providers
- **HTML dashboard** with radar charts + speed comparison
- **Mock tool framework** for reproducible function-calling tests

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Copy and edit config
cp config.yaml.example config.yaml

# List available tasks
claweval tasks

# Run evaluation
claweval run

# Generate dashboard from results
claweval report

# Compare two models
claweval compare model-a model-b
```

## Configuration

Copy `config.yaml.example` and add your model endpoints:

```yaml
providers:
  - name: lmstudio
    base_url: http://localhost:1234/v1
    api_key: "lm-studio"
    models:
      - id: my-model
        name: "My Model"
```

## Task Categories

| Category | Tasks | Scoring |
|----------|-------|---------|
| Tool Calling | 5 | Deterministic (tool name + params match) |
| Coding | 5 | Keyword matching |
| Reasoning | 5 | Keyword matching |
| Writing | 5 | Keyword matching |
| Research | 5 | Keyword + tool call matching |
| Memory | 5 | Keyword matching |
| Speed | 5 | Keyword matching + timing metrics |

## Output

- **JSON results**: `results/results_<timestamp>.json`
- **HTML dashboard**: `results/dashboard_<timestamp>.html`

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
