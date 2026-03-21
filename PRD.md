# ClawEval — OpenClaw Model Evaluation Suite

## Overview
ClawEval is an automated benchmark suite that evaluates LLMs across the full spectrum of OpenClaw agent tasks. It runs models locally via any OpenAI-compatible API (LM Studio, Ollama, etc.) and produces a scored comparison dashboard.

## Goals
1. Compare open-weight models on YOUR hardware with YOUR workloads
2. Cover all 7 capability categories that matter for OpenClaw agents
3. Produce actionable data: which model for which task at what speed/cost
4. Easy to run, easy to extend with new tasks

## Architecture

```
claweval/
├── cli.py                  # Main CLI entry point
├── runner.py               # Orchestrates eval runs
├── config.py               # Model + provider configuration
├── scorer.py               # Scoring engine (deterministic + LLM-judge)
├── reporter.py             # Dashboard / results output
├── tasks/                  # YAML task definitions by category
│   ├── tool_calling/       # Function calling accuracy
│   ├── coding/             # Code generation, editing, debugging
│   ├── reasoning/          # Multi-step logic, math, planning
│   ├── writing/            # Emails, reports, summaries
│   ├── research/           # Web search → synthesis
│   ├── memory/             # Long context, recall accuracy
│   └── speed/              # Performance benchmarks
├── tools/                  # Mock tool definitions (OpenClaw format)
├── results/                # Run output (JSON + HTML dashboard)
└── templates/
    └── dashboard.html      # Results visualization
```

## Task Categories & Design

### 1. Tool Calling (10 tasks)
Test the model's ability to invoke tools correctly using OpenClaw's function-calling format.

**Tasks include:**
- Single tool call with simple params
- Multi-tool chain (tool A output → tool B input)
- Tool selection from 10+ available tools
- Error recovery (tool returns error, model should retry or adapt)
- Nested/complex parameter structures (JSON objects, arrays)
- Tool call with optional params (should omit correctly)
- Parallel tool calls (multiple independent calls)
- Refusing invalid tool requests gracefully
- Multi-turn tool dialogue (3+ rounds)
- Ambiguous request requiring clarification vs. tool call

**Scoring:** Deterministic — exact match on tool name, params, sequence. Partial credit for correct tool + wrong params.

### 2. Coding (10 tasks)
Test code generation, editing, and debugging abilities.

**Tasks include:**
- Generate a FastAPI endpoint from description
- Fix a bug in provided Python code (off-by-one)
- Write tests for existing function
- Refactor function to be async
- Complete a partially written React component
- Debug a failing SQL query
- Write a shell script for file processing
- Multi-file edit (update import + usage)
- Code review: identify issues in a PR diff
- Generate a Dockerfile from requirements

**Scoring:** 
- Automated: syntax check, test execution where applicable
- LLM-judge: code quality, correctness, completeness (0-10)

### 3. Reasoning (10 tasks)
Test multi-step logical thinking and problem solving.

**Tasks include:**
- Math word problem (multi-step arithmetic)
- Logic puzzle (constraint satisfaction)
- Planning: order of operations for a deployment
- Causal reasoning: debug from symptoms
- Analogical reasoning
- Data interpretation from a table
- Temporal reasoning (scheduling conflicts)
- Spatial reasoning (network topology)
- Probabilistic reasoning
- Meta-reasoning: when to ask for clarification vs. proceed

**Scoring:** Deterministic for tasks with known answers. LLM-judge for open-ended.

### 4. Writing (10 tasks)
Test professional communication and content generation.

**Tasks include:**
- Draft email reply (professional tone)
- Write a project status update
- Summarize a long document (provided as context)
- Write release notes from a changelog
- Create a meeting agenda from notes
- Draft a technical design document section
- Write error messages for an API
- Compose a Slack message for a team announcement
- Rewrite text for different audiences (executive vs. engineer)
- Write documentation for a code function

**Scoring:** LLM-judge on clarity, tone, completeness, brevity (0-10 each).

### 5. Research (8 tasks)
Test ability to search, synthesize, and present information.

**Tasks include:**
- Answer a factual question requiring web search
- Compare two technologies (pros/cons table)
- Find and summarize recent news on a topic
- Compile a list of resources on a subject
- Fact-check a claim with sources
- Extract structured data from unstructured text
- Cross-reference multiple sources for accuracy
- Synthesize conflicting information

**Scoring:** LLM-judge on accuracy, source quality, synthesis depth.
Note: Research tasks require web_search/web_fetch tool access or are tested with provided source material.

### 6. Memory & Context (8 tasks)  
Test handling of long contexts and conversation history.

**Tasks include:**
- Recall a fact from 20K tokens ago in conversation
- Follow instructions given at the start after long context
- Summarize a 50K-token document accurately
- Maintain persona consistency over 10+ turns
- Track multiple entities across a long narrative
- Detect contradictions in a long conversation
- Follow a complex multi-step instruction set
- Context window stress test (max supported context)

**Scoring:** Deterministic for recall tasks. LLM-judge for quality tasks.

### 7. Speed & Efficiency (per-model, not per-task)
Measured automatically during all other tasks.

**Metrics:**
- Tokens per second (generation)
- Time to first token (TTFT)
- Prompt processing speed (tokens/sec)
- Peak memory usage
- Total wall-clock time per category
- Cost estimate (if API pricing provided)

## Configuration

```yaml
# config.yaml
providers:
  - name: lmstudio
    base_url: http://mac-studio.local:1234/v1
    api_key: "lm-studio"
    models:
      - id: qwen3-coder-next-80b-8bit
        name: "Qwen3-Coder-Next (80B, 8-bit)"
      - id: qwen3.5-122b-a10b-4bit
        name: "Qwen3.5-122B-A10B (4-bit)"
      - id: qwen3.5-27b-8bit
        name: "Qwen3.5-27B (8-bit)"
      - id: gpt-oss-120b-4bit
        name: "GPT-OSS-120B (4-bit)"
  
  - name: ollama
    base_url: http://mac-studio.local:11434/v1
    api_key: "ollama"
    models:
      - id: qwen3.5:35b-a3b
        name: "Qwen3.5-35B-A3B"

  # Optional: include cloud models as baseline
  - name: anthropic
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    models:
      - id: claude-sonnet-4-6-20260514
        name: "Claude Sonnet 4.6 (baseline)"

settings:
  judge_model: "anthropic/claude-sonnet-4-6-20260514"  # LLM-as-judge
  parallel_tasks: 1          # Sequential by default (fair comparison)
  timeout_seconds: 300       # Per task
  warmup_requests: 2         # Warm up model before timing
  runs_per_task: 1           # Repeat for statistical significance (increase to 3 for publication)
  categories:                # Which categories to run
    - tool_calling
    - coding
    - reasoning
    - writing
    - research
    - memory
```

## CLI Interface

```bash
# Run full suite against all configured models
python -m claweval run

# Run specific category
python -m claweval run --category coding

# Run specific model only
python -m claweval run --model qwen3-coder-next-80b-8bit

# Run quick (subset of tasks, 2 per category)
python -m claweval run --quick

# Compare two models head-to-head
python -m claweval compare model-a model-b

# Generate dashboard from existing results
python -m claweval report

# List available tasks
python -m claweval tasks

# Add a custom task
python -m claweval add-task --category coding --file my_task.yaml
```

## Output

### results.json
```json
{
  "run_id": "2026-03-21T14:00:00Z",
  "hardware": "Mac Studio M3 Ultra 512GB",
  "models": {
    "qwen3-coder-next-80b-8bit": {
      "tool_calling": { "score": 8.7, "tasks": [...] },
      "coding": { "score": 7.9, "tasks": [...] },
      "reasoning": { "score": 6.5, "tasks": [...] },
      "writing": { "score": 7.2, "tasks": [...] },
      "research": { "score": 6.8, "tasks": [...] },
      "memory": { "score": 7.5, "tasks": [...] },
      "speed": {
        "avg_tok_s": 58.7,
        "avg_ttft_ms": 700,
        "peak_memory_gb": 80.1
      },
      "overall": 7.43
    }
  }
}
```

### HTML Dashboard
- Radar chart: model capabilities across categories
- Bar chart: speed comparison
- Table: detailed per-task scores
- Cost analysis (if pricing data provided)
- Recommendation: "Best for coding", "Best for general use", "Best speed/quality ratio"

## Tech Stack
- **Language:** Python 3.12+
- **HTTP Client:** httpx (async, for parallel model calls)
- **LLM Client:** openai Python SDK (works with any OpenAI-compatible API)
- **Dashboard:** Jinja2 + Chart.js (single HTML file, no server needed)
- **Task format:** YAML
- **Test runner:** pytest (for development/CI)

## Task YAML Format

```yaml
id: tool_calling_001
name: "Simple file read"
category: tool_calling
description: "Ask the model to read a specific file"
difficulty: easy

system_prompt: |
  You are a helpful assistant with access to tools.

user_message: |
  Read the contents of /workspace/README.md and tell me what the project is about.

tools:
  - name: read_file
    description: "Read the contents of a file"
    parameters:
      type: object
      properties:
        path:
          type: string
          description: "Path to the file to read"
      required: ["path"]

mock_tool_responses:
  read_file:
    path: "/workspace/README.md"
    response: "# MyProject\nA web application for tracking tasks. Built with FastAPI and React."

expected:
  tool_calls:
    - name: read_file
      args:
        path: "/workspace/README.md"
  response_contains:
    - "task"
    - "FastAPI"

scoring:
  method: deterministic
  weights:
    correct_tool: 0.4
    correct_params: 0.3
    response_quality: 0.3
```

## Development Plan

### Phase 1: Core Framework (MVP)
- [ ] CLI scaffolding + config loader
- [ ] Model client (OpenAI-compatible)
- [ ] Task loader (YAML → task objects)
- [ ] Runner (sequential execution, timing)
- [ ] Deterministic scorer
- [ ] JSON results output
- [ ] 5 tasks per category (35 total)

### Phase 2: LLM Judge + Dashboard
- [ ] LLM-as-judge scoring integration
- [ ] HTML dashboard generation
- [ ] Speed/memory profiling
- [ ] Mock tool execution framework
- [ ] 10 tasks per category (66 total)

### Phase 3: Polish + Community
- [ ] OpenClaw skill wrapper (run via /benchmark)
- [ ] Result sharing (upload to compare with community)
- [ ] CI integration (run on model update)
- [ ] Custom task authoring guide
- [ ] PinchBench task import (compatibility layer)

## Non-Goals (for now)
- Fine-tuning recommendations
- Training data contamination detection
- Multi-modal (image/audio) evaluation
- Distributed testing across multiple machines
