# ClawEval Phase 2 — PRD

## Overview
Phase 2 adds LLM-as-judge scoring, harder tasks, context length stress tests, multi-turn conversations, and better speed metrics.

## 1. LLM-as-Judge Scoring

### Architecture
Add `claweval/judge.py` — calls an external LLM (Claude Sonnet via Anthropic API) to score responses.

```python
class JudgeScorer:
    """Score responses using an LLM judge."""
    
    def score_response(self, task, response, criteria) -> JudgeScore:
        """Send task + response to judge model, get structured score."""
```

### Scoring Criteria (per category)
- **Coding:** correctness (0-10), readability (0-10), edge case handling (0-10), best practices (0-10)
- **Writing:** clarity (0-10), tone appropriateness (0-10), completeness (0-10), conciseness (0-10)  
- **Research:** accuracy (0-10), source synthesis (0-10), depth (0-10), organization (0-10)
- **Reasoning:** logical validity (0-10), step clarity (0-10), answer correctness (0-10)
- **Memory:** recall accuracy (0-10), coherence (0-10)
- **Tool Calling:** keep deterministic (no judge needed)
- **Speed:** keep deterministic (no judge needed)

### Judge Prompt Template
Each category gets a rubric template. The judge receives:
1. The original task prompt
2. The model's response
3. Category-specific rubric
4. "Score each criterion 0-10. Return JSON."

### Config
```yaml
settings:
  judge_model: "anthropic/claude-sonnet-4-6-20260514"
  judge_api_key_env: ANTHROPIC_API_KEY
  scoring_mode: "hybrid"  # "deterministic", "judge", or "hybrid"
```

Hybrid mode: deterministic score + judge score, weighted 40/60.

## 2. Harder Tasks (70 new tasks, 10 per category)

### Coding (10 new — hard)
- Multi-file refactor: update 3 files consistently (model, route, test)
- Debug a race condition in async code
- Implement a caching decorator with TTL and LRU eviction
- Port a Python function to TypeScript
- Review a 200-line PR diff and identify 3 planted bugs
- Write a migration script for a schema change
- Implement error handling + retry logic for an API client
- Optimize a slow SQL query (given schema + explain output)
- Complete a half-written state machine
- Generate a comprehensive test suite for a class (edge cases, mocks)

### Memory (10 new — hard)
- Recall 3 specific facts from a 30K-token conversation
- Follow persona instructions after 20K tokens of unrelated context
- Track 8 characters across a narrative with role changes
- Detect 3 subtle contradictions in a long business document
- Execute a 15-step workflow specified at conversation start
- Remember and apply user preferences stated 50 messages ago
- Maintain conversation coherence across topic switches
- Cross-reference facts from different parts of a long context
- Identify when context contains outdated vs current information
- Summarize key decisions from a 40K-token meeting transcript

### Reasoning (10 new — hard)
- Multi-constraint optimization (schedule 5 meetings with 8 constraints)
- Chain-of-thought math (3 nested word problems)
- Counterfactual reasoning ("If X hadn't happened, would Y?")
- Scientific hypothesis evaluation (given data, which theory fits?)
- Game theory scenario (prisoner's dilemma variant)
- Legal reasoning (apply 3 rules to a fact pattern)
- Systems thinking (trace cascading effects of a config change)
- Probabilistic reasoning with incomplete information
- Meta-reasoning: determine when to give up vs. keep trying
- Abstract pattern recognition (complete a complex sequence)

### Research (10 new — hard)
- Synthesize conflicting sources into a balanced summary
- Build a comparison matrix (5 products × 8 criteria) from provided docs
- Extract and normalize structured data from 3 different formats
- Evaluate credibility of 5 sources on the same topic
- Generate a literature review outline from abstracts
- Cross-reference claims across 4 documents for consistency
- Identify gaps in a research summary
- Summarize technical paper for executive audience
- Compare/contrast two competing methodologies
- Build a timeline from unstructured event descriptions

### Writing (10 new — hard)
- Write a technical RFC (problem, proposal, alternatives, risks)
- Draft a sensitive HR email (performance concern, empathetic tone)
- Write API documentation with examples for 5 endpoints
- Convert a verbose 2000-word report into a 200-word executive summary
- Write a compelling product changelog for a developer audience
- Draft a customer apology email (data breach scenario)
- Write onboarding documentation for a complex system
- Create a decision matrix document with pros/cons
- Write a technical blog post explaining a complex concept simply
- Adapt a message for 4 audiences: CEO, engineer, customer, press

### Tool Calling (10 new — hard)
- 5-tool chain with conditional branching (if tool A returns X, call B, else C)
- Recover from a tool error (retry with modified params)
- Select correct tool from 15 available (high ambiguity)
- Parallel tool calls (3 independent calls, then merge results)
- Tool call with deeply nested JSON params
- Chain where tool B's params depend on parsing tool A's response
- Handle pagination (call tool repeatedly until done)
- Refuse an unsafe tool call (delete_all_files with confirmation)
- Multi-turn: user corrects tool params mid-chain
- Compose tool results into a structured final answer

### Speed (10 new — hard)
- Generate 1000-word technical document
- Produce structured JSON with 20+ fields
- Stream a multi-part response (intro, analysis, conclusion)
- Code generation: full CRUD module (~300 lines)
- Rapid-fire: 5 short answers in sequence (test consistency)
- Long-form reasoning with visible chain-of-thought
- Generate a complete config file from requirements
- Produce markdown table with 10 rows from description
- Write + format a complete README.md
- Generate API response examples for 5 endpoints

## 3. Context Length Stress Tests

New category: `context_stress`

Run the same 5 core tasks at increasing context sizes:
- **8K context**: baseline
- **32K context**: normal use
- **64K context**: extended sessions  
- **128K context**: heavy use
- **256K context**: max (your setup)

Pad context with realistic filler (code files, documentation, conversation history — not garbage text).

Measure:
- Score degradation curve
- TTFT increase curve  
- tok/s degradation curve

### Implementation
Add `--context-stress` flag to CLI. Tasks defined with `context_sizes` field.

## 4. Multi-Turn Conversations

New task format: `conversation` type with multiple turns.

```yaml
type: conversation
turns:
  - role: user
    content: "Set up a FastAPI project with auth"
  - expected_behavior: "Should ask clarifying questions or start coding"
  - role: user  
    content: "Use JWT tokens, PostgreSQL"
  - expected_behavior: "Should build on previous context"
  - role: user
    content: "Actually, switch to SQLite for now"
  - expected_behavior: "Should adapt without losing earlier context"
```

Score: coherence across turns, context retention, adaptation to corrections.

## 5. Better Speed Metrics

### LM Studio Stats Integration
After each task, query LM Studio's stats endpoint if available.

### Fallback: Calculate from Streaming
- Count chunks × avg chunk size for estimated tok/s
- Measure inter-chunk timing for generation speed
- Track prompt length → TTFT correlation

### Per-Task Metrics
```json
{
  "generation_tok_s": 54.2,
  "prompt_processing_tok_s": 1461.0,
  "ttft_ms": 533,
  "total_tokens": 1847,
  "tokens_per_dollar": null
}
```

## 6. Cost-Efficiency Score

New derived metric:
- `quality_per_gb = overall_score / ram_gb`
- `quality_per_second = overall_score / avg_wall_clock`
- `efficiency_rank` combining both

Requires RAM usage per model in config:
```yaml
models:
  - id: nvidia/nemotron-3-nano
    name: "Nemotron-3-Nano"
    ram_gb: 33.6
```

## 7. Updated Dashboard

- Add judge scores as separate radar overlay
- Context stress test: line chart (context size → score/speed)
- Multi-turn: timeline visualization
- Cost-efficiency quadrant chart (quality vs speed, bubble size = RAM)
- Per-task drill-down: click task to see full response + judge feedback

## Tech Additions
- `claweval/judge.py` — LLM judge scorer
- `claweval/context_stress.py` — context length test runner
- `claweval/multi_turn.py` — multi-turn conversation runner
- 70 new YAML task files
- 5 context stress YAML files
- 5 multi-turn YAML files
- Updated dashboard template

## CLI Changes
```bash
# Run with judge scoring
claweval run --scoring hybrid

# Run context stress tests only
claweval run --context-stress

# Run only new hard tasks
claweval run --difficulty hard

# Run everything
claweval run --all --scoring hybrid
```

## Development Plan
1. [ ] Judge scorer (`judge.py`) + rubric templates
2. [ ] Update scorer to support hybrid mode
3. [ ] Write 70 new hard tasks (YAML)
4. [ ] Context stress test framework
5. [ ] Multi-turn conversation runner
6. [ ] Better speed metrics (chunk counting)
7. [ ] RAM config + efficiency scoring
8. [ ] Updated dashboard with new charts
9. [ ] Tests for all new code
10. [ ] Run full Phase 2 benchmark
