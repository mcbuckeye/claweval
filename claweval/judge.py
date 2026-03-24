"""LLM-as-judge scoring engine for ClawEval."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


JUDGE_MODEL = "gpt-5.4"
JUDGE_BASE_URL = "https://api.openai.com/v1"


@dataclass
class JudgeScore:
    """Structured score from the LLM judge."""

    task_id: str
    criteria_scores: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "criteria_scores": {k: round(v, 2) for k, v in self.criteria_scores.items()},
            "overall": round(self.overall, 4),
            "feedback": self.feedback,
        }


# Category-specific rubrics
RUBRICS: dict[str, dict[str, str]] = {
    "coding": {
        "correctness": "Does the code correctly solve the problem? Are there bugs or logic errors? (0=completely wrong, 10=flawless)",
        "readability": "Is the code clean, well-structured, and easy to understand? Good naming, formatting? (0=unreadable, 10=exemplary)",
        "edge_cases": "Does the code handle edge cases, invalid input, and boundary conditions? (0=no handling, 10=comprehensive)",
        "best_practices": "Does the code follow language idioms, design patterns, and best practices? (0=anti-patterns, 10=idiomatic)",
    },
    "writing": {
        "clarity": "Is the writing clear, unambiguous, and easy to follow? (0=confusing, 10=crystal clear)",
        "tone": "Is the tone appropriate for the audience and context? (0=completely wrong tone, 10=perfect tone)",
        "completeness": "Does the response address all aspects of the request? (0=misses everything, 10=fully complete)",
        "conciseness": "Is the response appropriately concise without unnecessary filler? (0=extremely verbose/sparse, 10=perfectly balanced)",
    },
    "research": {
        "accuracy": "Are the facts and claims accurate? (0=mostly wrong, 10=fully accurate)",
        "source_synthesis": "Does the response synthesize information from multiple angles/sources? (0=single perspective, 10=excellent synthesis)",
        "depth": "Does the response go beyond surface-level and provide meaningful depth? (0=superficial, 10=deeply insightful)",
        "organization": "Is the information well-organized and logically structured? (0=chaotic, 10=perfectly organized)",
    },
    "reasoning": {
        "logical_validity": "Is the reasoning logically sound with no fallacies? (0=invalid logic, 10=airtight reasoning)",
        "step_clarity": "Are the reasoning steps clearly laid out and easy to follow? (0=opaque, 10=transparent)",
        "answer_correctness": "Is the final answer/conclusion correct? (0=wrong, 10=correct)",
    },
    "memory": {
        "recall_accuracy": "Does the response accurately recall facts from the conversation? (0=wrong recalls, 10=perfect recall)",
        "coherence": "Is the response coherent and consistent with prior context? (0=contradicts context, 10=fully coherent)",
    },
}


def _build_judge_prompt(
    category: str,
    task_prompt: str,
    model_response: str,
) -> str:
    """Build the judge prompt with category-specific rubric."""
    rubric = RUBRICS.get(category, RUBRICS["writing"])

    criteria_text = "\n".join(
        f"- **{name}**: {description}"
        for name, description in rubric.items()
    )

    return f"""You are an expert evaluator scoring AI model responses. Be rigorous and fair.

## Task Given to the Model
{task_prompt}

## Model's Response
{model_response}

## Scoring Rubric
Score each criterion from 0 to 10:

{criteria_text}

## Instructions
1. Evaluate the response against each criterion independently.
2. Be strict — reserve 9-10 for genuinely excellent work.
3. Provide brief feedback explaining your scores.
4. Return your evaluation as JSON in exactly this format:

```json
{{
  "scores": {{
{chr(10).join(f'    "{name}": <0-10>,' for name in rubric.keys())}
  }},
  "feedback": "<2-3 sentence summary of strengths and weaknesses>"
}}
```

Return ONLY the JSON block, no other text."""


def _parse_judge_response(raw: str, criteria: list[str]) -> tuple[dict[str, float], str]:
    """Parse the judge's JSON response into scores and feedback."""
    # Extract JSON from response (handle markdown code blocks)
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        # Fallback: try to find any JSON object in the response
        import re
        match = re.search(r'\{[^{}]*"scores"[^{}]*\{[^}]+\}[^}]*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return {c: 5.0 for c in criteria}, "Failed to parse judge response"

    scores = {}
    raw_scores = data.get("scores", {})
    for criterion in criteria:
        val = raw_scores.get(criterion, 5.0)
        scores[criterion] = max(0.0, min(10.0, float(val)))

    feedback = data.get("feedback", "")
    return scores, feedback


class JudgeScorer:
    """Score responses using an LLM judge (Claude via API or CLI)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JUDGE_MODEL,
        base_url: str = JUDGE_BASE_URL,
        use_cli: bool = False,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._use_cli = use_cli or (not self._api_key)
        self._client: OpenAI | None = None

        if not self._use_cli:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=base_url,
            )

    def _call_claude_cli(self, prompt: str, retries: int = 3) -> str:
        """Call claude CLI with OAuth auth, with retries. Uses temp file for prompt."""
        import subprocess
        import tempfile
        import time as _time
        for attempt in range(retries):
            try:
                # Write prompt to temp file to avoid shell escaping issues
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(prompt)
                    tmppath = f.name
                
                # Read prompt from file via shell
                result = subprocess.run(
                    ["bash", "-c", f'cat "{tmppath}" | claude --permission-mode bypassPermissions --print -'],
                    capture_output=True, text=True, timeout=180,
                    env={**os.environ, "HOME": os.path.expanduser("~")},
                )
                
                # Clean up
                try:
                    os.unlink(tmppath)
                except OSError:
                    pass
                
                output = result.stdout.strip()
                if output:
                    return output
                # Log stderr for debugging
                if result.stderr:
                    import sys
                    print(f"  [judge] stderr: {result.stderr[:200]}", file=sys.stderr)
                if attempt < retries - 1:
                    _time.sleep(2)
            except subprocess.TimeoutExpired:
                if attempt < retries - 1:
                    _time.sleep(5)
            except Exception as e:
                import sys
                print(f"  [judge] CLI error: {e}", file=sys.stderr)
                if attempt < retries - 1:
                    _time.sleep(2)
        return ""

    def _call_api(self, prompt: str) -> str:
        """Call via OpenAI-compatible API."""
        assert self._client is not None
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def score_response(
        self,
        task_id: str,
        category: str,
        task_prompt: str,
        model_response: str,
    ) -> JudgeScore:
        """Send task + response to judge model, get structured score."""
        # Categories that stay deterministic
        if category in ("tool_calling", "speed"):
            return JudgeScore(task_id=task_id, overall=0.0, feedback="deterministic-only category")

        rubric = RUBRICS.get(category, RUBRICS["writing"])
        criteria = list(rubric.keys())
        prompt = _build_judge_prompt(category, task_prompt, model_response)

        try:
            if self._use_cli:
                raw_text = self._call_claude_cli(prompt)
            else:
                raw_text = self._call_api(prompt)

            # Debug: log raw judge output
            with open("/tmp/claweval-judge-debug.log", "a") as _dbg:
                _dbg.write(f"\n=== {task_id} ===\n")
                _dbg.write(f"RAW ({len(raw_text)} chars): {repr(raw_text[:500])}\n")

            scores, feedback = _parse_judge_response(raw_text, criteria)

            # Overall = average of criteria scores, normalized to 0-1
            overall = (sum(scores.values()) / (len(scores) * 10)) if scores else 0.0

            return JudgeScore(
                task_id=task_id,
                criteria_scores=scores,
                overall=overall,
                feedback=feedback,
            )

        except Exception as e:
            import sys
            print(f"  [judge] ERROR on {task_id}: {e}", file=sys.stderr)
            return JudgeScore(
                task_id=task_id,
                overall=0.0,
                feedback=f"Judge error: {e}",
            )
