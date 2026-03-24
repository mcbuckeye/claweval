# Post-Run Fixes — Apply After Current Run Completes

## 1. Remove response truncation
- File: `claweval/runner.py` line 62
- Change: `self.response_text[:2000]` → `self.response_text`
- Why: No reason to truncate saved responses. Judge scores full text at runtime, but post-analysis review needs full responses too.

## 2. Estimate tok/s from response length
- File: `claweval/runner.py` in `run_task()`
- LM Studio doesn't return `usage` in streaming responses
- Fix: Count response chars, estimate tokens (chars/4), compute tok/s from wall clock
- Also try: non-streaming fallback to get usage stats, or hit LM Studio stats endpoint

## 3. Re-run Coder-Next with fixes
- Current Coder-Next results have 57 truncated responses in saved JSON
- Scores are accurate (judge saw full text) but saved responses are cut off
- Re-run to get full response text saved for post-analysis

## 4. Commit and push all fixes + final results
- Push to mcbuckeye/claweval
- Include final dashboard HTML
