"""Reporter — JSON results + HTML dashboard generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

from claweval.runner import TaskResult


RESULTS_DIR = Path(__file__).parent.parent / "results"

DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClawEval Results — {{ run_id }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 2rem; }
  h1, h2, h3 { color: #e6edf3; }
  h1 { margin-bottom: 0.5rem; font-size: 1.8rem; }
  .meta { color: #8b949e; margin-bottom: 2rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; }
  .chart-container { position: relative; height: 400px; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; cursor: pointer; user-select: none; }
  th:hover { color: #c9d1d9; }
  td { font-size: 0.9rem; }
  .score { font-weight: 700; }
  .score-high { color: #3fb950; }
  .score-mid { color: #d29922; }
  .score-low { color: #f85149; }
  .model-badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
                 font-size: 0.75rem; font-weight: 600; margin-right: 4px; }
  .efficiency { color: #bc8cff; font-weight: 600; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>🦞 ClawEval Results</h1>
<p class="meta">Run: {{ run_id }} &bull; {{ model_count }} model(s) &bull; {{ task_count }} task(s)</p>

<div class="grid">
  <div class="card">
    <h3>Category Scores (Radar)</h3>
    <div class="chart-container">
      <canvas id="radarChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h3>Speed Comparison</h3>
    <div class="chart-container">
      <canvas id="speedChart"></canvas>
    </div>
  </div>
</div>

<div class="card">
  <h3>Overall Scores</h3>
  <table>
    <thead><tr><th>Model</th><th>Overall</th>
    {% for cat in categories %}<th>{{ cat }}</th>{% endfor %}
    <th>Avg Time</th><th>Avg TTFT</th><th>RAM (GB)</th><th>Q/GB</th>
    </tr></thead>
    <tbody>
    {% for model_id, data in models.items() %}
    <tr>
      <td><strong>{{ data.name }}</strong></td>
      <td class="score {{ 'score-high' if data.overall >= 0.7 else ('score-mid' if data.overall >= 0.4 else 'score-low') }}">
        {{ "%.1f"|format(data.overall * 10) }}
      </td>
      {% for cat in categories %}
      <td class="score {{ 'score-high' if data.categories.get(cat, 0) >= 0.7 else ('score-mid' if data.categories.get(cat, 0) >= 0.4 else 'score-low') }}">
        {{ "%.1f"|format(data.categories.get(cat, 0) * 10) }}
      </td>
      {% endfor %}
      <td>{{ "%.2f"|format(data.speed.avg_wall_clock_ms / 1000) }}s</td>
      <td>{{ "%.0f"|format(data.speed.avg_ttft_ms) }}ms</td>
      <td>{{ "%.1f"|format(data.get('ram_gb', 0)) if data.get('ram_gb', 0) else '—' }}</td>
      <td class="efficiency">{{ "%.2f"|format(data.get('efficiency', {}).get('quality_per_gb', 0)) if data.get('efficiency', {}).get('quality_per_gb', 0) else '—' }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<div class="card" style="margin-top: 2rem;">
  <h3>Per-Task Detail</h3>
  <table>
    <thead><tr><th>Task</th><th>Category</th>
    {% for model_id in models %}<th>{{ model_id }}</th>{% endfor %}
    </tr></thead>
    <tbody>
    {% for task_id, task_data in task_details.items() %}
    <tr>
      <td>{{ task_id }}</td>
      <td>{{ task_data.category }}</td>
      {% for model_id in models %}
      <td class="score {{ 'score-high' if task_data.scores.get(model_id, 0) >= 0.7 else ('score-mid' if task_data.scores.get(model_id, 0) >= 0.4 else 'score-low') }}">
        {{ "%.2f"|format(task_data.scores.get(model_id, 0)) }}
      </td>
      {% endfor %}
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<script>
const categories = {{ categories_json }};
const modelColors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff', '#79c0ff', '#d2a8ff'];

// Radar chart
const radarData = {
  labels: categories,
  datasets: [
    {% for model_id, data in models.items() %}
    {
      label: '{{ data.name }}',
      data: [{% for cat in categories %}{{ "%.3f"|format(data.categories.get(cat, 0)) }}{{ "," if not loop.last }}{% endfor %}],
      borderColor: modelColors[{{ loop.index0 }} % modelColors.length],
      backgroundColor: modelColors[{{ loop.index0 }} % modelColors.length] + '20',
      pointRadius: 4,
    },
    {% endfor %}
  ]
};

new Chart(document.getElementById('radarChart'), {
  type: 'radar',
  data: radarData,
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2, color: '#8b949e' },
                    grid: { color: '#21262d' }, pointLabels: { color: '#c9d1d9' } } },
    plugins: { legend: { labels: { color: '#c9d1d9' } } }
  }
});

// Speed bar chart
const speedData = {
  labels: [{% for model_id, data in models.items() %}'{{ data.name }}'{{ "," if not loop.last }}{% endfor %}],
  datasets: [{
    label: 'Avg seconds per task',
    data: [{% for model_id, data in models.items() %}{{ "%.2f"|format(data.speed.avg_wall_clock_ms / 1000) }}{{ "," if not loop.last }}{% endfor %}],
    backgroundColor: modelColors.slice(0, {{ models|length }}),
  }]
};

new Chart(document.getElementById('speedChart'), {
  type: 'bar',
  data: speedData,
  options: {
    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
    scales: { x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
              y: { ticks: { color: '#c9d1d9' }, grid: { display: false } } },
    plugins: { legend: { display: false } }
  }
});

// Sortable tables
document.querySelectorAll('table').forEach(table => {
  const headers = table.querySelectorAll('th');
  let sortCol = -1, sortAsc = true;
  headers.forEach((th, idx) => {
    th.addEventListener('click', () => {
      if (sortCol === idx) { sortAsc = !sortAsc; } else { sortCol = idx; sortAsc = true; }
      headers.forEach(h => h.textContent = h.textContent.replace(/ [▲▼]$/, ''));
      th.textContent += sortAsc ? ' ▲' : ' ▼';
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        const aT = a.cells[idx]?.textContent.trim() ?? '';
        const bT = b.cells[idx]?.textContent.trim() ?? '';
        const aN = parseFloat(aT), bN = parseFloat(bT);
        if (!isNaN(aN) && !isNaN(bN)) return sortAsc ? aN - bN : bN - aN;
        return sortAsc ? aT.localeCompare(bT) : bT.localeCompare(aT);
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
});
</script>
</body>
</html>
"""


@dataclass
class ModelSummary:
    """Aggregated results for a single model."""

    name: str = ""
    overall: float = 0.0
    categories: dict[str, float] = None
    speed: dict[str, float] = None
    task_results: list[dict[str, Any]] = None
    efficiency: dict[str, float] = None
    ram_gb: float = 0.0

    def __post_init__(self):
        if self.categories is None:
            self.categories = {}
        if self.speed is None:
            self.speed = {"avg_tok_s": 0, "avg_ttft_ms": 0, "avg_gen_tok_s": 0, "avg_wall_clock_ms": 0}
        if self.task_results is None:
            self.task_results = []
        if self.efficiency is None:
            self.efficiency = {"quality_per_gb": 0, "quality_per_second": 0}


def aggregate_results(
    results: list[TaskResult],
    model_names: dict[str, str] | None = None,
) -> dict[str, ModelSummary]:
    """Aggregate TaskResults into per-model summaries."""
    model_names = model_names or {}
    by_model: dict[str, list[TaskResult]] = {}

    for r in results:
        by_model.setdefault(r.model_id, []).append(r)

    summaries: dict[str, ModelSummary] = {}

    for model_id, task_results in by_model.items():
        # Group by category
        by_cat: dict[str, list[TaskResult]] = {}
        for tr in task_results:
            cat = tr.task_id.rsplit("_", 1)[0] if "_" in tr.task_id else "unknown"
            # Extract category from task_id prefix (e.g., tool_calling_001 -> tool_calling)
            parts = tr.task_id.split("_")
            if len(parts) >= 3:
                cat = "_".join(parts[:-1])
            by_cat.setdefault(cat, []).append(tr)

        cat_scores: dict[str, float] = {}
        for cat, cat_results in by_cat.items():
            scores = [
                r.score.total_score
                for r in cat_results
                if r.score is not None
            ]
            cat_scores[cat] = sum(scores) / len(scores) if scores else 0.0

        # Speed metrics
        all_tok_s = [r.timing.tokens_per_second for r in task_results if r.timing.tokens_per_second > 0]
        all_ttft = [r.timing.ttft_ms for r in task_results if r.timing.ttft_ms > 0]
        all_gen_tok_s = [r.timing.estimated_gen_tok_s for r in task_results if r.timing.estimated_gen_tok_s > 0]

        overall = sum(cat_scores.values()) / len(cat_scores) if cat_scores else 0.0

        avg_tok_s = sum(all_tok_s) / len(all_tok_s) if all_tok_s else 0
        avg_ttft = sum(all_ttft) / len(all_ttft) if all_ttft else 0
        avg_gen_tok_s = sum(all_gen_tok_s) / len(all_gen_tok_s) if all_gen_tok_s else 0
        avg_wall_s = (
            sum(r.timing.wall_clock_ms for r in task_results) / len(task_results) / 1000
            if task_results else 0
        )

        # Get ram_gb from first result's model config if available
        ram_gb = 0.0
        # ram_gb is passed through model_names dict extended format or set externally

        # Efficiency metrics
        quality_per_gb = overall / ram_gb if ram_gb > 0 else 0
        quality_per_second = overall / avg_wall_s if avg_wall_s > 0 else 0

        summaries[model_id] = ModelSummary(
            name=model_names.get(model_id, model_id),
            overall=overall,
            categories=cat_scores,
            speed={
                "avg_tok_s": avg_tok_s,
                "avg_ttft_ms": avg_ttft,
                "avg_gen_tok_s": avg_gen_tok_s,
                "avg_wall_clock_ms": (
                    sum(r.timing.wall_clock_ms for r in task_results) / len(task_results)
                    if task_results else 0
                ),
            },
            task_results=[r.to_dict() for r in task_results],
            efficiency={
                "quality_per_gb": round(quality_per_gb, 4),
                "quality_per_second": round(quality_per_second, 4),
            },
            ram_gb=ram_gb,
        )

    return summaries


def save_json_results(
    results: list[TaskResult],
    model_names: dict[str, str] | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    filename: str | None = None,
) -> Path:
    """Save results as JSON."""
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summaries = aggregate_results(results, model_names)

    data = {
        "run_id": run_id,
        "models": {
            model_id: {
                "name": s.name,
                "overall": round(s.overall, 4),
                "categories": {k: round(v, 4) for k, v in s.categories.items()},
                "speed": s.speed,
                "efficiency": s.efficiency,
                "ram_gb": s.ram_gb,
                "tasks": s.task_results,
            }
            for model_id, s in summaries.items()
        },
    }

    if filename:
        output_path = output_dir / filename
    else:
        output_path = output_dir / f"results_{run_id.replace(':', '-')}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def generate_dashboard(
    results: list[TaskResult],
    model_names: dict[str, str] | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Generate an HTML dashboard from results."""
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summaries = aggregate_results(results, model_names)

    # Collect all categories
    all_cats: set[str] = set()
    for s in summaries.values():
        all_cats.update(s.categories.keys())
    categories = sorted(all_cats)

    # Build task details
    task_details: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.task_id not in task_details:
            parts = r.task_id.split("_")
            cat = "_".join(parts[:-1]) if len(parts) >= 2 else "unknown"
            task_details[r.task_id] = {"category": cat, "scores": {}}
        task_details[r.task_id]["scores"][r.model_id] = (
            r.score.total_score if r.score else 0.0
        )

    template = Template(DASHBOARD_TEMPLATE)
    html = template.render(
        run_id=run_id,
        model_count=len(summaries),
        task_count=len(task_details),
        categories=categories,
        categories_json=json.dumps(categories),
        models={
            mid: {
                "name": s.name,
                "overall": s.overall,
                "categories": s.categories,
                "speed": s.speed,
                "efficiency": s.efficiency,
                "ram_gb": s.ram_gb,
            }
            for mid, s in summaries.items()
        },
        task_details=task_details,
    )

    output_path = output_dir / f"dashboard_{run_id.replace(':', '-')}.html"
    with open(output_path, "w") as f:
        f.write(html)

    return output_path
