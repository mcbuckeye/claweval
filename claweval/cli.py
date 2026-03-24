"""CLI entry point for ClawEval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from claweval import __version__
from claweval.config import load_config
from claweval.task_loader import load_tasks, list_tasks, TASKS_DIR
from claweval.runner import run_tasks, TaskResult, TimingInfo
from claweval.reporter import save_json_results, generate_dashboard


DEFAULT_CONFIG = Path("config.yaml")


@click.group()
@click.version_option(version=__version__)
def cli():
    """ClawEval — OpenClaw Model Evaluation Suite."""
    pass


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[TaskResult], dict[str, str], set[str]]:
    """Load checkpoint file and return results, model names, and completed task keys."""
    results: list[TaskResult] = []
    model_names: dict[str, str] = {}
    completed: set[str] = set()

    if not checkpoint_path.exists():
        return results, model_names, completed

    with open(checkpoint_path) as f:
        data = json.load(f)

    for model_id, model_data in data.get("models", {}).items():
        model_names[model_id] = model_data.get("name", model_id)
        for task_data in model_data.get("tasks", []):
            from claweval.scorer import ScoreResult

            score_data = task_data.get("score")
            score = ScoreResult(
                task_id=task_data["task_id"],
                total_score=score_data["total_score"],
                breakdown=score_data.get("breakdown", {}),
                details=score_data.get("details", {}),
            ) if score_data else None

            timing_data = task_data.get("timing", {})
            timing = TimingInfo(
                wall_clock_ms=timing_data.get("wall_clock_ms", 0),
                ttft_ms=timing_data.get("ttft_ms", 0),
                total_tokens=timing_data.get("total_tokens", 0),
                prompt_tokens=timing_data.get("prompt_tokens", 0),
                completion_tokens=timing_data.get("completion_tokens", 0),
                tokens_per_second=timing_data.get("tokens_per_second", 0),
            )

            result = TaskResult(
                task_id=task_data["task_id"],
                model_id=model_id,
                score=score,
                timing=timing,
                response_text=task_data.get("response_text", ""),
                tool_calls_made=task_data.get("tool_calls_made", []),
                error=task_data.get("error", ""),
            )
            results.append(result)
            completed.add(f"{model_id}::{task_data['task_id']}")

    return results, model_names, completed


@cli.command()
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG),
              help="Path to config.yaml", type=click.Path())
@click.option("--category", multiple=True, help="Run only specific categories")
@click.option("--model", "model_filter", help="Run only a specific model ID")
@click.option("--quick", is_flag=True, help="Quick mode: 2 tasks per category")
@click.option("--output", "output_dir", default="results",
              help="Output directory for results")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
@click.option("--scoring", type=click.Choice(["deterministic", "judge", "hybrid"]),
              default=None, help="Scoring mode (overrides config)")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard"]),
              default=None, help="Filter tasks by difficulty")
@click.option("--context-stress", is_flag=True, help="Run context stress tests only")
@click.option("--all", "run_all", is_flag=True, help="Run everything including stress tests")
def run(config_path: str, category: tuple[str, ...], model_filter: str | None,
        quick: bool, output_dir: str, resume: bool, scoring: str | None,
        difficulty: str | None, context_stress: bool, run_all: bool):
    """Run evaluation suite against configured models."""
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Determine scoring mode
    scoring_mode = scoring or cfg.settings.scoring_mode

    # Set up judge scorer if needed
    judge_scorer = None
    if scoring_mode in ("judge", "hybrid"):
        import os
        from claweval.judge import JudgeScorer
        use_cli = cfg.settings.raw.get("judge_use_cli", False)
        api_key = os.environ.get(cfg.settings.judge_api_key_env, "") if not use_cli else ""
        if not api_key and not use_cli:
            click.echo(f"Warning: {cfg.settings.judge_api_key_env} not set, falling back to deterministic scoring", err=True)
            scoring_mode = "deterministic"
        else:
            judge_scorer = JudgeScorer(api_key=api_key, use_cli=use_cli)

    categories = list(category) if category else cfg.settings.categories

    if context_stress:
        categories = ["context_stress"]
    elif run_all:
        categories = cfg.settings.categories + ["context_stress"]

    tasks = load_tasks(categories)

    if not tasks:
        click.echo("No tasks found. Check your tasks directory.", err=True)
        sys.exit(1)

    # Filter by difficulty
    if difficulty:
        tasks = [t for t in tasks if t.difficulty == difficulty]
        if not tasks:
            click.echo(f"No tasks with difficulty '{difficulty}' found.", err=True)
            sys.exit(1)

    if quick:
        by_cat: dict[str, list] = {}
        for t in tasks:
            by_cat.setdefault(t.category, []).append(t)
        tasks = []
        for cat_tasks in by_cat.values():
            tasks.extend(cat_tasks[:2])

    models = cfg.models
    if model_filter:
        models = [m for m in models if m.id == model_filter]
        if not models:
            click.echo(f"Model '{model_filter}' not found in config.", err=True)
            sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.json"

    # Load checkpoint if resuming
    all_results: list[TaskResult] = []
    model_names: dict[str, str] = {}
    completed_keys: set[str] = set()

    if resume:
        all_results, model_names, completed_keys = _load_checkpoint(checkpoint_path)
        if completed_keys:
            click.echo(f"Resuming: {len(completed_keys)} tasks already completed")

    click.echo(f"ClawEval v{__version__}")
    click.echo(f"Running {len(tasks)} tasks against {len(models)} model(s)")
    click.echo(f"Scoring mode: {scoring_mode}\n")

    for model in models:
        model_names[model.id] = model.name
        
        # Filter out already-completed tasks for this model
        model_tasks = [t for t in tasks if f"{model.id}::{t.id}" not in completed_keys]
        
        if not model_tasks:
            click.echo(f"▸ Model: {model.name} ({model.id}) — all tasks complete, skipping")
            continue

        skipped = len(tasks) - len(model_tasks)
        skip_msg = f" (skipping {skipped} completed)" if skipped else ""
        click.echo(f"▸ Model: {model.name} ({model.id}){skip_msg}")

        def on_complete(result: TaskResult):
            status = "✓" if (result.score and result.score.total_score >= 0.5) else "✗"
            score_str = f"{result.score.total_score:.2f}" if result.score else "ERR"
            click.echo(f"  {status} {result.task_id}: {score_str} ({result.timing.wall_clock_ms:.0f}ms)")
            if result.error:
                click.echo(f"    Error: {result.error}")

            # Save checkpoint after every task
            all_results.append(result)
            try:
                save_json_results(all_results, model_names, out, filename="checkpoint.json")
            except Exception:
                pass  # Don't let checkpoint save failure kill the run

        run_tasks(
            tasks=model_tasks, model=model, settings=cfg.settings,
            on_complete=on_complete, scoring_mode=scoring_mode,
            judge_scorer=judge_scorer,
        )
        click.echo()

    # Save final results
    json_path = save_json_results(all_results, model_names, out)
    click.echo(f"Results saved: {json_path}")

    html_path = generate_dashboard(all_results, model_names, out, tasks=tasks)
    click.echo(f"Dashboard:     {html_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


@cli.command()
@click.option("--results-dir", default="results", help="Results directory")
@click.option("--results-file", help="Specific results JSON file")
def report(results_dir: str, results_file: str | None):
    """Generate HTML dashboard from existing results."""
    results_path = Path(results_dir)

    if results_file:
        json_file = Path(results_file)
    else:
        # Find most recent results file
        json_files = sorted(results_path.glob("results_*.json"))
        if not json_files:
            click.echo("No results files found.", err=True)
            sys.exit(1)
        json_file = json_files[-1]

    click.echo(f"Loading: {json_file}")
    with open(json_file) as f:
        data = json.load(f)

    # Reconstruct TaskResults from JSON
    all_results: list[TaskResult] = []
    model_names: dict[str, str] = {}

    for model_id, model_data in data.get("models", {}).items():
        model_names[model_id] = model_data.get("name", model_id)
        for task_data in model_data.get("tasks", []):
            from claweval.runner import TaskResult as TR, TimingInfo
            from claweval.scorer import ScoreResult

            score_data = task_data.get("score")
            score = ScoreResult(
                task_id=task_data["task_id"],
                total_score=score_data["total_score"],
                breakdown=score_data.get("breakdown", {}),
                details=score_data.get("details", {}),
                judge_score=score_data.get("judge_score"),
            ) if score_data else None

            timing_data = task_data.get("timing", {})
            timing = TimingInfo(
                wall_clock_ms=timing_data.get("wall_clock_ms", 0),
                ttft_ms=timing_data.get("ttft_ms", 0),
                total_tokens=timing_data.get("total_tokens", 0),
                prompt_tokens=timing_data.get("prompt_tokens", 0),
                completion_tokens=timing_data.get("completion_tokens", 0),
                tokens_per_second=timing_data.get("tokens_per_second", 0),
            )

            all_results.append(TR(
                task_id=task_data["task_id"],
                model_id=model_id,
                score=score,
                timing=timing,
                response_text=task_data.get("response_text", ""),
                tool_calls_made=task_data.get("tool_calls_made", []),
                error=task_data.get("error", ""),
            ))

    # Load task metadata from YAML files
    all_tasks = load_tasks()

    html_path = generate_dashboard(
        all_results, model_names, results_path,
        run_id=data.get("run_id", "unknown"),
        tasks=all_tasks,
    )
    click.echo(f"Dashboard generated: {html_path}")


@cli.command()
@click.option("--category", multiple=True, help="Filter by category")
def tasks(category: tuple[str, ...]):
    """List available evaluation tasks."""
    cats = list(category) if category else None
    grouped = list_tasks(cats)

    if not grouped:
        click.echo("No tasks found.")
        return

    total = 0
    for cat, cat_tasks in sorted(grouped.items()):
        click.echo(f"\n📁 {cat} ({len(cat_tasks)} tasks)")
        for t in cat_tasks:
            diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(t.difficulty, "⚪")
            click.echo(f"  {diff_emoji} {t.id}: {t.name}")
            total += 1

    click.echo(f"\nTotal: {total} tasks")


@cli.command()
@click.argument("model_a")
@click.argument("model_b")
@click.option("--results-dir", default="results", help="Results directory")
def compare(model_a: str, model_b: str, results_dir: str):
    """Compare two models from existing results."""
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("results_*.json"))

    if not json_files:
        click.echo("No results files found.", err=True)
        sys.exit(1)

    # Load the most recent results
    with open(json_files[-1]) as f:
        data = json.load(f)

    models = data.get("models", {})

    if model_a not in models:
        click.echo(f"Model '{model_a}' not found in results.", err=True)
        sys.exit(1)
    if model_b not in models:
        click.echo(f"Model '{model_b}' not found in results.", err=True)
        sys.exit(1)

    a_data = models[model_a]
    b_data = models[model_b]

    click.echo(f"\n{'Category':<20} {'Model A':>10} {'Model B':>10} {'Winner':>10}")
    click.echo("─" * 55)

    a_cats = a_data.get("categories", {})
    b_cats = b_data.get("categories", {})
    all_cats = sorted(set(list(a_cats.keys()) + list(b_cats.keys())))

    a_wins = 0
    b_wins = 0

    for cat in all_cats:
        a_score = a_cats.get(cat, 0)
        b_score = b_cats.get(cat, 0)
        if a_score > b_score:
            winner = "← A"
            a_wins += 1
        elif b_score > a_score:
            winner = "B →"
            b_wins += 1
        else:
            winner = "tie"

        click.echo(f"{cat:<20} {a_score:>10.3f} {b_score:>10.3f} {winner:>10}")

    click.echo("─" * 55)
    a_overall = a_data.get("overall", 0)
    b_overall = b_data.get("overall", 0)
    overall_winner = "← A" if a_overall > b_overall else ("B →" if b_overall > a_overall else "tie")
    click.echo(f"{'Overall':<20} {a_overall:>10.3f} {b_overall:>10.3f} {overall_winner:>10}")
    click.echo(f"\nCategory wins: A={a_wins} B={b_wins}")


if __name__ == "__main__":
    cli()
