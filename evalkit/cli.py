"""evalkit CLI entry point — installed as the `evalkit` command."""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

import click
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

load_dotenv()
console = Console()

_RUNS_DIR = Path(".evalkit/runs")


# ──────────────────────────────────────────────────────────────────────────────
# Shared display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pass_cell(passed: bool) -> Text:
    return Text("PASS", style="bold green") if passed else Text("FAIL", style="bold red")


def _build_results_table(run) -> Table:
    table = Table(
        title=f"[bold]{run.suite.name}[/bold]  ·  {run.model}  ·  {run.timestamp}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("ID", style="bold dim", no_wrap=True)
    table.add_column("Prompt", max_width=36)
    table.add_column("Output", max_width=36)
    table.add_column("Scorers")
    table.add_column("Latency", justify="right", style="cyan")
    table.add_column("Result", justify="center")

    for r in run.case_results:
        scorer_text = "  ".join(
            f"{sr.scorer_name} {sr.score:.2f} {'[green]✓[/green]' if sr.passed else '[red]✗[/red]'}"
            for sr in r.scorer_results
        )
        table.add_row(
            r.case.id,
            r.case.prompt[:80],
            r.output.strip()[:80],
            scorer_text,
            f"{r.latency_ms:.0f}ms",
            _pass_cell(r.passed),
        )
    return table


def _build_compare_table(runs: list) -> Table:
    """Comparison table: rows = cases, columns = models."""
    table = Table(
        title=f"[bold]{runs[0].suite.name}[/bold]  ·  Model Comparison",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Case", style="bold dim", no_wrap=True)
    for run in runs:
        # Shorten model name to fit: strip common prefixes
        label = run.model.replace("claude-", "").replace("-20251001", "")[:22]
        table.add_column(label, justify="center")

    # Ordered case IDs from first run
    case_ids = [r.case.id for r in runs[0].case_results]
    # {model: {case_id: passed}}
    lookup = {
        run.model: {r.case.id: r.passed for r in run.case_results}
        for run in runs
    }

    for case_id in case_ids:
        cells: list = [case_id]
        for run in runs:
            passed = lookup[run.model].get(case_id)
            if passed is None:
                cells.append(Text("N/A", style="dim"))
            else:
                cells.append(_pass_cell(passed))
        table.add_row(*cells)

    table.add_section()

    def _rate_text(run) -> Text:
        rate = run.pass_rate
        color = "green" if rate == 1.0 else ("yellow" if rate >= 0.5 else "red")
        return Text(f"{rate:.0%}", style=f"bold {color}")

    table.add_row("Pass rate",  *[_rate_text(r) for r in runs])
    table.add_row("Avg latency", *[Text(f"{r.avg_latency_ms:.0f}ms", style="cyan") for r in runs])
    table.add_row("Total tokens", *[str(r.total_tokens) for r in runs])

    return table


def _print_run_summary(run, diff=None) -> None:
    rate_color = "green" if run.pass_rate == 1.0 else ("yellow" if run.pass_rate >= 0.5 else "red")
    console.print(
        f"\n  [{rate_color}]{'█' * int(run.pass_rate * 20)}[/]"
        f"{'░' * (20 - int(run.pass_rate * 20))}  "
        f"[bold]{run.pass_rate:.0%}[/bold]  "
        f"{run.passed}/{run.total} passed  ·  "
        f"avg {run.avg_latency_ms:.0f}ms  ·  "
        f"{run.total_tokens} tokens"
    )
    if diff:
        parts = []
        if diff.regressions:
            parts.append(f"[bold red]{len(diff.regressions)} regression{'s' if len(diff.regressions) != 1 else ''}[/]")
        if diff.improvements:
            parts.append(f"[bold blue]{len(diff.improvements)} improvement{'s' if len(diff.improvements) != 1 else ''}[/]")
        if parts:
            console.print(f"  Diff vs previous: {', '.join(parts)}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI group
# ──────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="evalkit")
def main() -> None:
    """evalkit — production-grade LLM evaluation harness."""


# ──────────────────────────────────────────────────────────────────────────────
# evalkit run
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="Override the suite's model field.")
@click.option("--output-dir", "-o", default=".evalkit/runs", show_default=True)
@click.option("--no-report", is_flag=True, help="Skip HTML report.")
def run(suite_path: str, model: str | None, output_dir: str, no_report: bool) -> None:
    """Run a YAML eval suite against one model."""
    from evalkit.runner import run_suite
    from evalkit.report.json_writer import write_results, find_previous_run, load_results, to_dict
    from evalkit.report.html import generate_html_report
    from evalkit.diff import compare_runs

    runs_dir = Path(output_dir)

    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console, transient=True) as progress:
        task = progress.add_task("Starting…", total=None)

        def on_progress(done: int, total: int, result) -> None:
            icon = "✓" if result.passed else "✗"
            progress.update(task, total=total, completed=done,
                            description=f"{icon}  [{done}/{total}]  {result.case.id}")

        try:
            result = run_suite(suite_path, model=model, progress_callback=on_progress)
        except Exception as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            sys.exit(1)

    console.print()
    console.print(_build_results_table(result))

    json_path = None
    try:
        json_path = write_results(result, runs_dir)
        console.print(f"\n[dim]Results saved →[/dim] {json_path}")
    except Exception as exc:
        console.print(f"\n[yellow]Warning: could not save results: {exc}[/yellow]")

    diff = None
    if json_path:
        prev = find_previous_run(result.suite.name, runs_dir)
        if prev and prev != json_path:
            try:
                diff = compare_runs(load_results(prev), to_dict(result))
            except Exception:
                pass

    _print_run_summary(result, diff)

    if not no_report:
        report_path = (runs_dir / f"{result.run_id}_report.html").resolve()
        try:
            generate_html_report(result, report_path, diff=diff)
            console.print(f"[dim]HTML report  →[/dim] {report_path}")
            webbrowser.open(report_path.as_uri())
        except Exception as exc:
            console.print(f"[yellow]Warning: HTML report failed: {exc}[/yellow]")

    console.print()
    sys.exit(0 if result.failed == 0 else 1)


# ──────────────────────────────────────────────────────────────────────────────
# evalkit compare
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--models", "-m", required=True,
              help="Comma-separated model names, e.g. gpt-4o-mini,gpt-4o,claude-haiku-4-5-20251001")
@click.option("--output-dir", "-o", default=".evalkit/runs", show_default=True)
def compare(suite_path: str, models: str, output_dir: str) -> None:
    """Run a suite against multiple models and print a side-by-side comparison."""
    from evalkit.runner import run_suite

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        console.print("[bold red]Error:[/bold red] --models needs at least two comma-separated model names.")
        sys.exit(1)

    runs_dir = Path(output_dir)
    runs = []

    console.print(f"\n[bold]Comparing {len(model_list)} models[/bold] on [cyan]{Path(suite_path).name}[/cyan]\n")

    for i, model in enumerate(model_list, 1):
        console.print(f"  [{i}/{len(model_list)}] [cyan]{model}[/cyan] …", end="")

        with Progress(SpinnerColumn(), TextColumn("  {task.description}"),
                      console=console, transient=True) as progress:
            task = progress.add_task("running", total=None)

            def on_progress(done: int, total: int, result, _task=task, _prog=progress, _m=model) -> None:
                _prog.update(_task, total=total, completed=done,
                             description=f"{done}/{total}  {result.case.id}")

            try:
                result = run_suite(suite_path, model=model, progress_callback=on_progress)
            except Exception as exc:
                console.print(f"\n  [bold red]Error with {model}:[/bold red] {exc}")
                sys.exit(1)

        rate_color = "green" if result.pass_rate == 1.0 else ("yellow" if result.pass_rate >= 0.5 else "red")
        console.print(
            f"\r  [{i}/{len(model_list)}] [cyan]{model}[/cyan]"
            f"  [{rate_color}]{result.pass_rate:.0%}[/]"
            f"  {result.passed}/{result.total} passed"
            f"  avg [cyan]{result.avg_latency_ms:.0f}ms[/cyan]"
        )
        runs.append(result)

        # Save individual run
        try:
            from evalkit.report.json_writer import write_results
            write_results(result, runs_dir)
        except Exception:
            pass

    console.print()
    console.print(_build_compare_table(runs))

    # Save comparison summary JSON
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        from evalkit.config import load_suite
        suite = load_suite(suite_path)
        ts = runs[0].timestamp if runs else ""
        compare_path = runs_dir / f"{runs[0].run_id.split('_')[0]}_compare_{suite.name}.json"
        compare_data = {
            "suite_name": suite.name,
            "timestamp": ts,
            "models": [
                {
                    "model": r.model,
                    "provider": r.provider,
                    "pass_rate": r.pass_rate,
                    "passed": r.passed,
                    "total": r.total,
                    "avg_latency_ms": r.avg_latency_ms,
                    "total_tokens": r.total_tokens,
                    "cases": {cr.case.id: cr.passed for cr in r.case_results},
                }
                for r in runs
            ],
        }
        compare_path.write_text(json.dumps(compare_data, indent=2), encoding="utf-8")
        console.print(f"\n[dim]Comparison saved →[/dim] {compare_path}")
    except Exception:
        pass

    console.print()
    # Exit 1 if any model had failures
    sys.exit(0 if all(r.failed == 0 for r in runs) else 1)


# ──────────────────────────────────────────────────────────────────────────────
# evalkit diff
# ──────────────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
def diff(run_a: str, run_b: str) -> None:
    """Compare two run JSON files and show regressions/improvements."""
    from evalkit.report.json_writer import load_results
    from evalkit.diff import compare_runs

    a = load_results(Path(run_a))
    b = load_results(Path(run_b))
    result = compare_runs(a, b)

    console.print(f"\n[bold]Diff:[/bold]  {Path(run_a).stem}  →  {Path(run_b).stem}\n")

    if result.regressions:
        console.print(f"[bold red]Regressions ({len(result.regressions)}):[/bold red]")
        for cid in result.regressions:
            console.print(f"  [red]✗[/red]  {cid}")

    if result.improvements:
        console.print(f"\n[bold blue]Improvements ({len(result.improvements)}):[/bold blue]")
        for cid in result.improvements:
            console.print(f"  [blue]✓[/blue]  {cid}")

    if result.new_cases:
        console.print(f"\n[bold green]New ({len(result.new_cases)}):[/bold green]")
        for cid in result.new_cases:
            console.print(f"  [green]+[/green]  {cid}")

    if result.removed_cases:
        console.print(f"\n[bold yellow]Removed ({len(result.removed_cases)}):[/bold yellow]")
        for cid in result.removed_cases:
            console.print(f"  [yellow]−[/yellow]  {cid}")

    unchanged = len(result.unchanged_pass) + len(result.unchanged_fail)
    console.print(f"\n[bold]Summary:[/bold]  {result.summary}"
                  + (f"  ({unchanged} unchanged)" if unchanged and "unchanged" not in result.summary else ""))
    console.print()


# ──────────────────────────────────────────────────────────────────────────────
# evalkit list-runs
# ──────────────────────────────────────────────────────────────────────────────

@main.command("list-runs")
@click.option("--runs-dir", default=".evalkit/runs", show_default=True)
def list_runs(runs_dir: str) -> None:
    """List all saved runs."""
    from evalkit.report.json_writer import load_results

    runs_path = Path(runs_dir)
    if not runs_path.exists():
        console.print("[dim]No runs yet. Run: evalkit run evals/example_factual.yaml[/dim]")
        return

    runs = sorted(runs_path.glob("*.json"), reverse=True)
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Run ID", style="dim")
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Cases", justify="right")
    table.add_column("Timestamp")

    for run_file in runs:
        try:
            data = load_results(run_file)
        except Exception:
            continue
        rate = data.get("pass_rate", 0.0)
        color = "green" if rate == 1.0 else ("yellow" if rate >= 0.5 else "red")
        table.add_row(
            data.get("run_id", run_file.stem),
            data.get("suite_name", "?"),
            data.get("model", "?"),
            Text(f"{rate:.0%}", style=color),
            str(data.get("total", "?")),
            data.get("timestamp", "?"),
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# evalkit show
# ──────────────────────────────────────────────────────────────────────────────

@main.command("show")
@click.argument("run_id")
@click.option("--runs-dir", default=".evalkit/runs", show_default=True)
def show(run_id: str, runs_dir: str) -> None:
    """Show case-level detail for a saved run."""
    from evalkit.report.json_writer import load_results

    matches = sorted(Path(runs_dir).glob(f"{run_id}*.json"), reverse=True)
    if not matches:
        console.print(f"[bold red]No run found matching '{run_id}'[/bold red]")
        sys.exit(1)

    data = load_results(matches[0])
    rate = data.get("pass_rate", 0)
    color = "green" if rate == 1.0 else ("yellow" if rate >= 0.5 else "red")

    console.print(f"\n[bold]{data['suite_name']}[/bold]  ·  {data['model']}")
    console.print(f"Timestamp : {data['timestamp']}")
    console.print(f"Pass rate : [{color}]{rate:.0%}[/]  ({data['passed']}/{data['total']})\n")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="bold dim")
    table.add_column("Output", max_width=60)
    table.add_column("Passed", justify="center")

    for case in data["cases"]:
        table.add_row(case["id"], case["output"].strip()[:80], _pass_cell(case["passed"]))

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
