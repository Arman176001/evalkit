"""Render a self-contained HTML report from a RunResult."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..runner import RunResult
from ..diff import DiffResult

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_html_report(
    run: RunResult,
    output_path: Path,
    diff: DiffResult | None = None,
) -> Path:
    """Render and write the HTML report. Returns the output path."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("report.html.j2")
    html = template.render(run=run, diff=diff)
    output_path.write_text(html, encoding="utf-8")
    return output_path
