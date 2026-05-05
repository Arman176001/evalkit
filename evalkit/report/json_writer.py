"""Write and load run result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from ..runner import RunResult


def to_dict(run: RunResult) -> dict:
    """Serialise a RunResult to a JSON-compatible dict."""
    return {
        "run_id": run.run_id,
        "suite_name": run.suite.name,
        "model": run.model,
        "provider": run.provider,
        "timestamp": run.timestamp,
        "pass_rate": run.pass_rate,
        "total": run.total,
        "passed": run.passed,
        "failed": run.failed,
        "avg_latency_ms": run.avg_latency_ms,
        "total_tokens": run.total_tokens,
        "cases": [
            {
                "id": r.case.id,
                "prompt": r.case.prompt,
                "expected": r.case.expected,
                "output": r.output,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "scorer_results": [
                    {
                        "scorer_name": sr.scorer_name,
                        "score": sr.score,
                        "passed": sr.passed,
                        "reason": sr.reason,
                    }
                    for sr in r.scorer_results
                ],
            }
            for r in run.case_results
        ],
    }


def write_results(run: RunResult, output_dir: Path) -> Path:
    """Serialise run and save to <output_dir>/<run_id>.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = to_dict(run)
    path = output_dir / f"{run.run_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_results(path: Path) -> dict:
    """Load a previously saved run JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def find_previous_run(suite_name: str, runs_dir: Path) -> Path | None:
    """Return the most recent JSON for suite_name, or None."""
    if not runs_dir.exists():
        return None
    matches = sorted(runs_dir.glob(f"*_{suite_name}.json"), reverse=True)
    return matches[0] if matches else None
