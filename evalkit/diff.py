"""Compare two run JSON snapshots and classify per-case changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiffResult:
    regressions: list[str] = field(default_factory=list)       # pass → fail
    improvements: list[str] = field(default_factory=list)      # fail → pass
    unchanged_pass: list[str] = field(default_factory=list)    # pass → pass
    unchanged_fail: list[str] = field(default_factory=list)    # fail → fail
    new_cases: list[str] = field(default_factory=list)         # in B, not A
    removed_cases: list[str] = field(default_factory=list)     # in A, not B

    @property
    def summary(self) -> str:
        parts = []
        if self.regressions:
            n = len(self.regressions)
            parts.append(f"{n} regression{'s' if n != 1 else ''}")
        if self.improvements:
            n = len(self.improvements)
            parts.append(f"{n} improvement{'s' if n != 1 else ''}")
        if not parts:
            return "no changes"
        unchanged = len(self.unchanged_pass) + len(self.unchanged_fail)
        if unchanged:
            parts.append(f"{unchanged} unchanged")
        return ", ".join(parts)


def compare_runs(run_a: dict, run_b: dict) -> DiffResult:
    """Classify each case in run_b relative to its status in run_a."""
    a_by_id: dict[str, bool] = {c["id"]: c["passed"] for c in run_a.get("cases", [])}
    b_by_id: dict[str, bool] = {c["id"]: c["passed"] for c in run_b.get("cases", [])}

    result = DiffResult()

    for case_id, b_passed in b_by_id.items():
        if case_id not in a_by_id:
            result.new_cases.append(case_id)
            continue
        a_passed = a_by_id[case_id]
        if a_passed and not b_passed:
            result.regressions.append(case_id)
        elif not a_passed and b_passed:
            result.improvements.append(case_id)
        elif a_passed and b_passed:
            result.unchanged_pass.append(case_id)
        else:
            result.unchanged_fail.append(case_id)

    for case_id in a_by_id:
        if case_id not in b_by_id:
            result.removed_cases.append(case_id)

    return result
