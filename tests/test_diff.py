"""Tests for the diff/regression detection system."""

import pytest

from evalkit.diff import compare_runs, DiffResult


def _run(*cases: tuple[str, bool]) -> dict:
    """Build a minimal run dict from (case_id, passed) pairs."""
    return {"cases": [{"id": cid, "passed": passed} for cid, passed in cases]}


class TestComparRuns:
    def test_regression_detected(self):
        a = _run(("c1", True))
        b = _run(("c1", False))
        diff = compare_runs(a, b)
        assert "c1" in diff.regressions
        assert "c1" not in diff.improvements
        assert "c1" not in diff.unchanged_pass
        assert "c1" not in diff.unchanged_fail

    def test_improvement_detected(self):
        a = _run(("c1", False))
        b = _run(("c1", True))
        diff = compare_runs(a, b)
        assert "c1" in diff.improvements
        assert "c1" not in diff.regressions

    def test_unchanged_pass(self):
        a = _run(("c1", True))
        b = _run(("c1", True))
        diff = compare_runs(a, b)
        assert "c1" in diff.unchanged_pass

    def test_unchanged_fail(self):
        a = _run(("c1", False))
        b = _run(("c1", False))
        diff = compare_runs(a, b)
        assert "c1" in diff.unchanged_fail

    def test_new_case_in_b(self):
        a = _run(("old", True))
        b = _run(("old", True), ("new", True))
        diff = compare_runs(a, b)
        assert "new" in diff.new_cases
        assert "old" not in diff.new_cases

    def test_removed_case(self):
        a = _run(("old", True), ("gone", True))
        b = _run(("old", True))
        diff = compare_runs(a, b)
        assert "gone" in diff.removed_cases
        assert "old" not in diff.removed_cases

    def test_mixed_scenario(self):
        a = _run(("reg", True), ("imp", False), ("same", True))
        b = _run(("reg", False), ("imp", True), ("same", True), ("new", True))
        diff = compare_runs(a, b)

        assert "reg" in diff.regressions
        assert "imp" in diff.improvements
        assert "same" in diff.unchanged_pass
        assert "new" in diff.new_cases

    def test_empty_runs(self):
        diff = compare_runs(_run(), _run())
        assert not diff.regressions
        assert not diff.improvements
        assert not diff.new_cases
        assert not diff.removed_cases


class TestDiffResultSummary:
    def test_summary_with_regression(self):
        diff = DiffResult(regressions=["c1"], improvements=[])
        assert "regression" in diff.summary

    def test_summary_plural(self):
        diff = DiffResult(regressions=["c1", "c2"], improvements=["c3"])
        assert "2 regressions" in diff.summary
        assert "1 improvement" in diff.summary

    def test_summary_no_changes(self):
        diff = DiffResult(unchanged_pass=["c1", "c2"])
        assert diff.summary == "no changes"

    def test_summary_unchanged_count(self):
        diff = DiffResult(
            regressions=["c1"],
            unchanged_pass=["c2"],
            unchanged_fail=["c3"],
        )
        summary = diff.summary
        assert "regression" in summary
        assert "unchanged" in summary
