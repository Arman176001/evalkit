"""Exact match, substring, and regex scorers."""

from __future__ import annotations

import re

from .base import BaseScorer, ScoreResult


class ExactScorer(BaseScorer):
    """
    Scores string similarity with three levels:
      exact match → 1.0 | substring → 0.5 | no match → 0.0
    Optional regex pattern overrides substring check → 1.0 on match.
    Default threshold 0.7 means only exact and regex matches pass.
    """

    name = "exact"
    threshold = 0.7

    def score(
        self,
        prompt: str,
        output: str,
        expected: str | None = None,
        rubric: str | None = None,
        pattern: str | None = None,
        **kwargs,
    ) -> ScoreResult:
        if expected is None and pattern is None:
            return ScoreResult(score=0.0, passed=False, reason="No expected value or pattern", scorer_name=self.name)

        out = output.strip().lower()

        if pattern:
            matched = bool(re.search(pattern, output, re.IGNORECASE))
            score = 1.0 if matched else 0.0
            return ScoreResult(
                score=score,
                passed=score >= self.threshold,
                reason=f"Regex {'match' if matched else 'no match'}: {pattern!r}",
                scorer_name=self.name,
            )

        exp = expected.strip().lower()  # type: ignore[union-attr]

        if out == exp:
            return ScoreResult(score=1.0, passed=True, reason="Exact match", scorer_name=self.name)

        if exp in out:
            score = 0.5
            return ScoreResult(
                score=score,
                passed=score >= self.threshold,
                reason="Substring match",
                scorer_name=self.name,
            )

        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"No match — got {output.strip()[:60]!r}",
            scorer_name=self.name,
        )


class ContainsScorer(BaseScorer):
    """Pass iff expected appears anywhere in the output (case-insensitive). Score is binary 1/0."""

    name = "contains"
    threshold = 0.7

    def score(
        self,
        prompt: str,
        output: str,
        expected: str | None = None,
        rubric: str | None = None,
        **kwargs,
    ) -> ScoreResult:
        if expected is None:
            return ScoreResult(score=0.0, passed=False, reason="No expected value", scorer_name=self.name)

        found = expected.strip().lower() in output.strip().lower()
        score = 1.0 if found else 0.0
        return ScoreResult(
            score=score,
            passed=score >= self.threshold,
            reason="Found in output" if found else f"{expected!r} not found in output",
            scorer_name=self.name,
        )
