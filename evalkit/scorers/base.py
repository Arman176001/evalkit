"""Abstract base for scorers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScoreResult:
    score: float        # 0.0 – 1.0
    passed: bool        # score >= scorer threshold
    reason: str
    scorer_name: str


class BaseScorer(ABC):
    """Implement score() to add a new scoring strategy."""

    threshold: float = 0.7
    name: str = "base"

    @abstractmethod
    def score(
        self,
        prompt: str,
        output: str,
        expected: str | None = None,
        rubric: str | None = None,
        **kwargs,
    ) -> ScoreResult: ...
