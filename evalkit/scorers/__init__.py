"""Scorer registry and factory."""

from .base import BaseScorer, ScoreResult
from .exact import ExactScorer, ContainsScorer
from .llm_judge import LLMJudgeScorer
from .embed import EmbedScorer

SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    "exact": ExactScorer,
    "contains": ContainsScorer,
    "llm_judge": LLMJudgeScorer,
    "embed": EmbedScorer,
}


def get_scorer(name: str, suite=None) -> BaseScorer:
    """Return a configured scorer instance.

    Passes judge_model from the suite to LLMJudgeScorer when available.
    """
    if name == "exact":
        return ExactScorer()
    if name == "contains":
        return ContainsScorer()
    if name == "llm_judge":
        judge_model = getattr(suite, "judge_model", "claude-haiku-4-5-20251001") if suite else "claude-haiku-4-5-20251001"
        return LLMJudgeScorer(model=judge_model)
    if name == "embed":
        return EmbedScorer()
    raise ValueError(f"Unknown scorer: {name!r}. Available: {list(SCORER_REGISTRY)}")


__all__ = [
    "BaseScorer", "ScoreResult",
    "ExactScorer", "ContainsScorer", "LLMJudgeScorer", "EmbedScorer",
    "SCORER_REGISTRY", "get_scorer",
]
