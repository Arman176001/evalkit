"""Core benchmark runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import EvalCase, EvalSuite, load_suite
from .providers import get_provider
from .providers.base import BaseProvider, ProviderResponse
from .scorers import get_scorer
from .scorers.base import ScoreResult
from .utils import substitute_vars


@dataclass
class CaseResult:
    case: EvalCase
    output: str
    scorer_results: list[ScoreResult]
    passed: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RunResult:
    suite: EvalSuite
    model: str
    provider: str
    timestamp: str
    run_id: str
    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.case_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.case_results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.case_results) / len(self.case_results) if self.case_results else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.case_results)


ProgressCallback = Callable[[int, int, CaseResult], None]


def run_suite(
    suite_path: str | Path,
    model: str | None = None,
    provider: BaseProvider | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RunResult:
    """Load a YAML suite, run all cases, return aggregated results."""
    suite = load_suite(suite_path)
    effective_model = model or suite.model
    llm = provider or get_provider(effective_model)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = now.strftime("%Y%m%dT%H%M%S") + f"_{suite.name}"

    case_results: list[CaseResult] = []

    for i, case in enumerate(suite.cases):
        prompt = substitute_vars(case.prompt, case.input)

        try:
            t0 = time.perf_counter()
            resp: ProviderResponse = llm.complete(
                prompt,
                system=suite.system,
                temperature=suite.temperature,
                max_tokens=suite.max_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            output = resp.text
            prompt_tokens = resp.prompt_tokens
            completion_tokens = resp.completion_tokens
        except Exception as exc:
            output = f"[Provider error: {exc}]"
            latency_ms = 0.0
            prompt_tokens = 0
            completion_tokens = 0

        scorer_results: list[ScoreResult] = []
        for scorer_name in case.scorers:
            scorer = get_scorer(scorer_name, suite)
            try:
                result = scorer.score(
                    prompt=prompt,
                    output=output,
                    expected=case.expected,
                    rubric=case.rubric,
                    pattern=case.pattern,
                )
            except Exception as exc:
                result = ScoreResult(
                    score=0.0, passed=False,
                    reason=f"Scorer error: {exc}",
                    scorer_name=scorer_name,
                )
            scorer_results.append(result)

        passed = bool(scorer_results) and all(sr.passed for sr in scorer_results)
        cr = CaseResult(
            case=case,
            output=output,
            scorer_results=scorer_results,
            passed=passed,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        case_results.append(cr)

        if progress_callback:
            progress_callback(i + 1, len(suite.cases), cr)

    return RunResult(
        suite=suite,
        model=effective_model,
        provider=llm.name,
        timestamp=timestamp,
        run_id=run_id,
        case_results=case_results,
    )
