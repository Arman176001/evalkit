"""LLM-as-judge scorer using Anthropic Claude."""

from __future__ import annotations

import json
import re

from .base import BaseScorer, ScoreResult

_SYSTEM = (
    "You are an expert evaluator. Score the model response on a scale of 1–5. "
    "Return ONLY valid JSON with no explanation outside the JSON: "
    '{"score": <1-5>, "reason": "<one sentence>"}'
)


class LLMJudgeScorer(BaseScorer):
    """Calls Claude to score the model output on a 1–5 scale, normalised to 0–1."""

    name = "llm_judge"
    threshold = 0.7

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _build_user_msg(self, prompt: str, output: str, expected: str | None, rubric: str | None) -> str:
        parts = [f"**Prompt given to model:**\n{prompt}", f"**Model output:**\n{output}"]
        if expected:
            parts.append(f"**Reference / expected answer:**\n{expected}")
        if rubric:
            parts.append(f"**Evaluation rubric:**\n{rubric}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Strip markdown fences then parse JSON."""
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def score(
        self,
        prompt: str,
        output: str,
        expected: str | None = None,
        rubric: str | None = None,
        **kwargs,
    ) -> ScoreResult:
        user_msg = self._build_user_msg(prompt, output, expected, rubric)
        client = self._get_client()

        for attempt in range(2):
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                data = self._parse_json(resp.content[0].text)
                raw = float(data["score"])
                normalized = max(0.0, min(1.0, raw / 5.0))
                return ScoreResult(
                    score=normalized,
                    passed=normalized >= self.threshold,
                    reason=data.get("reason", ""),
                    scorer_name=self.name,
                )
            except Exception as exc:
                if attempt == 1:
                    return ScoreResult(
                        score=0.0,
                        passed=False,
                        reason=f"Judge error: {exc}",
                        scorer_name=self.name,
                    )
        # unreachable
        return ScoreResult(score=0.0, passed=False, reason="Unknown error", scorer_name=self.name)
