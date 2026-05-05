"""Tests for all scorer implementations."""

from unittest.mock import MagicMock, patch

import pytest

from evalkit.scorers.exact import ContainsScorer, ExactScorer
from evalkit.scorers.llm_judge import LLMJudgeScorer


# ──────────────────────────────────────────────────────────────────────────────
# ExactScorer
# ──────────────────────────────────────────────────────────────────────────────

class TestExactScorer:
    def setup_method(self):
        self.scorer = ExactScorer()

    def test_perfect_match(self):
        r = self.scorer.score("", "Paris", expected="Paris")
        assert r.score == 1.0
        assert r.passed

    def test_case_insensitive_exact(self):
        r = self.scorer.score("", "paris", expected="Paris")
        assert r.score == 1.0
        assert r.passed

    def test_strips_whitespace(self):
        r = self.scorer.score("", "  Paris  ", expected="Paris")
        assert r.score == 1.0
        assert r.passed

    def test_substring_score_is_half(self):
        r = self.scorer.score("", "The capital is Paris.", expected="Paris")
        assert r.score == 0.5

    def test_substring_does_not_pass_at_default_threshold(self):
        # threshold=0.7, substring score=0.5 → fail
        r = self.scorer.score("", "The capital is Paris.", expected="Paris")
        assert not r.passed

    def test_no_match(self):
        r = self.scorer.score("", "London", expected="Paris")
        assert r.score == 0.0
        assert not r.passed

    def test_regex_match(self):
        r = self.scorer.score("", "The answer is 42", expected="ignored", pattern=r"\d+")
        assert r.score == 1.0
        assert r.passed

    def test_regex_no_match(self):
        r = self.scorer.score("", "no digits here", expected="ignored", pattern=r"\d+")
        assert r.score == 0.0
        assert not r.passed

    def test_no_expected_no_pattern(self):
        r = self.scorer.score("", "anything")
        assert r.score == 0.0
        assert not r.passed


# ──────────────────────────────────────────────────────────────────────────────
# ContainsScorer
# ──────────────────────────────────────────────────────────────────────────────

class TestContainsScorer:
    def setup_method(self):
        self.scorer = ContainsScorer()

    def test_exact_hit(self):
        r = self.scorer.score("", "Paris", expected="Paris")
        assert r.score == 1.0
        assert r.passed

    def test_substring_hit(self):
        r = self.scorer.score("", "The answer is mitochondria.", expected="mitochondria")
        assert r.score == 1.0
        assert r.passed

    def test_case_insensitive(self):
        r = self.scorer.score("", "MITOCHONDRIA", expected="mitochondria")
        assert r.score == 1.0
        assert r.passed

    def test_miss(self):
        r = self.scorer.score("", "London", expected="Paris")
        assert r.score == 0.0
        assert not r.passed

    def test_no_expected(self):
        r = self.scorer.score("", "output")
        assert r.score == 0.0
        assert not r.passed


# ──────────────────────────────────────────────────────────────────────────────
# LLMJudgeScorer
# ──────────────────────────────────────────────────────────────────────────────

def _mock_client(response_text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


class TestLLMJudgeScorer:
    def setup_method(self):
        self.scorer = LLMJudgeScorer()

    def _score_with_mock(self, response_text: str, **kwargs):
        with patch.object(self.scorer, "_get_client", return_value=_mock_client(response_text)):
            return self.scorer.score("prompt", "output", **kwargs)

    def test_score_normalised(self):
        r = self._score_with_mock('{"score": 4, "reason": "Good answer"}')
        assert abs(r.score - 0.8) < 1e-6
        assert r.passed  # 0.8 >= 0.7

    def test_score_5_is_1(self):
        r = self._score_with_mock('{"score": 5, "reason": "Perfect"}')
        assert r.score == 1.0

    def test_score_1_is_0_2(self):
        r = self._score_with_mock('{"score": 1, "reason": "Bad"}')
        assert abs(r.score - 0.2) < 1e-6
        assert not r.passed

    def test_reason_propagated(self):
        r = self._score_with_mock('{"score": 3, "reason": "Partially correct"}')
        assert "Partially correct" in r.reason

    def test_strips_markdown_fences(self):
        r = self._score_with_mock("```json\n{\"score\": 3, \"reason\": \"ok\"}\n```")
        assert abs(r.score - 0.6) < 1e-6

    def test_handles_parse_error_gracefully(self):
        # Both attempts return invalid JSON
        client = _mock_client("not json at all")
        with patch.object(self.scorer, "_get_client", return_value=client):
            r = self.scorer.score("prompt", "output")
        assert r.score == 0.0
        assert not r.passed
        assert "error" in r.reason.lower()

    def test_scorer_name(self):
        r = self._score_with_mock('{"score": 4, "reason": "good"}')
        assert r.scorer_name == "llm_judge"


# ──────────────────────────────────────────────────────────────────────────────
# EmbedScorer (light smoke test — avoids downloading the model)
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbedScorer:
    def test_no_expected(self):
        from evalkit.scorers.embed import EmbedScorer
        scorer = EmbedScorer()
        r = scorer.score("prompt", "output", expected=None)
        assert r.score == 0.0
        assert not r.passed

    def test_scorer_name(self):
        from evalkit.scorers.embed import EmbedScorer
        scorer = EmbedScorer()
        assert scorer.name == "embed"

    def test_high_similarity_passes(self):
        """Mocks the ST model to verify score/pass logic without downloading weights."""
        import numpy as np
        from evalkit.scorers.embed import EmbedScorer

        scorer = EmbedScorer()
        fake_model = MagicMock()
        # Two identical unit vectors → cosine similarity 1.0
        vec = np.array([1.0, 0.0, 0.0])
        fake_model.encode.return_value = np.array([vec, vec])

        with patch.object(EmbedScorer, "_get_model", return_value=fake_model):
            r = scorer.score("p", "same sentence", expected="same sentence")

        assert abs(r.score - 1.0) < 1e-6
        assert r.passed

    def test_low_similarity_fails(self):
        import numpy as np
        from evalkit.scorers.embed import EmbedScorer

        scorer = EmbedScorer()
        fake_model = MagicMock()
        # Orthogonal vectors → cosine similarity 0.0
        fake_model.encode.return_value = np.array([
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ])

        with patch.object(EmbedScorer, "_get_model", return_value=fake_model):
            r = scorer.score("p", "apples", expected="skyscraper theory")

        assert abs(r.score) < 1e-6
        assert not r.passed
