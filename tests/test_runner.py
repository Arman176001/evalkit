"""Tests for the benchmark runner."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
import pytest

from evalkit.providers.base import ProviderResponse


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_suite(cases: list[dict], **overrides) -> dict:
    base = {"name": "test-suite", "model": "gpt-4o-mini"}
    base.update(overrides)
    base["cases"] = cases
    return base


def _write_suite(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def _mock_provider(text: str = "mocked output") -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.complete.return_value = ProviderResponse(
        text=text, model="mock-model", prompt_tokens=10, completion_tokens=5
    )
    return provider


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRunner:
    def test_variable_substitution(self):
        """Prompt template {input} must be replaced before the provider call."""
        suite = _make_suite([{
            "id": "t1",
            "prompt": "Summarise: {input}",
            "input": "The sky is blue.",
            "expected": "sky",
            "scorers": ["contains"],
        }])
        path = _write_suite(suite)
        provider = _mock_provider("The sky is indeed blue.")

        from evalkit.runner import run_suite
        with patch("evalkit.runner.get_provider", return_value=provider):
            run_suite(path, provider=provider)

        call_prompt = provider.complete.call_args[0][0]
        assert "The sky is blue." in call_prompt
        assert "{input}" not in call_prompt

    def test_case_passes_when_all_scorers_pass(self):
        suite = _make_suite([{
            "id": "t1", "prompt": "What is 2+2?", "expected": "4", "scorers": ["exact"],
        }])
        path = _write_suite(suite)
        provider = _mock_provider("4")

        from evalkit.runner import run_suite
        with patch("evalkit.runner.get_provider", return_value=provider):
            result = run_suite(path, provider=provider)

        assert result.case_results[0].passed

    def test_case_fails_when_any_scorer_fails(self):
        suite = _make_suite([{
            "id": "t1", "prompt": "What is 2+2?", "expected": "4", "scorers": ["exact"],
        }])
        path = _write_suite(suite)
        provider = _mock_provider("five")

        from evalkit.runner import run_suite
        with patch("evalkit.runner.get_provider", return_value=provider):
            result = run_suite(path, provider=provider)

        assert not result.case_results[0].passed

    def test_model_override(self):
        """--model CLI flag must override the suite's model field."""
        suite = _make_suite([{
            "id": "t1", "prompt": "ping", "expected": "pong", "scorers": ["contains"],
        }], model="gpt-4o")
        path = _write_suite(suite)
        provider = _mock_provider("pong")

        from evalkit.runner import run_suite
        with patch("evalkit.runner.get_provider", return_value=provider) as gp:
            result = run_suite(path, model="claude-haiku-4-5-20251001", provider=provider)

        assert result.model == "claude-haiku-4-5-20251001"

    def test_progress_callback_called(self):
        suite = _make_suite([
            {"id": "t1", "prompt": "A", "expected": "a", "scorers": ["contains"]},
            {"id": "t2", "prompt": "B", "expected": "b", "scorers": ["contains"]},
        ])
        path = _write_suite(suite)
        provider = _mock_provider("a b")

        calls = []
        def cb(done, total, result):
            calls.append((done, total, result.case.id))

        from evalkit.runner import run_suite
        with patch("evalkit.runner.get_provider", return_value=provider):
            run_suite(path, provider=provider, progress_callback=cb)

        assert len(calls) == 2
        assert calls[0] == (1, 2, "t1")
        assert calls[1] == (2, 2, "t2")

    def test_provider_error_captured_as_failed_case(self):
        """A provider exception must not crash the runner; it yields a failed case."""
        suite = _make_suite([{
            "id": "t1", "prompt": "hello", "expected": "world", "scorers": ["exact"],
        }])
        path = _write_suite(suite)
        provider = MagicMock()
        provider.name = "mock"
        provider.complete.side_effect = RuntimeError("API is down")

        from evalkit.runner import run_suite
        result = run_suite(path, provider=provider)

        assert not result.case_results[0].passed
        assert "Provider error" in result.case_results[0].output

    def test_run_result_aggregates_correctly(self):
        suite = _make_suite([
            {"id": "t1", "prompt": "A", "expected": "4", "scorers": ["exact"]},
            {"id": "t2", "prompt": "B", "expected": "4", "scorers": ["exact"]},
        ])
        path = _write_suite(suite)

        provider = _mock_provider()
        # First call returns "4" (pass), second returns "wrong" (fail)
        provider.complete.side_effect = [
            ProviderResponse(text="4",     model="m", prompt_tokens=5, completion_tokens=1),
            ProviderResponse(text="wrong", model="m", prompt_tokens=5, completion_tokens=1),
        ]

        from evalkit.runner import run_suite
        result = run_suite(path, provider=provider)

        assert result.total == 2
        assert result.passed == 1
        assert result.failed == 1
        assert result.pass_rate == 0.5
