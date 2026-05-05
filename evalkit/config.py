"""Pydantic models for the YAML eval suite schema."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single test case within an eval suite."""

    id: str
    prompt: str
    expected: str | None = None
    input: str | None = None
    scorers: list[str] = Field(default_factory=lambda: ["exact"])
    rubric: str | None = None
    tags: list[str] = Field(default_factory=list)
    pattern: str | None = None  # regex pattern for ExactScorer


class EvalSuite(BaseModel):
    """Top-level eval suite loaded from a YAML file."""

    name: str
    description: str | None = None
    model: str = "gpt-4o-mini"
    judge_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    max_tokens: int = 512
    system: str | None = None
    cases: list[EvalCase]


def load_suite(path: str | Path) -> EvalSuite:
    """Load and validate a YAML eval suite file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvalSuite.model_validate(data)
