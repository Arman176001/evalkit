"""Embedding cosine-similarity scorer. Lazy-loads sentence-transformers."""

from __future__ import annotations

import numpy as np

from .base import BaseScorer, ScoreResult

_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbedScorer(BaseScorer):
    """Scores output vs expected using cosine similarity of sentence embeddings."""

    name = "embed"
    threshold = 0.7
    _st_model = None  # class-level cache; loaded on first use

    @classmethod
    def _get_model(cls):
        if cls._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the embed scorer. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            cls._st_model = SentenceTransformer(_MODEL_NAME)
        return cls._st_model

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

        model = self._get_model()
        embeddings = model.encode([output.strip(), expected.strip()])
        a, b = embeddings[0], embeddings[1]
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        similarity = float(np.dot(a, b) / norm) if norm > 0 else 0.0
        similarity = max(0.0, min(1.0, similarity))

        return ScoreResult(
            score=similarity,
            passed=similarity >= self.threshold,
            reason=f"Cosine similarity: {similarity:.3f}",
            scorer_name=self.name,
        )
