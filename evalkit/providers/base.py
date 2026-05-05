"""Abstract base for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class BaseProvider(ABC):
    """Implement complete() to add a new provider — nothing else required."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ProviderResponse: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={getattr(self, 'model', '?')})"
