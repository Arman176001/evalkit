"""Provider registry and factory."""

from .base import BaseProvider, ProviderResponse
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def infer_provider(model: str) -> str:
    """Infer provider name from model string."""
    if model.startswith(("gpt-", "o1-", "o3-", "o4-", "text-")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return "openai"


def get_provider(model: str, api_key: str | None = None) -> BaseProvider:
    """Return a configured provider instance for the given model name."""
    provider_name = infer_provider(model)
    cls = PROVIDER_REGISTRY[provider_name]
    return cls(model=model, api_key=api_key)


__all__ = [
    "BaseProvider", "ProviderResponse",
    "OpenAIProvider", "AnthropicProvider",
    "PROVIDER_REGISTRY", "get_provider", "infer_provider",
]
