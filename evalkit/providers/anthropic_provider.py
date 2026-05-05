"""Anthropic provider."""

from anthropic import Anthropic

from .base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
    """Wraps the Anthropic messages API."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        self._client = Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ProviderResponse:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ProviderResponse(
            text=text,
            model=self.model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
