"""OpenAI provider."""

from openai import OpenAI

from .base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    """Wraps the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self._client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ProviderResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = resp.usage
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
