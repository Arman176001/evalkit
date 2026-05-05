"""Shared utilities."""

from __future__ import annotations


def substitute_vars(template: str, input_val: str | None = None, **kwargs: str) -> str:
    """Replace {input} and other {key} placeholders in a prompt template."""
    if input_val is not None:
        template = template.replace("{input}", input_val)
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate a string with ellipsis if it exceeds max_len."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"
