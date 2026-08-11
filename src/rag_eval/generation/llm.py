"""The language model boundary: a protocol, and a stdlib client behind it."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_GENERATION_MODEL",
    "ClaudeLanguageModel",
    "GenerationError",
    "LanguageModel",
]

DEFAULT_GENERATION_MODEL = "claude-sonnet-4-6"

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class GenerationError(RuntimeError):
    """The model could not be reached, or returned something unusable."""


@runtime_checkable
class LanguageModel(Protocol):
    """Turns a system prompt and a user message into text.

    Injected for the same reason encoders and rerankers are: no unit test may reach the
    network. The protocol is deliberately this thin — generation needs completion and
    nothing else, and a wider interface would only be harder to fake.
    """

    def complete(self, system: str, prompt: str) -> str:
        """Return the model's reply as plain text."""
        ...


class ClaudeLanguageModel:
    """Anthropic Messages API client built on ``urllib``.

    No SDK, for the same reason there is no vector database here: this is one HTTPS POST
    to one endpoint, and writing it out keeps the dependency list honest and the
    mechanics visible. The tradeoff is real and accepted — no streaming, no automatic
    retries, no connection pooling. None of those matter for answering one question at
    a time behind a retriever that costs more than the request does.

    The API key is read from the environment at construction so a missing key fails
    immediately rather than after a document has already been indexed.
    """

    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key.strip():
            raise GenerationError(
                "no API key: set ANTHROPIC_API_KEY in the environment or pass api_key"
            )

        self.model = model
        self._api_key = key
        self._max_tokens = max_tokens
        # Zero by default: an answer that changes between runs cannot be checked against
        # its own citations, and grounding is the whole point.
        self._temperature = temperature
        self._timeout = timeout

    def complete(self, system: str, prompt: str) -> str:
        """POST one message and return the concatenated text blocks of the reply."""
        payload = json.dumps(
            {
                "model": self.model,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            _API_URL,
            data=payload,
            headers={
                "content-type": "application/json",
                "anthropic-version": _API_VERSION,
                "x-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise GenerationError(f"the API returned {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GenerationError(f"could not reach the API: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GenerationError(f"the API returned invalid JSON: {exc}") from exc

        return _extract_text(body)

    def __repr__(self) -> str:
        return f"ClaudeLanguageModel({self.model!r})"


def _extract_text(body: object) -> str:
    if not isinstance(body, dict):
        raise GenerationError("the API response was not a JSON object")

    blocks = body.get("content")
    if not isinstance(blocks, list):
        raise GenerationError("the API response had no content blocks")

    text = "".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )
    if not text.strip():
        raise GenerationError("the API returned an empty completion")
    return text
