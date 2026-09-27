"""Provider-agnostic LLM layer.

Supports:
  - "none"      : extractive answer built directly from retrieved context
  - "openai"    : OpenAI Chat Completions
  - "anthropic" : Anthropic Messages
  - "ollama"    : local Ollama /api/chat

All providers receive the same system prompt that constrains them to answer
ONLY from the supplied support-doc context.
"""

from __future__ import annotations

from typing import List, Protocol

import httpx

from .config import Settings

SYSTEM_PROMPT = (
    "You are a customer support assistant. You must answer the user's question "
    "USING ONLY the information contained in the CONTEXT provided below, which "
    "comes from the official support documentation.\n\n"
    "Rules:\n"
    "1. If the answer is not contained in the CONTEXT, reply exactly with: "
    "\"__OUT_OF_SCOPE__\".\n"
    "2. Never use outside knowledge or make assumptions.\n"
    "3. Be concise, accurate, and friendly.\n"
    "4. When helpful, reference the relevant section titles from the context.\n"
)

OUT_OF_SCOPE_TOKEN = "__OUT_OF_SCOPE__"


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above."
    )


class LLMProvider(Protocol):
    def generate(self, question: str, context: str) -> str: ...


class NoneProvider:
    """No external LLM. Returns an extractive answer from the context.

    This keeps the project fully functional without any API keys. The chat
    engine already guarantees the context is relevant; here we simply surface
    the most relevant passages as the answer.
    """

    def generate(self, question: str, context: str) -> str:
        # Context is pre-joined relevant chunks. Present it directly.
        return (
            "Based on the support documentation:\n\n"
            f"{context.strip()}"
        )


class OpenAIProvider:
    def __init__(self, settings: Settings, api_key: str = "") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from exc
        key = (api_key or "").strip() or settings.openai_api_key
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=key)
        self._model = settings.openai_chat_model
        # True when the key came from the request (BYOK) rather than server config.
        self.ephemeral_key = bool((api_key or "").strip())

    def generate(self, question: str, context: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider:
    def __init__(self, settings: Settings, api_key: str = "") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc
        key = (api_key or "").strip() or settings.anthropic_api_key
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = settings.anthropic_chat_model
        # True when the key came from the request (BYOK) rather than server config.
        self.ephemeral_key = bool((api_key or "").strip())

    def generate(self, question: str, context: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(question, context)},
            ],
        )
        parts: List[str] = [
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ]
        return "".join(parts)


class OllamaProvider:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_chat_model

    def generate(self, question: str, context: str) -> str:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_user_prompt(question, context),
                        },
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")


def get_provider(settings: Settings, provider_override: str = "",
                   api_key_override: str = "") -> LLMProvider:
    provider = (provider_override or settings.llm_provider).lower().strip()
    if provider == "openai":
        return OpenAIProvider(settings, api_key=api_key_override)
    if provider == "anthropic":
        return AnthropicProvider(settings, api_key=api_key_override)
    if provider == "ollama":
        return OllamaProvider(settings)
    return NoneProvider()


def infer_provider_from_key(api_key: str) -> str:
    """Guess openai vs anthropic from key shape. Defaults to openai."""
    key = (api_key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    return "openai"
