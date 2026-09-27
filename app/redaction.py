"""Key redaction guard — defense in depth for BYOK deployments.

Anything that looks like an API key is replaced with [REDACTED] before it is
written to SQLite (runs, spans, llm/tool calls, evaluations). Keys sent via
the X-LLM-Key header are used in-memory only and never reach storage, but a
visitor could also paste a key into a chat/run input — this guarantees it can
never leak into stored traces or the dashboard.
"""

from __future__ import annotations

import re

_PATTERNS: list[re.Pattern] = [
    # Provider-masked key fragments, e.g. "sk-fake1***********cdef" as echoed
    # in OpenAI 401 messages. The asterisk makes it unambiguous — plain words
    # like "risk-free" never contain one, so this can't false-positive.
    re.compile(r"sk-[A-Za-z0-9\-_]*\*[A-Za-z0-9*\-_]*"),
    # Anthropic / OpenAI style keys
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{8,}"),
    re.compile(r"sk-proj-[A-Za-z0-9\-_]{8,}"),
    re.compile(r"sk-[A-Za-z0-9\-_]{16,}"),
    # Generic `api_key = "secret..."` assignments / headers
    re.compile(
        r"(?i)(api[_-]?key|secret|bearer)\s*([:=]\s*| +)(['\"]?)[A-Za-z0-9\-_.~+/=]{16,}\3"
    ),
    # OpenAI x-api-key-ish long tokens in headers dumps
    re.compile(r"(?i)(x-api-key:\s*)([A-Za-z0-9\-_.~+/=]{16,})"),
]

REDACTED = "[REDACTED]"


def _sub_keyed(m: re.Match) -> str:
    # Keep the label ("api_key=", "Bearer ") so context stays readable.
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}"


def redact(text: str) -> str:
    """Replace key-like secrets in `text` with [REDACTED]. Non-strings pass through."""
    if not isinstance(text, str) or not text:
        return text
    out = _PATTERNS[0].sub(REDACTED, text)
    out = _PATTERNS[1].sub(REDACTED, out)
    out = _PATTERNS[2].sub(REDACTED, out)
    out = _PATTERNS[3].sub(REDACTED, out)
    out = _PATTERNS[4].sub(_sub_keyed, out)
    out = _PATTERNS[5].sub(lambda m: f"{m.group(1)}{REDACTED}", out)
    return out
