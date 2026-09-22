"""
LLMClient
---------
Optional real-LLM backend with two pluggable providers:

  - Groq       (GROQ_API_KEY)       -> `groq` SDK, OpenAI-style chat API
  - Anthropic  (ANTHROPIC_API_KEY)  -> `anthropic` SDK, Messages API

Provider selection is automatic: if GROQ_API_KEY is set it's used first;
otherwise ANTHROPIC_API_KEY is used if present. If neither is set,
`available` is False and every expert in app/moe/experts.py falls back to
its deterministic offline responder, so the whole platform stays runnable
with zero external dependencies or secrets.

You can force a provider with LLM_PROVIDER=groq|anthropic, and override
the model with GROQ_MODEL / ANTHROPIC_MODEL (see .env.example).
"""
from __future__ import annotations

import os
from typing import Optional

_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


class LLMClient:
    def __init__(self, model: Optional[str] = None) -> None:
        groq_key = os.environ.get("GROQ_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if provider not in ("groq", "anthropic"):
            provider = "groq" if groq_key else ("anthropic" if anthropic_key else "")

        self.provider: str = provider
        self.available: bool = False
        self.model: str = model or ""
        self._client = None

        if provider == "groq" and groq_key:
            try:
                from groq import Groq  # imported lazily, optional dependency

                self._client = Groq(api_key=groq_key)
                self.model = model or os.environ.get("GROQ_MODEL", _DEFAULT_GROQ_MODEL)
                self.available = True
            except ImportError:
                # groq SDK not installed; degrade gracefully to offline mode
                self.available = False

        elif provider == "anthropic" and anthropic_key:
            try:
                import anthropic  # imported lazily, optional dependency

                self._client = anthropic.Anthropic(api_key=anthropic_key)
                self.model = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_ANTHROPIC_MODEL)
                self.available = True
            except ImportError:
                self.available = False

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if not self.available or self._client is None:
            raise RuntimeError("LLMClient is not configured with a real backend.")

        if self.provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        # provider == "anthropic"
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(parts).strip()
