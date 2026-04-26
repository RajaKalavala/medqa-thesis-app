"""Async Groq client with rate-limiting, retry, and disk-cached responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from medqa_rag.core.config import LLMConfig, get_settings
from medqa_rag.core.exceptions import LLMError, RateLimitError
from medqa_rag.llm.cache import LLMCache
from medqa_rag.llm.rate_limiter import TokenBucket
from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: dict[str, int] = {}
    raw: dict[str, Any] = {}


@dataclass
class _ClientHandles:
    """Cached singleton handles per process."""

    groq: Any = None
    rate_limiter: TokenBucket | None = None
    cache: LLMCache | None = None
    config: LLMConfig | None = None


_state: _ClientHandles = _ClientHandles()


# ---------------------------------------------------------------------------
class GroqClient:
    """Thin async wrapper over the official ``groq`` SDK.

    Features:
        * token-bucket rate limit (RPM)
        * exponential-backoff retry on transient errors
        * disk cache (request-keyed) — bypassed if ``temperature > 0``
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or get_settings().llm
        self._init_handles()

    # --- handle init ----------------------------------------------------
    def _init_handles(self) -> None:
        global _state
        if _state.config is not None and _state.config == self.config:
            return  # already initialised

        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover
            raise LLMError("groq SDK missing. `pip install groq`.") from exc

        api_key = get_settings().groq_api_key
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set in env / .env")

        _state.groq = AsyncGroq(api_key=api_key, timeout=self.config.timeout_seconds)
        _state.rate_limiter = TokenBucket(self.config.rate_limit_rpm)
        _state.cache = LLMCache(self.config.cache_dir) if self.config.cache_enabled else None
        _state.config = self.config

    # --- public API -----------------------------------------------------
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a chat completion."""
        cfg = self.config
        model = model or cfg.model
        temperature = cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or cfg.max_tokens

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [m.model_dump() for m in messages],
        }

        # ---- cache (only safe at temperature == 0) ----
        if _state.cache and temperature == 0.0:
            hit = _state.cache.get(payload)
            if hit is not None:
                logger.debug("llm_cache_hit", model=model)
                return LLMResponse(**hit)

        # ---- rate limit + retry ----
        assert _state.rate_limiter is not None
        await _state.rate_limiter.acquire()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(min=1, max=30),
            retry=retry_if_exception_type((RateLimitError, LLMError)),
            reraise=True,
        ):
            with attempt:
                response = await self._raw_chat(payload)

        if _state.cache and temperature == 0.0:
            _state.cache.set(payload, response.model_dump())

        return response

    # --- internal --------------------------------------------------------
    async def _raw_chat(self, payload: dict[str, Any]) -> LLMResponse:
        try:
            completion = await _state.groq.chat.completions.create(**payload)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg:
                raise RateLimitError(str(exc)) from exc
            raise LLMError(str(exc)) from exc

        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        usage_dict: dict[str, int] = {}
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            usage=usage_dict,
            raw={"id": completion.id, "finish_reason": choice.finish_reason},
        )

    # ----------------------------------------------------------------
    async def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        """Convenience: single (system, user) message pair."""
        return await self.chat(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            **kwargs,
        )
