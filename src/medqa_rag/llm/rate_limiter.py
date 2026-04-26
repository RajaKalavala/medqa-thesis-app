"""Async token-bucket rate limiter."""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Simple async token-bucket rate limiter (RPM granularity)."""

    def __init__(self, rate_per_minute: int) -> None:
        self.capacity = max(1, rate_per_minute)
        self.tokens: float = float(self.capacity)
        self.refill_rate: float = self.capacity / 60.0
        self.last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                deficit = 1 - self.tokens
                wait = deficit / self.refill_rate
            await asyncio.sleep(min(wait, 1.0))
