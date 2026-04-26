"""Async helpers."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from typing import TypeVar

T = TypeVar("T")


async def gather_with_concurrency(
    n: int,
    aws: Iterable[Awaitable[T]],
    *,
    return_exceptions: bool = False,
) -> list[T]:
    """``asyncio.gather`` with a bounded semaphore."""
    semaphore = asyncio.Semaphore(n)

    async def _bound(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *(_bound(a) for a in aws),
        return_exceptions=return_exceptions,
    )
