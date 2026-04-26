"""Timing helpers."""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class Timer(AbstractContextManager["Timer"]):
    """Lightweight monotonic-clock timer.

    Example:
        with Timer() as t:
            do_work()
        logger.info("done", elapsed_ms=t.elapsed_ms)
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = self.end or time.perf_counter()
        return (end - self.start) * 1000.0


def timed(label: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that logs execution time of the wrapped function."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        name = label or func.__qualname__

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with Timer() as t:
                result = func(*args, **kwargs)
            logger.debug("timed", op=name, elapsed_ms=round(t.elapsed_ms, 2))
            return result

        return wrapper

    return decorator
