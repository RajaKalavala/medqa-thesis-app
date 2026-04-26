"""Structured logging via structlog.

All log lines are JSON in non-development environments and include
``request_id`` / ``experiment_id`` / ``architecture`` when set via
:func:`bind_context`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from medqa_rag.core.config import get_settings

_CONFIGURED = False


def configure_logging(force: bool = False) -> None:
    """Configure stdlib and structlog. Idempotent unless ``force=True``."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    # Ensure log dir exists
    Path(settings.paths.log_dir).mkdir(parents=True, exist_ok=True)

    # ---- stdlib root logger ----
    handlers: list[logging.Handler] = []
    if settings.logging.console:
        handlers.append(logging.StreamHandler(sys.stdout))
    file_handler = logging.FileHandler(
        Path(settings.paths.log_dir) / "medqa_rag.log", encoding="utf-8"
    )
    handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    # ---- structlog ----
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.logging.json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind keys into the logging context for the current task."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear context-bound values."""
    structlog.contextvars.clear_contextvars()
