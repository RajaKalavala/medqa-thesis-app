"""Startup / shutdown lifecycle: pre-warm logging and (optionally) indices."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from medqa_rag.observability.logger import configure_logging, get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info("api_startup", routes=len(app.routes))
    try:
        yield
    finally:
        logger.info("api_shutdown")
