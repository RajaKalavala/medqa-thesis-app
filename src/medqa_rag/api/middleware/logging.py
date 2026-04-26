"""Request-id + structured logging middleware."""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from medqa_rag.observability.logger import bind_context, clear_context, get_logger

logger = get_logger("api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach a UUID request id, log start/end, expose ``X-Request-ID`` header."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        bind_context(request_id=request_id, path=request.url.path, method=request.method)
        logger.info("request_received")
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed")
            clear_context()
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("request_completed", status=response.status_code, duration_ms=round(duration_ms, 2))
        response.headers["X-Request-ID"] = request_id
        clear_context()
        return response
