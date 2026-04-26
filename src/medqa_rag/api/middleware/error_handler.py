"""Map domain exceptions to JSON envelopes."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from medqa_rag.api.schemas.errors import ErrorResponse
from medqa_rag.core.exceptions import (
    ConfigError,
    DataError,
    EmbeddingError,
    EvaluationError,
    ExplainabilityError,
    LLMError,
    MedQARAGError,
    RateLimitError,
    RetrievalError,
)


_STATUS_MAP: dict[type[Exception], int] = {
    ConfigError: 500,
    DataError: 400,
    EmbeddingError: 500,
    RetrievalError: 500,
    LLMError: 502,
    RateLimitError: 429,
    EvaluationError: 500,
    ExplainabilityError: 500,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MedQARAGError)
    async def _handle_domain(request: Request, exc: MedQARAGError) -> JSONResponse:
        status = _STATUS_MAP.get(type(exc), 500)
        body = ErrorResponse(
            error=type(exc).__name__,
            detail=str(exc),
            request_id=request.headers.get("X-Request-ID"),
        )
        return JSONResponse(status_code=status, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        body = ErrorResponse(
            error="InternalError",
            detail=str(exc) or "Unexpected error",
            request_id=request.headers.get("X-Request-ID"),
        )
        return JSONResponse(status_code=500, content=body.model_dump())
