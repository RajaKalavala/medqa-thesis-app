"""FastAPI application entry point. Run with: uvicorn medqa_rag.api.main:app --reload"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medqa_rag.__version__ import __version__
from medqa_rag.api.docs import OPENAPI_TAGS
from medqa_rag.api.lifespan import lifespan
from medqa_rag.api.middleware import RequestLoggingMiddleware, register_exception_handlers
from medqa_rag.api.routers import evaluation, explainability, health, qa
from medqa_rag.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="medqa-rag",
        version=__version__,
        description=(
            "Comparative framework for four RAG architectures on MedQA USMLE "
            "(Naive, Self, Hybrid, Multi-Hop). Swagger UI at /docs."
        ),
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(qa.router)
    app.include_router(evaluation.router)
    app.include_router(explainability.router)

    return app


app = create_app()
