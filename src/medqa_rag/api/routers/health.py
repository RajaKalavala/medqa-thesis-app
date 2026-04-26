"""Liveness + readiness probes."""
from __future__ import annotations

from fastapi import APIRouter

from medqa_rag.__version__ import __version__

router = APIRouter(prefix="", tags=["health"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/readyz", summary="Readiness probe")
async def readyz() -> dict[str, str]:
    return {"status": "ready"}
