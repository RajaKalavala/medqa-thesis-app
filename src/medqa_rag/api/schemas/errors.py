"""Error envelope schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["RetrievalError"])
    detail: str = Field(..., examples=["FAISS index not found at /data/indices/faiss"])
    request_id: str | None = None
