"""Disk-backed response cache for LLM calls.

Keyed by sha256 of (model + messages + temperature + max_tokens).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medqa_rag.utils.io import ensure_dir


class LLMCache:
    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(root)

    @staticmethod
    def _key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        p = self._path(self._key(payload))
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, payload: dict[str, Any], response: dict[str, Any]) -> None:
        p = self._path(self._key(payload))
        with p.open("w", encoding="utf-8") as fh:
            json.dump(response, fh, ensure_ascii=False)
