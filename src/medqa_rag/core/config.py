"""Single source of runtime configuration.

Loads from `config/settings.yaml`, then overlays environment variables
prefixed with ``MEDQA_`` (use ``__`` as the nesting delimiter, e.g.
``MEDQA_LLM__MODEL``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from medqa_rag.core.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class Paths(BaseModel):
    data_dir: Path = Path("./data")
    raw_medqa: Path = Path("./data/raw/medqa")
    raw_textbooks: Path = Path("./data/raw/textbooks")
    processed_dir: Path = Path("./data/processed")
    embeddings_dir: Path = Path("./data/embeddings")
    index_dir: Path = Path("./data/indices")
    faiss_dir: Path = Path("./data/indices/faiss")
    bm25_dir: Path = Path("./data/indices/bm25")
    log_dir: Path = Path("./logs")
    results_dir: Path = Path("./results")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json: bool = True
    console: bool = True


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    judge_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_retries: int = 5
    rate_limit_rpm: int = 30
    cache_enabled: bool = True
    cache_dir: Path = Path("./data/processed/llm_cache")


class EmbedderConfig(BaseModel):
    model_name: str = "NeuML/pubmedbert-base-embeddings"
    device: str = "auto"  # auto | cpu | cuda
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"  # recursive | fixed
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    top_k: int = 5
    faiss_metric: str = "cosine"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class EvaluationConfig(BaseModel):
    ragas_metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_correctness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
        ]
    )
    judge_temperature: float = 0.0
    test_set_size: int | None = None
    random_seed: int = 42


class ExplainabilityConfig(BaseModel):
    sample_size: int = 400
    stratify_by: str = "subject"
    lime_num_samples: int = 50
    shap_num_samples: int = 50


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout: float = 120.0


class MLflowConfig(BaseModel):
    tracking_uri: str = "./results/mlruns"
    experiment_name: str = "medqa-rag-comparison"


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="MEDQA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    paths: Paths = Field(default_factory=Paths)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Secrets (env-only)
    groq_api_key: str | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML in {path} must be a mapping")
    return data


def _resolve_settings_path() -> Path:
    here = Path(__file__).resolve()
    # walk up to project root and look for config/settings.yaml
    for parent in (here, *here.parents):
        candidate = parent / "config" / "settings.yaml"
        if candidate.exists():
            return candidate
    return Path("config/settings.yaml")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance.

    YAML provides defaults; environment variables override.
    """
    yaml_path = _resolve_settings_path()
    yaml_data = _load_yaml(yaml_path)
    try:
        return Settings(**yaml_data)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to construct Settings: {exc}") from exc


def reload_settings() -> Settings:
    """Drop the cache and reload (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()
