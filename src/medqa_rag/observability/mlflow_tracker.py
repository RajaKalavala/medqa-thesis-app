"""MLflow experiment tracking wrapper."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import mlflow

from medqa_rag.core.config import get_settings
from medqa_rag.observability.logger import get_logger

logger = get_logger(__name__)


def init_mlflow() -> None:
    """Configure MLflow tracking URI and experiment."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager wrapping a single MLflow run with safe defaults."""
    init_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        logger.info("mlflow_run_started", run_name=run_name, run_id=run.info.run_id)
        try:
            yield run
        finally:
            logger.info("mlflow_run_ended", run_id=run.info.run_id)


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)
