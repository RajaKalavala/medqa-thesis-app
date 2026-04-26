"""Ingestion pipeline: textbooks → cleaned chunks → FAISS + BM25 indices."""
from __future__ import annotations

import argparse
from pathlib import Path

from medqa_rag.core.config import get_settings
from medqa_rag.data.chunking.factory import build_chunker
from medqa_rag.data.loaders.textbook_loader import load_textbooks
from medqa_rag.data.preprocessing.cleaners import clean_medical_text
from medqa_rag.embeddings.factory import build_embedder
from medqa_rag.observability.logger import configure_logging, get_logger
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever

logger = get_logger(__name__)


def run() -> None:
    configure_logging()
    settings = get_settings()

    logger.info("ingestion_start", textbooks=str(settings.paths.raw_textbooks))
    docs = list(load_textbooks(settings.paths.raw_textbooks))
    if not docs:
        logger.warning("no_documents_to_index")
        return

    cleaned = [d.model_copy(update={"text": clean_medical_text(d.text)}) for d in docs]

    chunker = build_chunker()
    chunks = chunker.split(cleaned)
    logger.info("chunked", n_chunks=len(chunks))

    # ---- FAISS ----
    embedder = build_embedder()
    faiss = FaissRetriever(embedder)
    faiss.build(chunks)
    faiss.save(settings.paths.faiss_dir)

    # ---- BM25 ----
    bm25 = BM25Retriever(k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b)
    bm25.build(chunks)
    bm25.save(settings.paths.bm25_dir)

    logger.info(
        "ingestion_done",
        faiss_dir=str(settings.paths.faiss_dir),
        bm25_dir=str(settings.paths.bm25_dir),
        n_chunks=len(chunks),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indices from textbook corpus")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
