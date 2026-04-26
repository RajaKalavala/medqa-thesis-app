"""Naive RAG pipeline."""
from __future__ import annotations

from medqa_rag.core.types import Architecture, Question, RAGOutput
from medqa_rag.embeddings.base import Embedder
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.observability.logger import get_logger
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.naive_rag.config import NaiveRAGConfig
from medqa_rag.rags.naive_rag.generator import generate_naive_answer
from medqa_rag.rags.naive_rag.retriever import retrieve_naive
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever
from medqa_rag.utils.timing import Timer

logger = get_logger(__name__)


class NaiveRAGPipeline(RAGPipeline):
    """Single-pass dense retrieve → stuff → generate."""

    architecture = Architecture.NAIVE
    package = "medqa_rag.rags.naive_rag"

    def __init__(
        self,
        llm: GroqClient,
        embedder: Embedder,
        faiss: FaissRetriever | None = None,
        bm25: BM25Retriever | None = None,
        config: NaiveRAGConfig | None = None,
    ) -> None:
        super().__init__(llm=llm, embedder=embedder, faiss=faiss, bm25=bm25)
        self.config = config or NaiveRAGConfig()
        self._template = self.load_prompt("naive_qa.jinja2")

    async def answer(self, question: Question) -> RAGOutput:
        if self.faiss is None:
            raise RuntimeError("NaiveRAGPipeline requires a FAISS retriever")

        with Timer() as t:
            docs = retrieve_naive(self.faiss, question.stem, self.config.top_k)
            response = await generate_naive_answer(
                llm=self.llm,
                template=self._template,
                question_block=self.format_question(question),
                docs=docs,
                context_text=self.format_context(docs),
            )

        letter = self.parse_letter(response.text)
        idx = self.letter_to_index(letter)

        logger.info(
            "naive_rag_done",
            qid=question.id,
            n_docs=len(docs),
            predicted=letter,
            latency_ms=round(t.elapsed_ms, 1),
        )

        return RAGOutput(
            question_id=question.id,
            architecture=self.architecture,
            retrieved_docs=docs,
            generated_answer=response.text,
            predicted_letter=letter,
            predicted_index=idx,
            latency_ms=t.elapsed_ms,
            token_usage=response.usage,
        )
