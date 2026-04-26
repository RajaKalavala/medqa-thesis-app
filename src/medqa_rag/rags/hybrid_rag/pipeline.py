"""Hybrid RAG pipeline."""
from __future__ import annotations

from medqa_rag.core.types import Architecture, Question, RAGOutput
from medqa_rag.embeddings.base import Embedder
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.observability.logger import get_logger
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.hybrid_rag.config import HybridRAGConfig
from medqa_rag.rags.hybrid_rag.generator import generate_hybrid_answer
from medqa_rag.rags.hybrid_rag.retriever import retrieve_hybrid
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever
from medqa_rag.utils.timing import Timer

logger = get_logger(__name__)


class HybridRAGPipeline(RAGPipeline):
    """Dense + sparse → RRF fusion → answer."""

    architecture = Architecture.HYBRID
    package = "medqa_rag.rags.hybrid_rag"

    def __init__(
        self,
        llm: GroqClient,
        embedder: Embedder,
        faiss: FaissRetriever | None = None,
        bm25: BM25Retriever | None = None,
        config: HybridRAGConfig | None = None,
    ) -> None:
        super().__init__(llm=llm, embedder=embedder, faiss=faiss, bm25=bm25)
        self.config = config or HybridRAGConfig()
        self._template = self.load_prompt("hybrid_qa.jinja2")

    async def answer(self, question: Question) -> RAGOutput:
        if self.faiss is None or self.bm25 is None:
            raise RuntimeError("HybridRAGPipeline needs both FAISS and BM25 retrievers")

        with Timer() as t:
            docs = retrieve_hybrid(
                self.faiss,
                self.bm25,
                question.stem,
                pool_k=self.config.pool_k,
                top_k=self.config.top_k,
                rrf_k=self.config.rrf_k,
            )
            response = await generate_hybrid_answer(
                self.llm,
                self._template,
                self.format_question(question),
                docs,
                self.format_context(docs),
            )

        letter = self.parse_letter(response.text)
        idx = self.letter_to_index(letter)

        logger.info(
            "hybrid_rag_done",
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
            extras={"pool_k": self.config.pool_k, "rrf_k": self.config.rrf_k},
        )
