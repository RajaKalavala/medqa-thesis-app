"""Multi-Hop Explainable RAG pipeline."""
from __future__ import annotations

from medqa_rag.core.types import Architecture, Question, RAGOutput
from medqa_rag.embeddings.base import Embedder
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.observability.logger import get_logger
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.multihop_rag.chain_aggregator import aggregate_chain
from medqa_rag.rags.multihop_rag.config import MultiHopRAGConfig
from medqa_rag.rags.multihop_rag.decomposer import decompose_question
from medqa_rag.rags.multihop_rag.generator import generate_multihop_answer
from medqa_rag.rags.multihop_rag.iterative_retriever import iterative_retrieve
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever
from medqa_rag.utils.timing import Timer

logger = get_logger(__name__)


class MultiHopRAGPipeline(RAGPipeline):
    """Decompose → iterate retrieval → aggregate → reason → answer."""

    architecture = Architecture.MULTIHOP
    package = "medqa_rag.rags.multihop_rag"

    def __init__(
        self,
        llm: GroqClient,
        embedder: Embedder,
        faiss: FaissRetriever | None = None,
        bm25: BM25Retriever | None = None,
        config: MultiHopRAGConfig | None = None,
    ) -> None:
        super().__init__(llm=llm, embedder=embedder, faiss=faiss, bm25=bm25)
        self.config = config or MultiHopRAGConfig()
        self._decompose_template = self.load_prompt("decompose.jinja2")
        self._final_template = self.load_prompt("final_answer.jinja2")

    async def answer(self, question: Question) -> RAGOutput:
        if self.faiss is None:
            raise RuntimeError("MultiHopRAGPipeline requires a FAISS retriever")

        question_block = self.format_question(question)
        with Timer() as t:
            sub_queries = await decompose_question(
                self.llm, self._decompose_template, question_block, self.config.max_subqueries
            )
            # Always include the original stem as the first hop, then sub-queries
            queries = [question.stem, *sub_queries][: self.config.max_hops]

            runs = iterative_retrieve(
                self.faiss, queries, top_k=self.config.top_k_per_hop
            )
            aggregated = aggregate_chain(
                runs, top_n=self.config.top_k_per_hop * len(queries)
            )

            response = await generate_multihop_answer(
                self.llm,
                self._final_template,
                question_block,
                sub_queries=queries,
                docs=aggregated,
                context_text=self.format_context(aggregated),
            )

        letter = self.parse_letter(response.text)
        idx = self.letter_to_index(letter)

        logger.info(
            "multihop_rag_done",
            qid=question.id,
            hops=len(queries),
            n_subqueries=len(sub_queries),
            n_docs=len(aggregated),
            predicted=letter,
            latency_ms=round(t.elapsed_ms, 1),
        )

        return RAGOutput(
            question_id=question.id,
            architecture=self.architecture,
            retrieved_docs=aggregated,
            generated_answer=response.text,
            predicted_letter=letter,
            predicted_index=idx,
            latency_ms=t.elapsed_ms,
            token_usage=response.usage,
            hop_count=len(queries),
            sub_queries=queries,
        )
