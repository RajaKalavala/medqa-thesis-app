"""Self-RAG pipeline: retrieve only when confidence is low."""
from __future__ import annotations

from medqa_rag.core.config import get_settings
from medqa_rag.core.types import Architecture, Question, RAGOutput
from medqa_rag.embeddings.base import Embedder
from medqa_rag.llm.groq_client import GroqClient
from medqa_rag.observability.logger import get_logger
from medqa_rag.rags.base import RAGPipeline
from medqa_rag.rags.self_rag.config import SelfRAGConfig
from medqa_rag.rags.self_rag.confidence_gate import estimate_confidence
from medqa_rag.rags.self_rag.generator import generate_self_rag_answer
from medqa_rag.rags.self_rag.retriever import retrieve_self_rag
from medqa_rag.retrieval.dense_faiss import FaissRetriever
from medqa_rag.retrieval.sparse_bm25 import BM25Retriever
from medqa_rag.utils.timing import Timer

logger = get_logger(__name__)


class SelfRAGPipeline(RAGPipeline):
    """Adaptive: gate retrieval on a cheap self-confidence estimate."""

    architecture = Architecture.SELF
    package = "medqa_rag.rags.self_rag"

    def __init__(
        self,
        llm: GroqClient,
        embedder: Embedder,
        faiss: FaissRetriever | None = None,
        bm25: BM25Retriever | None = None,
        config: SelfRAGConfig | None = None,
    ) -> None:
        super().__init__(llm=llm, embedder=embedder, faiss=faiss, bm25=bm25)
        self.config = config or SelfRAGConfig()
        self._gate_template = self.load_prompt("confidence_check.jinja2")
        self._qa_template = self.load_prompt("self_rag_qa.jinja2")

    async def answer(self, question: Question) -> RAGOutput:
        if self.faiss is None:
            raise RuntimeError("SelfRAGPipeline requires a FAISS retriever")

        judge = get_settings().llm.judge_model
        question_block = self.format_question(question)

        with Timer() as t:
            confidence, conf_text = await estimate_confidence(
                self.llm, self._gate_template, question_block, judge_model=judge
            )

            if confidence >= self.config.confidence_threshold:
                docs: list = []
                context = None
                retrieval_used = False
            else:
                docs = retrieve_self_rag(self.faiss, question.stem, self.config.top_k)
                context = self.format_context(docs)
                retrieval_used = True

            response = await generate_self_rag_answer(
                self.llm,
                self._qa_template,
                question_block,
                docs=docs if retrieval_used else None,
                context_text=context,
            )

        letter = self.parse_letter(response.text)
        idx = self.letter_to_index(letter)

        logger.info(
            "self_rag_done",
            qid=question.id,
            confidence=round(confidence, 3),
            retrieval_used=retrieval_used,
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
            retrieval_used=retrieval_used,
            extras={"confidence": confidence, "confidence_raw": conf_text[:300]},
        )
