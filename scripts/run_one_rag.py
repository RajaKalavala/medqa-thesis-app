"""CLI: run a single RAG architecture over the test set."""
from __future__ import annotations

import argparse
import asyncio

from medqa_rag.core.types import Architecture
from medqa_rag.pipelines.evaluation_pipeline import run_architecture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag",
        required=True,
        choices=[a.value for a in Architecture],
        help="Which architecture to run (naive | self | hybrid | multihop)",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS evaluation")
    args = parser.parse_args()
    asyncio.run(
        run_architecture(
            Architecture(args.rag),
            n_questions=args.n,
            use_ragas=not args.no_ragas,
        )
    )


if __name__ == "__main__":
    main()
