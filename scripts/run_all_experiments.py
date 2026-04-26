"""CLI: run all four architectures + comparison report."""
from __future__ import annotations

import argparse
import asyncio

from medqa_rag.pipelines.comparison_pipeline import run_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_all(args.n))


if __name__ == "__main__":
    main()
