"""Run all four architectures, then compute paired statistical tests + report."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from medqa_rag.core.config import get_settings
from medqa_rag.core.types import Architecture
from medqa_rag.evaluation.reporters.latex_reporter import render_latex
from medqa_rag.evaluation.reporters.markdown_reporter import render_markdown
from medqa_rag.evaluation.statistical_tests import cochran_q, mcnemar
from medqa_rag.observability.logger import configure_logging, get_logger
from medqa_rag.pipelines.evaluation_pipeline import run_architecture

logger = get_logger(__name__)


async def run_all(n_questions: int | None = None) -> Path:
    configure_logging()
    settings = get_settings()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    architectures = [Architecture.NAIVE, Architecture.SELF, Architecture.HYBRID, Architecture.MULTIHOP]
    paths: dict[str, Path] = {}
    for arch in architectures:
        paths[arch.value] = await run_architecture(arch, n_questions=n_questions)

    # ---- Aggregate metrics ----
    metrics: dict[str, dict[str, float]] = {}
    latency: dict[str, dict[str, float]] = {}
    halluc_rate: dict[str, float] = {}
    correctness_by_arch: dict[str, list[bool]] = {}
    correctness_maps: dict[str, dict[str, bool]] = {}
    for arch_name, p in paths.items():
        record = json.loads(p.read_text())
        metrics[arch_name] = {"accuracy": record["accuracy"], **record.get("ragas", {})}
        latency[arch_name] = record["latency"]
        halluc_rate[arch_name] = record["hallucination_rate"]
        correctness_maps[arch_name] = record.get("correctness", {})

    # Align correctness vectors across architectures via the intersection of qids
    if correctness_maps:
        shared_qids = sorted(set.intersection(*(set(m.keys()) for m in correctness_maps.values())))
        for arch_name, cm in correctness_maps.items():
            correctness_by_arch[arch_name] = [bool(cm[q]) for q in shared_qids]

    # ---- Statistical tests ----
    stats: dict[str, dict] = {}
    arch_names = list(correctness_by_arch.keys())
    if all(len(correctness_by_arch[a]) == len(correctness_by_arch[arch_names[0]]) for a in arch_names):
        try:
            q_res = cochran_q([correctness_by_arch[a] for a in arch_names])
            stats["cochran_q"] = {
                "statistic": q_res.statistic,
                "pvalue": q_res.pvalue,
                "description": q_res.description,
            }
        except Exception:
            logger.exception("cochran_q_failed")

        pairwise: dict[str, dict] = {}
        for i, a in enumerate(arch_names):
            for b in arch_names[i + 1 :]:
                try:
                    res = mcnemar(correctness_by_arch[a], correctness_by_arch[b])
                    pairwise[f"{a}_vs_{b}"] = {
                        "statistic": res.statistic,
                        "pvalue": res.pvalue,
                        "description": res.description,
                    }
                except Exception:
                    logger.exception("mcnemar_failed", a=a, b=b)
        stats["pairwise"] = pairwise

    report = {
        "architectures": arch_names,
        "metrics": metrics,
        "latency": latency,
        "hallucination_rate": halluc_rate,
        "stats": stats,
    }

    out_dir = Path(settings.paths.results_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"comparison_{timestamp}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    md_path = out_dir / f"comparison_{timestamp}.md"
    md_path.write_text(render_markdown(report))

    tex_path = out_dir / f"comparison_{timestamp}.tex"
    tex_path.write_text(render_latex(report))

    logger.info("comparison_done", json=str(json_path), md=str(md_path), tex=str(tex_path))
    return json_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_all(args.n))


if __name__ == "__main__":
    main()
