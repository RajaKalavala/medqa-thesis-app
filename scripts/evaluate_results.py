"""CLI: re-render reports from existing per-architecture metric JSON files."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from medqa_rag.core.config import get_settings
from medqa_rag.evaluation.reporters.latex_reporter import render_latex
from medqa_rag.evaluation.reporters.markdown_reporter import render_markdown
from medqa_rag.evaluation.statistical_tests import cochran_q, mcnemar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        default=str(Path(get_settings().paths.results_dir) / "metrics"),
    )
    args = parser.parse_args()

    files = sorted(Path(args.metrics_dir).glob("*.json"))
    by_arch: dict[str, dict] = {}
    for f in files:
        record = json.loads(f.read_text())
        # keep only the latest run per architecture
        by_arch[record["architecture"]] = record

    if not by_arch:
        print("No metric files found")
        return

    metrics = {a: {"accuracy": r["accuracy"], **r.get("ragas", {})} for a, r in by_arch.items()}
    latency = {a: r["latency"] for a, r in by_arch.items()}
    halluc = {a: r["hallucination_rate"] for a, r in by_arch.items()}

    correctness_maps = {a: r.get("correctness", {}) for a, r in by_arch.items()}
    shared = sorted(set.intersection(*(set(c.keys()) for c in correctness_maps.values())))
    correctness = {a: [bool(c[q]) for q in shared] for a, c in correctness_maps.items()}

    stats: dict[str, dict] = {}
    if shared:
        archs = list(correctness.keys())
        q_res = cochran_q([correctness[a] for a in archs])
        stats["cochran_q"] = {
            "statistic": q_res.statistic,
            "pvalue": q_res.pvalue,
            "description": q_res.description,
        }
        pairwise: dict[str, dict] = {}
        for i, a in enumerate(archs):
            for b in archs[i + 1 :]:
                res = mcnemar(correctness[a], correctness[b])
                pairwise[f"{a}_vs_{b}"] = {
                    "statistic": res.statistic,
                    "pvalue": res.pvalue,
                }
        stats["pairwise"] = pairwise

    report = {
        "architectures": list(by_arch.keys()),
        "metrics": metrics,
        "latency": latency,
        "hallucination_rate": halluc,
        "stats": stats,
    }

    out_dir = Path(get_settings().paths.results_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"report_{timestamp}.md").write_text(render_markdown(report))
    (out_dir / f"report_{timestamp}.tex").write_text(render_latex(report))
    (out_dir / f"report_{timestamp}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Reports written to {out_dir}")


if __name__ == "__main__":
    main()
