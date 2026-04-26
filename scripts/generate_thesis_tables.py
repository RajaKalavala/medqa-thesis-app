"""Render LaTeX tables from the latest comparison report into thesis/appendix snippets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medqa_rag.core.config import get_settings
from medqa_rag.evaluation.reporters.latex_reporter import render_latex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=None,
        help="Path to comparison_*.json. Default: latest under results/reports.",
    )
    parser.add_argument(
        "--out",
        default=str(Path(get_settings().paths.results_dir) / "reports" / "thesis_tables.tex"),
    )
    args = parser.parse_args()

    reports_dir = Path(get_settings().paths.results_dir) / "reports"
    if args.report is None:
        candidates = sorted(reports_dir.glob("comparison_*.json"))
        if not candidates:
            raise SystemExit("No comparison report found; run the comparison pipeline first.")
        report_path = candidates[-1]
    else:
        report_path = Path(args.report)

    report = json.loads(report_path.read_text())
    Path(args.out).write_text(render_latex(report))
    print(f"LaTeX written to {args.out}")


if __name__ == "__main__":
    main()
