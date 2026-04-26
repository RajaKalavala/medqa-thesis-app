"""Render comparison results as LaTeX (thesis-ready)."""
from __future__ import annotations

from typing import Any


def render_latex(report: dict[str, Any]) -> str:
    archs = report["architectures"]
    metrics = report.get("metrics", {})
    metric_names: list[str] = sorted({m for d in metrics.values() for m in d}) if metrics else []

    lines: list[str] = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{RAG architecture comparison on MedQA USMLE.}")
    lines.append(r"\label{tab:rag-comparison}")
    cols = "l" + "c" * len(metric_names)
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\hline")
    header = " & ".join(["Architecture", *[m.replace("_", r"\_") for m in metric_names]]) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    for a in archs:
        row = [a.replace("_", r"\_"), *[f"{metrics[a].get(m, float('nan')):.3f}" for m in metric_names]]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
