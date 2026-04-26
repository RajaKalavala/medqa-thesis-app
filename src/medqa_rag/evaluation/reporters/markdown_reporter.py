"""Render comparison results as a Markdown report."""
from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    """Render the comparison report.

    Expects:
        report = {
            "architectures": ["naive", "self", "hybrid", "multihop"],
            "metrics": {arch: {metric: value, ...}, ...},
            "latency": {arch: {p50_ms, p95_ms, ...}},
            "hallucination_rate": {arch: float},
            "stats": {"cochran_q": {...}, "pairwise": {...}},
        }
    """
    archs = report["architectures"]
    metrics = report.get("metrics", {})
    latency = report.get("latency", {})
    halluc = report.get("hallucination_rate", {})

    metric_names: list[str] = sorted({m for d in metrics.values() for m in d}) if metrics else []

    lines: list[str] = ["# RAG Comparison Report", ""]

    # ---- accuracy + RAGAS table ----
    if metric_names:
        lines.append("## Metric scores (mean over test set)")
        header = "| Architecture | " + " | ".join(metric_names) + " |"
        sep = "|" + "---|" * (len(metric_names) + 1)
        lines.extend([header, sep])
        for a in archs:
            row = [a] + [f"{metrics[a].get(m, float('nan')):.3f}" for m in metric_names]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ---- latency ----
    if latency:
        lines.append("## Latency")
        lines.append("| Architecture | n | mean_ms | p50_ms | p95_ms | p99_ms |")
        lines.append("|---|---|---|---|---|---|")
        for a in archs:
            d = latency.get(a, {})
            lines.append(
                f"| {a} | {d.get('n', 0)} | {d.get('mean_ms', 0):.1f} | "
                f"{d.get('p50_ms', 0):.1f} | {d.get('p95_ms', 0):.1f} | {d.get('p99_ms', 0):.1f} |"
            )
        lines.append("")

    # ---- hallucination ----
    if halluc:
        lines.append("## Hallucination flag rate")
        lines.append("| Architecture | flag_rate |")
        lines.append("|---|---|")
        for a in archs:
            lines.append(f"| {a} | {halluc.get(a, 0.0):.3f} |")
        lines.append("")

    # ---- stats ----
    stats = report.get("stats", {})
    if stats:
        lines.append("## Statistical tests")
        lines.append("```json")
        import json

        lines.append(json.dumps(stats, indent=2))
        lines.append("```")
    return "\n".join(lines)
