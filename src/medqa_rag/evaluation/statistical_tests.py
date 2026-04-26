"""Paired statistical tests across architectures."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TestResult:
    statistic: float
    pvalue: float
    description: str


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> TestResult:
    """McNemar's test for paired binary correctness on the same questions."""
    if len(correct_a) != len(correct_b):
        raise ValueError("Lists must have equal length")
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(~a & b))   # A wrong, B right
    b10 = int(np.sum(a & ~b))   # A right, B wrong
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    except ImportError:
        # Fallback: exact binomial via scipy
        from scipy.stats import binom

        n = b01 + b10
        if n == 0:
            return TestResult(0.0, 1.0, "no discordant pairs")
        k = min(b01, b10)
        pvalue = float(2 * binom.cdf(k, n, 0.5))
        pvalue = min(1.0, pvalue)
        return TestResult(float(abs(b01 - b10)), pvalue, "mcnemar exact (scipy)")

    table = [[0, b01], [b10, 0]]
    res = sm_mcnemar(table, exact=True)
    return TestResult(
        float(res.statistic), float(res.pvalue), "mcnemar exact (statsmodels)"
    )


def cochran_q(correctness_matrix: list[list[bool]]) -> TestResult:
    """Cochran's Q across K systems on the same N questions.

    Args:
        correctness_matrix: list of K lists of length N, each item bool.
    """
    arr = np.asarray(correctness_matrix, dtype=int)  # shape (K, N)
    if arr.ndim != 2:
        raise ValueError("Expected 2D matrix (K systems x N questions)")
    k, n = arr.shape
    if k < 2:
        raise ValueError("Need at least 2 systems")

    col_sums = arr.sum(axis=0)         # per-question, in [0..K]
    row_sums = arr.sum(axis=1)         # per-system, in [0..N]
    grand_total = int(arr.sum())

    num = (k - 1) * (k * float((row_sums**2).sum()) - grand_total**2)
    denom = k * grand_total - int((col_sums**2).sum())
    if denom == 0:
        return TestResult(0.0, 1.0, "all-equal columns; Q undefined")

    q = num / denom
    from scipy.stats import chi2

    pvalue = float(chi2.sf(q, df=k - 1))
    return TestResult(float(q), pvalue, f"cochran-q over {k} systems × {n} questions")
