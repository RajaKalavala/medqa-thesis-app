"""Statistical test wrappers."""
from __future__ import annotations

import pytest

from medqa_rag.evaluation.statistical_tests import cochran_q, mcnemar


@pytest.mark.unit
def test_mcnemar_no_disagreement_returns_high_pvalue():
    a = [True, True, False, False]
    b = [True, True, False, False]
    res = mcnemar(a, b)
    assert res.pvalue >= 0.05


@pytest.mark.unit
def test_mcnemar_clear_disagreement_returns_low_statistic():
    # 10 questions; A right on 9, B right on 1
    a = [True] * 9 + [False]
    b = [False] * 9 + [True]
    res = mcnemar(a, b)
    # Should be statistically significant given strong disagreement
    assert res.pvalue < 0.05


@pytest.mark.unit
def test_cochran_q_three_systems():
    matrix = [
        [True, True, False, True, True],
        [False, True, False, True, True],
        [False, False, False, True, True],
    ]
    res = cochran_q(matrix)
    assert res.statistic >= 0
    assert 0.0 <= res.pvalue <= 1.0
