"""Nadeau-Bengio corrected paired t-test.

The default cross-method significance test for both papers
(per ``evaluation-protocol.md``).

Reference:
    Nadeau & Bengio (2003), "Inference for the Generalization Error",
    Machine Learning 52(3), 239-281.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TTestResult:
    t_stat: float
    p_value: float
    mean_diff: float
    df: int


class NadeauBengioCorrectedTTest:
    """Paired t-test with the Nadeau-Bengio variance correction.

    The correction inflates the sample variance by ``1 + n2/n1`` where
    ``n1`` is the size of the training subset and ``n2`` the test
    subset, accounting for the dependence between repeated runs that
    share random subsets of the data.
    """

    def __init__(
        self, alpha: float = 0.05, n_train: int | None = None, n_test: int | None = None
    ) -> None:
        self.alpha = alpha
        self.n_train = n_train
        self.n_test = n_test

    def test(self, a: np.ndarray, b: np.ndarray) -> TTestResult:
        """Test ``H_0: mean(a) = mean(b)`` against the two-sided alternative."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        assert a.shape == b.shape, "paired inputs must align"
        diffs = a - b
        n = diffs.size
        mean = diffs.mean()
        var = diffs.var(ddof=1)
        if self.n_train is not None and self.n_test is not None and self.n_train > 0:
            var *= 1.0 + (self.n_test / self.n_train)
        se = np.sqrt(var / n)
        if se == 0:
            return TTestResult(t_stat=0.0, p_value=1.0, mean_diff=float(mean), df=n - 1)
        t_stat = mean / se
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))
        return TTestResult(
            t_stat=float(t_stat), p_value=float(p_value), mean_diff=float(mean), df=n - 1
        )
