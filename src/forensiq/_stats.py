"""Dependency-free statistics shared across ForensiQ.

No scipy/sklearn: percentiles, mean/std and the chi-square survival function
(regularized upper incomplete gamma, Numerical Recipes gser/gcf) are
implemented by hand — that is the point of this project.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of a non-empty sequence, q in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Population mean and standard deviation (ddof=0). Empty -> (0.0, 0.0)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def zscore(value: float, mean: float, std: float, eps: float = 1e-12) -> float:
    """Z-score with a degenerate-spread guard.

    When the baseline spread is ~0 (e.g. every trace sets top_k=5) a value that
    differs at all is maximally surprising, so we return a large signed z
    instead of dividing by zero; identical values score exactly 0.
    """
    if std <= eps:
        if math.isclose(value, mean, rel_tol=1e-9, abs_tol=1e-9):
            return 0.0
        return 10.0 if value > mean else -10.0
    return (value - mean) / std


_FPMIN = 1e-300


def _gser(a: float, x: float, itmax: int = 300, eps: float = 3e-9) -> float:
    """Series representation of P(a, x) (regularized lower incomplete gamma)."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float, itmax: int = 300, eps: float = 3e-9) -> float:
    """Continued-fraction representation of Q(a, x) (Lentz's method)."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b if b != 0 else 1.0 / _FPMIN
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError(f"invalid gammq(a={a}, x={x})")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution: P(X > x) with df DOF."""
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return 1.0
    return gammq(df / 2.0, x / 2.0)
