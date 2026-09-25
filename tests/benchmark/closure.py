"""
Closure-tier benchmark assertion.

A *closure test* answers a single question:
"Does the truth fall inside the (1 − α) credible interval of the
posterior, for every parameter we care about?"

The cheap version (no posterior-coverage statistics, no bias
estimation) — see the framework docstring for why bias and coverage
tiers are deliberately deferred until they have a use case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Default credible level.  0.997 ≈ ±3σ for a Gaussian; chosen so a
# single closure failure is statistically meaningful even for ~10
# parameters per fit (~3 % chance of any false positive across the
# whole table by union bound).
_DEFAULT_CREDIBLE = 0.997


@dataclass(frozen=True)
class ClosureResult:
    """Per-parameter outcome of a closure check."""

    name: str
    truth: float
    median: float
    lower: float
    upper: float
    credible: float
    inside: bool

    def __str__(self) -> str:
        flag = "PASS" if self.inside else "FAIL"
        return (
            f"  [{flag}]  {self.name:<24s}  truth={self.truth:>12.5g}  "
            f"med={self.median:>12.5g}  "
            f"[{self.lower:>12.5g}, {self.upper:>12.5g}]  "
            f"({100 * self.credible:.1f}% CI)"
        )


def closure_test(
    truth: dict,
    chains: dict,
    *,
    credible: float = _DEFAULT_CREDIBLE,
    raise_on_fail: bool = True,
) -> list[ClosureResult]:
    """
    Verify the posterior chains enclose the truth for each named param.

    Parameters
    ----------
    truth : dict
        Map ``param_name → truth_value``.  Only the keys present in
        both ``truth`` and ``chains`` are checked; missing keys raise
        ``KeyError`` so silent under-testing is impossible.
    chains : dict
        Posterior chains (output of ``fit_*_orbit(...)["chains"]``).
        Each value is a 1-D numpy array of MCMC samples.
    credible : float, default 0.997
        Credible-interval level (two-sided).  Quantiles taken at
        ``(1 − credible)/2`` and ``1 − (1 − credible)/2``.
    raise_on_fail : bool, default True
        If True, raise :class:`AssertionError` on any failure with
        a detailed multi-line message — convenient for pytest.
        If False, return the result list and let the caller inspect.

    Returns
    -------
    results : list of :class:`ClosureResult`
        One entry per parameter, in the order keys appear in ``truth``.

    Raises
    ------
    KeyError
        If a key in ``truth`` is absent from ``chains``.
    AssertionError
        If any parameter fails the closure (only when
        ``raise_on_fail`` is True).
    """
    if not 0.0 < credible < 1.0:
        raise ValueError(f"credible must be in (0, 1); got {credible}")

    half_alpha = 0.5 * (1.0 - credible)
    q_lo = half_alpha
    q_hi = 1.0 - half_alpha

    results: list[ClosureResult] = []
    for name, truth_val in truth.items():
        if name not in chains:
            raise KeyError(
                f"Closure test: parameter {name!r} requested in `truth` "
                f"but missing from `chains`. Available chain keys: "
                f"{sorted(chains.keys())}"
            )
        samples = np.asarray(chains[name], dtype=float)
        med = float(np.median(samples))
        lo = float(np.quantile(samples, q_lo))
        hi = float(np.quantile(samples, q_hi))
        # NaN truth → automatic fail (avoid silent passes).
        inside = (
            math.isfinite(truth_val)
            and lo <= float(truth_val) <= hi
        )
        results.append(
            ClosureResult(
                name=name, truth=float(truth_val),
                median=med, lower=lo, upper=hi,
                credible=credible, inside=inside,
            )
        )

    if raise_on_fail:
        failed = [r for r in results if not r.inside]
        if failed:
            lines = [
                f"Closure test failed: {len(failed)}/{len(results)} parameters "
                f"outside their {100 * credible:.1f}% credible interval.",
                *[str(r) for r in results],
            ]
            raise AssertionError("\n".join(lines))

    return results
