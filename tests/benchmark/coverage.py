"""
Coverage-tier benchmark assertion (truth-based, sampler-agnostic).

Where ``closure_test`` answers "does ONE posterior enclose the truth?",
``coverage_test`` answers the calibration question across MANY fits:

    "Over N independent fits of the SAME truth (different noise draws),
     does the truth land inside the (1 − α) credible interval about the
     nominal fraction (1 − α) of the time, for each parameter?"

This is a small-N CALIBRATION SANITY CHECK, not full Simulation-Based
Calibration (SBC).  It is deliberately sampler-agnostic: it consumes a
list of already-computed chain dicts and never invokes Julia / NUTS or
any particular fitter.  Honest credible intervals mean empirical
coverage sits inside the binomial band of the nominal level; persistent
UNDER-coverage at both levels flags a biased/over-tight posterior, while
one level off but not the other points at interval-WIDTH miscalibration.

All inputs synthetic; no data files read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ParamCoverage:
    """Per-parameter, per-level empirical coverage outcome."""

    name: str
    level: float
    n_fits: int
    n_inside: int
    fraction: float
    expected: float
    band_half_width: float
    in_band: bool
    # Truth's rank within each posterior (0..n_samples), one per fit —
    # a flat rank histogram across fits is the SBC self-consistency check.
    ranks: tuple[int, ...] = field(default_factory=tuple)

    @property
    def band_lo(self) -> float:
        return self.expected - self.band_half_width

    @property
    def band_hi(self) -> float:
        return self.expected + self.band_half_width

    def __str__(self) -> str:
        flag = "OK " if self.in_band else "OUT"
        return (
            f"  [{flag}] {self.name:<20s} L={self.level:.2f}  "
            f"cov={self.fraction:.3f} ({self.n_inside}/{self.n_fits})  "
            f"expect {self.expected:.3f} ± {self.band_half_width:.3f}"
        )


def _ci_bounds(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided equal-tailed credible interval at ``level``."""
    half_alpha = 0.5 * (1.0 - level)
    return (
        float(np.quantile(samples, half_alpha)),
        float(np.quantile(samples, 1.0 - half_alpha)),
    )


def _binomial_band_half_width(level: float, n: int, *, k_sigma: float = 2.0) -> float:
    """Half-width of the ``k_sigma``·σ binomial band on the coverage fraction.

    For ``n`` fits each in/out at nominal probability ``p = level``, the
    inside-COUNT is Binomial(n, p); its std is ``sqrt(n p (1−p))``.  As a
    FRACTION that is ``sqrt(p(1−p)/n)``.  The default ``k_sigma = 2``
    gives the ~95 % acceptance band used by the gates.
    """
    p = level
    return k_sigma * math.sqrt(p * (1.0 - p) / n)


def coverage_test(
    truth: dict,
    chains_list: list[dict],
    *,
    levels: tuple[float, ...] = (0.68, 0.90),
    k_sigma: float = 2.0,
    raise_on_fail: bool = True,
) -> dict[str, list[ParamCoverage]]:
    """
    Empirical coverage of the truth across many fits, per parameter & level.

    Parameters
    ----------
    truth : dict
        Map ``param_name → truth_value`` (the SAME truth for every fit).
    chains_list : list of dict
        One posterior-chain dict per fit (each value a 1-D array of MCMC
        samples).  Only keys present in BOTH ``truth`` and every chain
        dict are checked; a missing key raises ``KeyError`` so silent
        under-testing is impossible.
    levels : tuple of float, default (0.68, 0.90)
        Credible levels at which to measure coverage.
    k_sigma : float, default 2.0
        Width of the binomial acceptance band in standard deviations.
    raise_on_fail : bool, default True
        If True, raise :class:`AssertionError` listing every
        (param, level) whose empirical coverage fell outside the band.

    Returns
    -------
    results : dict
        ``param_name → list[ParamCoverage]`` (one entry per level), in the
        key order of ``truth``.

    Raises
    ------
    KeyError
        If a key in ``truth`` is absent from any chain dict.
    AssertionError
        If any (param, level) coverage is out of band (only when
        ``raise_on_fail`` is True).
    """
    if not chains_list:
        raise ValueError("coverage_test: chains_list is empty")
    for lvl in levels:
        if not 0.0 < lvl < 1.0:
            raise ValueError(f"levels must lie in (0, 1); got {lvl}")

    n_fits = len(chains_list)
    results: dict[str, list[ParamCoverage]] = {}

    for name, truth_val in truth.items():
        truth_val = float(truth_val)
        # Inside-flags + truth ranks, collected once across fits, reused
        # for every level (rank is level-independent).
        inside_by_level = {lvl: 0 for lvl in levels}
        ranks: list[int] = []
        for chains in chains_list:
            if name not in chains:
                raise KeyError(
                    f"coverage_test: parameter {name!r} requested in `truth` "
                    f"but missing from a chain dict. Available keys: "
                    f"{sorted(chains.keys())}"
                )
            samples = np.asarray(chains[name], dtype=float)
            # Rank of the truth among the posterior samples (0..n_samples).
            ranks.append(int(np.sum(samples < truth_val)))
            for lvl in levels:
                lo, hi = _ci_bounds(samples, lvl)
                if lo <= truth_val <= hi:
                    inside_by_level[lvl] += 1

        per_level: list[ParamCoverage] = []
        for lvl in levels:
            n_in = inside_by_level[lvl]
            frac = n_in / n_fits
            half = _binomial_band_half_width(lvl, n_fits, k_sigma=k_sigma)
            per_level.append(
                ParamCoverage(
                    name=name, level=lvl, n_fits=n_fits, n_inside=n_in,
                    fraction=frac, expected=lvl, band_half_width=half,
                    in_band=abs(frac - lvl) <= half,
                    ranks=tuple(ranks),
                )
            )
        results[name] = per_level

    if raise_on_fail:
        failed = [
            pc for pcs in results.values() for pc in pcs if not pc.in_band
        ]
        if failed:
            lines = [
                f"Coverage test failed: {len(failed)} (param, level) pairs "
                f"outside the {k_sigma:.0f}σ binomial band over "
                f"{n_fits} fits.",
                *[str(pc) for pcs in results.values() for pc in pcs],
            ]
            raise AssertionError("\n".join(lines))

    return results
