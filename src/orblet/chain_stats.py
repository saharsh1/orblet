"""
Chain-summary helpers for posterior samples.

Linear-quantity helpers (:func:`chain_quantiles`,
:func:`chain_credible_interval`, :func:`chain_summary_table`) are
straightforward wrappers over :func:`numpy.quantile` with strict
input validation.

Angular-quantity helper (:func:`chain_circular_summary`) computes the
circular mean and circular standard deviation of an angle chain.  Two
implementation notes worth preserving:

- Inputs are wrapped to ``[0, 2π)`` before the circular-std call
  because :func:`scipy.stats.circstd` does NOT internally fold values
  outside ``[low, high]``.
- For known-multimodal posteriors (e.g., the ``(ω + π, Ω + π)`` 180°
  degeneracy in retrograde-inclination astrometric fits), the circular
  mean collapses to a meaningless midpoint between modes; the docstring
  flags this so the helper is not used naively for such cases.

*Dependencies.*  Three of the four helpers need only numpy and scipy.
:func:`chain_summary_table` returns a :class:`pandas.DataFrame`, and pandas is
an OPTIONAL dependency (``pip install orblet[tables]``) imported inside that
function rather than at module scope — otherwise one unused table helper would
make the whole module unimportable for anyone without pandas.  The numbers are
available without it: :func:`chain_quantiles` returns the same values as a
plain dict, and :func:`chain_summary_table` only arranges them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import circstd

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    import pandas as pd


def _quantile_key(level: float) -> str:
    """Encode a quantile level as a 3-digit-millis key (e.g., ``q025``)."""
    return f"q{int(round(level * 1000)):03d}"


def chain_quantiles(
    samples: np.ndarray,
    levels: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> dict[str, float]:
    """
    Compute quantiles of a 1-D chain.

    Parameters
    ----------
    samples : np.ndarray, shape (n,)
        1-D array of posterior samples.  Empty arrays raise
        :class:`ValueError`.  NaN samples raise :class:`ValueError`.
    levels : tuple of floats in (0, 1), default ``(0.025, 0.5, 0.975)``
        Quantile levels.  Returned dict keys are integer-encoded:
        ``f"q{int(round(lvl*1000)):03d}"`` → ``"q025"``, ``"q500"``,
        ``"q975"``.

    Returns
    -------
    dict[str, float]
        Mapping of quantile-key to value.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("samples is empty")
    if np.any(np.isnan(samples)):
        raise ValueError("samples contains NaN")
    qs = np.quantile(samples, list(levels))
    return {_quantile_key(lvl): float(qs[i]) for i, lvl in enumerate(levels)}


def chain_credible_interval(
    samples: np.ndarray,
    level: float = 0.99,
) -> dict[str, float]:
    """
    Compute the central credible interval at the given level.

    Parameters
    ----------
    samples : np.ndarray, shape (n,)
        1-D array of posterior samples.  Empty arrays raise
        :class:`ValueError`.
    level : float in (0, 1), default 0.99
        Credible-interval fraction (e.g., 0.68, 0.95, 0.99).
        Computes quantiles at ``(1-level)/2`` and ``(1+level)/2``.

    Returns
    -------
    dict[str, float]
        ``{"low": float, "high": float, "level": float}``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("samples is empty")
    if np.any(np.isnan(samples)):
        raise ValueError("samples contains NaN")
    lo_q = (1.0 - level) / 2.0
    hi_q = (1.0 + level) / 2.0
    lo, hi = np.quantile(samples, [lo_q, hi_q])
    return {"low": float(lo), "high": float(hi), "level": float(level)}


def chain_circular_summary(samples: np.ndarray) -> dict[str, float]:
    """
    Compute circular-mean and circular-std of an angular chain.

    Parameters
    ----------
    samples : np.ndarray, shape (n,)
        1-D array of angle samples in radians.  Convention may be
        ``[0, 2π)`` or ``[-π, π)``; the function pre-wraps to
        ``[0, 2π)`` before computing.

    Returns
    -------
    dict[str, float]
        ``{"circmean_rad": float, "circstd_rad": float}``.  Boundary
        convention: ``circmean_rad`` is in ``[0, 2π)``.

    Notes
    -----
    Pre-wrap to ``[0, 2π)`` is required because
    :func:`scipy.stats.circstd` does NOT internally re-wrap; values
    outside ``[low, high]`` are not folded.

    For posteriors known to be multi-modal (e.g., ω with the
    ``(ω + π, Ω + π)`` 180° degeneracy in retrograde-inclination
    astrometric fits), the circular mean collapses bimodal
    posteriors to a meaningless midpoint.  Report per-mode summaries
    instead in those cases.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("samples is empty")
    if np.any(np.isnan(samples)):
        raise ValueError("samples contains NaN")
    # Pre-wrap to [0, 2π) — scipy.stats.circstd does not re-wrap.
    samples = samples % (2.0 * np.pi)
    # Circular mean via complex-exponential trick: arg(mean(exp(i·θ)))
    # is the canonical wrap-aware estimator and ignores additive 2π
    # shifts in any individual sample.
    circmean = float(np.angle(np.mean(np.exp(1j * samples))) % (2.0 * np.pi))
    circstd_val = float(circstd(samples, low=0.0, high=2.0 * np.pi))
    return {"circmean_rad": circmean, "circstd_rad": circstd_val}


def chain_summary_table(
    chain_dict: dict[str, np.ndarray],
    levels: tuple[float, ...] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Tabular summary across multiple chain keys.

    Parameters
    ----------
    chain_dict : dict[str, np.ndarray]
        Mapping of parameter name → 1-D sample array.
    levels : tuple of floats, default ``(0.025, 0.5, 0.975)``
        Same as :func:`chain_quantiles`.

    Returns
    -------
    pd.DataFrame
        Columns: ``parameter``, plus one column per level (integer-
        encoded keys: ``q025``, ``q500``, ``q975``).
    """
    # pandas is imported HERE, not at module scope, so that the three helpers
    # above — which need only numpy and scipy — stay usable when pandas is not
    # installed.  A top-of-file import would take all four down over this one
    # call.  See the module docstring for the install story.
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "chain_summary_table returns a pandas DataFrame, and orblet does "
            "not install pandas by default. Either `pip install orblet[tables]`, "
            "or call chain_quantiles per parameter — it returns the same "
            "numbers as a plain dict and needs only numpy."
        ) from exc

    rows = []
    for name, samples in chain_dict.items():
        qs = chain_quantiles(samples, levels=levels)
        row = {"parameter": name}
        row.update(qs)
        rows.append(row)
    columns = ["parameter"] + [_quantile_key(lvl) for lvl in levels]
    return pd.DataFrame(rows, columns=columns)
