"""The PDC periodogram engine: a data distance matrix against the phase
distance matrix, swept over trial periods."""

from __future__ import annotations

import numpy as np

from orblet.periodogram._common import _as_1d_float
from orblet.periodogram.distances import _circular_distance
from orblet.periodogram.dcor import (
    _pdc_score_from_centered,
    _semipartial_pdc_score_from_centered,
    _u_center,
    _u_inner,
)

# ===================================================
# Phase distance correlation (PDC) periodograms
# ===================================================
#
# The PDC periodogram detects periodicity by measuring the distance
# correlation between an observation-space distance matrix and a
# phase-space distance matrix, swept over trial periods.
#
# The partial PDC controls for a confounding variable (e.g. the scan
# angle): the nuisance component is projected out of one space before the
# correlation is taken.  Because only one space is residualised it is
# formally a semi-partial distance correlation; it is called the partial
# PDC throughout.
#
# The design separates distance-matrix construction (data-type-specific)
# from the periodogram engine (generic).  The user builds distance
# matrices explicitly, then passes them to compute_pdc_periodogram.
#
# References:
#   Szekely & Rizzo (2014), Annals of Statistics 42(6).
#   Zucker (2018), MNRAS 474(1).


def _phase_distance_matrix(t: np.ndarray, period: float) -> np.ndarray:
    """
    Circular phase distance matrix.

    d_ij = phi_ij × (P - phi_ij),  where phi_ij = (t_i - t_j) mod P.
    Maximal at half-period separation, zero at full-period.
    """
    return _circular_distance(t, period)


def compute_pdc_periodogram(
    t_days: np.ndarray,
    periods_days: np.ndarray,
    obs_dist: np.ndarray,
    *,
    nuisance_dist: np.ndarray | None = None,
    partial_mode: str = "none",
) -> dict:
    """
    Compute a PDC (phase distance correlation) periodogram.

    Measures the distance correlation between a precomputed
    observation distance matrix and a circular phase distance matrix,
    swept over a grid of trial periods.

    The observation distance matrix encodes pairwise dissimilarity
    of the data (e.g. RV, astrometry, spectra) and is built by the
    user with one of the distance-matrix functions:

    - :func:`scalar_distance_matrix` — for radial velocities
    - :func:`astrometric_segment_distance_matrix` — for Gaia AL data
    - :func:`spectral_distance_matrix` — for epoch spectra
    - Any user-supplied (N, N) symmetric matrix

    Parameters
    ----------
    t_days : 1-D array
        Observation times (days).  Length N.
    periods_days : 1-D array
        Trial periods to evaluate (days).
    obs_dist : (N, N) array
        Observation distance matrix.
    nuisance_dist : (N, N) array, optional
        Confounding-variable distance matrix for the partial PDC.
        Built with e.g. :func:`scan_angle_distance_matrix` or
        :func:`spectral_distance_matrix`.
    partial_mode : str, default ``"none"``
        ``"none"`` — ordinary PDC (no nuisance correction).
        ``"semi"`` — partial PDC, in its semi-partial form: the nuisance
        is projected out of the observation matrix (requires
        *nuisance_dist*).

    Returns
    -------
    result : dict
        ``"periods_days"``     — trial period grid.
        ``"scores"``           — PDC score at each period.
        ``"best_period_days"`` — period of highest score.
        ``"best_score"``       — peak score value.
        ``"partial_mode"``     — configuration echo.
        ``"coupling"``         — ONLY with ``partial_mode="semi"``: the
        distance correlation between the observation and nuisance
        matrices (for the scan angle, the coupling D_cpl of
        :func:`scan_angle_coupling`).
    """
    t = _as_1d_float(t_days, "t_days")
    periods = _as_1d_float(periods_days, "periods_days")
    obs_dist = np.asarray(obs_dist, dtype=float)

    n = len(t)
    if obs_dist.shape != (n, n):
        raise ValueError(
            f"obs_dist shape {obs_dist.shape} does not match "
            f"number of epochs ({n})."
        )

    if nuisance_dist is not None:
        nuisance_dist = np.asarray(nuisance_dist, dtype=float)
        if nuisance_dist.shape != (n, n):
            raise ValueError(
                f"nuisance_dist shape {nuisance_dist.shape} does not "
                f"match number of epochs ({n})."
            )

    # ── Pre-center fixed matrices (done once, not per period) ───────────
    A = _u_center(obs_dist)
    aa = _u_inner(A, A)

    Z = None
    zz = az = 0.0
    if partial_mode == "semi":
        if nuisance_dist is None:
            raise ValueError("partial_mode='semi' requires nuisance_dist")
        Z = _u_center(nuisance_dist)
        zz = _u_inner(Z, Z)
        az = _u_inner(A, Z)
        # The same arithmetic as distance_correlation(obs_dist, nuisance_dist).
        coupling = _pdc_score_from_centered(A, Z, aa)

    # ── Sweep over trial periods ──────────────────────────────────────────
    scores = np.full_like(periods, np.nan, dtype=float)

    for i, P in enumerate(periods):
        if not np.isfinite(P) or P <= 0:
            continue

        B = _u_center(_phase_distance_matrix(t, P))

        if partial_mode == "none":
            scores[i] = _pdc_score_from_centered(A, B, aa)
        elif partial_mode == "semi":
            scores[i] = _semipartial_pdc_score_from_centered(
                A, B, Z, zz, az,
            )
        else:
            raise ValueError("partial_mode must be 'none' or 'semi'")

    best_idx = int(np.nanargmax(scores))

    result = {
        "periods_days": periods,
        "scores": scores,
        "best_period_days": float(periods[best_idx]),
        "best_score": float(scores[best_idx]),
        "partial_mode": partial_mode,
    }
    if partial_mode == "semi":
        result["coupling"] = float(coupling)
    return result


def pdc_false_alarm_probability(score, n: int):
    """
    Pointwise false-alarm probability of a PDC score.

    Uses the χ² approximation to the null distribution of the unbiased
    distance correlation (Shen, Panda & Vogelstein 2022):

        FAP(D) = P(χ²₁ > n·D + 1).

    Parameters
    ----------
    score : float or array
        PDC score(s) — one value, or a whole periodogram.  NaN passes
        through as NaN.
    n : int
        Number of epochs the score was computed from (n ≥ 4).

    Returns
    -------
    fap : float or ndarray
        Same shape as ``score``.

    Notes
    -----
    **Pointwise.**  This is the probability at ONE trial period.  Searching
    a grid of periods and reporting the best one is not corrected for (no
    look-elsewhere factor); the chance that some period in the grid beats
    a threshold is larger.  A scramble null over the same epochs is the
    calibrated alternative.

    **Partial PDC.**  The approximation is derived for the ordinary
    distance correlation; applying it to the partial (semi-partial) score
    is an assumption.
    """
    # Lazy scipy import: importing the periodograms must stay cheap.
    from scipy.stats import chi2

    n = int(n)
    if n < 4:
        raise ValueError(f"n must be >= 4 (the unbiased statistic needs it), got {n}.")
    fap = chi2.sf(n * np.asarray(score, dtype=float) + 1.0, 1)
    return float(fap) if np.ndim(fap) == 0 else fap

