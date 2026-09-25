"""Peak measurements on a periodogram of any kind."""

from __future__ import annotations

import numpy as np

from orblet.periodogram._common import _as_1d_float


def peak_fwhm_days(
    periods_days: np.ndarray,
    scores: np.ndarray,
    *,
    level: float = 0.5,
) -> tuple[float, float]:
    """
    Full width at half-maximum of the tallest peak in a periodogram.

    Walks outward from ``argmax(scores)`` in both directions until the
    score drops below ``level × max``, and reports the width in
    **days**.  Useful as a data-driven width for Gaussian priors or
    penalty terms centred on the periodogram peak — e.g. when injecting
    a period constraint into a Thiele-Innes astrometric fit.

    Parameters
    ----------
    periods_days : 1-D array
        Trial period grid (days).  Needs to match ``scores`` in length.
        Works for either ascending or descending ordering.
    scores : 1-D array
        Periodogram score at each period.  Must be finite at the peak.
    level : float, default 0.5
        Fraction of the peak score that defines the "half-max" contour.
        Use 0.5 for FWHM, or a smaller fraction for a looser width.

    Returns
    -------
    peak_period_days : float
        Period at which the score is maximum.
    width_days : float
        Width of the peak at ``level × max``, measured as
        ``|periods_days[i_right] − periods_days[i_left]|`` where
        ``i_left, i_right`` are the first indices outside the contour.

    Notes
    -----
    **Period vs frequency width.**  If the periodogram is sampled
    uniformly in frequency, the period-space FWHM is asymmetric about
    the peak (tighter on the short-period side, looser on the long-period
    side).  Pragmatically fine for narrow peaks; use the frequency-space
    equivalent ``σ_P ≈ P² · σ_f`` for a symmetric Gaussian approximation.

    **Degenerate peaks.**  If the peak is poorly isolated — multi-peak
    region, broad low-score hump, or alias blending — the outward walk
    can run to a grid edge and return an inflated width comparable to
    the peak period itself.  Sanity-check
    ``width_days / peak_period_days``; values near or above 1 indicate
    the periodogram isn't providing a usable localisation.

    Examples
    --------
    >>> pg = compute_pdc_periodogram(t, periods_days, obs_dist)
    >>> P0, fwhm = peak_fwhm_days(pg["periods_days"], pg["scores"])
    >>> sigma_P_days = fwhm / 2.355    # Gaussian-equivalent σ
    """
    periods_arr = _as_1d_float(periods_days, "periods_days")
    scores_arr = _as_1d_float(scores, "scores")
    if periods_arr.shape != scores_arr.shape:
        raise ValueError(
            f"periods_days shape {periods_arr.shape} does not match "
            f"scores shape {scores_arr.shape}."
        )
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}.")

    i_peak = int(np.nanargmax(scores_arr))
    peak_score = float(scores_arr[i_peak])
    threshold = level * peak_score
    n = len(scores_arr)

    # Walk outward from the peak in both index directions.  The direction
    # in period space depends on the ordering of periods_arr but the
    # result — a width in days — is independent of it.
    i_left = i_peak
    while i_left > 0 and scores_arr[i_left] >= threshold:
        i_left -= 1
    i_right = i_peak
    while i_right < n - 1 and scores_arr[i_right] >= threshold:
        i_right += 1

    peak_period = float(periods_arr[i_peak])
    width = abs(float(periods_arr[i_right]) - float(periods_arr[i_left]))
    return peak_period, width
