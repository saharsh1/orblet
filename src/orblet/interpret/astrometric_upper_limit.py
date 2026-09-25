"""SB1 astrometric upper-limit consistency primitives.

For a single-lined spectroscopic binary (SB1), when the astrometric
orbit is weak or **undetected** we can still put an **upper limit** on the
astrometric semi-major axis and compare it to the spectroscopic
``a1·sin i``.  This module holds the auditable, engine-agnostic
primitives for that cross-check; the notebook only calls them.

Dark-companion (β = 0) assumption (load-bearing — see the reference
manual, ``docs/model_and_likelihoods.md`` §7, item 1)
------------------------------------------------------------------------
With a DARK companion the photocentre IS the primary, so the astrometric
orbit is the primary's true orbit about the barycentre and the
photocentre semi-major axis equals the primary axis, ``a_phot = a1``.
A luminous companion (β > 0) shrinks ``a_phot`` by photocentre
cancellation and breaks the identities below; that scenario is out of
scope for this module.

Conventions and units
---------------------
- Spectroscopy (primary RV): projected primary axis
  ``a1·sin i [AU] = K1·P·√(1−e²) / (2π)`` with ``K1`` in km/s and ``P``
  in days (= ``P_yr × DAYS_PER_KEPLER_YEAR``).  This is a **lower bound**
  on ``a_phot`` because ``sin i ≤ 1``.
- Astrometry: deprojected true axis ``a_phot [AU] = a_phot_mas /
  parallax_mas`` (angular mas over parallax mas → AU, since
  1 arcsec × 1 pc ≡ 1 AU).  ``a_phot_mas`` is the TI→Campbell
  ``a_phot_mas`` (a FOLDED, non-negative magnitude).
- Model-independent inequality: ``a_phot ≥ a1·sin i``.
- Mass function (Kepler-III normalised, AU / yr / M_sun): ``C = a³ /
  P_yr²`` with ``P_yr = P_days / DAYS_PER_KEPLER_YEAR``.  NO ``G`` /
  ``MSUN`` enters here; ``C`` equals ``(M2 sin i)³/(M1+M2)²`` for
  ``a = a1 sin i`` and ``M2³/(M1+M2)²`` for ``a = a_phot`` (β = 0).

The companion-mass inversion REUSES
:func:`orblet.interpret.companion_mass.solve_companion_mass` (no new
solver).  ``a_up`` (the upper limit) carries a POSITIVE folded-magnitude
noise floor and is NOT a detection upper limit — it must be read
together with the RV shape it is conditioned on (see the reference
manual, §6.2).

This module is numpy-only at top level, persists nothing, and emits no
logs (mirrors ``companion_mass.py`` I/O hygiene).
"""

from __future__ import annotations

import numpy as np

from orblet.constants import AU_M, DAYS_PER_KEPLER_YEAR
from orblet.interpret.companion_mass import solve_companion_mass

# AU per (km/s · day): 86400 s/day × 1000 m/km / AU_M [m].  Derived
# LOCALLY from the PUBLIC AU_M (arch#4): this module deliberately does
# NOT import ``rv_chain._AU_PER_KMS_DAY``.  The two are bit-equal by
# construction; the equality is guarded from the test side by
# ``tests/test_astrometric_upper_limit.py`` (T-AUL-21, divergence guard).
_AU_PER_KMS_DAY: float = 86400.0 / (AU_M / 1000.0)


def a1sini_au(
    K1_kms: np.ndarray | float,
    P_days: np.ndarray | float,
    e: np.ndarray | float,
) -> np.ndarray:
    """Projected primary semi-major axis ``a1·sin i`` in AU.

    ``a1·sin i [AU] = K1·P·√(1−e²) / (2π)``.  This is the spectroscopic
    LOWER bound on the photocentre axis ``a_phot`` (β = 0).

    Parameters
    ----------
    K1_kms : array or float
        Primary radial-velocity semi-amplitude (km/s).
    P_days : array or float
        Orbital period in days (= ``P_yr × DAYS_PER_KEPLER_YEAR``).
    e : array or float
        Eccentricity (dimensionless).  No clamping: ``e = 1`` gives
        exactly 0 and ``e > 1`` gives NaN via ``√(1−e²)``.

    Returns
    -------
    numpy.ndarray
        ``a1·sin i`` in AU, elementwise over broadcast inputs.
    """
    K = np.asarray(K1_kms, dtype=float)
    P = np.asarray(P_days, dtype=float)
    ecc = np.asarray(e, dtype=float)
    return K * P * np.sqrt(1.0 - ecc**2) / (2.0 * np.pi) * _AU_PER_KMS_DAY


def a_phot_au(
    a_phot_mas: np.ndarray | float,
    parallax_mas: np.ndarray | float,
) -> np.ndarray:
    """Deproject a photocentre amplitude from mas to AU.

    ``a_phot [AU] = a_phot_mas / parallax_mas`` (1 arcsec × 1 pc ≡ 1 AU).
    Under β = 0 this is the true primary axis about the barycentre.

    Parameters
    ----------
    a_phot_mas : array or float
        Photocentre semi-major axis on the sky (mas), e.g. the
        TI→Campbell ``a_phot_mas`` (a folded, non-negative magnitude).
    parallax_mas : array or float
        Parallax (mas).

    Returns
    -------
    numpy.ndarray
        ``a_phot`` in AU.  Elements with ``parallax_mas ≤ 0`` are NaN
        (no physical distance).  Never raises.
    """
    alpha = np.asarray(a_phot_mas, dtype=float)
    plx = np.asarray(parallax_mas, dtype=float)
    # Guarded divide: parallax ≤ 0 has no physical distance → NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = alpha / plx
    return np.where(plx > 0.0, ratio, np.nan)


def mass_function_au(
    a_au: np.ndarray | float,
    P_days: np.ndarray | float,
) -> np.ndarray:
    """Kepler-III normalised mass function ``C = a³ / P_yr²`` (M_sun).

    ``P_yr = P_days / DAYS_PER_KEPLER_YEAR``.  No ``G`` / ``MSUN``
    appears (AU / yr / M_sun units).  ``C`` equals ``(M2 sin i)³ /
    (M1+M2)²`` when ``a = a1 sin i`` and ``M2³ / (M1+M2)²`` when
    ``a = a_phot`` (β = 0).

    Parameters
    ----------
    a_au : array or float
        Semi-major axis (AU): either ``a1·sin i`` or ``a_phot``.
    P_days : array or float
        Orbital period in days.

    Returns
    -------
    numpy.ndarray
        Mass function ``C`` in M_sun, elementwise.
    """
    a = np.asarray(a_au, dtype=float)
    P_yr = np.asarray(P_days, dtype=float) / DAYS_PER_KEPLER_YEAR
    return a**3 / P_yr**2


def astrometric_upper_limit_au(
    a_phot_au_draws: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """One-sided upper bound on ``a_phot`` from posterior draws (AU).

    Returns ``np.percentile(finite draws, percentile)`` (default 95%,
    editable).  This is an RV-shape-conditioned CREDIBLE upper bound,
    NOT a detection upper limit: because ``a_phot_mas`` is a folded,
    non-negative magnitude the draws carry a POSITIVE noise floor even
    on a pure non-detection, so ``a_up`` can only be pushed UPWARD (see
    the reference manual, §6.2).

    Parameters
    ----------
    a_phot_au_draws : numpy.ndarray
        Posterior draws of ``a_phot`` in AU.  NaN draws are dropped.
    percentile : float, default 95.0
        One-sided percentile for the upper bound.

    Returns
    -------
    float
        The percentile of the finite draws, or NaN if none are finite.
    """
    draws = np.asarray(a_phot_au_draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def companion_mass_bracket(
    a1sini_med_au: float,
    a_up_au: float,
    P_days: float,
    M1_msun: float,
) -> tuple[float, float]:
    """Interpretive companion-mass bracket ``(M2_min, M2_max)`` in M_sun.

    Given an assumed primary mass ``M1``:

    - ``M2_min`` = ``solve_companion_mass(mass_function_au(a1sini_med_au,
      P_days), M1, sin i = 1)`` — the representative MINIMUM companion
      mass at the MEDIAN spectroscopic axis (``sin i = 1`` floors it).
    - ``M2_max`` = ``solve_companion_mass(mass_function_au(a_up_au,
      P_days), M1, sin i = 1)`` — the mass at the astrometric upper
      limit ``a_up``.

    REUSES :func:`orblet.interpret.companion_mass.solve_companion_mass`
    (no new solver).  Returns ``(nan, nan)`` on NaN operands; never
    raises.

    Notebook contract (v2.1 D1): the bracket INVERTS (``M2_max <
    M2_min``, equivalently ``a_up < a1sini_med``) in the window where
    ``a_up`` falls below the median spectroscopic axis.  Notebook Cell C
    keys its interval-vs-inconsistent branch on ``M2_max >= M2_min`` and
    never renders an inverted bracket in interval form.  This is a
    DIFFERENT question from the Layer-1 q16 ``consistency_verdict`` and
    the two may legitimately disagree in that window.

    Parameters
    ----------
    a1sini_med_au : float
        Median spectroscopic ``a1·sin i`` (AU) → ``M2_min``.
    a_up_au : float
        Astrometric upper limit ``a_up`` (AU) → ``M2_max``.
    P_days : float
        Orbital period (days).
    M1_msun : float
        Assumed primary mass (M_sun).

    Returns
    -------
    tuple of float
        ``(M2_min, M2_max)`` in M_sun.
    """
    c_min = mass_function_au(a1sini_med_au, P_days)
    c_max = mass_function_au(a_up_au, P_days)
    M2_min = float(solve_companion_mass(c_min, M1_msun, 1.0))
    M2_max = float(solve_companion_mass(c_max, M1_msun, 1.0))
    return M2_min, M2_max


def consistency_verdict(a_up_au: float, a1sini_lower_au: float) -> bool:
    """Model-independent Layer-1 consistency check (heuristic).

    Returns ``True`` iff ``a_up_au >= a1sini_lower_au``.  The caller is
    expected to pass a LOWER percentile (q16) of the spectroscopic
    ``a1·sin i`` draws so that a ``True`` verdict is CONSERVATIVE:
    "falsified" fires only when even the astrometric upper bound falls
    below a plausibly-low spectroscopic axis.  The check is a heuristic,
    not a calibrated test.  NaN in either operand → ``False``.

    Parameters
    ----------
    a_up_au : float
        Astrometric upper limit ``a_up`` (AU).
    a1sini_lower_au : float
        Lower percentile (q16) of the spectroscopic ``a1·sin i`` (AU).

    Returns
    -------
    bool
        ``True`` if consistent with a dark single companion (β = 0),
        else ``False``.
    """
    a_up = float(a_up_au)
    lower = float(a1sini_lower_au)
    if not (np.isfinite(a_up) and np.isfinite(lower)):
        return False
    return bool(a_up >= lower)
