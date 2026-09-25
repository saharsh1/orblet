"""
Shared Kepler-equation solver and Campbell-elements orbit projection.

These helpers are consolidated here so that ``orbit/postprocess.py``,
``orbit/plotting.py`` and ``batch/report.py`` all share one
implementation.

Naming note
-----------
The sky-plane projector here is :func:`campbell_xy` — it consumes
Campbell elements (ω, i, Ω, a).  Do not confuse it with the genuinely
Thiele-Innes-amplitude-parametrised
:func:`orblet.model.thiele_innes_xy`
(which consumes A, B, F, G).  The two are independent
implementations with different input semantics; this module's function
is an independent reimplementation oracle and is NOT the production
forward model.
"""

from __future__ import annotations

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR


def _wrap_to_pi(x):
    """Wrap angle (rad) into ``[-π, π)``.  Vectorised over numpy arrays.

    The single source of truth for the wrap convention: every site that
    wraps an angle (chain export, :mod:`orblet.rv_chain`'s prior
    bridge) imports this helper, so round trips stay stable.  Note the
    half-open side: an input of exactly ``+π`` maps to ``-π``.
    """
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def solve_kepler(
    mean_anomaly: np.ndarray,
    ecc: float,
    *,
    tol: float = 1e-10,
    maxiter: int = 50,
) -> np.ndarray:
    """
    Newton iteration for the Kepler equation ``M = E - e sin E``.

    Parameters
    ----------
    mean_anomaly : array
        Mean anomaly in radians.
    ecc : float
        Eccentricity (``0 ≤ e < 1``).
    tol : float, default 1e-10
        Stopping tolerance on ``max|dE|``.
    maxiter : int, default 50
        Hard cap on iterations.

    Returns
    -------
    E : array
        Eccentric anomaly in radians, same shape as ``mean_anomaly``.

    Notes
    -----
    Entries the Newton loop fails to converge are repaired by bisection
    (see below), so the returned ``E`` satisfies ``|E - e sin E - M| <= 1e-9``
    for every entry at any ``e < 1``.  Entries that DO converge are
    byte-identical to the plain-Newton result (the repair never touches
    them) — this byte-identity assumes ``tol <= 1e-9`` (a converged step
    has equation residual ``|dE|·|1 - e cos E| < 2·tol``, safely under the
    repair gate at the default ``tol``).
    """
    M_arr = np.asarray(mean_anomaly, dtype=float)
    E = M_arr.copy()
    for _ in range(maxiter):
        dE = (M_arr - E + ecc * np.sin(E)) / (1.0 - ecc * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break

    # Newton from the E0 = M start diverges for e ≳ 0.96 near periastron
    # (M ~ 0 mod 2π): the first denominator 1 - e·cos M ≈ 1 - e is tiny and
    # the iterate is thrown out of the convergence basin (residuals up to
    # ~1e20 at e = 0.99, visible as a sudden "phase jump" in a plotted orbit).
    # Repair ONLY the non-converged entries by bisection: f(E) = E - e sin E
    # is strictly increasing (f' = 1 - e cos E ≥ 1 - e > 0), so for wrapped
    # M ∈ [0, 2π) the root is bracketed by [0, 2π] and bisection cannot fail.
    # Converged entries pass through untouched (byte-identical fast path).
    # The residual gate is sufficient: f is a global bijection for e < 1, so
    # the equation has exactly ONE real root per M — a diverged Newton entry
    # always shows a large residual, never a spurious exact solution.
    f_resid = E - ecc * np.sin(E) - M_arr
    bad = np.abs(f_resid) > 1e-9
    if np.any(bad):
        M_wrap = M_arr % (2.0 * np.pi)
        lo = np.zeros_like(M_wrap)
        hi = np.full_like(M_wrap, 2.0 * np.pi)
        # 64 halvings of [0, 2π] reach interval width ~3e-19 < float eps.
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            high_side = mid - ecc * np.sin(mid) > M_wrap
            hi = np.where(high_side, mid, hi)
            lo = np.where(high_side, lo, mid)
        # Shift back onto the caller's (possibly unwrapped) M branch:
        # E(M + 2πk) = E(M) + 2πk.
        E = np.where(bad, 0.5 * (lo + hi) + (M_arr - M_wrap), E)
    return E


def campbell_xy(
    t_mjd: np.ndarray,
    period_yr: float,
    ecc: float,
    omega_rad: float,
    inc_rad: float,
    Omega_rad: float,
    tp_mjd: float,
    a_mas: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sky-plane offsets (Δα*, Δδ) [mas] for a Campbell-elements orbit.

    This is an **independent textbook/reference** Thiele-Innes
    projector (it builds A, B, F, G from the Campbell elements ω, i, Ω
    and projects the in-plane orbit onto the sky).  It is used as a
    cross-check oracle for the production astrometric engine; it is
    NOT the production forward model.

    Frame convention (load-bearing)
    -------------------------------
    This function uses the **primary-frame argument of periastron**
    ``omega_rad`` directly and a **positive** ``a_mas`` semi-major axis.

    The production photocenter forward
    :func:`orblet.model.campbell_xy`
    uses the SAME convention as this oracle: primary-frame ``omega``
    as-is and a POSITIVE photocenter amplitude.

    Callers therefore pass ``omega_rad = omega_primary`` and
    ``a_mas = +a_phot_mas`` directly.  Do NOT apply an
    ``omega -> omega + pi`` shift together with a negative amplitude:
    that compensating pair would introduce the very mismatch it looks
    like it cancels.

    Parameters
    ----------
    t_mjd : array
        Epochs in MJD.
    period_yr : float
        Period in Keplerian years (365.25 d).
    ecc : float
        Eccentricity (``0 ≤ e < 1``).
    omega_rad : float
        Primary-frame argument of periastron ω, radians.
    inc_rad : float
        Inclination i, radians.
    Omega_rad : float
        Longitude of ascending node Ω, radians.
    tp_mjd : float
        Time of periastron, MJD.
    a_mas : float
        Photocenter semi-major axis in mas (POSITIVE for the textbook
        primary-frame convention used here).

    Returns
    -------
    d_ra : array
        Δα* (= Δα · cos δ) in mas, same shape as ``t_mjd``.
    d_dec : array
        Δδ in mas, same shape as ``t_mjd``.
    """
    P_days = period_yr * DAYS_PER_KEPLER_YEAR
    # Wrap mean anomaly into [0, 2π) for numerical stability of the solver.
    M_anom = (2.0 * np.pi * (t_mjd - tp_mjd) / P_days) % (2.0 * np.pi)
    E = solve_kepler(M_anom, ecc)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    r_norm = (1.0 - ecc ** 2) / (1.0 + ecc * np.cos(nu))
    x_orb = r_norm * np.cos(nu)
    y_orb = r_norm * np.sin(nu)

    cos_O, sin_O = np.cos(Omega_rad), np.sin(Omega_rad)
    cos_w, sin_w = np.cos(omega_rad), np.sin(omega_rad)
    cos_i = np.cos(inc_rad)

    A = a_mas * (cos_O * cos_w - sin_O * sin_w * cos_i)
    B = a_mas * (sin_O * cos_w + cos_O * sin_w * cos_i)
    F = a_mas * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    G = a_mas * (-sin_O * sin_w + cos_O * cos_w * cos_i)

    d_ra = B * x_orb + G * y_orb
    d_dec = A * x_orb + F * y_orb
    return d_ra, d_dec


def thiele_innes_al(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    period_yr: float,
    ecc: float,
    omega_rad: float,
    inc_rad: float,
    Omega_rad: float,
    tp_mjd: float,
    a_mas: float,
) -> np.ndarray:
    """
    Along-scan projection of the photocenter orbit, in mas.

    Equivalent to ``Δα* sin ψ + Δδ cos ψ`` evaluated from
    :func:`campbell_xy`.  The angle arguments use the same primary-frame
    convention and positive ``a_mas`` as :func:`campbell_xy` (see that
    docstring for the oracle-vs-engine frame distinction).
    """
    d_ra, d_dec = campbell_xy(
        t_mjd, period_yr, ecc, omega_rad, inc_rad, Omega_rad, tp_mjd,
        a_mas,
    )
    return d_ra * np.sin(psi) + d_dec * np.cos(psi)
