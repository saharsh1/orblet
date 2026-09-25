"""The design-matrix COLUMN functions — the arithmetic behind the builders.

The six functions every design in orblet is assembled from:

============================== ================================
function                       what it builds
============================== ================================
``astrometric_columns``        the five single-star columns
``acceleration_columns``       the two sky-plane acceleration columns
``ti_design_matrix``           the nine Thiele-Innes columns
``rv_design_matrix``           the three linear RV columns
``reduced_rv_design``          the reduced joint RV design
``reduced_astro_design``       the reduced joint astrometric design
============================== ================================

Three of the six (``acceleration_columns``, ``ti_design_matrix``,
``rv_design_matrix``) are PUBLIC names on orblet's front door; that the
front-door name is the same OBJECT as the one defined here is pinned by
the front-door identity test.

**Why here.** Every composed design draws its columns from this one
module, so a single-star block is the same object everywhere it appears.
The module sits BELOW the channel solvers: the solvers import from it,
one-way and at module scope, and it imports nothing from any solver. That
ordering is load-bearing. If the column functions lived in the channel
modules instead, each builder would have to import back into the module
that calls it, and the package would stay importable only through lazy
imports — which a natural-looking tidy-up (moving one to module scope)
would silently break.

The rule, then: the column arithmetic has one home, and no channel can
reach it by importing another channel's solver.
Each function's docstring states its column order and units; the
reference manual's §2.2 (RV) and §3.3 (Thiele-Innes) state the same
order, and the design baseline test pins the values.

**Layering** (strictly one-way, top depends on bottom)::

    orblet.constants
      orblet.kepler                 (Kepler solver)
        orblet.model                (forward models, _true_anomaly)
          orblet.design             (THIS package: columns, then builders)
            orblet.solve, orblet.search
                                    (the linear solves and the frequency scan)

Nothing below a layer imports from above it; that ordering, not lazy
imports, is what keeps the package importable, and a module-scope import
in the allowed direction is always safe.

The values are pinned byte for byte by
``tests/test_design_columns_baseline.py``.

Naming: the column functions carry no leading underscore because they are
the shared column source for the whole package, not private to one
channel.  Where an underscored spelling (``_astrometric_columns``,
``_reduced_rv_design_engine``) exists at a channel site, it is an alias of
the function defined here, so both spellings are the same object.


Conventions pinned here (documented, do NOT reorder)
---------------------------------------------------
- single-star block, 5 columns: ``[ra_offset, dec_offset, pmra, pmdec, plx]``
- RV design, 3 columns: ``[γ, K cos ω, K sin ω]``
- reduced RV design, 2 columns: ``[γ, K]`` at FIXED ω
- proper motion is linear in ``Δt_yr = (t − epoch_ref_mjd) /
  DAYS_PER_KEPLER_YEAR``; ``epoch_ref_mjd`` and ``t_mjd`` MUST share one
  time scale (Gaia DR4: TCB)
- periastron time is ``tp = τ · P_days + epoch_ref_mjd`` on both RV paths,
  exactly as ``rv_model`` does it
"""

from __future__ import annotations

import math

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import (
    _kepler_xy_orbit,
    _tp_from_disk_angle,
)
from orblet.model import _true_anomaly

# Column counts, derived from the documented column orders above.  Kept as
# literals here (rather than ``len`` of a name list) because this module is
# the definition site: the channel modules' own ``_N_*`` constants remain
# theirs, and a mismatch between them and these is caught by
# ``tests/test_design_columns_baseline.py::test_column_counts_are_the_pinned_conventions``.
_N_ASTROMETRIC = 5
_N_RV_BETA = 3
_N_TI_BETA = 9


def astrometric_columns(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    *,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Single-star (5-parameter) along-scan design block, shape ``(n, 5)``.

    These are the partial derivatives of the engine along-scan model w.r.t.
    the five single-star amplitudes, in the fixed order
    ``[ra_offset, dec_offset, pmra, pmdec, plx]`` — the block that is
    INDEPENDENT of the orbit shape ``(frequency, e, τ)``:

        ∂model/∂ra_offset  = sinψ
        ∂model/∂dec_offset = cosψ
        ∂model/∂pmra       = sinψ · Δt_yr
        ∂model/∂pmdec      = cosψ · Δt_yr
        ∂model/∂plx        = parallax_factor_al

    with ``Δt_yr = (t_mjd − epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR``.  This is
    exactly columns 4-8 of the 9-column Thiele-Innes design (which calls
    this helper to fill them) and the body of the public
    :func:`orblet.astrometric_5param_design_matrix`.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation epochs (MJD), shape ``(n,)``.  Same time scale as
        ``epoch_ref_mjd`` (Gaia DR4: TCB).
    psi : np.ndarray
        Along-scan angle ``ψ`` (radians, counterclockwise from north), one
        per epoch.
    parallax_factor_al : np.ndarray
        Dimensionless along-scan parallax factor per epoch (the ``plx``
        column).
    epoch_ref_mjd : float
        Proper-motion / parallax zero-point epoch (MJD), REQUIRED.

    Returns
    -------
    np.ndarray
        ``(n, 5)`` float64 block in the fixed single-star column order.
    """
    t = np.asarray(t_mjd, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    pf = np.asarray(parallax_factor_al, dtype=float)

    sin_psi = np.sin(psi_arr)
    cos_psi = np.cos(psi_arr)
    # Proper-motion uses the SAME named constant + epoch_ref as the engine.
    dt_yr = (t - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR

    n = t.shape[0]
    block = np.empty((n, _N_ASTROMETRIC), dtype=float)
    block[:, 0] = sin_psi           # ra_offset
    block[:, 1] = cos_psi           # dec_offset
    block[:, 2] = sin_psi * dt_yr   # pmra
    block[:, 3] = cos_psi * dt_yr   # pmdec
    block[:, 4] = pf                # plx
    return block


def rv_design_matrix(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    tau: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Design matrix ``X`` (shape ``(n, 3)``, float64) for the linear SB1
    RV solve at a FIXED non-linear shape ``(P, e, τ)``.

    The three columns are the partial derivatives of the engine RV model
    w.r.t. the linear amplitudes ``β = (γ, C, S) = (γ, K cosω, K sinω)``:

        ∂v/∂γ = 1        ∂v/∂C = cosν + e        ∂v/∂S = −sinν,

    i.e. ``X = column_stack([1, cosν + e, −sinν])``.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation epochs (MJD), shape ``(n,)``.  Same time scale as
        ``epoch_ref_mjd``.
    period_yr : float
        Period in Keplerian years (matches ``rv_model``).
    ecc : float
        Eccentricity (``0 ≤ e < 1``).
    tau : float
        Periastron-phase fraction (``τ ∈ [0, 1)``); the periastron time is
        ``tp = τ · P_days + epoch_ref_mjd``, EXACTLY as ``rv_model``.
    epoch_ref_mjd : float
        Reference epoch (MJD), REQUIRED (no default).  Periastron-time
        zero only.

    Returns
    -------
    np.ndarray
        ``X`` of shape ``(n, 3)``, float64.
    """
    t = np.asarray(t_mjd, dtype=float)
    period_yr = float(period_yr)
    ecc = float(ecc)

    # tp via the SAME τ → tp map rv_model uses.
    P_days = period_yr * DAYS_PER_KEPLER_YEAR
    tp_mjd = float(tau) * P_days + float(epoch_ref_mjd)

    # ν via the RV-LOCAL atom (single source of truth with rv_model).
    nu = _true_anomaly(t, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd)

    n = t.shape[0]
    X = np.empty((n, _N_RV_BETA), dtype=float)
    X[:, 0] = 1.0               # γ
    X[:, 1] = np.cos(nu) + ecc  # C = K cosω
    X[:, 2] = -np.sin(nu)       # S = K sinω
    return X


def reduced_rv_design(
    t_mjd: np.ndarray,
    *,
    P_yr: float,
    e: float,
    omega: float,
    tau: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Two-amplitude REDUCED RV design ``(n, 2)`` at a FIXED ``(P, e, omega, tau)``.

    Columns are the partials of ``v = gamma + K[cos(nu+omega) + e cos omega]``
    w.r.t. the two reduced amplitudes ``beta = (gamma, K)`` at fixed omega::

        col 0 = 1
        col 1 = cos(nu + omega) + e cos(omega),

    with ``nu`` from the ENGINE atom ``rv.forward._true_anomaly`` and
    ``tp = tau * P_days + epoch_ref_mjd`` (the ``P_yr`` argument is in
    Keplerian years).
    """
    t = np.asarray(t_mjd, dtype=float)
    P_days = float(P_yr) * DAYS_PER_KEPLER_YEAR
    tp = float(tau) * P_days + float(epoch_ref_mjd)
    nu = _true_anomaly(t, period_yr=float(P_yr), ecc=float(e), tp_mjd=tp)
    col_gamma = np.ones_like(t)
    col_K = np.cos(nu + float(omega)) + float(e) * math.cos(float(omega))
    return np.column_stack([col_gamma, col_K])


def reduced_astro_design(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    *,
    P_yr: float,
    e: float,
    omega: float,
    Omega: float,
    cos_i: float,
    tau: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Two-amplitude REDUCED astrometric design ``(n, 6)`` at fixed geometry.

    Column order matches the single-star block order of
    :func:`astrometric_columns`::

        col 0 = d(model)/d(a_phot)   (the orbit column)
        col 1 = sin(psi)             (ra_offset)
        col 2 = cos(psi)             (dec_offset)
        col 3 = sin(psi) * dt_yr     (pmra)
        col 4 = cos(psi) * dt_yr     (pmdec)
        col 5 = parallax_factor      (plx)

    The orbit column is built from the ENGINE atom ``_ti_constants_unit``
    times the ENGINE in-plane point ``(x, y)`` from ``_kepler_xy_orbit``::

        col0 = (B_u x + G_u y) sin(psi) + (A_u x + F_u y) cos(psi),

    with ``(A_u, B_u, F_u, G_u) = _ti_constants_unit(omega, Omega, cos_i)``
    so that ``a_phot * col0`` reproduces the full photocentre along-scan
    orbit.  Cols 1-5 are :func:`astrometric_columns` VERBATIM (one source of
    truth for the single-star convention), called MODULE-LOCALLY rather
    than through an import reaching back into the astrometric channel.

    """
    from orblet.model import (
        _kepler_xy_orbit,
        _ti_constants_unit,
    )

    t = np.asarray(t_mjd, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    pf = np.asarray(parallax_factor_al, dtype=float)

    P_days = float(P_yr) * DAYS_PER_KEPLER_YEAR
    tp = float(tau) * P_days + float(epoch_ref_mjd)
    x, y = _kepler_xy_orbit(
        t, period_yr=float(P_yr), ecc=float(e), tp_mjd=tp,
    )

    A_u, B_u, F_u, G_u = _ti_constants_unit(
        float(omega), float(Omega), float(cos_i),
    )
    d_ra_unit = B_u * x + G_u * y
    d_dec_unit = A_u * x + F_u * y

    sin_psi, cos_psi = np.sin(psi_arr), np.cos(psi_arr)
    col_orbit = d_ra_unit * sin_psi + d_dec_unit * cos_psi

    # Cols 1-5 VERBATIM from the shared single-star block helper.
    block = astrometric_columns(
        t, psi_arr, pf, epoch_ref_mjd=float(epoch_ref_mjd),
    )
    return np.column_stack(
        [col_orbit, block[:, 0], block[:, 1], block[:, 2], block[:, 3],
         block[:, 4]]
    )
def acceleration_columns(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    *,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Along-scan acceleration design block, shape ``(n, 2)``.

    The partial derivatives of the along-scan model w.r.t. a sky-plane
    acceleration ``(a_ra*, a_dec)`` under the ½-IN-COLUMN convention (the
    amplitude IS the physical acceleration in mas/yr²)::

        ∂model/∂a_ra*  = sinψ · Δt_yr²/2
        ∂model/∂a_dec  = cosψ · Δt_yr²/2

    with ``Δt_yr = (t_mjd − epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR`` — the
    SAME time origin and year definition as the proper-motion columns of
    :func:`astrometric_columns`.  Sharing one ``epoch_ref_mjd`` between the
    PM and acceleration blocks is LOAD-BEARING: under a common origin shift
    t₀ → t₀+Δ the acceleration is invariant but the recovered PM absorbs
    +a·Δ and the offset absorbs μ·Δ + a·Δ²/2 (an affine reparametrization —
    the fitted model is unchanged, only the parameter meaning moves).

    Shared primitive: used by the relative-motion engine
    and intended for the future t² single-star engine.  Same sky-axis
    convention as the single-star block (sinψ ↔ Δα*, cosψ ↔ Δδ).

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation epochs (MJD), shape ``(n,)``; same time scale as
        ``epoch_ref_mjd`` (Gaia DR4: TCB).
    psi : np.ndarray
        Along-scan angle ``ψ`` (radians, counterclockwise from north).
    epoch_ref_mjd : float
        The shared PM/acceleration time origin (MJD), REQUIRED.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` float64 block ``[sinψ·Δt²/2, cosψ·Δt²/2]`` (Δt in
        Keplerian years).
    """
    t = np.asarray(t_mjd, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    dt_yr = (t - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR
    half_dt2 = 0.5 * dt_yr * dt_yr

    n = t.shape[0]
    block = np.empty((n, 2), dtype=float)
    block[:, 0] = np.sin(psi_arr) * half_dt2   # a_ra*
    block[:, 1] = np.cos(psi_arr) * half_dt2   # a_dec
    return block


# Private alias for the engine-internal callers; the public spelling is
# the documented one.
def ti_design_matrix(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    *,
    f_per_day: float,
    ecc: float,
    tau: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Design matrix ``X`` (shape ``(n, 9)``, float64) for the linear TI
    pre-search at a FIXED non-linear shape.

    The columns are the partial derivatives of the engine along-scan
    model w.r.t. the nine linear amplitudes, in the fixed order
    ``[A, B, F, G, ra_offset, dec_offset, pmra, pmdec, plx]`` (see the
    module docstring for the explicit expressions and conventions).

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation epochs (MJD), shape ``(n,)``.  Same time scale as
        ``epoch_ref_mjd`` (Gaia DR4: TCB).
    psi : np.ndarray
        Along-scan angle (radians, counterclockwise from north), one per
        epoch.
    parallax_factor_al : np.ndarray
        Dimensionless parallax factor per epoch (the ``plx`` column).
    f_per_day : float
        Orbital frequency in cycles/day; ``P_days = 1 / f_per_day``.
    ecc : float
        Eccentricity (``0 ≤ e < 1``).
    tau : float
        Periastron-phase fraction on the unit disk (``τ ∈ [0, 1)``); the
        engine atoms reconstruct ``tp`` from the disk point
        ``(cos 2πτ, sin 2πτ)``.
    epoch_ref_mjd : float
        Reference epoch (MJD), REQUIRED.  Sets BOTH the periastron-time
        zero and the proper-motion zero-point (see module docstring).

    Returns
    -------
    np.ndarray
        ``X`` of shape ``(n, 9)``, float64.
    """
    t = np.asarray(t_mjd, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    pf = np.asarray(parallax_factor_al, dtype=float)

    period_days = 1.0 / f_per_day
    period_yr = period_days / DAYS_PER_KEPLER_YEAR

    # τ-on-the-disk → tp via the ENGINE atom (one source of truth).  The
    # disk point at angle 2πτ has angle == 2πτ, so the atom recovers τ.
    disk_x = np.cos(2.0 * np.pi * tau)
    disk_y = np.sin(2.0 * np.pi * tau)
    tp_mjd = _tp_from_disk_angle(
        disk_x, disk_y, period_days=period_days, epoch_ref_mjd=epoch_ref_mjd,
    )

    # Orbit shape from the ENGINE atom (normalised by the semi-major axis).
    x_orb, y_orb = _kepler_xy_orbit(
        t, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd,
    )

    sin_psi = np.sin(psi_arr)
    cos_psi = np.cos(psi_arr)

    n = t.shape[0]
    X = np.empty((n, _N_TI_BETA), dtype=float)
    X[:, 0] = x_orb * cos_psi   # A
    X[:, 1] = x_orb * sin_psi   # B
    X[:, 2] = y_orb * cos_psi   # F
    X[:, 3] = y_orb * sin_psi   # G
    # Columns 4-8 are the single-star block (independent of orbit shape);
    # fill them from the shared helper so the 5-parameter fit and the TI
    # fit use ONE source of truth for the single-star convention.
    X[:, 4:9] = astrometric_columns(
        t, psi_arr, pf, epoch_ref_mjd=epoch_ref_mjd,
    )
    return X

