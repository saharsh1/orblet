"""Where the Earth is, turned into how far the star appears to move.

Parallax is the only reason a single star's apparent position wanders on a
one-year cycle, and it is the largest signal in epoch astrometry. To fit an
orbit you must first be able to predict that wander, which means knowing the
observer's position relative to the solar-system barycentre and projecting it
onto the sky at the target.

:func:`per_direction_parallax_factors` does exactly that, returning the two
on-sky components

    Δα*(t) = π · F_α*(t),    Δδ(t) = π · F_δ(t),

with (Lindegren et al. 2012; Gaia DR3 §3.3)

    F_α*(t) = -ê_α* · r_O(t),    F_δ(t) = -ê_δ · r_O(t),

where ``r_O(t)`` is the observer position relative to the solar-system
BARYCENTRE in AU, and

    ê_α* = (-sin α,  cos α,            0),
    ê_δ  = (-sin δ cos α, -sin δ sin α, cos δ).

The minus sign converts observer position into apparent stellar shift: the
star appears displaced opposite to the observer's motion.

NOTE: ê_α* is a UNIT VECTOR; it does NOT carry an explicit cos δ factor. The
cos δ scaling is folded into the basis vectors' unitarity, not into F_α* as a
multiplicative scalar, so F_α* is in the same on-sky-angle units as F_δ.

:func:`along_scan_parallax_factor` is the projection onto a scan angle ψ,

    parallax_factor_al = F_α* · sin ψ + F_δ · cos ψ,

the single number per transit that the along-scan measurement equation uses.
The demo cadence (:class:`orblet.simulate.cadence.DemoParallaxConsistentCadence`)
is built on it.

Observer: L2 by default, ``observer="geocentre"`` available
-----------------------------------------------------------
The default observer sits at the Sun-Earth L2 point, where Gaia is: about
0.01 AU beyond the geocentre, anti-Sunward. MEASURED on the public Gaia BH3
pre-release bundle (558 transits, 5.1 yr): projected on the scan angle, the L2
factor agrees with Gaia's own ``parallax_factor_al`` to slope 1.00007 and
0.18 % rms residual (the spacecraft's orbit about L2 is not modelled, plus
ephemeris vintage); the geocentric factor is 1 % low (slope 0.990). Pass
``observer="geocentre"`` for an Earth-bound observer.

Ephemeris: astropy's built-in one by default
--------------------------------------------
The Earth's position comes from :data:`orblet.constants.SOLAR_SYSTEM_EPHEMERIS_PIN`
(``"builtin"``, astropy's analytic ERFA ephemeris): no download, no extra
package, within ~5 km of JPL DE432s. To use a JPL kernel instead:

    pip install "orblet[ephemeris]"          # adds jplephem
    python -c "from orblet.parallax import fetch_ephemeris; fetch_ephemeris('de432s')"

and then pass ``ephemeris="de432s"``. The fetch is a one-time ~10 MB download
into astropy's cache; afterwards it runs offline.

None of this replaces a mission's own bundled parallax-factor column when
fitting REAL epoch astrometry (geocentre: 1 % parallax bias, ~3 % in
``fm_ast``; L2: ~0.2 %). Use the bundled column. These helpers are for
simulation, for the 2-D ``(ra_offset, dec_offset)`` decomposition of a fitted
parallax, and for pedagogy.

**Lazy imports required**: astropy must NOT be imported at module scope.
Every astropy import here lives inside a function body and is marked
``# DO NOT LIFT``.

Time-scale contract: TCB-MJD epochs only. The anchor is
:data:`orblet.constants.MJD_J2010_TCB` and the convention is that Gaia
DR3/DR4 epochs are TCB-MJD already. Passing TDB or UTC silently produces
wrong factors at the few-mas level, because the Earth-Sun position drifts a
few hundred km per second across the TCB-TDB rate offset.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from orblet.constants import SOLAR_SYSTEM_EPHEMERIS_PIN


__all__ = [
    "per_direction_parallax_factors",
    "along_scan_parallax_factor",
    "fetch_ephemeris",
    "OBSERVER_GEOCENTRE",
    "OBSERVER_L2",
    "L2_OFFSET_AU",
]

# Observer placement for :func:`per_direction_parallax_factors`.
OBSERVER_GEOCENTRE = "geocentre"
OBSERVER_L2 = "l2"

# Sun–Earth L2 distance beyond the geocentre, in AU, anti-Sunward.  The
# restricted-three-body (Hill) estimate ``a_E · (μ/3)^{1/3}`` with
# μ = M_Earth/M_Sun = 3.003e-6 gives 0.010003 AU ≈ 1.496e6 km; the
# accepted mean value is 1.5e6 km.  Gaia's Lissajous orbit about L2
# (~1e5–3e5 km, i.e. ~1e-3 AU) is NOT modelled — that is the 0.18 % rms
# residual measured against Gaia's factor on BH3.
L2_OFFSET_AU = 0.010003

_BAD_OBSERVER_MSG = (
    "observer must be 'l2' (default) or 'geocentre'; see "
    "per_direction_parallax_factors."
)


def _ephemeris_failure_msg(ephemeris: str) -> str:
    # Value-free: names the ephemeris the caller chose, nothing else.
    if ephemeris == "builtin":
        return "Solar-system ephemeris lookup failed with astropy's built-in ephemeris."
    return (
        f"Solar-system ephemeris lookup failed for ephemeris={ephemeris!r}. "
        "A JPL kernel needs the jplephem package "
        "(pip install \"orblet[ephemeris]\") and a one-time download: "
        f"orblet.parallax.fetch_ephemeris({ephemeris!r}). "
        "Or omit `ephemeris` to use astropy's built-in one."
    )


# ── Helper: per-direction parallax factors ───────────────────────────


def per_direction_parallax_factors(
    t_mjd_tcb: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    *,
    observer: str = OBSERVER_L2,
    ephemeris: str = SOLAR_SYSTEM_EPHEMERIS_PIN,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute ``(F_α*, F_δ)`` at TCB-MJD epochs for ``(α, δ)``.

    Parameters
    ----------
    t_mjd_tcb : np.ndarray
        Epoch array in **TCB-MJD** (NOT TDB, NOT UTC).  Passing epochs in
        another time scale silently produces wrong parallax factors at the
        few-mas level (the Earth-Sun position drifts a few-hundred km per
        second under the TCB↔TDB rate offset).
    ra_deg, dec_deg : float
        Target right ascension and declination (degrees, ICRS).
    observer : {"l2", "geocentre"}, default "l2"
        Where the observer is placed.  ``"l2"`` (the default) moves the
        observer along the Sun→Earth line to the Sun–Earth L2 point,
        ``L2_OFFSET_AU`` beyond the geocentre, which is where Gaia sits (its
        ~1e5–3e5 km Lissajous orbit about L2 is NOT modelled); measured
        agreement with Gaia's factor: slope 1.00007, 0.18 % rms residual.
        ``"geocentre"`` uses the barycentric geocentre and under-predicts
        Gaia's factor by ~1 %.  Any other value raises ``ValueError``.
    ephemeris : str, default :data:`orblet.constants.SOLAR_SYSTEM_EPHEMERIS_PIN`
        Any name :class:`astropy.coordinates.solar_system_ephemeris` accepts.
        The default, ``"builtin"``, needs no download and no extra package.
        A JPL kernel such as ``"de432s"`` needs ``pip install
        "orblet[ephemeris]"`` and a one-time :func:`fetch_ephemeris`.

    Returns
    -------
    (F_alpha_star, F_delta) : tuple of np.ndarray
        Two arrays of shape ``(len(t_mjd_tcb),)`` and dtype float64,
        carrying the per-direction parallax factors at each epoch.
        Dimensionless; magnitudes are bounded by ~1 AU (Earth-Sun
        distance in AU).

    Raises
    ------
    ValueError
        For an unknown ``observer``.
    RuntimeError
        When the ephemeris lookup fails (for a JPL kernel: jplephem missing
        or the kernel not fetched); the message says how to fix it.

    Sign convention (Lindegren et al. 2012; Gaia DR3 §3.3)
    -----------------------------------------------------
    With ``ê_α* = (-sin α, cos α, 0)`` and
    ``ê_δ = (-sin δ cos α, -sin δ sin α, cos δ)``,

        F_α*(t) = -ê_α* · r_O(t),    F_δ(t) = -ê_δ · r_O(t),

    where ``r_O(t)`` is the BARYCENTRIC observer position (AU; relative to
    the solar-system barycentre, NOT to the Sun).  The apparent stellar
    displacement (mas) is ``π_mas · F_α*`` and ``π_mas · F_δ`` for the two
    on-sky components.
    """
    if observer not in (OBSERVER_GEOCENTRE, OBSERVER_L2):
        raise ValueError(_BAD_OBSERVER_MSG)
    t_arr = np.asarray(t_mjd_tcb, dtype=float)

    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    e_alpha = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=float)
    e_delta = np.array(
        [
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        ],
        dtype=float,
    )

    # DO NOT LIFT — astropy stays out of the module top.  See module
    # docstring lazy-import contract.
    try:
        from astropy.coordinates import (  # DO NOT LIFT
            get_body_barycentric,
            solar_system_ephemeris,
        )
        from astropy.time import Time  # DO NOT LIFT

        with solar_system_ephemeris.set(ephemeris):
            times = Time(t_arr, format="mjd", scale="tcb")
            pos = get_body_barycentric("earth", times)
            if observer == OBSERVER_L2:
                # The Sun is needed only to orient the Sun→Earth line
                # along which L2 lies; same ephemeris.
                sun = get_body_barycentric("sun", times)
        bx = np.asarray(pos.x.to_value("AU"), dtype=float)
        by = np.asarray(pos.y.to_value("AU"), dtype=float)
        bz = np.asarray(pos.z.to_value("AU"), dtype=float)
        if observer == OBSERVER_L2:
            sx = np.asarray(sun.x.to_value("AU"), dtype=float)
            sy = np.asarray(sun.y.to_value("AU"), dtype=float)
            sz = np.asarray(sun.z.to_value("AU"), dtype=float)
    except Exception:
        # Value-free, severance idiom (``from None``): the underlying
        # exception may carry filesystem paths or numeric values.
        raise RuntimeError(_ephemeris_failure_msg(str(ephemeris))) from None

    # Stack to (N, 3) for the dot product.  Axis-meaning comment:
    # b[:, 0:3] = (b_x, b_y, b_z) in AU at each epoch (axis 0 = epoch,
    # axis 1 = ICRS-cartesian).
    b = np.stack([bx, by, bz], axis=-1)  # shape (n_epoch, 3)
    if observer == OBSERVER_L2:
        # Observer = geocentre + L2_OFFSET_AU along the heliocentric
        # Earth direction (Sun→Earth unit vector, anti-Sunward).  The
        # direction is taken from the barycentric difference so the
        # frame stays barycentric; only the observer moves.
        helio = b - np.stack([sx, sy, sz], axis=-1)  # (n_epoch, 3), AU
        helio_hat = helio / np.linalg.norm(helio, axis=1, keepdims=True)
        b = b + L2_OFFSET_AU * helio_hat

    F_alpha_star = -(b @ e_alpha)
    F_delta = -(b @ e_delta)
    return (
        np.asarray(F_alpha_star, dtype=float),
        np.asarray(F_delta, dtype=float),
    )


def along_scan_parallax_factor(
    t_mjd_tcb: np.ndarray,
    psi_rad: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    *,
    observer: str = OBSERVER_L2,
    ephemeris: str = SOLAR_SYSTEM_EPHEMERIS_PIN,
) -> np.ndarray:
    """Along-scan parallax factor at TCB-MJD epochs and scan angles.

    ``parallax_factor_al = F_α* · sin ψ + F_δ · cos ψ``, with ``(F_α*, F_δ)``
    from :func:`per_direction_parallax_factors` (same ``observer`` and
    ``ephemeris``, same defaults: Gaia at L2, astropy's built-in ephemeris).
    The ψ assignment (``sin ψ ↔ α*``, ``cos ψ ↔ δ``) is the one the along-scan
    measurement equation uses.

    Parameters
    ----------
    t_mjd_tcb : np.ndarray
        Epochs in TCB-MJD.
    psi_rad : np.ndarray
        Scan position angles ψ (radians), same shape as ``t_mjd_tcb``.
    ra_deg, dec_deg : float
        Target position (degrees, ICRS).
    observer, ephemeris : str
        As in :func:`per_direction_parallax_factors`.

    Returns
    -------
    np.ndarray
        Dimensionless along-scan factor, one per epoch.
    """
    psi = np.asarray(psi_rad, dtype=float)
    f_alpha, f_delta = per_direction_parallax_factors(
        t_mjd_tcb, ra_deg, dec_deg, observer=observer, ephemeris=ephemeris,
    )
    return f_alpha * np.sin(psi) + f_delta * np.cos(psi)


def fetch_ephemeris(name: str = "de432s") -> None:
    """Download a JPL ephemeris kernel into astropy's cache, once.

    Afterwards ``ephemeris=name`` works offline in
    :func:`per_direction_parallax_factors` and
    :func:`along_scan_parallax_factor`.  Needs the ``jplephem`` package
    (``pip install "orblet[ephemeris]"``) and, this once, the network
    (DE432s is ~10 MB).  Not needed for the default built-in ephemeris.

    Parameters
    ----------
    name : str, default "de432s"
        A JPL kernel name astropy recognises (e.g. ``"de432s"``, ``"de440"``).

    Raises
    ------
    ImportError
        If jplephem is not installed.
    """
    try:
        import jplephem  # noqa: F401  DO NOT LIFT — optional extra
    except ImportError:
        raise ImportError(
            "A JPL ephemeris needs the jplephem package: "
            "pip install \"orblet[ephemeris]\"."
        ) from None

    from astropy.coordinates import (  # DO NOT LIFT
        get_body_barycentric,
        solar_system_ephemeris,
    )
    from astropy.time import Time  # DO NOT LIFT

    # One lookup triggers astropy's download-and-cache of the kernel.
    with solar_system_ephemeris.set(name):
        get_body_barycentric("earth", Time(55197.0, format="mjd", scale="tcb"))
