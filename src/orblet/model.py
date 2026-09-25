"""The forward models: orbital elements and epochs in, observables out.

This is the layer everything else is built on. Give it a set of elements and
the times you observed, and it tells you what you should have measured — a
radial velocity curve, a photocentre track on the sky, an along-scan
abscissa. Nothing here fits anything: no optimiser, no sampler, no data. Pure
functions of their arguments.

Three channels, one file
------------------------
Radial velocity, astrometry, and the joint case that couples them share the
same Kepler solution and the same conventions, so they live together. Reading
the astrometric photocentre model directly under the RV model is the point:
the two differ in what they project the orbit onto, not in the orbit.

    radial velocity   rv_curve, rv_model, semi_amplitude_kms, fm_from_K
    astrometry        campbell_xy, thiele_innes_xy, kepler_xy_orbit,
                      along_scan_model, along_scan_from_theta,
                      tp_from_disk_angle
    joint             rv_model_with_inclination, joint_theta_view

Conventions, stated once because every function below assumes them
-----------------------------------------------------------------
- **Radial velocity sign**: positive means receding from the observer — the
  standard spectroscopic convention.
- **ω** is the argument of periastron in the **primary's** frame; the forward
  models use it directly (textbook binary-star convention).
- **Ω** is the longitude of the ascending node, measured from North toward
  East.
- **Inclination** *i* runs [0, π]; the isotropic prior is uniform in cos *i*.
- **Period** is in Julian years (``period_yr``, 365.25 d/yr) wherever the
  argument is named so; days where named ``period_days``.
- **Angles** are radians, **times** are TCB-MJD, **angular sizes** are mas.
- **Thiele-Innes** (A, B, F, G) pair as Δα* ← B·x + G·y and Δδ ← A·x + F·y.

Import cost
-----------
Module scope imports ``math``, ``numpy``, and orblet's own constants and
Kepler solver — nothing else. No scipy, no emcee, no astropy. That is a
CONTRACT, not an accident: a user computing a model curve should not pay for
a sampler. Enforced by ``tests/test_public_composable_api.py``.

All forward models live in this one module, for every channel.  Their
values are pinned byte for byte by
``tests/test_forward_model_baseline.py`` — 17 cases, ``np.array_equal``.
"""

from __future__ import annotations

import math

import numpy as np

from orblet.constants import AU_M, DAYS_PER_KEPLER_YEAR, GM_SUN_SI
from orblet.kepler import solve_kepler


# ══════════════════════════════════════════════════════════════════════
# RADIAL VELOCITY
# ══════════════════════════════════════════════════════════════════════


def fm_from_K(K_kms, period_yr, ecc):
    """SB1 spectroscopic mass function ``f(m)`` from the semi-amplitude.

    Single source of truth for the mass function::

        f(m) = K³ · P · (1 − e²)^{3/2} / (2π · GM☉)   [M☉]

    Both export sites — the non-linear RV packaging
    (``...rv.chains.vectors_to_chains``) and the linearised RV packaging
    (``...rv.linear_sampler.fit_rv_orbit_linear``) — route through this
    function instead of inlining the arithmetic, so the formula has a
    single home and cannot drift between them.

    Mass unit
    ---------
    The denominator is ``2π · GM_SUN_SI`` — the GM implied by the project's
    AU / Keplerian-year / M☉ unit system, i.e. the same solar mass the
    astrometric ``fm_ast = a_phot³ / P²`` uses.  Re-deriving it as
    ``2π · G_SI × MSUN_KG`` (2.19e-4 larger) would put this mass
    function on a slightly different solar mass from the astrometric
    one — and the deficit ``D`` compares exactly those two channels.

    Parameters
    ----------
    K_kms : float or np.ndarray
        Primary RV semi-amplitude ``K`` in km/s.
    period_yr : float or np.ndarray
        Orbital period in Keplerian years.
    ecc : float or np.ndarray
        Eccentricity, ``0 ≤ ecc < 1``.

    Returns
    -------
    fm_msun : float or np.ndarray
        Spectroscopic mass function in solar masses (M☉).
    """
    # K km/s → m/s; P_sec = P_yr · DAYS_PER_KEPLER_YEAR · 86400.
    K_ms = K_kms * 1000.0
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    P_sec = period_yr * sec_per_yr
    # Divide by GM_SUN_SI directly, NOT by G_SI * MSUN_KG.  With a typed
    # solar mass those two differ by ~2e-4, which would put this
    # spectroscopic mass function on a different solar mass from the
    # astrometric `fm_ast = a_phot³ / P²` — and the deficit D compares
    # exactly those two.  GM is also the measured quantity; G alone is
    # known only to ~2e-5.
    return (K_ms ** 3) * P_sec * (1.0 - ecc * ecc) ** 1.5 / (
        2.0 * np.pi * GM_SUN_SI
    )


def semi_amplitude_kms(
    *,
    mass_msun: float,
    period_yr: float,
    ecc: float,
    M_total_msun: float,
) -> float:
    """Return the primary RV semi-amplitude ``K`` (km/s) for sin(i)=1.

    Units and assumptions: ``mass_msun`` is the COMPANION mass and
    ``M_total_msun`` the total mass, both in solar masses; ``period_yr``
    in Keplerian years (``DAYS_PER_KEPLER_YEAR`` days); ``ecc``
    dimensionless.  Edge-on (``sin i = 1``); multiply by ``sin i`` for
    an inclined orbit.  Kepler III with the TOTAL mass sets the
    relative orbit; the primary's share is ``mass_msun / M_total_msun``.

    Mirrors :func:`orblet.elements._stellar_K_from_elements`
    but takes companion mass in **solar masses** (no MJup detour),
    matching the native-units design of the Python engine.
    """
    # Kepler III with TOTAL mass: a [AU] = (P^2 [yr] * M_total [M☉])^{1/3}.
    a_au = (period_yr ** 2 * M_total_msun) ** (1.0 / 3.0)
    # Companion velocity semi-amplitude in AU/yr.
    J = 2.0 * math.pi * a_au / period_yr / math.sqrt(1.0 - ecc * ecc)
    # AU/yr → m/s.
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    K_comp_ms = J * AU_M / sec_per_yr
    # Primary K = (m_comp / M_total) × K_companion.
    K_star_ms = (mass_msun / M_total_msun) * K_comp_ms
    return K_star_ms / 1000.0


# Private alias for the engine-internal callers; the public spelling is
# the documented one.
_semi_amplitude_kms = semi_amplitude_kms


def _true_anomaly(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    tp_mjd: float,
) -> np.ndarray:
    """True anomaly ``ν`` (radians) at ``t_mjd`` for a Keplerian orbit.

    RV-LOCAL Kepler→true-anomaly atom: the single source of truth shared
    by :func:`rv_model` and the linearised RV marginal core
    (:mod:`orblet.solve.rv`).  The mean-anomaly wrap and half-angle
    ``atan2`` form live here once, so :func:`rv_model` and the linear
    core cannot drift apart.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation times in MJD.
    period_yr : float
        Period in Keplerian years.
    ecc : float
        Eccentricity, ``0 ≤ ecc < 1``.
    tp_mjd : float
        Time of periastron, MJD (``tp = τ × P_days + epoch_ref_mjd``).

    Returns
    -------
    nu : np.ndarray
        True anomaly in radians, same shape as ``t_mjd``.
    """
    P_days = period_yr * DAYS_PER_KEPLER_YEAR
    # Mean anomaly wrapped into [0, 2π) for numerical stability of the
    # Newton solver (matches the convention in
    # ``orblet.kepler.campbell_xy``).
    M_anom = (2.0 * np.pi * (np.asarray(t_mjd, dtype=float) - tp_mjd) / P_days) % (
        2.0 * np.pi
    )
    E = solve_kepler(M_anom, ecc)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    return nu


def rv_curve(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    omega_rad: float,
    tau: float,
    K_kms: float,
    offset_kms: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Predicted line-of-sight RV in km/s, parametrised by ``K`` directly.

    K-native Keplerian core: the textbook primary-frame RV form::

        v_r = γ + K [cos(ν + ω) + e cos(ω)]

    ``ν`` comes from the shared :func:`_true_anomaly` atom (Kepler solve
    via :func:`orblet.kepler.solve_kepler`).
    :func:`rv_model` is a thin wrapper that maps
    ``(mass, M) → K`` via :func:`_semi_amplitude_kms` then delegates here.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation times in MJD.
    period_yr : float
        Period in Keplerian years.
    ecc : float
        Eccentricity, ``0 ≤ ecc < 1``.
    omega_rad : float
        Argument of periastron in radians (primary frame).
    tau : float
        Phase fraction in ``[0, 1)``.
    K_kms : float
        Primary RV semi-amplitude ``K`` in km/s.
    offset_kms : float
        RV zero-point offset (systemic velocity ``γ``) in km/s.
    epoch_ref_mjd : float
        Reference MJD for the τ → tp conversion: ``tp = τ × P_days +
        epoch_ref_mjd`` where ``P_days = period_yr × DAYS_PER_KEPLER_YEAR``.

    Returns
    -------
    rv : np.ndarray
        Line-of-sight radial velocity at each ``t_mjd`` (km/s).
    """
    P_yr = float(period_yr)
    e = float(ecc)
    omega = float(omega_rad)
    tau = float(tau)
    K = float(K_kms)
    gamma = float(offset_kms)

    P_days = P_yr * DAYS_PER_KEPLER_YEAR
    tp_mjd = tau * P_days + epoch_ref_mjd

    nu = _true_anomaly(t_mjd, period_yr=P_yr, ecc=e, tp_mjd=tp_mjd)

    return gamma + K * (np.cos(nu + omega) + e * np.cos(omega))


def rv_model(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    omega_rad: float,
    tau: float,
    mass_msun: float,
    M_msun: float,
    offset_kms: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Predicted line-of-sight RV in km/s at ``t_mjd``.

    Thin mass-native wrapper around :func:`rv_curve`: it maps the
    companion/total mass pair ``(mass_msun, M_msun)`` to the semi-amplitude
    ``K`` via :func:`_semi_amplitude_kms` (sin(i)=1) and delegates the
    Keplerian forward to :func:`rv_curve`.  The joint engine
    (``...joint.forward.joint_rv_model``) wraps this.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation times in MJD.
    period_yr : float
        Period in Keplerian years.
    ecc : float
        Eccentricity, ``0 ≤ ecc < 1``.
    omega_rad : float
        Argument of periastron in radians (primary frame).
    tau : float
        Phase fraction in ``[0, 1)``.
    mass_msun : float
        Projected companion mass ``m_2 × sin(i_true)`` in M☉.
    M_msun : float
        Total system mass in M☉.
    offset_kms : float
        RV zero-point offset in km/s.
    epoch_ref_mjd : float
        Reference MJD for the τ → tp conversion: ``tp = τ × P_days +
        epoch_ref_mjd`` where ``P_days = period_yr × DAYS_PER_KEPLER_YEAR``.

    Returns
    -------
    rv : np.ndarray
        Line-of-sight radial velocity at each ``t_mjd`` (km/s).
    """
    K = _semi_amplitude_kms(
        mass_msun=float(mass_msun),
        period_yr=float(period_yr),
        ecc=float(ecc),
        M_total_msun=float(M_msun),
    )
    return rv_curve(
        t_mjd,
        period_yr=period_yr,
        ecc=ecc,
        omega_rad=omega_rad,
        tau=tau,
        K_kms=K,
        offset_kms=offset_kms,
        epoch_ref_mjd=epoch_ref_mjd,
    )

# ══════════════════════════════════════════════════════════════════════
# ASTROMETRY
# ══════════════════════════════════════════════════════════════════════


# ── Shared latent-decode atoms (canonical, single-site) ──────────────
#
# These tiny numpy-only helpers hold the projected-disk-angle → tp and
# the u → inclination arithmetic in ONE place.  Both the chain decoders
# (``...astro.chains.vector_to_theta_campbell`` / ``_ti``) AND the
# public :func:`along_scan_from_theta` helper call them, so the
# arccos/τ→tp arithmetic is never re-inlined a third time.


def _inc_from_u_latent(u_inc: float) -> float:
    """Inclination (rad) from the latent: ``i = arccos(1 − 2·u_inc)``.

    ``u_inc ∈ [0, 1]`` ⇒ ``i ∈ [0, π]`` (the full sphere).
    ``cos i = 1 − 2 u_inc`` is uniform on ``[-1, +1]`` for uniform
    ``u_inc``; the implied density on ``i`` is ``p(i) = (1/2) sin(i)``
    on ``[0, π]`` — isotropic orientation, the standard uninformative
    prior on an orbit's inclination.

    The full sphere, not the half ``i = arccos(1 − u_inc) ∈ [0, π/2]``:
    a half-sphere map would silently pick one side of the i ↔ π − i
    mirror.  See :class:`orblet.priors.CosUniformInclinationPrior` and
    the reference manual, §4.3.
    """
    return math.acos(1.0 - 2.0 * float(u_inc))


def _tau_from_disk_angle(disk_x: float, disk_y: float) -> float:
    """Periastron-phase fraction on the unit disk:
    ``τ = (atan2(disk_y, disk_x) / 2π) mod 1`` ∈ [0, 1).

    Only the ANGLE of (disk_x, disk_y) is identifiable (radius dropped).
    NOTE the argument order: atan2 takes (y, x), so disk_y is first.
    A swapped atan2(disk_x, disk_y) silently rotates τ by 90° and is
    GOLDEN-INVISIBLE (both engines share this atom) — guarded only by
    the directed argument-order test in test_c4h_joint_consolidation.
    """
    return (math.atan2(float(disk_y), float(disk_x)) / (2.0 * math.pi)) % 1.0


def tp_from_disk_angle(
    disk_x: float, disk_y: float, *, period_days: float, epoch_ref_mjd: float,
) -> float:
    """Periastron time (MJD) from a unit-disk (x, y) phase latent.

    ``τ = (atan2(y, x) / 2π) mod 1`` then ``tp = τ · P_days + t_ref``.
    Only the angle of (x, y) is identifiable (the radius is dropped).
    The τ fraction is delegated to :func:`_tau_from_disk_angle` so the
    angle→fraction arithmetic lives in exactly one place.
    """
    tau = _tau_from_disk_angle(disk_x, disk_y)
    return tau * float(period_days) + float(epoch_ref_mjd)


# ── Campbell-elements forward model ──────────────────────────────────


# Private alias for the engine-internal callers; the public spelling is
# the documented one.
_tp_from_disk_angle = tp_from_disk_angle


def kepler_xy_orbit(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    tp_mjd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the in-plane orbit coordinates ``(x_orb, y_orb)``,
    normalised by the semi-major axis (so ``r_norm = (1−e²)/(1+e cos ν)``).
    Helper shared by Campbell and Thiele-Innes forward models.
    """
    P_days = period_yr * DAYS_PER_KEPLER_YEAR
    M_anom = (2.0 * np.pi * (np.asarray(t_mjd, dtype=float) - tp_mjd) / P_days) % (
        2.0 * np.pi
    )
    E = solve_kepler(M_anom, ecc)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    r_norm = (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(nu))
    x_orb = r_norm * np.cos(nu)
    y_orb = r_norm * np.sin(nu)
    return x_orb, y_orb


# Private alias for the engine-internal callers; the public spelling is
# the documented one.
_kepler_xy_orbit = kepler_xy_orbit


def _ti_constants_unit(
    omega: float, Omega: float, cos_i: float,
) -> tuple[float, float, float, float]:
    """Unit-amplitude Thiele-Innes constants ``(A, B, F, G)`` (dimensionless).

    Returns the four Thiele-Innes constants for a photocenter amplitude
    ``a_phot ≡ 1`` — the factored, unit form of the inline ``A, B, F, G``
    block inside :func:`campbell_xy` (which multiplies each of these by the
    amplitude ``a_mas`` — positive by construction in ``campbell_xy``; the
    Tier-2 sampled ``a_phot`` MAY carry sign under the U5 Klein-4 gauge,
    folded non-negative at chain export).  The full amplitudes are recovered as
    ``(A, B, F, G) = a_phot · _ti_constants_unit(ω, Ω, cos i)`` and the sky
    mapping is ``Δα* = B x_orb + G y_orb``, ``Δδ = A x_orb + F y_orb``.

    Units / conventions
    -------------------
    - ``omega`` — the PRIMARY's argument of periastron (rad), textbook
      binary-star / primary-frame convention, used as-is (NO ``ω → ω + π``
      companion-frame flip).
    - ``Omega`` — longitude of the ascending node (rad), measured
      counterclockwise from north (the Gaia convention).
    - ``cos_i`` — the COSINE of the inclination, passed directly (not the
      angle).  The inclination is ``i ∈ [0, π]`` (full sphere), so
      ``cos_i ∈ [−1, 1]``; the caller supplies ``cos i`` already computed.
    - ``a_phot ≡ 1`` — unit photocenter amplitude (the amplitude scaling is
      applied by the caller).

    Reference: Hilditch (2001) *An Introduction to Close Binary Stars*,
    §5.12 (textbook Thiele-Innes constants, primary frame).

    This is the factored unit form of :func:`campbell_xy`'s inline
    ``A, B, F, G`` (kept duplicated deliberately — ``campbell_xy`` is NOT
    refactored to call this atom; a directed equality test guards the two
    copies against drift).

    Independence guard (LOAD-BEARING)
    ---------------------------------
    The A0 oracle's independent forwards
    (:func:`orblet.kepler.campbell_xy` and the test-side textbook
    Thiele-Innes forms) must NEVER be routed through this atom — collapsing
    them would make the A0 oracle compare the engine against itself,
    destroying the "simulator ≠ fitter" cross-check that oracle exists for.
    """
    cos_O, sin_O = math.cos(Omega), math.sin(Omega)
    cos_w, sin_w = math.cos(omega), math.sin(omega)
    # Duplicated verbatim from campbell_xy's inline block (Hilditch §5.12,
    # primary frame); a_phot ≡ 1 here (the caller multiplies by a_mas).
    A = cos_O * cos_w - sin_O * sin_w * cos_i
    B = sin_O * cos_w + cos_O * sin_w * cos_i
    F = -cos_O * sin_w - sin_O * cos_w * cos_i
    G = -sin_O * sin_w + cos_O * cos_w * cos_i
    return A, B, F, G


def campbell_xy(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    omega: float,
    inc: float,
    Omega: float,
    tp_mjd: float,
    m_comp_msun: float,
    M_total_msun: float,
    plx_mas: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Photocenter sky offsets ``(Δα*, Δδ)`` in mas for a Campbell-elements
    orbit.

    Photocenter amplitude (primary frame, positive)
    -----------------------------------------------
    This forward computes the Thiele-Innes constants A, B, F, G DIRECTLY
    in the PRIMARY frame, with a POSITIVE photocenter amplitude
    ``a_phot = (m_comp / M_total) · a · plx`` and primary-frame ``omega``
    used as-is.  This matches
    :func:`orblet.kepler.campbell_xy` (the independent oracle)
    and the Gaia DR3 / textbook Hilditch (2001, §5.12) convention — there
    is NO internal companion-frame ``ω → ω + π`` flip and NO negative
    amplitude sign.

    β = 0 assumed: the amplitude computed here is ``a_1`` (the primary's
    own orbit), which is the photocentre orbit ONLY for a dark companion.
    A luminous companion shrinks the photocentre orbit below ``a_1``, and
    a fit built on this forward then under-estimates ``m_comp / M_total``.
    See the module docstring.

    Returns
    -------
    d_ra : np.ndarray
        Δα* (= Δα · cos δ) in mas, shape ``t_mjd.shape``.
    d_dec : np.ndarray
        Δδ in mas, shape ``t_mjd.shape``.
    """
    # Kepler III with TOTAL mass: a_AU = (P_yr² × M_total_msun)^{1/3}.
    a_au = (period_yr ** 2 * M_total_msun) ** (1.0 / 3.0)
    # Positive primary-frame photocenter amplitude in mas (see docstring;
    # no negative-amplitude / ω+π pair — Wave-1 engine-simplification).
    a_mas = (m_comp_msun / M_total_msun) * a_au * plx_mas

    x_orb, y_orb = _kepler_xy_orbit(
        t_mjd, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd,
    )

    cos_O, sin_O = math.cos(Omega), math.sin(Omega)
    # Primary-frame ω used directly (no ω+π companion-frame flip).
    cos_w, sin_w = math.cos(omega), math.sin(omega)
    cos_i = math.cos(inc)

    # Standard Thiele-Innes constants (Hilditch 2001, §5.12) computed in
    # the PRIMARY frame; d_ra is Δα* (= Δα cos δ); d_dec is Δδ.
    A = a_mas * (cos_O * cos_w - sin_O * sin_w * cos_i)
    B = a_mas * (sin_O * cos_w + cos_O * sin_w * cos_i)
    F = a_mas * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    G = a_mas * (-sin_O * sin_w + cos_O * cos_w * cos_i)

    d_ra = B * x_orb + G * y_orb
    d_dec = A * x_orb + F * y_orb
    return d_ra, d_dec


# ── Thiele-Innes-amplitudes forward model ────────────────────────────


def thiele_innes_xy(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    A_mas: float,
    B_mas: float,
    F_mas: float,
    G_mas: float,
    tp_mjd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Photocenter sky offsets ``(Δα*, Δδ)`` in mas, T-I parametrisation.

    The (A, B, F, G) amplitudes are PHOTOCENTER amplitudes in mas — the
    user has absorbed the ``+m_comp / M_total · plx`` factor into the
    sampled ``(A, B, F, G)``.  This forward model applies neither sign
    flip nor mass-ratio scaling.

    Returns
    -------
    d_ra : np.ndarray
        Δα* in mas.
    d_dec : np.ndarray
        Δδ in mas.
    """
    x_orb, y_orb = _kepler_xy_orbit(
        t_mjd, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd,
    )
    d_ra = B_mas * x_orb + G_mas * y_orb
    d_dec = A_mas * x_orb + F_mas * y_orb
    return d_ra, d_dec


# ── Along-scan projection + 5-parameter astrometry ───────────────────


def along_scan_model(
    *,
    d_ra: np.ndarray,
    d_dec: np.ndarray,
    psi: np.ndarray,
    t_mjd: np.ndarray,
    epoch_ref_mjd: float,
    ra_offset_mas: float,
    dec_offset_mas: float,
    pmra_masyr: float,
    pmdec_masyr: float,
    plx_mas: float,
    parallax_factor_al: np.ndarray,
) -> np.ndarray:
    """Project the orbital ``(Δα*, Δδ)`` and the 5-parameter astrometry
    onto the along-scan direction ``ψ`` (radians, counterclockwise from
    north).

    Returns the model along-scan offset in mas at each epoch::

        model = Δα* sin(ψ) + Δδ cos(ψ)
              + ra_offset_mas · sin(ψ)
              + dec_offset_mas · cos(ψ)
              + (pmra · sin(ψ) + pmdec · cos(ψ)) · Δt_yr
              + plx_mas · parallax_factor_al

    where ``Δt_yr = (t_mjd − epoch_ref_mjd) / 365.25``.

    Conventions
    -----------
    - ``pmra_masyr`` is μ_α* (already ``× cos δ``); no extra ``cos δ``.
    - ``parallax_factor_al`` is dimensionless; the term is added
      directly (no extra trig).
    """
    psi = np.asarray(psi, dtype=float)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # Orbital component: Δα* sin(ψ) + Δδ cos(ψ).
    model = d_ra * sin_psi + d_dec * cos_psi

    # Position offset (constant across epochs, projected onto ψ).
    model = model + ra_offset_mas * sin_psi + dec_offset_mas * cos_psi

    # Proper motion: linear in time relative to the reference epoch.
    # μ_α* is already cos-δ-corrected (Gaia convention).
    dt_yr = (np.asarray(t_mjd, dtype=float) - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR
    model = model + (pmra_masyr * sin_psi + pmdec_masyr * cos_psi) * dt_yr

    # Parallax: additive plx × parallax_factor_al, no extra trig.
    model = model + plx_mas * np.asarray(parallax_factor_al, dtype=float)

    return model


# ── Internal-theta → along-scan model (engine model path) ────────────
#
# ``_along_scan_for_theta_campbell`` and ``_along_scan_for_theta_ti`` are
# pure compositions of the forward primitives ``campbell_xy`` /
# ``thiele_innes_xy`` / ``along_scan_model`` defined above: they add no
# math of their own, only the decoding of a sampler's internal ``theta``
# into the arguments those primitives take.  The joint likelihood uses
# the Campbell one (:func:`orblet.likelihood.loglike_joint`); the
# public entry point for either basis is :func:`along_scan_from_theta`.
#
#
#
# They consume the INTERNAL ``theta`` dict produced by
# ``...astro.chains.vector_to_theta_campbell`` / ``_ti`` (period in
# YEARS via ``P_yr``, inclination in radians via ``inc_rad``,
# periastron time via ``tp_mjd``).  The public
# :func:`along_scan_from_theta` adapts the PROJECTED theta to this
# internal contract.


def _along_scan_for_theta_campbell(
    theta: dict,
    *,
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Engine along-scan model for a Campbell internal ``theta`` dict."""
    d_ra, d_dec = campbell_xy(
        t_mjd,
        period_yr=theta["P_yr"],
        ecc=theta["e"],
        omega=theta["omega_rad"],
        inc=theta["inc_rad"],
        Omega=theta["Omega_rad"],
        tp_mjd=theta["tp_mjd"],
        m_comp_msun=theta["m2_msun"],
        M_total_msun=theta["M_total_msun"],
        plx_mas=theta["plx_mas"],
    )
    return along_scan_model(
        d_ra=d_ra, d_dec=d_dec,
        psi=psi, t_mjd=t_mjd, epoch_ref_mjd=epoch_ref_mjd,
        ra_offset_mas=theta["ra_offset_mas"],
        dec_offset_mas=theta["dec_offset_mas"],
        pmra_masyr=theta["pmra_masyr"],
        pmdec_masyr=theta["pmdec_masyr"],
        plx_mas=theta["plx_mas"],
        parallax_factor_al=parallax_factor_al,
    )


def _along_scan_for_theta_ti(
    theta: dict,
    *,
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Engine along-scan model for a Thiele-Innes internal ``theta``."""
    d_ra, d_dec = thiele_innes_xy(
        t_mjd,
        period_yr=theta["P_yr"],
        ecc=theta["e"],
        A_mas=theta["A_mas"],
        B_mas=theta["B_mas"],
        F_mas=theta["F_mas"],
        G_mas=theta["G_mas"],
        tp_mjd=theta["tp_mjd"],
    )
    return along_scan_model(
        d_ra=d_ra, d_dec=d_dec,
        psi=psi, t_mjd=t_mjd, epoch_ref_mjd=epoch_ref_mjd,
        ra_offset_mas=theta["ra_offset_mas"],
        dec_offset_mas=theta["dec_offset_mas"],
        pmra_masyr=theta["pmra_masyr"],
        pmdec_masyr=theta["pmdec_masyr"],
        plx_mas=theta["plx_mas"],
        parallax_factor_al=parallax_factor_al,
    )


# ── Public: projected theta → engine along-scan model ────────────────

# Accepted public basis spellings (project-wide; NO campbell/ti
# aliases).  Exposed for the wrong-basis error message.
_ACCEPTED_BASES = ("kepler", "thiele_innes")


def along_scan_from_theta(
    theta: dict,
    *,
    basis: str,
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    epoch_ref_mjd: float,
) -> np.ndarray:
    r"""Engine along-scan model (mas) from a PROJECTED astrometric theta.

    This is the safe way to recompute the Python astrometric engine's
    own along-scan model from the flat ``theta`` mapping a custom astro
    ``log_lik_fn`` receives (the
    :func:`...astro.nonlinear_sampler._vector_to_theta_for_prior` projection).
    A hand-rolled forward model that does NOT call this helper must
    reproduce the FULL convention stack (period units, the D-1
    inclination latent, the τ-on-the-disk periastron-time map, the
    positive primary-frame photocenter amplitude + mass-ratio scaling,
    the parallax-factor additive term) — getting any of those
    wrong silently mis-fits and yields an invalid companion-mass /
    compact-object inference.  Prefer this helper.

    Parameters
    ----------
    theta : dict
        The PROJECTED theta produced by
        ``_vector_to_theta_for_prior(vec, basis=...)``.  This is NOT the
        internal forward-model theta: the period is in DAYS
        (``P_days``), the inclination is the raw ``u_inc`` latent
        (Campbell), the periastron time is ABSENT (it is reconstructed
        here from the disk-angle latents), and the scalar physics
        parameters carry physics-unit names.  Expected keys per basis:

        ``basis="kepler"`` (Campbell):
            ``P_days`` (days), ``e``, ``omega_rad`` (rad, primary
            frame), ``u_inc`` (∈ [0, 1]), ``Omega_rad`` (rad),
            ``theta_x``, ``theta_y`` (periastron disk latents),
            ``m2_msun`` (M☉, companion mass), ``M_total_msun`` (M☉,
            TOTAL mass), ``plx_mas`` (mas), ``ra_offset_mas``,
            ``dec_offset_mas`` (mas), ``pmra_masyr``, ``pmdec_masyr`` (mas/yr).

        ``basis="thiele_innes"`` (TI):
            ``P_days`` (days), ``e``, ``A_mas``, ``B_mas``, ``F_mas``,
            ``G_mas`` (PHOTOCENTER amplitudes in mas — already × plx and ×
            m_comp/M_total; NO further scaling applied), ``phase_x``,
            ``phase_y`` (periastron disk latents), ``ra_offset_mas``,
            ``dec_offset_mas`` (mas), ``pmra_masyr``, ``pmdec_masyr`` (mas/yr),
            ``plx_mas`` (mas, used only in the parallax term).

        The projected theta and the internal forward model share the
        ``A_mas`` … ``G_mas`` amplitude spelling.  The
        ``astro_jitter_mas`` key is ignored here (jitter enters the
        likelihood, not the model).
    basis : str
        One of ``"kepler"`` or ``"thiele_innes"`` (the project-wide
        spellings; there are NO ``"campbell"`` / ``"ti"`` aliases).
    t_mjd : np.ndarray
        Observation epochs (MJD), shape ``(n_epoch,)``.
    psi : np.ndarray
        Along-scan angle (radians, counterclockwise from north), one per
        epoch.
    parallax_factor_al : np.ndarray
        Dimensionless parallax factor per epoch; the term
        ``plx_mas × parallax_factor_al`` is ADDED directly (no extra
        trig).
    epoch_ref_mjd : float
        Reference epoch (MJD).  MUST equal the fit's value: it sets BOTH
        the periastron-time zero (via the τ-on-the-disk map) AND the
        proper-motion zero-point.  Passing a different value silently
        shifts both.  ``t_mjd`` and ``epoch_ref_mjd`` must be on the SAME
        time scale as the fit's epochs (Gaia DR4: TCB); mixing scales
        (e.g. TDB/UTC ``t_mjd`` with a TCB ``epoch_ref_mjd``) silently
        biases the proper-motion and parallax phase.

    Returns
    -------
    np.ndarray
        Along-scan model offset (mas), shape ``(n_epoch,)``, ``float64``.
        Nothing here logs, prints or persists it; what the caller does
        with it is the caller's business.

    Photocenter vs relative orbit (per basis)
    -----------------------------------------
    - ``"kepler"`` (Campbell): pass ``m2_msun`` / ``M_total_msun`` /
      ``plx_mas``; this helper applies the
      ``+m_comp / M_total · plx`` photocenter scaling internally (via
      :func:`campbell_xy`).  The scaling is POSITIVE, with primary-frame
      ω used as-is.
    - ``"thiele_innes"``: ``A_mas, B_mas, F_mas, G_mas`` must ALREADY be photocenter
      amplitudes in mas; NO scaling is applied (matches
      :func:`thiele_innes_xy`).

    Conventions
    -----------
    ``ω`` is the primary's argument of periastron (textbook frame), used
    directly in :func:`campbell_xy` with the positive primary-frame
    photocenter amplitude.  ``Ω`` is measured CCW from north.  Period is
    converted days → years with the named constant
    :data:`orblet.constants.DAYS_PER_KEPLER_YEAR`.

    Example (Campbell)
    ------------------
    >>> # proj = _vector_to_theta_for_prior(vec, basis="kepler")
    >>> # model = along_scan_from_theta(
    ... #     proj, basis="kepler", t_mjd=t, psi=psi,
    ... #     parallax_factor_al=pf, epoch_ref_mjd=57388.5,
    ... # )

    No θ value is logged, printed or persisted in this function, and
    no error message echoes one.  Value-carrying delegate failures are re-raised as
    generic, digit-free messages at the helper boundary.
    """
    if basis not in _ACCEPTED_BASES:
        # VALUE-FREE: name the accepted spellings only, no theta value.
        raise ValueError(
            "basis must be 'kepler' or 'thiele_innes'"
        )

    # Build the INTERNAL theta the hoisted ``_along_scan_for_theta_*``
    # helpers consume, then evaluate.  Missing-key access raises a bare
    # KeyError naming the KEY (no value) — data-safe, re-raised unchanged.
    # Math/delegate failures that could carry a theta VALUE (e.g. an
    # out-of-range ``u_inc`` reaching ``arccos``, or Kepler
    # non-convergence) are caught and re-raised with a generic,
    # digit-free message, so no input value is echoed through the
    # exception.  The whole decode + delegate is inside the try so the
    # ``arccos`` / ``τ→tp`` math is covered too.
    try:
        period_yr = theta["P_days"] / DAYS_PER_KEPLER_YEAR
        if basis == "kepler":
            internal = {
                "P_yr": period_yr,
                "e": theta["e"],
                "omega_rad": theta["omega_rad"],
                "inc_rad": _inc_from_u_latent(theta["u_inc"]),
                "Omega_rad": theta["Omega_rad"],
                "tp_mjd": _tp_from_disk_angle(
                    theta["theta_x"], theta["theta_y"],
                    period_days=theta["P_days"],
                    epoch_ref_mjd=epoch_ref_mjd,
                ),
                "m2_msun": theta["m2_msun"],
                "M_total_msun": theta["M_total_msun"],
                "plx_mas": theta["plx_mas"],
                "ra_offset_mas": theta["ra_offset_mas"],
                "dec_offset_mas": theta["dec_offset_mas"],
                "pmra_masyr": theta["pmra_masyr"],
                "pmdec_masyr": theta["pmdec_masyr"],
            }
            delegate = _along_scan_for_theta_campbell
        else:  # basis == "thiele_innes"
            internal = {
                "P_yr": period_yr,
                "e": theta["e"],
                # Internal forward and the projected theta share the
                # ``A_mas`` … ``G_mas`` key spelling.
                "A_mas": theta["A_mas"],
                "B_mas": theta["B_mas"],
                "F_mas": theta["F_mas"],
                "G_mas": theta["G_mas"],
                "tp_mjd": _tp_from_disk_angle(
                    theta["phase_x"], theta["phase_y"],
                    period_days=theta["P_days"],
                    epoch_ref_mjd=epoch_ref_mjd,
                ),
                "plx_mas": theta["plx_mas"],
                "ra_offset_mas": theta["ra_offset_mas"],
                "dec_offset_mas": theta["dec_offset_mas"],
                "pmra_masyr": theta["pmra_masyr"],
                "pmdec_masyr": theta["pmdec_masyr"],
            }
            delegate = _along_scan_for_theta_ti

        return delegate(
            internal, t_mjd=t_mjd, psi=psi,
            parallax_factor_al=parallax_factor_al,
            epoch_ref_mjd=epoch_ref_mjd,
        )
    except KeyError:
        # KeyError names a KEY (no value) — re-raise unchanged.
        raise
    except Exception:
        # Fixed exception type + fixed message: re-raising ``type(exc)`` is
        # fragile (some exception classes have non-string ``__init__``
        # signatures) and would leak the delegate's class.  ``from None``
        # drops the value-bearing __cause__/__context__ chain.
        raise RuntimeError(
            "along_scan_from_theta failed evaluating the model "
            "(see basis/geometry inputs); value suppressed for "
            "data-safety"
        ) from None

# ══════════════════════════════════════════════════════════════════════
# JOINT RV + ASTROMETRY
#
# The joint views reuse the RV and astrometric forward models above,
# with the inclination shared between the channels.
# ══════════════════════════════════════════════════════════════════════


def rv_model_with_inclination(
    t_mjd: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    omega_rad: float,
    tau: float,
    mass_msun: float,
    inc_rad: float,
    M_msun: float,
    offset_kms: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Predicted line-of-sight RV in km/s under joint-fit sin(i) scaling.

    The RV-only forward model fixes ``sin(i) = 1`` and treats its
    ``mass_msun`` argument as the projected mass ``m_2 × sin(i_true)``.
    The joint forward must instead scale K by an explicit ``sin(i)``
    because the joint posterior is over the TRUE companion mass.

    Implementation detail: ``rv_model`` computes ``K`` as a linear
    function of ``mass_msun`` (see :func:`...rv.forward._semi_amplitude_kms`).
    Substituting ``mass × sin(i)`` for ``mass`` therefore scales the
    predicted RV (about γ) by ``sin(i)`` exactly.  The transformation
    is algebraic; no Newton-iteration is duplicated.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation times in MJD.
    period_yr, ecc, omega_rad, tau, mass_msun, M_msun, offset_kms : float
        Orbital parameters; see
        :func:`orblet.model.rv_model`.
        ``mass_msun`` here is the TRUE companion mass; the wrapper
        applies the ``sin(i)`` projection before calling ``rv_model``.
    inc_rad : float
        Orbital inclination in radians.  The RV channel sees K scaled
        by ``sin(inc_rad)``.
    epoch_ref_mjd : float
        Reference MJD for the τ → tp conversion.

    Returns
    -------
    rv : np.ndarray
        Line-of-sight radial velocity at each ``t_mjd`` (km/s).
    """
    sin_inc = math.sin(float(inc_rad))
    # Forward the orbit to ``rv_model`` with mass replaced by
    # mass × sin(i).  Every other parameter passes through unchanged so
    # the Kepler solve in :func:`rv_model` sees the same
    # ``(P, e, ω, τ, M, γ)`` and the same Newton-iteration path.
    return rv_model(
        t_mjd,
        period_yr=period_yr,
        ecc=ecc,
        omega_rad=omega_rad,
        tau=tau,
        mass_msun=float(mass_msun) * sin_inc,
        M_msun=M_msun,
        offset_kms=offset_kms,
        epoch_ref_mjd=epoch_ref_mjd,
    )


def joint_theta_view(theta: dict, *, epoch_ref_mjd: float) -> tuple[dict, dict]:
    """Return ``(rv_theta, astro_theta)`` sub-dicts for downstream engines.

    The joint sampler operates on a single ``theta`` dict that carries
    every parameter both engines need.  The downstream RV and astro
    forwards / likelihoods expect their own narrow key schemas;
    :func:`joint_theta_view` provides those narrowed views without
    copying the full theta.

    The RV view's ``m2_msun`` is overridden by ``mass × sin(i)``
    (matches the convention used in :func:`rv_model_with_inclination`)
    so callers that bypass the inclination-aware RV forward and use
    the raw :func:`...rv.likelihood.loglike` see the correct K-scaling
    for free.

    Returns
    -------
    rv_theta : dict
        Keys: ``"P_yr", "e", "omega_rad", "tau",
        "m2_msun" (= mass × sin i), "M_total_msun", "gamma_kms",
        "rv_jitter_kms"``.  Shape-compatible with
        :func:`...rv.likelihood.loglike`.
    astro_theta : dict
        Keys: ``"P_yr", "e", "omega_rad", "inc_rad",
        "Omega_rad", "m2_msun" (true companion mass),
        "M_total_msun", "plx_mas", "ra_offset_mas", "dec_offset_mas",
        "pmra_masyr", "pmdec_masyr", "astro_jitter_mas", "tp_mjd"``.
        Shape-compatible with
        :func:`...astro.nonlinear_sampler._along_scan_for_theta_campbell`.

    Notes
    -----
    ``tp_mjd`` is reconstructed from ``tau × P_days + epoch_ref_mjd``
    so both engines agree on the periastron time.  The joint sampler
    samples a single ``tau`` latent (via the τ-disk reparametrisation
    in :mod:`...joint.chains`).
    """
    inc = float(theta["inc_rad"])
    sin_inc = math.sin(inc)

    P_yr = float(theta["P_yr"])
    P_days = P_yr * DAYS_PER_KEPLER_YEAR
    tau = float(theta["tau"])
    tp_mjd = tau * P_days + float(epoch_ref_mjd)

    mass_true = float(theta["m2_msun"])

    rv_theta = {
        "P_yr":       P_yr,
        "e":          float(theta["e"]),
        "omega_rad":  float(theta["omega_rad"]),
        "tau":        tau,
        "m2_msun":  mass_true * sin_inc,           # RV-engine projects via sin(i)
        "M_total_msun": float(theta["M_total_msun"]),
        "gamma_kms":    float(theta["gamma_kms"]),
        "rv_jitter_kms": float(theta.get("rv_jitter_kms", 0.0)),
    }
    astro_theta = {
        "P_yr":         P_yr,
        "e":            float(theta["e"]),
        "omega_rad":    float(theta["omega_rad"]),
        "inc_rad":      inc,
        "Omega_rad":    float(theta["Omega_rad"]),
        "m2_msun":    mass_true,                    # astro consumes true mass
        "M_total_msun":   float(theta["M_total_msun"]),
        "plx_mas":        float(theta["plx_mas"]),
        "ra_offset_mas":  float(theta["ra_offset_mas"]),
        "dec_offset_mas": float(theta["dec_offset_mas"]),
        "pmra_masyr":     float(theta["pmra_masyr"]),
        "pmdec_masyr":    float(theta["pmdec_masyr"]),
        "astro_jitter_mas": float(theta["astro_jitter_mas"]),
        "tp_mjd":         tp_mjd,
    }
    return rv_theta, astro_theta


#: The names this module advertises (``from orblet.model import *``).
#: `rv_curve`, `semi_amplitude_kms`, `fm_from_K`, `joint_theta_view` and
#: `rv_model_with_inclination` are defined above and importable by name,
#: but are not advertised here; listing them would be a decision about
#: the public surface, not a tidy-up.  The package's public names are the
#: ones on orblet's front door (``orblet.__all__``); this list governs
#: star-imports from this module only.
__all__ = [
    "rv_model",
    "thiele_innes_xy",
    "campbell_xy",
    "along_scan_model",
    "along_scan_from_theta",
    "kepler_xy_orbit",
    "tp_from_disk_angle",
]
