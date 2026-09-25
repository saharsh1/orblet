"""
Phase A diagnostic: Python-side simulator roundtrip.

Status
------
This file is the **diagnostic deliverable** of Batch 4c Phase A.  It
verifies that the along-scan astrometric signal produced by
:meth:`benchmark.simulator.OrbitSimulator.simulate_astrometry` is
byte-for-byte equivalent to an independently-assembled forward model
built directly from :func:`orblet.kepler.campbell_xy` plus
parallax-factor, proper-motion, and tangent-plane-offset terms.

**A failing test in this file is an acceptable Phase A deliverable.**
The diagnostic-first stance forbids modifying the simulator, the
``_kepler`` helpers, or the engines to make these tests pass.  A
test failure localises a Python-side convention bug to a specific term
(orbit, parallax, PM, photocenter, AL projection); a passing suite
indicates the bug is downstream in the engines.

Scope
-----
- Astrometric channel only.
- Light ratio β = 0 (simulator's ``__post_init__`` enforces this).
- Synthetic cadences only (BH3-like and toy-circular).
- No real cadence files, no ``@benchmark`` marker.

Conventions tested
------------------
- AL projection ``Δα*·sin ψ + Δδ·cos ψ``.
- ``campbell_xy`` argument order and Thiele-Innes (A, B, F, G)
  signs, *roundtrip-pinned only* (both expected and actual call the
  same helper for tests 6.1–6.6).  Test 6.7 is the only fully
  independent textbook cross-check.
- Proper-motion reference epoch MJD 57388.5 (J2016.0 TCB).
- Parallax-factor sign.
- Photocenter vs relative-orbit magnitude (see test 6.6).

Time conventions
----------------
- ``cadence.astro_obs_time``: days from J2010.0 TCB.  Convert to MJD
  via ``+55197.0`` (``MJD_J2010_TCB``).
- ``MJD_J2016_TCB = 57388.5`` is the PM reference epoch (TT-MJD; the
  ~22 s TCB-vs-TT shift is below the 1e-9 mas test tolerances).

Fixtures
--------
``cad_bh3`` and ``cad_circ`` live in ``tests/conftest.py``, shared by
name across test files.  Their sky positions are derived from the
presets (``bh3_like().ra_deg`` etc.) rather than hardcoded literals, so the tangent-plane offset is structurally zero unless a
test explicitly overrides ``ra0_deg``/``dec0_deg``.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import OrbitSimulator
from benchmark.presets import bh3_like, circular_toy
from orblet.constants import MJD_J2010_TCB, MJD_J2016_TCB
from orblet.kepler import campbell_xy
from orblet.priors import DAYS_PER_KEPLER_YEAR

# The synthetic-cadence dataclass lives in conftest.py alongside the
# fixtures.  Tests that need to construct ad-hoc single-epoch cadences
# import the type from there.
from conftest import _SyntheticCadence


# ── Helper: independently-assembled forward model ──────────────────────────


def _orbit_xy(sim: OrbitSimulator, t_j2010_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Independently-assembled (Δα*, Δδ) photocenter offsets in mas.

    Note (Phase A diagnostic limitation): both this expected path and
    the simulator's ``along_scan_orbit`` call ``campbell_xy`` (or
    ``thiele_innes_al`` which delegates to it).  This pins the
    simulator's *use* of the helper (argument order, units) but does
    NOT isolate the helper's internal sign/formula correctness.  A
    separate ``test_kepler.py`` plan must cover that.
    """
    t_mjd = t_j2010_days + MJD_J2010_TCB
    return campbell_xy(
        t_mjd=t_mjd,
        period_yr=sim.P_yr,
        ecc=sim.e,
        omega_rad=sim.omega_rad,
        inc_rad=sim.i_rad,
        Omega_rad=sim.Omega_rad,
        tp_mjd=sim.tp_mjd,
        a_mas=sim.a_phot_mas,
    )


def _project_al(d_ra: np.ndarray, d_dec: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """AL projection: ``Δα*·sin ψ + Δδ·cos ψ`` in mas."""
    return d_ra * np.sin(psi) + d_dec * np.cos(psi)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_orbit_only_residual_matches_thiele_innes(cad_bh3: _SyntheticCadence) -> None:
    """
    6.1: AL-projected orbit-only residual matches the independently-
    assembled AL projection of ``campbell_xy`` output.

    Configuration: BH3-like simulator, noise off (centroid_sigma=0),
    parallax/PM/offset disabled by passing ``include_5param_model=
    False`` so the simulator emits the orbit signal alone.
    """
    # Disable noise so simulate_astrometry returns the deterministic
    # signal.  ra0/dec0 are taken from the cadence; with
    # include_5param_model=False they are unused (no offset / PM /
    # parallax assembly).
    sim = bh3_like(centroid_sigma_mas=0.0)
    out = sim.simulate_astrometry(cad_bh3, seed=0, include_5param_model=False)

    sim_al_orbit = np.asarray(out["centroid_pos"], dtype=float)

    # Independent assembly: Thiele-Innes Δα*, Δδ → AL projection.
    d_ra, d_dec = _orbit_xy(sim, cad_bh3.astro_obs_time)
    ref_al_orbit = _project_al(d_ra, d_dec, cad_bh3.astro_scan_angle)

    np.testing.assert_allclose(sim_al_orbit, ref_al_orbit, atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "sim_factory_name",
    ["bh3_like", "circular_toy"],
    ids=["bh3_like", "circular_toy"],
)
def test_full_assembly_matches_simulate(
    sim_factory_name: str,
    cad_bh3: _SyntheticCadence,
    cad_circ: _SyntheticCadence,
) -> None:
    """
    6.2: Full simulator output matches an independently-assembled
    forward model summing orbit + parallax + PM + tangent-plane offset
    and AL-projecting via ``Δα*·sin ψ + Δδ·cos ψ``.

    The simulator (``simulate_astrometry``) applies ``cos δ`` to the
    tangent-plane *position offset* only; the proper-motion term is
    assembled WITHOUT a ``cos δ`` factor (``μ_α*`` is already declared
    to include ``cos δ`` in the project's PM convention).  The
    independent reference here therefore matches that convention.
    """
    if sim_factory_name == "bh3_like":
        sim = bh3_like(centroid_sigma_mas=0.0)
        cad = cad_bh3
    else:
        sim = circular_toy(centroid_sigma_mas=0.0)
        cad = cad_circ

    out = sim.simulate_astrometry(cad, seed=0, include_5param_model=True)
    sim_al = np.asarray(out["centroid_pos"], dtype=float)

    # ── Independent assembly ────────────────────────────────────────
    t_j2010 = np.asarray(cad.astro_obs_time, dtype=float)
    psi = np.asarray(cad.astro_scan_angle, dtype=float)
    f_pi = np.asarray(cad.astro_parallax_factor_al, dtype=float)
    cos_d = np.cos(np.deg2rad(sim.dec_deg))

    # Orbit term.
    d_ra_orb, d_dec_orb = _orbit_xy(sim, t_j2010)

    # Tangent-plane offset (Δα* with cos δ; Δδ no cos factor).  Matches
    # simulator lines 301–306 in form.
    ra0 = cad.ra0_deg if cad.ra0_deg is not None else sim.ra_deg
    dec0 = cad.dec0_deg if cad.dec0_deg is not None else sim.dec_deg
    d_ra_off = (sim.ra_deg - ra0) * cos_d * 3_600_000.0
    d_dec_off = (sim.dec_deg - dec0) * 3_600_000.0

    # Proper motion.  μ_α* already includes cos δ in the project's PM
    # convention (matches simulator's `simulate_astrometry`, which
    # applies no cos δ here).  Δt in years from J2016.0 TCB.
    dt_yr = (t_j2010 + MJD_J2010_TCB - MJD_J2016_TCB) / DAYS_PER_KEPLER_YEAR
    d_ra_pm = sim.pmra_masyr * dt_yr
    d_dec_pm = sim.pmdec_masyr * dt_yr

    d_ra_total = d_ra_orb + d_ra_off + d_ra_pm
    d_dec_total = d_dec_orb + d_dec_off + d_dec_pm

    ref_al = _project_al(d_ra_total, d_dec_total, psi) + sim.parallax_mas * f_pi

    np.testing.assert_allclose(sim_al, ref_al, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    "sim_factory_name",
    ["bh3_like", "circular_toy"],
    ids=["bh3_like", "circular_toy"],
)
def test_pm_isolated_decomposition(
    sim_factory_name: str,
    cad_bh3: _SyntheticCadence,
    cad_circ: _SyntheticCadence,
) -> None:
    """
    6.3: The PM contribution, isolated by subtracting orbit, parallax,
    and tangent-plane offset from the simulator output, matches an
    independently-assembled PM term.

    Strategy: parallax_mas and m2_msun cannot be set to zero
    (``__post_init__`` rejects both).  Instead, subtract orbit +
    parallax + offset analytically from the simulator output and
    compare residual to the PM-only reference.

    PM convention: ``μ_α*`` includes ``cos δ`` already, so the
    independent reference applies NO additional ``cos δ`` factor on RA
    (matches the simulator's ``simulate_astrometry`` PM block).
    """
    if sim_factory_name == "bh3_like":
        # Use a finite, non-trivial PM for diagnostic clarity.
        sim = bh3_like(centroid_sigma_mas=0.0)
        cad = cad_bh3
    else:
        # circular_toy() defaults pmra=pmdec=0; override so the PM
        # signal is non-zero and the test is informative.
        sim = circular_toy(
            centroid_sigma_mas=0.0,
            pmra_masyr=20.0,
            pmdec_masyr=-15.0,
        )
        cad = cad_circ

    out = sim.simulate_astrometry(cad, seed=0, include_5param_model=True)
    sim_al = np.asarray(out["centroid_pos"], dtype=float)

    t_j2010 = np.asarray(cad.astro_obs_time, dtype=float)
    psi = np.asarray(cad.astro_scan_angle, dtype=float)
    f_pi = np.asarray(cad.astro_parallax_factor_al, dtype=float)
    cos_d = np.cos(np.deg2rad(sim.dec_deg))

    # Orbit AL contribution.
    d_ra_orb, d_dec_orb = _orbit_xy(sim, t_j2010)
    al_orbit = _project_al(d_ra_orb, d_dec_orb, psi)

    # Tangent-plane offset AL contribution.
    ra0 = cad.ra0_deg if cad.ra0_deg is not None else sim.ra_deg
    dec0 = cad.dec0_deg if cad.dec0_deg is not None else sim.dec_deg
    d_ra_off = (sim.ra_deg - ra0) * cos_d * 3_600_000.0
    d_dec_off = (sim.dec_deg - dec0) * 3_600_000.0
    al_offset = d_ra_off * np.sin(psi) + d_dec_off * np.cos(psi)

    # Parallax AL contribution.
    al_parallax = sim.parallax_mas * f_pi

    # Subtract everything except PM.
    sim_al_pm_only = sim_al - al_orbit - al_offset - al_parallax

    # Independent PM reference: no cos δ (μ_α* already includes it).
    dt_yr = (t_j2010 + MJD_J2010_TCB - MJD_J2016_TCB) / DAYS_PER_KEPLER_YEAR
    d_ra_pm = sim.pmra_masyr * dt_yr
    d_dec_pm = sim.pmdec_masyr * dt_yr
    ref_al_pm_only = _project_al(d_ra_pm, d_dec_pm, psi)

    np.testing.assert_allclose(sim_al_pm_only, ref_al_pm_only, atol=1e-7, rtol=0)


def test_pm_reference_epoch_is_j2016() -> None:
    """
    6.4: At a single epoch placed exactly at MJD 57388.5 (J2016.0 TCB),
    the PM contribution is zero to within 1e-9 mas — pinning the PM
    reference-epoch literal in the simulator/_constants module.

    Construction: BH3-like simulator with pmra=10, pmdec=0, orbit and
    parallax not directly disable-able (mass and parallax must be > 0
    by ``__post_init__``).  Strategy mirrors test 6.3: place a single
    epoch at MJD 57388.5 (J2010 days = 57388.5 - 55197.0 = 2191.5),
    let the simulator emit centroid_pos, then subtract orbit +
    parallax + offset analytically and check the residual (the PM
    contribution) is zero.

    Cadence sentinels (``ra0_deg``, ``dec0_deg``) are pinned to the
    simulator's truth (zero tangent-plane offset) so the offset term
    drops out cleanly.
    """
    sim = bh3_like(
        centroid_sigma_mas=0.0,
        pmra_masyr=10.0,
        pmdec_masyr=0.0,
    )

    # Single epoch exactly at J2016.0 TCB (MJD 57388.5).
    t_j2010_single = MJD_J2016_TCB - MJD_J2010_TCB  # = 2191.5

    # Cadence: N=1 with all arrays length 1.  Scan angle chosen so the
    # parallax factor's AL contribution is well-defined (any value
    # works since we subtract it analytically).  ``ra0_deg``/
    # ``dec0_deg`` set to simulator truth → zero tangent-plane offset.
    cad = _SyntheticCadence(
        astro_obs_time=np.array([t_j2010_single], dtype=np.float64),
        astro_scan_angle=np.array([0.5], dtype=np.float64),
        astro_parallax_factor_al=np.array([0.3], dtype=np.float64),
        astro_transit_id=np.array([0], dtype=np.int64),
        astro_centroid_pos_err=np.array([0.1], dtype=np.float64),
        ra0_deg=sim.ra_deg,
        dec0_deg=sim.dec_deg,
        rv_obs_time=np.array([], dtype=float),
        rv_scan_angle=np.array([], dtype=float),
        rv_transit_id=np.array([], dtype=np.int64),
        rv_err=np.array([], dtype=float),
    )

    out = sim.simulate_astrometry(cad, seed=0, include_5param_model=True)
    sim_al = np.asarray(out["centroid_pos"], dtype=float)

    # Analytic subtraction of orbit, offset, parallax to leave PM only.
    psi = cad.astro_scan_angle
    f_pi = cad.astro_parallax_factor_al
    cos_d = np.cos(np.deg2rad(sim.dec_deg))

    d_ra_orb, d_dec_orb = _orbit_xy(sim, cad.astro_obs_time)
    al_orbit = _project_al(d_ra_orb, d_dec_orb, psi)

    ra0 = cad.ra0_deg if cad.ra0_deg is not None else sim.ra_deg
    dec0 = cad.dec0_deg if cad.dec0_deg is not None else sim.dec_deg
    d_ra_off = (sim.ra_deg - ra0) * cos_d * 3_600_000.0
    d_dec_off = (sim.dec_deg - dec0) * 3_600_000.0
    al_offset = d_ra_off * np.sin(psi) + d_dec_off * np.cos(psi)

    al_parallax = sim.parallax_mas * f_pi

    al_pm_only = sim_al - al_orbit - al_offset - al_parallax

    # At MJD 57388.5 the PM contribution must vanish.
    np.testing.assert_allclose(al_pm_only, np.zeros_like(al_pm_only), atol=1e-9, rtol=0)


def test_parallax_factor_sign_convention() -> None:
    """
    6.5: With orbit and PM held fixed (parallax cannot be set to zero
    by the simulator's ``__post_init__``, so we cancel orbit and PM by
    differencing two epochs that share the same time and scan angle),
    flipping the sign of ``astro_parallax_factor_al`` flips the sign
    of the parallax AL contribution by exactly
    ``2 × |f_pi| × parallax_mas``.

    Construction: two epochs at the same time, same scan angle, with
    opposite-sign parallax factors (+0.7 and −0.7).  Orbit and PM
    contributions are identical at both epochs, so they cancel in the
    difference.  Difference should equal 1.4 × parallax_mas.

    Cadence sentinels (``ra0_deg``, ``dec0_deg``) are pinned to the
    simulator's truth (zero tangent-plane offset) so the offset term
    drops out cleanly.
    """
    sim = bh3_like(
        centroid_sigma_mas=0.0,
        parallax_mas=1.0,
        pmra_masyr=0.0,
        pmdec_masyr=0.0,
    )

    # Same time, same scan angle, opposite parallax factors.
    t_j2010 = np.array([2000.0, 2000.0], dtype=np.float64)
    psi = np.array([1.0, 1.0], dtype=np.float64)
    f_pi = np.array([+0.7, -0.7], dtype=np.float64)

    cad = _SyntheticCadence(
        astro_obs_time=t_j2010,
        astro_scan_angle=psi,
        astro_parallax_factor_al=f_pi,
        astro_transit_id=np.array([0, 1], dtype=np.int64),
        astro_centroid_pos_err=np.full(2, 0.1, dtype=np.float64),
        ra0_deg=sim.ra_deg,
        dec0_deg=sim.dec_deg,
        rv_obs_time=np.array([], dtype=float),
        rv_scan_angle=np.array([], dtype=float),
        rv_transit_id=np.array([], dtype=np.int64),
        rv_err=np.array([], dtype=float),
    )

    out = sim.simulate_astrometry(cad, seed=0, include_5param_model=True)
    sim_al = np.asarray(out["centroid_pos"], dtype=float)

    # Difference: only the parallax term differs between the two
    # epochs (orbit, PM, offset all identical).
    diff = sim_al[0] - sim_al[1]
    expected = 2.0 * 0.7 * sim.parallax_mas  # = 1.4 mas

    np.testing.assert_allclose(diff, expected, atol=1e-8, rtol=0)


def test_photocenter_sign_convention(cad_circ: _SyntheticCadence) -> None:
    """
    6.6: The photocenter (Δα*, Δδ) and the standard "relative orbit"
    (built from ``+a_rel_mas`` with the same orbital elements) are
    related by ``d_phot = (M_comp / M_total) × d_rel`` — same sign,
    scaled by the mass fraction.

    Convention: the simulator's
    ``a_phot_mas = +a × m2/M_total × plx`` is a positive magnitude,
    feeding the SAME orbital elements (P, e, ω, i, Ω, tp) into
    ``campbell_xy`` as the relative-orbit reference.  Both
    therefore produce same-sign sky-plane offsets per epoch; only
    their magnitudes differ by ``m2/M_total``.

    The "negative sign" sometimes attributed to photocenter motion
    refers to the photocenter trajectory vs the SECONDARY's relative
    motion (primary moves opposite to secondary), which is a
    different comparison than this test makes.  See the simulator's
    ``a_phot_au`` property and the reference manual, §3.2.

    This test asserts both elementwise sign equality and the mass-
    fraction magnitude relation.
    """
    # Override M_companion via the circular_toy preset.  The factory
    # default has m2_msun=2.0; we keep that explicitly here for
    # clarity, plus a 0-noise override.
    sim = circular_toy(
        centroid_sigma_mas=0.0,
        m1_msun=1.0,
        m2_msun=2.0,
    )
    cad = cad_circ

    # Simulator photocenter offsets (Δα*, Δδ) via the same path the
    # simulator uses internally — namely ``campbell_xy`` with
    # a_mas = sim.a_phot_mas (whatever sign the simulator assigns).
    d_ra_phot, d_dec_phot = _orbit_xy(sim, cad.astro_obs_time)

    # Reference relative-orbit offsets: same Kepler geometry but with
    # a_mas = +a_rel_mas (positive total semi-major axis in mas).
    a_rel_mas = sim.a_au * sim.parallax_mas
    d_ra_rel, d_dec_rel = campbell_xy(
        t_mjd=cad.astro_obs_time + MJD_J2010_TCB,
        period_yr=sim.P_yr,
        ecc=sim.e,
        omega_rad=sim.omega_rad,
        inc_rad=sim.i_rad,
        Omega_rad=sim.Omega_rad,
        tp_mjd=sim.tp_mjd,
        a_mas=a_rel_mas,
    )

    # Mass-fraction magnitude relation (still required).
    mass_frac = sim.m2_msun / sim.M_total_msun
    np.testing.assert_allclose(
        np.abs(d_ra_phot), mass_frac * np.abs(d_ra_rel), atol=1e-9, rtol=0,
    )
    np.testing.assert_allclose(
        np.abs(d_dec_phot), mass_frac * np.abs(d_dec_rel), atol=1e-9, rtol=0,
    )

    # Elementwise sign equality: same orbital elements + positive
    # a_mas on both calls to campbell_xy → same sky-plane sign at
    # every epoch.  Pre-filter near-zero crossings to avoid float
    # round-off flipping the sign of essentially zero values.
    nonzero_ra = np.abs(d_ra_rel) > 1e-9
    nonzero_dec = np.abs(d_dec_rel) > 1e-9
    np.testing.assert_array_equal(
        np.sign(d_ra_phot[nonzero_ra]),
        np.sign(d_ra_rel[nonzero_ra]),
        err_msg="photocenter Δα* sign should match relative-orbit Δα* (same orbital elements + positive a_phot_mas).",
    )
    np.testing.assert_array_equal(
        np.sign(d_dec_phot[nonzero_dec]),
        np.sign(d_dec_rel[nonzero_dec]),
        err_msg="photocenter Δδ sign should match relative-orbit Δδ (same orbital elements + positive a_phot_mas).",
    )


def test_face_on_circular_textbook_limit(cad_circ: _SyntheticCadence) -> None:
    """
    6.7: A face-on (i = 0) circular (e = 0) orbit traces a circle of
    radius ``a_phot_mas`` in the (Δα*, Δδ) plane, with angular
    position advancing linearly in mean anomaly at rate 2π/P.

    This is the only test in the suite whose expected value is
    computed without calling ``campbell_xy`` — it provides true
    cross-check coverage for the orbit term.
    """
    # Toy-circular factory defaults: e=0 already.  Override i_rad=0
    # for the face-on limit, plus zero noise.
    sim = circular_toy(
        centroid_sigma_mas=0.0,
        i_rad=0.0,
    )
    cad = cad_circ

    # Use the simulator's own forward-model helper to get (Δα*, Δδ),
    # then verify the textbook geometric properties.  We could also
    # call the simulator's simulate_astrometry with
    # include_5param_model=False, but along_scan_orbit returns the AL
    # projection only — we need the (Δα*, Δδ) components separately,
    # so we go via _orbit_xy (which calls campbell_xy directly).
    #
    # To keep this test fully independent of campbell_xy
    # *correctness*, we instead invoke simulate_astrometry with
    # include_5param_model=False and then back out (Δα*, Δδ) by
    # solving two linearly-independent scan-angle pairs at each epoch
    # — but that requires additional cadence machinery.  A simpler
    # textbook check that still bypasses campbell_xy is to verify
    # the AL signal magnitude directly: |s_AL(t)| ≤ a_phot, with peak
    # |s_AL| = a_phot when scan angle aligns with the orbit position.
    #
    # We do both: the geometric circle/angular-velocity check against
    # _orbit_xy (covers the (Δα*, Δδ) pair), and an AL-magnitude bound
    # check against simulate_astrometry (no campbell_xy in expected
    # path for the bound).
    d_ra, d_dec = _orbit_xy(sim, cad.astro_obs_time)

    # Radius check: face-on circular orbit traces a circle of radius
    # a_phot_mas exactly.
    radius = np.hypot(d_ra, d_dec)
    np.testing.assert_allclose(radius, sim.a_phot_mas, atol=1e-9, rtol=0)

    # Angular advance check: in mean anomaly, position advances at
    # rate 2π/P_days per day.  arctan2(Δδ, Δα*) returns the angular
    # position in radians; differences across epochs must equal
    # 2π·Δt/P_days.
    angle = np.unwrap(np.arctan2(d_dec, d_ra))
    dt_days = np.diff(cad.astro_obs_time)
    expected_dphi = 2.0 * np.pi * dt_days / sim.P_days
    actual_dphi = np.diff(angle)
    # Sign of dphi may be ±2π/P (depends on Ω).  Compare absolute
    # values for the rate; the sign is a separate convention not
    # asserted here.
    np.testing.assert_allclose(
        np.abs(actual_dphi), np.abs(expected_dphi), atol=1e-9, rtol=0,
    )

    # AL bound check (independent of campbell_xy in the expected
    # path): the AL projection magnitude cannot exceed a_phot_mas for
    # a face-on circular orbit.
    out = sim.simulate_astrometry(cad, seed=0, include_5param_model=False)
    sim_al = np.asarray(out["centroid_pos"], dtype=float)
    assert np.all(np.abs(sim_al) <= sim.a_phot_mas + 1e-9), (
        f"AL signal exceeds a_phot bound: max|s_AL|={np.abs(sim_al).max():.6e}, "
        f"a_phot_mas={sim.a_phot_mas:.6e}"
    )
