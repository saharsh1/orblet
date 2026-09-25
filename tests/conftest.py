"""
Shared pytest configuration and fixtures for the ``tests/`` tree.

Two responsibilities:

1. Make ``tests/`` importable so test files can write
   ``from benchmark import OrbitSimulator`` without sys.path gymnastics.
   Without this, the ``tests/`` directory is *not* on the import path
   (pytest adds the rootdir, not subdirs, by default).  Putting it on
   sys.path here keeps the synth-data benchmark framework reachable from
   every test file under ``tests/`` regardless of how pytest is invoked
   (``pytest tests/`` or ``pytest -k`` or ``python -m pytest …``).

2. Host shared synthetic-cadence fixtures (``cad_bh3``, ``cad_circ``)
   so multiple test files (``test_simulator_python_roundtrip.py``,
   ``test_simulator_loader_contract.py``) can consume them by name
   without redefining.  Pytest's canonical mechanism for cross-module
   fixtures is conftest.py at the package root.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


# Imports below depend on sys.path having ``tests/`` on it (for
# ``benchmark``) — they are placed AFTER the sys.path mutation above.
from benchmark import OrbitSimulator  # noqa: E402
from benchmark.cadence import Cadence  # noqa: E402
from benchmark.presets import bh3_like, circular_toy  # noqa: E402
from orblet.constants import C_KMS, MJD_J2010_TCB, MJD_J2016_TCB  # noqa: E402


# ── Synthetic cadence dataclass ───────────────────────────────────────


@dataclass(frozen=True)
class _SyntheticCadence:
    """
    Inline synthetic cadence satisfying the :class:`Cadence` Protocol.

    All 11 Protocol attributes are declared.  Astrometric arrays carry
    real values; RV arrays are present-but-empty because the consumers
    (``test_simulator_python_roundtrip.py``,
    ``test_simulator_loader_contract.py``) target the astrometric
    channel only.

    Time arrays: days from J2010.0 TCB (matches Gaia's obs_time).
    Scan angles: radians.  Parallax factor: dimensionless.
    """

    astro_obs_time: np.ndarray
    astro_scan_angle: np.ndarray
    astro_parallax_factor_al: np.ndarray
    astro_transit_id: np.ndarray
    astro_centroid_pos_err: np.ndarray
    ra0_deg: float | None
    dec0_deg: float | None
    rv_obs_time: np.ndarray
    rv_scan_angle: np.ndarray
    rv_transit_id: np.ndarray
    rv_err: np.ndarray


def _make_cadence(
    *,
    n: int,
    seed: int,
    ra0_deg: float,
    dec0_deg: float,
    t_start_j2010_days: float,
    t_span_days: float,
) -> _SyntheticCadence:
    """
    Build a synthetic cadence over ``[t_start, t_start + t_span]`` in
    J2010-day units.  Scan angles and parallax factors come from a
    seeded ``np.random.default_rng`` — the global ``np.random`` state
    is never touched.
    """
    rng = np.random.default_rng(seed)
    # Astrometric arrays — populated (astrometric channel scope).
    t = np.linspace(t_start_j2010_days, t_start_j2010_days + t_span_days, n)
    psi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    f_pi = rng.uniform(-1.0, 1.0, size=n)
    return _SyntheticCadence(
        astro_obs_time=t.astype(np.float64),
        astro_scan_angle=psi.astype(np.float64),
        astro_parallax_factor_al=f_pi.astype(np.float64),
        astro_transit_id=np.arange(n, dtype=np.int64),
        astro_centroid_pos_err=np.full(n, 0.1, dtype=np.float64),
        ra0_deg=ra0_deg,
        dec0_deg=dec0_deg,
        # RV arrays — present-but-empty (astrometric scope).
        rv_obs_time=np.array([], dtype=float),
        rv_scan_angle=np.array([], dtype=float),
        rv_transit_id=np.array([], dtype=np.int64),
        rv_err=np.array([], dtype=float),
    )


# Cadence span: 5 Julian years starting around MJD 57389.0.  The
# simulator interprets ``cadence.astro_obs_time`` as days from J2010.0
# TCB, so subtract ``MJD_J2010_TCB`` from any MJD literal.
_BH3_T_START_J2010 = 57389.0 - MJD_J2010_TCB
_BH3_T_SPAN_DAYS = 5.0 * 365.25
_CIRC_T_START_J2010 = 57389.0 - MJD_J2010_TCB
_CIRC_T_SPAN_DAYS = 5.0 * 365.25


@pytest.fixture(scope="session")
def cad_bh3() -> _SyntheticCadence:
    """
    BH3-like cadence: N=50, seed=42, ra0/dec0 = factory truth (zero
    tangent-plane offset).  ``ra0_deg``/``dec0_deg`` are derived from
    ``bh3_like()`` rather than hardcoded so the cadence
    automatically tracks any factory edit.
    """
    sim = bh3_like()
    cad = _make_cadence(
        n=50,
        seed=42,
        ra0_deg=sim.ra_deg,
        dec0_deg=sim.dec_deg,
        t_start_j2010_days=_BH3_T_START_J2010,
        t_span_days=_BH3_T_SPAN_DAYS,
    )
    # Protocol-conformance check at fixture-resolution time so drift in
    # the Cadence Protocol surfaces here, not buried in a per-test body.
    assert isinstance(cad, Cadence)
    return cad


@pytest.fixture(scope="session")
def cad_circ() -> _SyntheticCadence:
    """
    Toy-circular cadence: N=30, seed=43, ra0/dec0 = factory truth (zero
    tangent-plane offset).  ``ra0_deg``/``dec0_deg`` are derived from
    ``circular_toy()`` rather than hardcoded.
    """
    sim = circular_toy()
    cad = _make_cadence(
        n=30,
        seed=43,
        ra0_deg=sim.ra_deg,
        dec0_deg=sim.dec_deg,
        t_start_j2010_days=_CIRC_T_START_J2010,
        t_span_days=_CIRC_T_SPAN_DAYS,
    )
    assert isinstance(cad, Cadence)
    return cad


# ── Truth fixtures (Batch 4c Phase B) ─────────────────────────────────
#
# Session-scoped truth dicts derived from the same factories that produce
# the cadence fixtures.  Keys are derived from the
# :class:`OrbitSimulator` properties — never hardcoded — so any factory
# edit propagates automatically.  Consumed by
# the engine contract tests.


def _truth_from_simulator(sim: OrbitSimulator) -> dict:
    """
    Return a dict of canonical truth quantities derived from ``sim``.

    Units follow the simulator's documented conventions:
    period in years, masses in M☉, angles in radians, axis in AU/mas,
    parallax in mas, sky position in degrees, periastron in MJD.
    """
    return {
        "period_yr":     float(sim.P_yr),
        "ecc":           float(sim.e),
        "inc_rad":       float(sim.i_rad),
        "omega_rad":     float(sim.omega_rad),
        "Omega_rad":     float(sim.Omega_rad),
        "tp_mjd":        float(sim.tp_mjd),
        "M_total_msun":  float(sim.M_total_msun),
        "m_comp_msun":   float(sim.m2_msun),
        "plx_mas":       float(sim.parallax_mas),
        "a_AU":          float(sim.a_au),
        "a_phot_mas":    float(sim.a_phot_mas),
        "ra_deg":        float(sim.ra_deg),
        "dec_deg":       float(sim.dec_deg),
    }


@pytest.fixture(scope="session")
def truth_bh3() -> dict:
    """
    Truth dict derived from ``bh3_like()``.

    Batch 4c Phase B.  Mirrors the ``cad_bh3`` pattern: factory is the
    single source of truth, no hardcoded duplicates.
    """
    return _truth_from_simulator(bh3_like())


@pytest.fixture(scope="session")
def truth_circ() -> dict:
    """
    Truth dict derived from ``circular_toy()``.

    Batch 4c Phase B.  Mirrors the ``cad_circ`` pattern: factory is the
    single source of truth, no hardcoded duplicates.
    """
    return _truth_from_simulator(circular_toy())


# ── Synthetic RV-dataset fixture (P1, Python RV engine) ──────────────
#
# Builds a prepared-RV dict (the shape produced by
# :func:`orblet.prepare.prepare_rv_for_orbit`) directly from
# Keplerian truth, using the Kepler solver from
# :mod:`orblet.kepler` as an *independent* forward model
# (the fitter under test must not be invoked to build its own data).
#
# All inputs synthetic; no data files are read.  Returns a callable
# fixture so per-test parameter overrides are cheap.


def _synthesize_rv_prepared(
    *,
    seed: int,
    n_epochs: int,
    P_yr: float,
    e: float,
    omega_rad: float,
    tau: float,
    K_kms: float,
    gamma_kms: float,
    sigma_kms: float,
    epoch_ref_mjd: float = 60000.0,
) -> dict:
    """Build a prepared-RV dict from Keplerian truth + Gaussian noise.

    Sampled via the canonical solver in :mod:`orblet.kepler`
    so this fixture is independent of the fitter under test.
    """
    from orblet.kepler import solve_kepler
    from orblet.constants import DAYS_PER_KEPLER_YEAR

    rng = np.random.default_rng(seed)
    P_days = P_yr * DAYS_PER_KEPLER_YEAR
    t_lo = epoch_ref_mjd
    t_hi = epoch_ref_mjd + 2.0 * P_days
    t = np.sort(rng.uniform(t_lo, t_hi, size=n_epochs))

    tp_mjd = tau * P_days + epoch_ref_mjd
    M_anom = (2.0 * np.pi * (t - tp_mjd) / P_days) % (2.0 * np.pi)
    E = solve_kepler(M_anom, e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )
    rv_clean = gamma_kms + K_kms * (
        np.cos(nu + omega_rad) + e * np.cos(omega_rad)
    )
    noise = rng.normal(0.0, sigma_kms, size=n_epochs)
    rv = rv_clean + noise
    rv_err = np.full(n_epochs, sigma_kms)

    return {
        "epochs_mjd": t.astype(np.float64),
        "rv": rv.astype(np.float64),
        "rv_err": rv_err.astype(np.float64),
        "n_epochs": int(n_epochs),
        # Truth attached for closure tests.  Not part of the fitter API.
        "_truth": {
            "P_yr": P_yr,
            "P_days": P_days,
            "e": e,
            "omega_rad": omega_rad,
            "tau": tau,
            "tp_mjd": tp_mjd,
            "K_kms": K_kms,
            "gamma_kms": gamma_kms,
            "sigma_kms": sigma_kms,
            "epoch_ref_mjd": epoch_ref_mjd,
        },
    }


@pytest.fixture(scope="session")
def synth_rv_dataset():
    """Session-scoped factory for synthetic prepared-RV dicts.

    Usage::

        prepared = synth_rv_dataset(
            seed=0, n_epochs=24, P_yr=1.0, e=0.2,
            omega_rad=0.7, tau=0.3, K_kms=5.0,
            gamma_kms=10.0, sigma_kms=0.3,
        )

    Returns a dict with the same keys as
    :func:`orblet.prepare.prepare_rv_for_orbit`, plus a private
    ``"_truth"`` sub-dict carrying the generating parameters.
    """
    return _synthesize_rv_prepared


# ── Synthetic astro-dataset fixture (P2, Python astrometric engine) ──
#
# Builds an astro_data dict (the epoch-astrometry loader shape)
# directly from Keplerian truth, using the canonical Thiele-Innes
# routine in :mod:`orblet.kepler` as the forward model.
#
# All inputs synthetic; no data files are read.  Returns a callable
# fixture so per-test parameter overrides are cheap.


def _synthesize_astro(
    *,
    seed: int,
    n_epochs: int = 80,
    P_days: float = 800.0,
    e: float = 0.3,
    omega_rad: float = 0.7,
    Omega_rad: float = 1.2,
    inc_rad: float = 0.9,
    tp_offset: float = 0.4,
    m_comp_msun: float = 0.5,
    M_total_msun: float = 1.5,
    plx_mas: float = 5.0,
    pmra_masyr: float = 2.0,
    pmdec_masyr: float = -3.0,
    ra_deg: float = 100.0,
    dec_deg: float = 20.0,
    sigma_mas: float = 0.4,
    epoch_ref_mjd: float = 57388.5,
    t_span_days: float | None = None,
) -> tuple[dict, dict]:
    """Build a synthetic astro_data dict + truth.

    Returns ``(astro_data, truth)`` where ``astro_data`` carries the
    keys of the epoch-astrometry loader shape
    (``obs_time``, ``centroid_pos``, ``centroid_pos_err``,
    ``scan_angle``, ``parallax_factor_al``).

    ``obs_time`` is in **MJD** (consistent with the Python engine's
    expectations after the J2010-day → MJD shift).  Scan angles are
    drawn uniform on ``[0, 2π)`` and parallax factors uniform on
    ``[-1, 1]``.
    """
    from orblet.kepler import campbell_xy
    from orblet.constants import DAYS_PER_KEPLER_YEAR

    rng = np.random.default_rng(seed)
    if t_span_days is None:
        t_span_days = 5.0 * P_days
    t_lo = epoch_ref_mjd - 0.5 * t_span_days
    t_hi = epoch_ref_mjd + 0.5 * t_span_days
    t_mjd = np.sort(rng.uniform(t_lo, t_hi, size=n_epochs))
    psi = rng.uniform(0.0, 2.0 * np.pi, size=n_epochs)
    plx_factor = rng.uniform(-1.0, 1.0, size=n_epochs)

    # Photocenter signature: a_phot = m_comp/M_total · a_au · plx (mas).
    P_yr = P_days / DAYS_PER_KEPLER_YEAR
    a_au = (P_yr ** 2 * M_total_msun) ** (1.0 / 3.0)
    a_phot_mas = (m_comp_msun / M_total_msun) * a_au * plx_mas

    tp_mjd = tp_offset * P_days + epoch_ref_mjd

    # Use campbell_xy as a forward-model emulator.  This canonical
    # helper takes ``omega`` directly (no internal translation).  The
    # astrometric forward also takes primary-frame ω directly with a
    # POSITIVE photocentre amplitude.  We use positive amplitude + primary-frame
    # ω here, matching the demo simulator's convention.
    d_ra, d_dec = campbell_xy(
        t_mjd=t_mjd,
        period_yr=P_yr,
        ecc=e,
        omega_rad=omega_rad,
        inc_rad=inc_rad,
        Omega_rad=Omega_rad,
        tp_mjd=tp_mjd,
        a_mas=a_phot_mas,  # POSITIVE — matches engine's net forward output
    )
    orbit_al = d_ra * np.sin(psi) + d_dec * np.cos(psi)

    # 5-parameter astrometric layer.  Position offsets are zero in the
    # synthetic (data is centred on (ra_deg, dec_deg)).
    dt_yr = (t_mjd - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR
    pm_al = (pmra_masyr * np.sin(psi) + pmdec_masyr * np.cos(psi)) * dt_yr
    plx_al = plx_mas * plx_factor

    centroid_clean = orbit_al + pm_al + plx_al
    noise = rng.normal(0.0, sigma_mas, size=n_epochs)
    centroid = centroid_clean + noise

    astro_data = {
        "obs_time":             t_mjd.astype(np.float64),
        "scan_angle":           psi.astype(np.float64),
        "centroid_pos":         centroid.astype(np.float64),
        "centroid_pos_err":     np.full(n_epochs, sigma_mas, dtype=np.float64),
        "parallax_factor_al":   plx_factor.astype(np.float64),
        "transit_id":           np.arange(n_epochs, dtype=np.int64),
        "ra0_deg":              ra_deg,
        "dec0_deg":             dec_deg,
        # J2016 in days-from-J2010 (2191.5): the bundle's reference epoch,
        # which fits read rather than assume.
        "ref_epoch_days":       MJD_J2016_TCB - MJD_J2010_TCB,
    }
    truth = {
        "P_days":         P_days,
        "P_yr":           P_yr,
        "e":              e,
        "omega_rad":      omega_rad,
        "Omega_rad":      Omega_rad,
        "inc_rad":        inc_rad,
        "tp_mjd":         tp_mjd,
        "tp_offset":      tp_offset,
        "m_comp_msun":    m_comp_msun,
        "M_total_msun":   M_total_msun,
        "plx_mas":        plx_mas,
        "pmra_masyr":     pmra_masyr,
        "pmdec_masyr":    pmdec_masyr,
        "ra_deg":         ra_deg,
        "dec_deg":        dec_deg,
        "sigma_mas":      sigma_mas,
        "epoch_ref_mjd":  epoch_ref_mjd,
        "a_au":           a_au,
        "a_phot_mas":     a_phot_mas,
    }
    return astro_data, truth


@pytest.fixture(scope="session")
def synth_astro_dataset():
    """Session-scoped factory for synthetic astrometric datasets.

    Usage::

        astro, truth = synth_astro_dataset(seed=0, n_epochs=80)

    Returns ``(astro_data, truth)`` where ``astro_data`` is in the
    epoch-astrometry loader shape (with ``obs_time`` in MJD).  ``truth`` carries the generating
    parameters.
    """
    return _synthesize_astro


def compose_seed_from_astro_data(
    astro_data: dict,
    *,
    epoch_ref_mjd: float,
    p_min_days: float,
    p_max_days: float,
) -> dict:
    """The two-call pattern, for tests: scan → compose → (pass).

    Exactly the three calls a user makes (see
    ``docs/model_and_likelihoods.md`` §3.9), so the acceptance
    gates certify the route users actually take.  Plain module-level
    helper (imported as ``from conftest import ...``, the
    ``require_notebook_or_skip`` precedent), not a fixture.

    - Observation times go through the sampler's OWN
      ``resolve_epochs_mjd`` (MJD vs days-from-J2010 is decided by the
      same median rule the fit uses) — never a re-implemented heuristic.
    - ``p_min_days`` / ``p_max_days`` are REQUIRED per call site: the scan
      window must sit strictly inside THAT site's period prior, because a
      seed outside the composed prior RAISES by design (T21C:
      ``LogUniform(P/2, 2P)``; W2 default ``LogUniform(P/3, 3P)``;
      coverage ``Uniform(0.5P, 1.5P)``).
    - The scan grid is B-1's audited one (``test_seed_bridge.py``),
      verbatim: a coarser eccentricity grid would seed an e = 0.1 truth
      at e = 0.0 — the trap's own location — and per-test grid tweaks
      invite the "tuned against the gate" charge.
    - A negative scan-fitted parallax would fall outside the default
      ``LogUniform`` parallax prior and RAISE; not expected at these
      fixtures' SNR — read such a failure as a seed-quality finding.
    """
    from orblet.search import (
        compose_ti_seed,
        scan_ti_frequency,
    )
    from orblet.prepare import resolve_epochs_mjd

    t_mjd = np.asarray(resolve_epochs_mjd(astro_data), dtype=float)
    peaks = scan_ti_frequency(
        t_mjd,
        np.asarray(astro_data["scan_angle"], dtype=float),
        np.asarray(astro_data["parallax_factor_al"], dtype=float),
        np.asarray(astro_data["centroid_pos"], dtype=float),
        np.asarray(astro_data["centroid_pos_err"], dtype=float),
        f_min_per_day=1.0 / float(p_max_days),
        f_max_per_day=1.0 / float(p_min_days),
        oversample=4.0,
        ecc_grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85),
        tau_grid=tuple(np.linspace(0.0, 0.875, 8)),
        epoch_ref_mjd=float(epoch_ref_mjd),
        top_k=3,
    )
    return compose_ti_seed(peaks[0], epoch_ref_mjd=float(epoch_ref_mjd))


# ── Synthetic joint-dataset fixture (P3, Python joint engine) ────────
#
# Builds a unified joint dataset from a SINGLE Keplerian truth:
#   - rv:    prepared-RV dict (same schema as ``synth_rv_dataset``).
#   - astro: astro_data dict (same schema as ``synth_astro_dataset``).
# Both forward models use the same canonical solver + campbell_xy
# helpers from :mod:`orblet.kepler` so the fitter under test is
# never invoked to build its own data.  No data files are read.
#
# The factory returns ``(joint, truth)`` where ``joint = {"rv": ...,
# "astro": ...}`` and ``truth`` is the merged Keplerian-+-system truth
# dict.  Returns a callable so per-test parameter overrides are cheap.


def _synthesize_joint(
    *,
    seed: int,
    # RV-channel knobs.
    n_rv_epochs: int = 24,
    rv_sigma_kms: float = 0.3,
    gamma_kms: float = 0.0,
    # Astro-channel knobs.
    n_astro_epochs: int = 80,
    centroid_sigma_mas: float = 0.4,
    pmra_masyr: float = 0.0,
    pmdec_masyr: float = 0.0,
    ra_deg: float = 100.0,
    dec_deg: float = 20.0,
    # Shared Keplerian truth.
    P_days: float = 400.0,
    e: float = 0.2,
    omega_rad: float = 0.7,
    Omega_rad: float = 1.2,
    inc_rad: float = np.pi / 3.0,
    tau: float = 0.3,
    m_comp_msun: float = 2.0,
    M_total_msun: float = 3.0,
    plx_mas: float = 5.0,
    epoch_ref_mjd: float = MJD_J2016_TCB,
    t_span_rv_periods: float = 2.0,
    t_span_astro_periods: float = 5.0,
) -> tuple[dict, dict]:
    """Build a synthetic joint (RV, astro) dataset from one Keplerian truth.

    Returns ``({"rv": rv_prepared, "astro": astro_data}, truth)``.

    The RV channel uses the standard textbook-primary form
    ``v(t) = γ + K [cos(ν + ω) + e cos ω]`` with
    ``K = (2π/P) · a₁·sin(i) / sqrt(1−e²)``, ``a₁ = a × m_comp / M_total``.
    The astrometric channel uses :func:`campbell_xy` with positive
    photocenter amplitude + primary-frame ω (matches the demo
    simulator's convention; see existing ``_synthesize_astro``).

    All randomness routes through ``np.random.default_rng(seed)``.  Two
    independent sub-streams (one per channel) keep the noise
    independent across RV and astro, mirroring the project's data
    model.

    The default ``epoch_ref_mjd`` is ``MJD_J2016_TCB`` (the Gaia DR4 catalog
    epoch), matching the ``fit_joint_orbit`` default so a consumer that does
    NOT pass ``epoch_ref_mjd`` cannot silently mismatch the proper-motion
    lever-arm origin (the C4e bug-#1 contract).  The error from such a
    mismatch scales as proper-motion × epoch-offset, so it is invisible for
    zero-PM fixtures but ~747 mas for the large-PM BH3-like fixture.
    """
    from orblet.kepler import solve_kepler, campbell_xy
    from orblet.constants import AU_M, DAYS_PER_KEPLER_YEAR

    P_yr = P_days / DAYS_PER_KEPLER_YEAR
    tp_mjd = tau * P_days + epoch_ref_mjd
    a_au = (P_yr ** 2 * M_total_msun) ** (1.0 / 3.0)
    a1_au = a_au * m_comp_msun / M_total_msun
    a_phot_mas = a1_au * plx_mas

    # K = (2π/P_days) · a1·sin(i) [AU] · (km / AU) / sqrt(1 − e²).
    #   AU per day → km/s : 1 AU/day = (AU_M / 1e3) km / 86400 s
    au_per_day_to_kms = (AU_M / 1000.0) / 86400.0
    K_kms = (
        2.0 * np.pi / P_days * a1_au * np.sin(inc_rad)
        * au_per_day_to_kms / np.sqrt(1.0 - e * e)
    )

    # Two independent sub-streams: integers from a master_rng pin them.
    master_rng = np.random.default_rng(seed)
    rv_seed, astro_seed = master_rng.integers(0, 2**31 - 1, size=2)
    rv_rng = np.random.default_rng(int(rv_seed))
    astro_rng = np.random.default_rng(int(astro_seed))

    # ── RV channel ────────────────────────────────────────────────────
    t_lo_rv = epoch_ref_mjd
    t_hi_rv = epoch_ref_mjd + float(t_span_rv_periods) * P_days
    t_rv = np.sort(rv_rng.uniform(t_lo_rv, t_hi_rv, size=n_rv_epochs))

    M_anom_rv = (2.0 * np.pi * (t_rv - tp_mjd) / P_days) % (2.0 * np.pi)
    E_rv = solve_kepler(M_anom_rv, e)
    nu_rv = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E_rv / 2.0),
        np.sqrt(1.0 - e) * np.cos(E_rv / 2.0),
    )
    rv_clean = gamma_kms + K_kms * (
        np.cos(nu_rv + omega_rad) + e * np.cos(omega_rad)
    )
    rv = rv_clean + rv_rng.normal(0.0, rv_sigma_kms, size=n_rv_epochs)
    rv_err = np.full(n_rv_epochs, rv_sigma_kms)

    rv_prepared = {
        "epochs_mjd": t_rv.astype(np.float64),
        "rv":         rv.astype(np.float64),
        "rv_err":     rv_err.astype(np.float64),
        "n_epochs":   int(n_rv_epochs),
    }

    # ── Astro channel ─────────────────────────────────────────────────
    t_span_astro = float(t_span_astro_periods) * P_days
    t_lo_a = epoch_ref_mjd - 0.5 * t_span_astro
    t_hi_a = epoch_ref_mjd + 0.5 * t_span_astro
    t_a = np.sort(
        astro_rng.uniform(t_lo_a, t_hi_a, size=n_astro_epochs)
    )
    psi = astro_rng.uniform(0.0, 2.0 * np.pi, size=n_astro_epochs)
    plx_factor = astro_rng.uniform(-1.0, 1.0, size=n_astro_epochs)

    # Use campbell_xy as a forward emulator (positive amplitude +
    # primary-frame ω; matches the existing ``_synthesize_astro`` and
    # the demo simulator).
    d_ra, d_dec = campbell_xy(
        t_mjd=t_a,
        period_yr=P_yr,
        ecc=e,
        omega_rad=omega_rad,
        inc_rad=inc_rad,
        Omega_rad=Omega_rad,
        tp_mjd=tp_mjd,
        a_mas=a_phot_mas,
    )
    orbit_al = d_ra * np.sin(psi) + d_dec * np.cos(psi)

    # 5-parameter astrometric layer: zero position offsets (data
    # centred on ra_deg/dec_deg), pmra/pmdec from kwargs, parallax from
    # plx_mas × plx_factor.
    dt_yr = (t_a - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR
    pm_al = (pmra_masyr * np.sin(psi) + pmdec_masyr * np.cos(psi)) * dt_yr
    plx_al = plx_mas * plx_factor

    centroid_clean = orbit_al + pm_al + plx_al
    noise = astro_rng.normal(0.0, centroid_sigma_mas, size=n_astro_epochs)
    centroid = centroid_clean + noise

    astro_data = {
        "obs_time":           t_a.astype(np.float64),
        "scan_angle":         psi.astype(np.float64),
        "centroid_pos":       centroid.astype(np.float64),
        "centroid_pos_err":   np.full(
            n_astro_epochs, centroid_sigma_mas, dtype=np.float64,
        ),
        "parallax_factor_al": plx_factor.astype(np.float64),
        "transit_id":         np.arange(n_astro_epochs, dtype=np.int64),
        "ra0_deg":            ra_deg,
        "dec0_deg":           dec_deg,
        # J2016 in days-from-J2010: the bundle's reference epoch, which fits
        # read rather than assume.
        "ref_epoch_days":     MJD_J2016_TCB - MJD_J2010_TCB,
    }

    truth = {
        "P_days":         P_days,
        "P_yr":           P_yr,
        "e":              e,
        "omega_rad":      omega_rad,
        "Omega_rad":      Omega_rad,
        "inc_rad":        inc_rad,
        "tau":            tau,
        "tp_mjd":         tp_mjd,
        "K_kms":          float(K_kms),
        "gamma_kms":      gamma_kms,
        "rv_sigma_kms":   rv_sigma_kms,
        "centroid_sigma_mas": centroid_sigma_mas,
        "m_comp_msun":    m_comp_msun,
        "M_total_msun":   M_total_msun,
        "plx_mas":        plx_mas,
        "pmra_masyr":     pmra_masyr,
        "pmdec_masyr":    pmdec_masyr,
        "ra_deg":         ra_deg,
        "dec_deg":        dec_deg,
        "a_au":           float(a_au),
        "a_phot_mas":     float(a_phot_mas),
        "epoch_ref_mjd":  epoch_ref_mjd,
    }
    return {"rv": rv_prepared, "astro": astro_data}, truth


@pytest.fixture(scope="session")
def synth_joint_dataset():
    """Session-scoped factory for synthetic joint RV+astrometric datasets.

    Usage::

        joint, truth = synth_joint_dataset(seed=0)
        prepared_rv = joint["rv"]
        astro_data  = joint["astro"]

    Returns ``(joint, truth)`` where ``joint["rv"]`` shape-matches
    :func:`orblet.prepare.prepare_rv_for_orbit`, ``joint["astro"]``
    is in the
    epoch-astrometry loader shape,
    and ``truth`` carries the shared Keplerian + system parameters.
    """
    return _synthesize_joint


# ── DE432s ephemeris-cache fixture ───────────────────────────────────
#
# Five new test files (``test_parallax_factors.py``,
# ``test_simulate_sub_dict_astro.py``, ``test_simulate_sub_dict_joint.py``,
# ``test_ephemeris_pin.py``, ``test_closure_simulate_sub_dict_python.py``)
# all guard their tests with a local ``_ephemeris_cache_available`` /
# ``_skip_if_no_ephemeris`` pair.  Hoisted into a single session-scoped
# fixture here so the cache probe runs at most once per session and the
# skip behavior lives in one place.


def _ephemeris_cache_probe() -> bool:
    """Return True iff the DE432s ephemeris kernel is locally cached.

    Touches astropy lazily inside the function body (no top-level
    import) so conftest collection stays cheap on machines without
    astropy installed for unrelated test runs.  Returns ``False`` on
    any network / cache / IO error.
    """
    try:
        from astropy.coordinates import (
            get_body_barycentric,
            solar_system_ephemeris,
        )
        from astropy.time import Time

        with solar_system_ephemeris.set("de432s"):
            get_body_barycentric(
                "earth", Time(58000.0, format="mjd", scale="tcb")
            )
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def ephemeris_or_skip() -> None:
    """Skip the calling test cleanly if DE432s is not cached.

    Usage::

        def test_something(ephemeris_or_skip):
            # body runs only when DE432s is cached
            ...

    Session-scoped: the cache probe is run at most once per pytest
    session.  On cache miss every dependent test is skipped via
    ``pytest.skip`` with a one-line setup hint.
    """
    if not _ephemeris_cache_probe():
        pytest.skip(
            "DE432s not cached; install orblet[ephemeris] and run "
            "orblet.parallax.fetch_ephemeris('de432s') once."
        )
