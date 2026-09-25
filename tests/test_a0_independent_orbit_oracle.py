"""A0 — INDEPENDENT closed-form orbit oracle (engine-fidelity keystone).

Why this test exists
--------------------
Every other fidelity test (model-at-truth, recovery, calibration) scores the
engine against data produced by the engine's OWN sibling simulator
(``_kepler.campbell_xy`` / ``OrbitSimulator``).  A convention or sign error
shared by BOTH the simulator and the fitter is invisible to all of them — the
codebase explicitly documents one such pair (the ``ω+π`` companion-frame flip
that "cancels exactly" against the negative photocenter amplitude;
see ``tests/test_joint_epoch_contract.py`` and ``astro/forward.py``).

This module builds a SMALL, GENUINELY INDEPENDENT closed-form orbit oracle —
textbook Kepler + Thiele-Innes, written from scratch below — and asserts the
ENGINE'S forward output agrees with the ORACLE'S at the SAME truth.  The oracle
deliberately:

- does NOT import ``astro.forward`` / ``rv.forward`` (the production forwards),
- does NOT import ``_kepler.campbell_xy`` / ``_kepler.thiele_innes_al``
  (the engine's sibling reference projector),
- does NOT import ``solve_kepler`` (it iterates the Kepler equation itself).

The only shared atom is ``DAYS_PER_KEPLER_YEAR`` (a unit constant, not physics)
and ``numpy``.

ω-convention reconciliation (the load-bearing boundary)
-------------------------------------------------------
The oracle computes everything in the PROJECT'S PRIMARY-frame ω (textbook
binary-star / spectroscopic convention) with a POSITIVE photocentre
amplitude ``a_phot = (m_comp/M_total)·a·plx``.  The astrometric forward
(``campbell_xy``) does the same.  (A companion-frame ``ω → ω + π`` paired
with a NEGATIVE amplitude would be equivalent — replacing ``ω → ω+π``
negates every ``cosω`` / ``sinω`` term and hence ``A,B,F,G``, and the ``−1``
on the amplitude negates them back — but orblet uses neither.)  If the two
derivations ever DISAGREE the assertions below fail loudly (that is the
whole point — do NOT loosen the tolerance to pass; a disagreement is a real
convention bug).

All inputs synthetic; no data files read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orblet.constants import DAYS_PER_KEPLER_YEAR


# ──────────────────────────────────────────────────────────────────────
# The INDEPENDENT oracle — textbook Kepler + Thiele-Innes from scratch.
# Nothing below imports the engine's forward models or its sibling
# _kepler projector; the Kepler equation is solved here directly.
# ──────────────────────────────────────────────────────────────────────


def _oracle_solve_kepler(M: np.ndarray, e: float) -> np.ndarray:
    """Solve ``M = E - e sin E`` for E (rad) via independent Newton iteration.

    Deliberately NOT ``orblet.kepler.solve_kepler`` — this is a
    from-scratch reimplementation so a bug in the shared solver could not
    hide a forward-model error.  Starts from ``E = M`` and iterates to a
    tight tolerance.
    """
    M = np.asarray(M, dtype=float)
    E = M.copy()
    for _ in range(100):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.max(np.abs(dE)) < 1e-13:
            break
    return E


def _oracle_true_anomaly(
    t_mjd: np.ndarray, *, period_yr: float, ecc: float, tp_mjd: float,
) -> np.ndarray:
    """True anomaly ν (rad), textbook half-angle form, independent solver."""
    P_days = period_yr * DAYS_PER_KEPLER_YEAR
    M = 2.0 * math.pi * (np.asarray(t_mjd, dtype=float) - tp_mjd) / P_days
    E = _oracle_solve_kepler(M, ecc)
    # ν from E via the standard half-angle tangent relation.
    nu = 2.0 * np.arctan2(
        math.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        math.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    return nu


def _oracle_thiele_innes(
    *, a: float, omega: float, inc: float, Omega: float,
) -> tuple[float, float, float, float]:
    """Textbook Thiele-Innes constants (Hilditch 2001, §5.12; ESA Hipparcos
    vol. 1, eq. 1.3.69) from Campbell elements, PRIMARY-frame ω, signed
    amplitude ``a``.

        A = a (cosω cosΩ − sinω sinΩ cos i)
        B = a (cosω sinΩ + sinω cosΩ cos i)
        F = a (−sinω cosΩ − cosω sinΩ cos i)
        G = a (−sinω sinΩ + cosω cosΩ cos i)

    Returns ``(A, B, F, G)`` with the project's sky mapping
    ``Δα* = B x + G y``, ``Δδ = A x + F y`` applied by the caller.
    """
    cw, sw = math.cos(omega), math.sin(omega)
    cO, sO = math.cos(Omega), math.sin(Omega)
    ci = math.cos(inc)
    A = a * (cw * cO - sw * sO * ci)
    B = a * (cw * sO + sw * cO * ci)
    F = a * (-sw * cO - cw * sO * ci)
    G = a * (-sw * sO + cw * cO * ci)
    return A, B, F, G


def oracle_astro_along_scan(
    *,
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    period_yr: float,
    ecc: float,
    omega_primary: float,
    inc: float,
    Omega: float,
    tp_mjd: float,
    a_phot_mas: float,
    plx_mas: float,
    pmra_masyr: float,
    pmdec_masyr: float,
    ra_offset_mas: float,
    dec_offset_mas: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Independent textbook along-scan model (mas).

    Uses PRIMARY-frame ω and POSITIVE ``a_phot_mas`` — the engine's internal
    ``ω→ω+π`` plus negative amplitude reduce to exactly this (see module
    docstring).  Sky mapping and along-scan / 5-parameter terms follow the
    project convention (CCW-from-north ψ; pmra is μ_α* already × cos δ;
    parallax additive ``plx × factor``).
    """
    nu = _oracle_true_anomaly(
        t_mjd, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd,
    )
    # Elliptical rectangular coordinates normalised by a (r = (1-e²)/(1+e cosν)).
    r = (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)

    A, B, F, G = _oracle_thiele_innes(
        a=a_phot_mas, omega=omega_primary, inc=inc, Omega=Omega,
    )
    d_ra = B * x + G * y   # Δα*
    d_dec = A * x + F * y  # Δδ

    psi = np.asarray(psi, dtype=float)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)

    model = d_ra * sin_psi + d_dec * cos_psi
    model = model + ra_offset_mas * sin_psi + dec_offset_mas * cos_psi
    dt_yr = (np.asarray(t_mjd, dtype=float) - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR
    model = model + (pmra_masyr * sin_psi + pmdec_masyr * cos_psi) * dt_yr
    model = model + plx_mas * np.asarray(parallax_factor_al, dtype=float)
    return model


def oracle_rv(
    *,
    t_mjd: np.ndarray,
    period_yr: float,
    ecc: float,
    omega_primary: float,
    tp_mjd: float,
    K_kms: float,
    gamma_kms: float,
) -> np.ndarray:
    """Independent textbook RV model (km/s): ``v = γ + K[cos(ν+ω) + e cosω]``
    with PRIMARY-frame ω (spectroscopic convention).  K is taken directly,
    independent of the engine's mass→K machinery.
    """
    nu = _oracle_true_anomaly(
        t_mjd, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd,
    )
    return gamma_kms + K_kms * (
        np.cos(nu + omega_primary) + ecc * math.cos(omega_primary)
    )


# ──────────────────────────────────────────────────────────────────────
# A0 ASTRO — densest convention stack first.
# ──────────────────────────────────────────────────────────────────────


def test_a0_astro_engine_agrees_with_independent_oracle(synth_astro_dataset):
    """The engine's astro forward at truth == the independent oracle, to a
    tight mas tolerance.  Disagreement => a shared convention/sign bug.
    """
    from orblet.model import (
        _along_scan_for_theta_campbell,
    )

    astro, truth = synth_astro_dataset(seed=7)
    t_mjd = np.asarray(astro["obs_time"], dtype=float)  # fixture is MJD already
    psi = np.asarray(astro["scan_angle"], dtype=float)
    plx_factor = np.asarray(astro["parallax_factor_al"], dtype=float)
    epoch_ref = float(truth["epoch_ref_mjd"])

    # Engine forward at truth (internal-theta path; primary-frame ω in,
    # engine applies ω+π and the negative photocenter amplitude itself).
    astro_theta = {
        "P_yr": truth["P_yr"], "e": truth["e"],
        "omega_rad": truth["omega_rad"], "inc_rad": truth["inc_rad"],
        "Omega_rad": truth["Omega_rad"], "m2_msun": truth["m_comp_msun"],
        "M_total_msun": truth["M_total_msun"], "plx_mas": truth["plx_mas"],
        "ra_offset_mas": 0.0, "dec_offset_mas": 0.0,
        "pmra_masyr": truth["pmra_masyr"], "pmdec_masyr": truth["pmdec_masyr"],
        "tp_mjd": truth["tp_mjd"],
    }
    engine_model = _along_scan_for_theta_campbell(
        astro_theta, t_mjd=t_mjd, psi=psi,
        parallax_factor_al=plx_factor, epoch_ref_mjd=epoch_ref,
    )

    # Independent oracle: primary-frame ω + POSITIVE a_phot (the +π/sign
    # reconciliation, done exactly once — see module docstring).
    oracle_model = oracle_astro_along_scan(
        t_mjd=t_mjd, psi=psi, parallax_factor_al=plx_factor,
        period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], inc=truth["inc_rad"],
        Omega=truth["Omega_rad"], tp_mjd=truth["tp_mjd"],
        a_phot_mas=truth["a_phot_mas"], plx_mas=truth["plx_mas"],
        pmra_masyr=truth["pmra_masyr"], pmdec_masyr=truth["pmdec_masyr"],
        ra_offset_mas=0.0, dec_offset_mas=0.0, epoch_ref_mjd=epoch_ref,
    )

    diff = np.max(np.abs(np.asarray(engine_model) - oracle_model))
    scale = np.max(np.abs(oracle_model))
    # Tolerance: 1e-6 mas absolute OR 1e-6 relative to the signal amplitude.
    # Both forwards iterate Kepler to ~1e-13, so the only residual is
    # float round-off; a convention bug would show as O(amplitude).
    assert diff <= 1e-6 + 1e-6 * scale, (
        f"engine astro forward disagrees with the independent oracle by "
        f"{diff:.3e} mas (signal amplitude {scale:.3f} mas) — a SHARED "
        "convention/sign bug; do NOT loosen this tolerance."
    )


def test_a0_astro_oracle_catches_a_wrong_omega_sign(synth_astro_dataset):
    """NEGATIVE control: feeding the oracle the WRONG ω convention (the raw
    companion-frame ω+π WITHOUT the compensating amplitude sign) makes it
    disagree with the engine — confirming the test has teeth (it is not
    trivially passing because both sides are identical code).
    """
    from orblet.model import (
        _along_scan_for_theta_campbell,
    )

    astro, truth = synth_astro_dataset(seed=11)
    t_mjd = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    plx_factor = np.asarray(astro["parallax_factor_al"], dtype=float)
    epoch_ref = float(truth["epoch_ref_mjd"])

    astro_theta = {
        "P_yr": truth["P_yr"], "e": truth["e"],
        "omega_rad": truth["omega_rad"], "inc_rad": truth["inc_rad"],
        "Omega_rad": truth["Omega_rad"], "m2_msun": truth["m_comp_msun"],
        "M_total_msun": truth["M_total_msun"], "plx_mas": truth["plx_mas"],
        "ra_offset_mas": 0.0, "dec_offset_mas": 0.0,
        "pmra_masyr": truth["pmra_masyr"], "pmdec_masyr": truth["pmdec_masyr"],
        "tp_mjd": truth["tp_mjd"],
    }
    engine_model = _along_scan_for_theta_campbell(
        astro_theta, t_mjd=t_mjd, psi=psi,
        parallax_factor_al=plx_factor, epoch_ref_mjd=epoch_ref,
    )
    # WRONG: ω+π but amplitude left POSITIVE (the un-cancelled half of the
    # documented pair).  The orbital part should flip sign and disagree.
    wrong_oracle = oracle_astro_along_scan(
        t_mjd=t_mjd, psi=psi, parallax_factor_al=plx_factor,
        period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"] + math.pi, inc=truth["inc_rad"],
        Omega=truth["Omega_rad"], tp_mjd=truth["tp_mjd"],
        a_phot_mas=truth["a_phot_mas"], plx_mas=truth["plx_mas"],
        pmra_masyr=truth["pmra_masyr"], pmdec_masyr=truth["pmdec_masyr"],
        ra_offset_mas=0.0, dec_offset_mas=0.0, epoch_ref_mjd=epoch_ref,
    )
    diff = np.max(np.abs(np.asarray(engine_model) - wrong_oracle))
    assert diff > 1.0, (
        "the wrong-ω oracle should disagree with the engine by >1 mas; "
        "if it does not, the test lacks teeth."
    )


# ──────────────────────────────────────────────────────────────────────
# A0 RV — primary-frame ω, K taken directly.
# ──────────────────────────────────────────────────────────────────────


def test_a0_rv_engine_agrees_with_independent_oracle(synth_rv_dataset):
    """The engine's RV forward at truth == the independent oracle, to a
    tight km/s tolerance.  The engine drives K through its mass→K machinery;
    we invert that ONCE to feed the mass that yields truth K, then compare.
    """
    from orblet.model import (
        rv_model, _semi_amplitude_kms,
    )

    prepared = synth_rv_dataset(
        seed=5, n_epochs=30, P_yr=1.3, e=0.35,
        omega_rad=0.9, tau=0.4, K_kms=7.0, gamma_kms=-12.0, sigma_kms=0.2,
    )
    tr = prepared["_truth"]
    t_mjd = np.asarray(prepared["epochs_mjd"], dtype=float)
    epoch_ref = float(tr["epoch_ref_mjd"])

    # Invert the engine's K(mass) ONCE: K is LINEAR in mass_msun, so the mass
    # that yields truth K at an arbitrary M_total is K_truth / K(unit mass).
    M_total = 1.0
    K_unit = _semi_amplitude_kms(
        mass_msun=1.0, period_yr=tr["P_yr"], ecc=tr["e"], M_total_msun=M_total,
    )
    mass_for_truth_K = tr["K_kms"] / K_unit

    engine_rv = rv_model(
        t_mjd, period_yr=tr["P_yr"], ecc=tr["e"], omega_rad=tr["omega_rad"],
        tau=tr["tau"], mass_msun=mass_for_truth_K, M_msun=M_total,
        offset_kms=tr["gamma_kms"], epoch_ref_mjd=epoch_ref,
    )

    oracle_rv_vals = oracle_rv(
        t_mjd=t_mjd, period_yr=tr["P_yr"], ecc=tr["e"],
        omega_primary=tr["omega_rad"], tp_mjd=tr["tp_mjd"],
        K_kms=tr["K_kms"], gamma_kms=tr["gamma_kms"],
    )

    diff = np.max(np.abs(np.asarray(engine_rv) - oracle_rv_vals))
    scale = np.max(np.abs(oracle_rv_vals))
    assert diff <= 1e-6 + 1e-6 * scale, (
        f"engine RV forward disagrees with the independent oracle by "
        f"{diff:.3e} km/s (signal {scale:.3f} km/s) — a SHARED convention "
        "bug; do NOT loosen this tolerance."
    )


# ──────────────────────────────────────────────────────────────────────
# A0 STRENGTHENING — asymmetric-truth sweep + negative-control battery.
#
# The single-truth tests above can be passed by a sign bug that CANCELS
# at that one geometry (i=90° zeros every cos i term; ω/Ω=0 zeros the a
# term; e=0 removes the e·cosω RV term).  The sweep below evaluates the
# SAME engine/oracle agreement over a curated grid of truths chosen so
# EVERY cos/sin term is non-zero and in both hemispheres, and the
# control battery perturbs each convention degree-of-freedom (DOF)
# individually in the oracle and asserts the engine DISAGREES — i.e. the
# engine is genuinely SENSITIVE to that DOF at the chosen truth, so the
# agreement test has provable teeth.  No data files read.
# ──────────────────────────────────────────────────────────────────────


def _a_phot_mas_for(*, P_yr: float, m_comp: float, M_total: float, plx: float) -> float:
    """Photocenter semi-major axis (mas), matching the engine's definition.

    The engine derives ``a_au = (P_yr² · M_total)^(1/3)`` and
    ``a_phot = (m_comp / M_total) · a_au · plx`` (positive; the engine
    carries the sign internally — see the module docstring).  Computed
    here identically so the oracle and the engine use the SAME amplitude,
    isolating the convention/sign comparison from any axis mismatch.
    """
    a_au = (P_yr ** 2 * M_total) ** (1.0 / 3.0)
    return (m_comp / M_total) * a_au * plx


def _engine_astro_forward(truth: dict, *, t_mjd, psi, plx_factor, epoch_ref):
    """Run the engine's internal-theta astro forward at ``truth`` (mas)."""
    from orblet.model import (
        _along_scan_for_theta_campbell,
    )

    astro_theta = {
        "P_yr": truth["P_yr"], "e": truth["e"],
        "omega_rad": truth["omega_rad"], "inc_rad": truth["inc_rad"],
        "Omega_rad": truth["Omega_rad"], "m2_msun": truth["m_comp_msun"],
        "M_total_msun": truth["M_total_msun"], "plx_mas": truth["plx_mas"],
        "ra_offset_mas": 0.0, "dec_offset_mas": 0.0,
        "pmra_masyr": truth["pmra_masyr"], "pmdec_masyr": truth["pmdec_masyr"],
        "tp_mjd": truth["tp_mjd"],
    }
    return np.asarray(_along_scan_for_theta_campbell(
        astro_theta, t_mjd=t_mjd, psi=psi,
        parallax_factor_al=plx_factor, epoch_ref_mjd=epoch_ref,
    ))


# Curated ~12-case asymmetric grid.  Each axis avoids the symmetric
# values that would zero a term: i ∉ {0,90,180}° and spans BOTH
# hemispheres; ω ∉ {0,90,180,270}°; Ω likewise; e spans 0.1–0.85; PM is
# nonzero; two sky positions.  Collectively every cos/sin term is
# non-degenerate in at least several cases, so a single-term sign bug
# cannot hide.
_ASTRO_SWEEP = [
    # (i_deg, omega_deg, Omega_deg, e, pmra, pmdec, ra_deg, dec_deg)
    (37.0, 23.0, 41.0, 0.10, 2.0, -3.0, 100.0, 20.0),
    (63.0, 147.0, 199.0, 0.40, -4.0, 1.5, 100.0, 20.0),
    (118.0, 251.0, 313.0, 0.70, 3.0, 5.0, 100.0, 20.0),
    (152.0, 23.0, 199.0, 0.85, -1.0, -2.0, 100.0, 20.0),
    (37.0, 147.0, 313.0, 0.40, 5.0, -4.0, 250.0, -55.0),
    (63.0, 251.0, 41.0, 0.70, -3.0, 2.0, 250.0, -55.0),
    (118.0, 23.0, 199.0, 0.10, 1.0, 4.0, 250.0, -55.0),
    (152.0, 147.0, 41.0, 0.40, -5.0, -1.0, 100.0, 20.0),
    (37.0, 251.0, 313.0, 0.70, 4.0, 3.0, 100.0, 20.0),
    (63.0, 23.0, 41.0, 0.85, -2.0, -5.0, 250.0, -55.0),
    (118.0, 147.0, 199.0, 0.40, 3.0, 1.0, 100.0, 20.0),
    (152.0, 251.0, 313.0, 0.10, -4.0, 2.0, 250.0, -55.0),
]


def _astro_truth(i_deg, omega_deg, Omega_deg, e, pmra, pmdec, ra_deg, dec_deg):
    """Assemble an astro truth dict for one sweep case (fixed P/masses/plx)."""
    P_yr = 800.0 / DAYS_PER_KEPLER_YEAR
    m_comp, M_total, plx = 0.5, 1.5, 5.0
    epoch_ref = 57388.5
    return {
        "P_yr": P_yr, "e": e,
        "omega_rad": math.radians(omega_deg), "inc_rad": math.radians(i_deg),
        "Omega_rad": math.radians(Omega_deg),
        "m_comp_msun": m_comp, "M_total_msun": M_total, "plx_mas": plx,
        "pmra_masyr": pmra, "pmdec_masyr": pmdec,
        "tp_mjd": 0.4 * 800.0 + epoch_ref,
        "epoch_ref_mjd": epoch_ref,
        "a_phot_mas": _a_phot_mas_for(
            P_yr=P_yr, m_comp=m_comp, M_total=M_total, plx=plx,
        ),
    }


@pytest.mark.parametrize("case", _ASTRO_SWEEP)
def test_a0_astro_agreement_sweep(case, synth_astro_dataset):
    """Engine astro forward == independent oracle across the asymmetric grid.

    Borrows ONLY the observing geometry (t, ψ, parallax factor) from the
    fixture; the orbit params are swept here so every cos/sin term is
    non-degenerate.  Disagreement => a shared convention/sign bug.
    """
    astro, _ = synth_astro_dataset(seed=7)
    t_mjd = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    plx_factor = np.asarray(astro["parallax_factor_al"], dtype=float)

    truth = _astro_truth(*case)
    epoch_ref = truth["epoch_ref_mjd"]

    engine_model = _engine_astro_forward(
        truth, t_mjd=t_mjd, psi=psi, plx_factor=plx_factor, epoch_ref=epoch_ref,
    )
    oracle_model = oracle_astro_along_scan(
        t_mjd=t_mjd, psi=psi, parallax_factor_al=plx_factor,
        period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], inc=truth["inc_rad"],
        Omega=truth["Omega_rad"], tp_mjd=truth["tp_mjd"],
        a_phot_mas=truth["a_phot_mas"], plx_mas=truth["plx_mas"],
        pmra_masyr=truth["pmra_masyr"], pmdec_masyr=truth["pmdec_masyr"],
        ra_offset_mas=0.0, dec_offset_mas=0.0, epoch_ref_mjd=epoch_ref,
    )

    diff = np.max(np.abs(engine_model - oracle_model))
    scale = np.max(np.abs(oracle_model))
    assert diff <= 1e-6 + 1e-6 * scale, (
        f"engine astro forward disagrees with the independent oracle by "
        f"{diff:.3e} mas (signal {scale:.3f} mas) at i={case[0]} ω={case[1]} "
        f"Ω={case[2]} e={case[3]} — a SHARED convention/sign bug; do NOT "
        "loosen this tolerance."
    )


# A single NON-degenerate truth for the control battery — every term is
# active (i off-axis in the southern hemisphere, ω/Ω off the cardinal
# values, moderate e, nonzero PM).
_CTRL_ASTRO_CASE = (118.0, 251.0, 313.0, 0.70, 3.0, 5.0, 100.0, 20.0)


def _oracle_astro_swapped(*, t_mjd, psi, parallax_factor_al, period_yr, ecc,
                          omega_primary, inc, Omega, tp_mjd, a_phot_mas,
                          plx_mas, pmra_masyr, pmdec_masyr,
                          ra_offset_mas, dec_offset_mas, epoch_ref_mjd):
    """Oracle variant with the A↔F / B↔G sky mapping SWAPPED.

    The engine uses ``Δα* = B x + G y``, ``Δδ = A x + F y``.  This variant
    builds ``Δα* = A x + F y``, ``Δδ = B x + G y`` instead — a wrong sky
    projection — so the engine must disagree, proving the projection
    assignment is pinned.
    """
    nu = _oracle_true_anomaly(t_mjd, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd)
    r = (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(nu))
    x, y = r * np.cos(nu), r * np.sin(nu)
    A, B, F, G = _oracle_thiele_innes(
        a=a_phot_mas, omega=omega_primary, inc=inc, Omega=Omega,
    )
    d_ra = A * x + F * y   # SWAPPED (should be B x + G y)
    d_dec = B * x + G * y  # SWAPPED (should be A x + F y)
    psi = np.asarray(psi, dtype=float)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    model = d_ra * sin_psi + d_dec * cos_psi
    model = model + ra_offset_mas * sin_psi + dec_offset_mas * cos_psi
    dt_yr = (np.asarray(t_mjd, dtype=float) - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR
    model = model + (pmra_masyr * sin_psi + pmdec_masyr * cos_psi) * dt_yr
    model = model + plx_mas * np.asarray(parallax_factor_al, dtype=float)
    return model


# Each entry: (label, kwarg-overrides-or-variant).  A perturbation is a
# dict applied on top of the truth kwargs, OR the sentinel "swap" / "proj"
# selecting a structural oracle variant.
def _astro_controls(truth):
    pi = math.pi
    return [
        ("omega+pi", {"omega_primary": truth["omega_rad"] + pi}),
        ("inc->pi-inc", {"inc": pi - truth["inc_rad"]}),
        ("Omega+pi", {"Omega": truth["Omega_rad"] + pi}),
        ("negate_pmra", {"pmra_masyr": -truth["pmra_masyr"]}),
        ("negate_pmdec", {"pmdec_masyr": -truth["pmdec_masyr"]}),
        ("negate_plx_factor", "negate_plx_factor"),
        ("swap_AF_BG", "swap"),
    ]


@pytest.mark.parametrize(
    "label", [c[0] for c in _astro_controls(_astro_truth(*_CTRL_ASTRO_CASE))]
)
def test_a0_astro_negative_control_battery(label, synth_astro_dataset):
    """NEGATIVE controls: each astro convention DOF, perturbed alone in the
    oracle, makes the engine DISAGREE by >1 mas — proving the engine is
    SENSITIVE to that DOF (the agreement test has teeth there).
    """
    astro, _ = synth_astro_dataset(seed=11)
    t_mjd = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    plx_factor = np.asarray(astro["parallax_factor_al"], dtype=float)

    truth = _astro_truth(*_CTRL_ASTRO_CASE)
    epoch_ref = truth["epoch_ref_mjd"]
    engine_model = _engine_astro_forward(
        truth, t_mjd=t_mjd, psi=psi, plx_factor=plx_factor, epoch_ref=epoch_ref,
    )

    base = dict(
        t_mjd=t_mjd, psi=psi, parallax_factor_al=plx_factor,
        period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], inc=truth["inc_rad"],
        Omega=truth["Omega_rad"], tp_mjd=truth["tp_mjd"],
        a_phot_mas=truth["a_phot_mas"], plx_mas=truth["plx_mas"],
        pmra_masyr=truth["pmra_masyr"], pmdec_masyr=truth["pmdec_masyr"],
        ra_offset_mas=0.0, dec_offset_mas=0.0, epoch_ref_mjd=epoch_ref,
    )
    spec = dict(_astro_controls(truth))[label]

    if spec == "swap":
        perturbed = _oracle_astro_swapped(**base)
    elif spec == "negate_plx_factor":
        kw = dict(base)
        kw["parallax_factor_al"] = -np.asarray(plx_factor)
        perturbed = oracle_astro_along_scan(**kw)
    else:
        kw = dict(base)
        kw.update(spec)
        perturbed = oracle_astro_along_scan(**kw)

    diff = np.max(np.abs(engine_model - perturbed))
    assert diff > 1.0, (
        f"control {label!r}: perturbed oracle should disagree with the engine "
        f"by >1 mas but max|Δ|={diff:.3e} mas — the engine is INSENSITIVE to "
        f"this DOF at this truth (degenerate truth or a real bug)."
    )


# ──────────────────────────────────────────────────────────────────────
# A0 RV — agreement sweep + negative-control battery (none existed before).
# ──────────────────────────────────────────────────────────────────────


def _engine_rv_forward(*, t_mjd, P_yr, e, omega_rad, tau, K_kms,
                       gamma_kms, epoch_ref):
    """Run the engine RV forward at K_kms by inverting K(mass) ONCE (linear)."""
    from orblet.model import (
        rv_model, _semi_amplitude_kms,
    )

    M_total = 1.0
    K_unit = _semi_amplitude_kms(
        mass_msun=1.0, period_yr=P_yr, ecc=e, M_total_msun=M_total,
    )
    mass_for_truth_K = K_kms / K_unit
    return np.asarray(rv_model(
        t_mjd, period_yr=P_yr, ecc=e, omega_rad=omega_rad,
        tau=tau, mass_msun=mass_for_truth_K, M_msun=M_total,
        offset_kms=gamma_kms, epoch_ref_mjd=epoch_ref,
    ))


# RV sweep: ω off the cardinal values, e spanning 0.1–0.85, varied K/γ/τ.
# (i, Ω, sky position do not enter the RV forward.)
_RV_SWEEP = [
    # (omega_deg, e, tau, K_kms, gamma_kms)
    (23.0, 0.10, 0.15, 7.0, -12.0),
    (147.0, 0.40, 0.40, 5.0, 3.0),
    (251.0, 0.70, 0.65, 9.0, 0.0),
    (23.0, 0.85, 0.90, 4.0, 20.0),
    (147.0, 0.10, 0.40, 6.0, -5.0),
    (251.0, 0.40, 0.15, 8.0, 11.0),
]


def _rv_truth(omega_deg, e, tau, K_kms, gamma_kms):
    """Assemble an RV truth dict for one sweep case (fixed P/epoch_ref)."""
    P_yr = 1.3
    epoch_ref = 60000.0
    P_days = P_yr * DAYS_PER_KEPLER_YEAR
    return {
        "P_yr": P_yr, "e": e, "omega_rad": math.radians(omega_deg),
        "tau": tau, "tp_mjd": tau * P_days + epoch_ref,
        "K_kms": K_kms, "gamma_kms": gamma_kms,
        "epoch_ref_mjd": epoch_ref, "P_days": P_days,
    }


@pytest.mark.parametrize("case", _RV_SWEEP)
def test_a0_rv_agreement_sweep(case, synth_rv_dataset):
    """Engine RV forward == independent oracle across the asymmetric grid.

    Borrows ONLY the epoch times from the fixture; orbit params swept here.
    """
    prepared = synth_rv_dataset(
        seed=5, n_epochs=30, P_yr=1.3, e=0.35,
        omega_rad=0.9, tau=0.4, K_kms=7.0, gamma_kms=-12.0, sigma_kms=0.2,
    )
    t_mjd = np.asarray(prepared["epochs_mjd"], dtype=float)

    truth = _rv_truth(*case)
    engine_rv = _engine_rv_forward(
        t_mjd=t_mjd, P_yr=truth["P_yr"], e=truth["e"],
        omega_rad=truth["omega_rad"], tau=truth["tau"], K_kms=truth["K_kms"],
        gamma_kms=truth["gamma_kms"], epoch_ref=truth["epoch_ref_mjd"],
    )
    oracle_vals = oracle_rv(
        t_mjd=t_mjd, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    diff = np.max(np.abs(engine_rv - oracle_vals))
    scale = np.max(np.abs(oracle_vals))
    assert diff <= 1e-6 + 1e-6 * scale, (
        f"engine RV forward disagrees with the independent oracle by "
        f"{diff:.3e} km/s (signal {scale:.3f} km/s) at ω={case[0]} e={case[1]} "
        "— a SHARED convention bug; do NOT loosen this tolerance."
    )


def _oracle_rv_drop_ecosw(*, t_mjd, period_yr, ecc, omega_primary, tp_mjd,
                          K_kms, gamma_kms):
    """Oracle RV variant that DROPS the ``e·cosω`` offset term (wrong)."""
    nu = _oracle_true_anomaly(t_mjd, period_yr=period_yr, ecc=ecc, tp_mjd=tp_mjd)
    return gamma_kms + K_kms * np.cos(nu + omega_primary)


_CTRL_RV_CASE = (251.0, 0.70, 0.65, 9.0, 0.0)


def _rv_controls(truth):
    return [
        ("omega+pi", {"omega_primary": truth["omega_rad"] + math.pi}),
        ("negate_K", {"K_kms": -truth["K_kms"]}),
        ("drop_ecosw", "drop_ecosw"),
        ("tp+P/2", {"tp_mjd": truth["tp_mjd"] + 0.5 * truth["P_days"]}),
    ]


@pytest.mark.parametrize(
    "label", [c[0] for c in _rv_controls(_rv_truth(*_CTRL_RV_CASE))]
)
def test_a0_rv_negative_control_battery(label, synth_rv_dataset):
    """NEGATIVE controls: each RV convention DOF, perturbed alone in the
    oracle, makes the engine DISAGREE by >0.5 km/s — proving sensitivity.
    """
    prepared = synth_rv_dataset(
        seed=5, n_epochs=30, P_yr=1.3, e=0.35,
        omega_rad=0.9, tau=0.4, K_kms=7.0, gamma_kms=-12.0, sigma_kms=0.2,
    )
    t_mjd = np.asarray(prepared["epochs_mjd"], dtype=float)

    truth = _rv_truth(*_CTRL_RV_CASE)
    engine_rv = _engine_rv_forward(
        t_mjd=t_mjd, P_yr=truth["P_yr"], e=truth["e"],
        omega_rad=truth["omega_rad"], tau=truth["tau"], K_kms=truth["K_kms"],
        gamma_kms=truth["gamma_kms"], epoch_ref=truth["epoch_ref_mjd"],
    )
    base = dict(
        t_mjd=t_mjd, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    spec = dict(_rv_controls(truth))[label]
    if spec == "drop_ecosw":
        perturbed = _oracle_rv_drop_ecosw(**base)
    else:
        kw = dict(base)
        kw.update(spec)
        perturbed = oracle_rv(**kw)

    diff = np.max(np.abs(engine_rv - perturbed))
    assert diff > 0.5, (
        f"control {label!r}: perturbed oracle should disagree with the engine "
        f"by >0.5 km/s but max|Δ|={diff:.3e} km/s — the engine is INSENSITIVE "
        f"to this DOF at this truth (degenerate truth or a real bug)."
    )
