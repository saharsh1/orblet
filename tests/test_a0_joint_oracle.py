"""A0 JOINT oracle — cross-channel convention keystone for the Tier-2 joint.

Why this test exists
--------------------
Tier 2 (the marginalised-amplitude joint, ``fit_joint_orbit``) ties the RV
and astrometry channels through ONE shared photocentre amplitude ``a₁`` and a
shared ``(ω, i)``.  Before that engine is built, this module pins — with an
INDEPENDENT textbook forward (no engine forward imported into the oracle) —
that the engine's EXISTING building blocks already agree on the conventions
ACROSS both channels.  A wrong convention here (an ω-frame mismatch, a
``sin i`` vs ``cos i`` slip, a units / parallax-bridge error) is exactly the
"simulator == fitter" blind spot the A0 oracle exists to catch: every other
fidelity test scores the engine against data made by the engine's own sibling
simulator, so a convention bug SHARED by both is invisible to them.

The oracle is independent
-------------------------
This module reuses the from-scratch scaffolding of
``tests/test_a0_independent_orbit_oracle.py``:

- ``_oracle_solve_kepler`` — Newton-iterates ``M = E − e sin E`` directly
  (NOT ``orblet.kepler.solve_kepler``);
- ``_oracle_true_anomaly`` — half-angle ``ν`` from that ``E``;
- ``_oracle_thiele_innes`` — textbook A,B,F,G (Hilditch 2001 §5.12) from
  Campbell elements in the PRIMARY frame with a POSITIVE ``a`` (the already-
  reconciled ω-frame / sign convention; see that module's docstring);
- ``oracle_rv`` — textbook ``v = γ + K[cos(ν+ω) + e cosω]`` with K taken
  directly.

The ONLY shared atoms are unit constants (``DAYS_PER_KEPLER_YEAR``,
``AU_M``) and ``numpy`` — never an engine forward.  Each identity asserts
``engine_value == oracle_value``.

The five identities (clean noiseless truth, β = 0, i≈60°, e≈0.3)
----------------------------------------------------------------
1. Sky-TI identity: the engine's TI built from the geometry (recovered from
   ``astro.forward.campbell_xy``) == the oracle's independent TI, per
   coefficient, rtol ≤ 1e-10.
2. RV-K identity: the engine's K (``rv.forward._semi_amplitude_kms`` from the
   geometry) == the oracle's independent ``K = 2π a₁ sin i /(P√(1−e²))``
   (AU/yr→km/s), rtol ≤ 1e-10.
3. Keystone (the load-bearing one): the SAME truth ``(a₁, ω, i)`` is
   consistent across channels — ω from the RV linear amplitudes
   (``recover_omega``) == ω from the astro TI (``ti_to_kepler``) == truth ω
   (primary frame, wrap-aware), and ``a₁`` implied by the RV (K → a₁ sin i →
   a₁ with the truth i) == ``a_phot/parallax`` from the astro (a_phot = a₁
   for β = 0), rtol ≤ 1e-10.
4. Marginal-evidence reduction: on NOISELESS oracle data, ``linear_solve_rv``
   and ``linear_solve_ti`` each recover the truth linear amplitudes (β̂ ≈
   truth) to tight tolerance and have finite ``log_evidence``.
5. Mirror / RV sign: the node mirror ``(ω,Ω) → (ω+π, Ω+π)`` leaves the astro
   along-scan model BIT-identical but flips the SIGN of the RV ``(C, S)`` —
   astrometry cannot break the mirror, RV does.

All inputs synthetic; no data files read.  No engine / source edited.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orblet.constants import AU_M, DAYS_PER_KEPLER_YEAR

# Reuse the INDEPENDENT oracle scaffolding (from-scratch Kepler + TI + RV).
# These helpers do NOT import any engine forward — see that module's
# docstring; importing them keeps the oracle genuinely independent.
from test_a0_independent_orbit_oracle import (  # noqa: E402  (tests/ on sys.path)
    _oracle_thiele_innes,
    _oracle_true_anomaly,
    oracle_rv,
)


# ──────────────────────────────────────────────────────────────────────
# A clean, unambiguous, NON-DEGENERATE shared truth.
#
# i≈60° (cos i and sin i both well away from 0/1 ⇒ the keystone is
# non-degenerate and the sin i / cos i slip is detectable), e≈0.3 (the
# e·cosω RV term is active and the Kepler eccentric anomaly is non-trivial),
# ω/Ω generic (off every cardinal value ⇒ every cos/sin term is live), a₁ a
# few mas·AU, parallax a few mas.  β = 0 ⇒ photocentre a_phot = a₁.
# ──────────────────────────────────────────────────────────────────────

_TRUTH = {
    "P_days": 540.0,
    "e": 0.3,
    "omega_rad": 0.9,        # generic, off cardinal
    "Omega_rad": 2.1,        # generic, off cardinal
    "inc_rad": math.radians(60.0),  # i ≈ 60° (away from edge-/face-on)
    "tau": 0.37,
    "m_comp_msun": 2.0,
    "M_total_msun": 3.0,
    "plx_mas": 5.0,
    "gamma_kms": -7.0,
    "epoch_ref_mjd": 57388.5,  # MJD_J2016_TCB
}


def _derived_truth(**overrides) -> dict:
    """Augment ``_TRUTH`` with derived geometry the oracle/engine compare on.

    All quantities here are INDEPENDENT textbook algebra (Kepler III for the
    relative axis, the mass-ratio split for ``a₁``, the AU/yr→km/s factor for
    K) — no engine import.  ``β = 0`` ⇒ ``a_phot = a₁``.

    ``overrides`` replace entries of ``_TRUTH`` BEFORE the derived geometry is
    recomputed, so the keystone sweep can vary ``inc_rad`` / ``e`` while every
    derived quantity (a₁, a_phot, K, tp) stays self-consistent.
    """
    t = dict(_TRUTH)
    t.update(overrides)
    P_yr = t["P_days"] / DAYS_PER_KEPLER_YEAR
    # Kepler III with TOTAL mass: a_rel [AU] = (P_yr² · M_total)^(1/3).
    a_rel_au = (P_yr ** 2 * t["M_total_msun"]) ** (1.0 / 3.0)
    # Primary (photocentre, β=0) semi-major axis: a₁ = a_rel · m_comp/M_total.
    a1_au = a_rel_au * t["m_comp_msun"] / t["M_total_msun"]
    a_phot_mas = a1_au * t["plx_mas"]
    tp_mjd = t["tau"] * t["P_days"] + t["epoch_ref_mjd"]

    # Independent K: K = 2π · a₁·sin i / (P · √(1−e²)), in AU/yr → km/s.
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    au_yr_to_kms = AU_M / 1000.0 / sec_per_yr
    K_kms = (
        2.0 * math.pi * a1_au * math.sin(t["inc_rad"])
        / (P_yr * math.sqrt(1.0 - t["e"] ** 2))
        * au_yr_to_kms
    )

    t.update(
        P_yr=P_yr, a_rel_au=a_rel_au, a1_au=a1_au, a_phot_mas=a_phot_mas,
        tp_mjd=tp_mjd, K_kms=K_kms,
    )
    return t


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle (rad) into ``(−π, π]``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _ang_diff_mod_pi(a: float, b: float) -> float:
    """Smallest |a − b| modulo π (rad).

    The TI inversion (``ti_to_kepler``) recovers ω only up to the node mirror
    ``(ω, Ω) → (ω+π, Ω+π)`` (two degenerate modes 180° apart — see the
    ``ti_to_kepler`` docstring and identity 5).  Comparing modulo π is the
    correct wrap-aware test for "same ω up to that mirror".
    """
    d = (float(a) - float(b)) % math.pi
    return min(d, math.pi - d)


# ──────────────────────────────────────────────────────────────────────
# Identity 1 — sky-Thiele-Innes: engine TI == oracle TI, per coefficient.
# ──────────────────────────────────────────────────────────────────────


def _engine_ti_coeffs_from_geometry(truth: dict) -> tuple[float, float, float, float]:
    """Recover the engine's (A, B, F, G) photocentre TI amplitudes (mas).

    The engine's ``astro.forward.campbell_xy`` returns only the projected
    ``(Δα*, Δδ)`` per epoch, with ``Δα* = B x_orb + G y_orb`` and
    ``Δδ = A x_orb + F y_orb``.  To extract the FOUR engine coefficients we
    evaluate the engine forward at two epochs whose engine in-plane points
    ``(x_orb, y_orb)`` (from the engine atom ``_kepler_xy_orbit``) are
    linearly independent, then solve the resulting 2×2 systems.  Both atoms
    are ENGINE atoms — the oracle's own forward is never used here.
    """
    from orblet.model import (
        _kepler_xy_orbit,
        campbell_xy,
    )

    # Pick two epochs with well-separated orbital phase ⇒ independent (x, y).
    t_pair = np.array(
        [truth["tp_mjd"] + 0.13 * truth["P_days"],
         truth["tp_mjd"] + 0.61 * truth["P_days"]],
        dtype=float,
    )
    x_orb, y_orb = _kepler_xy_orbit(
        t_pair, period_yr=truth["P_yr"], ecc=truth["e"], tp_mjd=truth["tp_mjd"],
    )
    d_ra, d_dec = campbell_xy(
        t_pair,
        period_yr=truth["P_yr"], ecc=truth["e"],
        omega=truth["omega_rad"], inc=truth["inc_rad"],
        Omega=truth["Omega_rad"], tp_mjd=truth["tp_mjd"],
        m_comp_msun=truth["m_comp_msun"], M_total_msun=truth["M_total_msun"],
        plx_mas=truth["plx_mas"],
    )
    # 2×2 solves: [x y] [B;G] = d_ra ; [x y] [A;F] = d_dec.
    basis = np.column_stack([x_orb, y_orb])  # rows are epochs
    B_eng, G_eng = np.linalg.solve(basis, d_ra)
    A_eng, F_eng = np.linalg.solve(basis, d_dec)
    return float(A_eng), float(B_eng), float(F_eng), float(G_eng)


def test_a0_joint_sky_ti_engine_equals_oracle():
    """Identity 1: the engine's photocentre TI built from the geometry equals
    the INDEPENDENT oracle TI, per coefficient, rtol ≤ 1e-10.
    """
    truth = _derived_truth()
    A_eng, B_eng, F_eng, G_eng = _engine_ti_coeffs_from_geometry(truth)

    # Oracle TI: primary-frame ω, POSITIVE a_phot (β=0 ⇒ a_phot = a₁).
    A_o, B_o, F_o, G_o = _oracle_thiele_innes(
        a=truth["a_phot_mas"], omega=truth["omega_rad"],
        inc=truth["inc_rad"], Omega=truth["Omega_rad"],
    )

    for name, eng, ora in (
        ("A", A_eng, A_o), ("B", B_eng, B_o),
        ("F", F_eng, F_o), ("G", G_eng, G_o),
    ):
        assert math.isclose(eng, ora, rel_tol=1e-10, abs_tol=1e-12), (
            f"sky-TI coefficient {name}: engine {eng:.12e} != oracle "
            f"{ora:.12e} — a SHARED astro convention/sign bug; do NOT loosen."
        )


# ──────────────────────────────────────────────────────────────────────
# Identity 2 — RV K: engine K(geometry) == oracle K.
# ──────────────────────────────────────────────────────────────────────


def test_a0_joint_rv_K_engine_equals_oracle():
    """Identity 2: the engine's semi-amplitude K from the SAME geometry equals
    the independent oracle ``K = 2π a₁ sin i /(P√(1−e²))``, rtol ≤ 1e-10.
    """
    from orblet.model import _semi_amplitude_kms

    truth = _derived_truth()
    # The engine fixes sin i ≡ 1, so its mass argument is the PROJECTED mass
    # m_comp · sin i; feed that so the engine K carries the truth inclination.
    engine_K = _semi_amplitude_kms(
        mass_msun=truth["m_comp_msun"] * math.sin(truth["inc_rad"]),
        period_yr=truth["P_yr"], ecc=truth["e"],
        M_total_msun=truth["M_total_msun"],
    )
    assert math.isclose(engine_K, truth["K_kms"], rel_tol=1e-10), (
        f"RV K: engine {engine_K:.12e} != oracle {truth['K_kms']:.12e} km/s "
        "— a SHARED RV geometry/units bug; do NOT loosen."
    )


# ──────────────────────────────────────────────────────────────────────
# Identity 3 — KEYSTONE: the same (a₁, ω, i) recovered from BOTH channels.
# ──────────────────────────────────────────────────────────────────────


def _assert_keystone_consistency(truth: dict) -> None:
    """Identity 3 (keystone) for one truth dict: the SAME ``(a₁, ω, i)`` is
    consistent across both channels.

    ω: recovered from the RV Cartesian amplitudes ``(C, S) = (K cosω, K sinω)``
    via ``recover_omega`` == recovered from the astro TI via ``ti_to_kepler``
    == the truth ω (primary frame), wrap-aware (mod π for the TI node mirror).

    a₁: the RV implies ``a₁ = K · P √(1−e²) / (2π sin i)`` (km/s → AU/yr) with
    the truth i; the astro implies ``a_phot / plx`` (= a₁ for β = 0).  These
    must agree, rtol ≤ 1e-10.

    Factored out so the canonical case, the (i, e) sweep, and the negative
    control share ONE implementation (no convention re-derivation drift).
    """
    from orblet.solve.rv import recover_omega
    from orblet.elements import ti_to_kepler


    # ── RV side: build the truth Cartesian amplitudes, recover ω. ──
    C = truth["K_kms"] * math.cos(truth["omega_rad"])
    S = truth["K_kms"] * math.sin(truth["omega_rad"])
    beta_rv = np.array([truth["gamma_kms"], C, S], dtype=float)
    omega_rv = recover_omega(beta_rv)

    # ── Astro side: build the truth TI amplitudes, recover ω via ti_to_kepler.
    A_o, B_o, F_o, G_o = _oracle_thiele_innes(
        a=truth["a_phot_mas"], omega=truth["omega_rad"],
        inc=truth["inc_rad"], Omega=truth["Omega_rad"],
    )
    ti_chain = {
        "A_mas": np.array([A_o]), "B_mas": np.array([B_o]),
        "F_mas": np.array([F_o]), "G_mas": np.array([G_o]),
        "plx_mas": np.array([truth["plx_mas"]]),
        "m2_msun": np.array([truth["m_comp_msun"]]),
        "M_total_msun": np.array([truth["M_total_msun"]]),
    }
    recovered = ti_to_kepler(ti_chain)
    omega_astro = float(recovered["omega_rad"][0])

    # ω agreement (primary frame).  RV recovers ω exactly (atan2 of the truth
    # Cartesian pair); TI recovers it up to the node mirror (mod π).
    assert _wrap_to_pi(omega_rv - truth["omega_rad"]) == pytest.approx(
        0.0, abs=1e-10,
    ), (
        f"RV ω {omega_rv:.12e} != truth ω {truth['omega_rad']:.12e} — RV "
        "amplitude→ω convention bug; do NOT loosen."
    )
    assert _ang_diff_mod_pi(omega_astro, truth["omega_rad"]) <= 1e-10, (
        f"astro TI ω {omega_astro:.12e} != truth ω {truth['omega_rad']:.12e} "
        "(mod π for the node mirror) — astro TI→ω convention bug."
    )

    # ── a₁ agreement across channels. ──
    # RV → a₁: invert K = 2π a₁ sin i /(P √(1−e²)) (AU/yr→km/s).
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    au_yr_to_kms = AU_M / 1000.0 / sec_per_yr
    a1_from_rv = (
        truth["K_kms"] / au_yr_to_kms
        * truth["P_yr"] * math.sqrt(1.0 - truth["e"] ** 2)
        / (2.0 * math.pi * math.sin(truth["inc_rad"]))
    )
    # Astro → a₁ = a_phot / plx  (= the recovered photocentre axis in AU).
    a1_from_astro = float(recovered["a_phot_au"][0])

    assert math.isclose(a1_from_rv, a1_from_astro, rel_tol=1e-10), (
        f"a₁ cross-channel: RV implies {a1_from_rv:.12e} AU, astro implies "
        f"{a1_from_astro:.12e} AU — a SHARED amplitude / sin i / parallax-"
        "bridge convention bug; do NOT loosen."
    )
    # And both equal the construction truth a₁.
    assert math.isclose(a1_from_rv, truth["a1_au"], rel_tol=1e-10)
    assert math.isclose(a1_from_astro, truth["a1_au"], rel_tol=1e-10)


def test_a0_joint_keystone_cross_channel_consistency():
    """Identity 3 (keystone) at the canonical non-degenerate truth
    (i≈60°, e=0.3)."""
    _assert_keystone_consistency(_derived_truth())


@pytest.mark.parametrize("inc_deg, e", [(30.0, 0.1), (45.0, 0.5), (75.0, 0.6)])
def test_a0_joint_keystone_sweep(inc_deg, e):
    """Identity 3 across several non-degenerate (i, e): a coincidental
    cancellation at the single canonical truth cannot satisfy the contract."""
    _assert_keystone_consistency(
        _derived_truth(inc_rad=math.radians(inc_deg), e=e)
    )


def test_a0_joint_keystone_negative_control():
    """Teeth for identity 3: feeding the WRONG inclination to the RV→a₁
    inversion must BREAK the cross-channel a₁ agreement — the keystone is not
    vacuously satisfied for any ``sin i``."""
    from orblet.elements import ti_to_kepler

    truth = _derived_truth()
    # Correct astro a₁ (= a_phot / plx, recovered from the truth TI).
    A_o, B_o, F_o, G_o = _oracle_thiele_innes(
        a=truth["a_phot_mas"], omega=truth["omega_rad"],
        inc=truth["inc_rad"], Omega=truth["Omega_rad"],
    )
    ti_chain = {
        "A_mas": np.array([A_o]), "B_mas": np.array([B_o]),
        "F_mas": np.array([F_o]), "G_mas": np.array([G_o]),
        "plx_mas": np.array([truth["plx_mas"]]),
        "m2_msun": np.array([truth["m_comp_msun"]]),
        "M_total_msun": np.array([truth["M_total_msun"]]),
    }
    a1_from_astro = float(ti_to_kepler(ti_chain)["a_phot_au"][0])

    # RV → a₁ with a DELIBERATELY WRONG inclination (truth + 15°).
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    au_yr_to_kms = AU_M / 1000.0 / sec_per_yr
    wrong_inc = truth["inc_rad"] + math.radians(15.0)
    a1_from_rv_wrong = (
        truth["K_kms"] / au_yr_to_kms
        * truth["P_yr"] * math.sqrt(1.0 - truth["e"] ** 2)
        / (2.0 * math.pi * math.sin(wrong_inc))
    )
    assert not math.isclose(a1_from_rv_wrong, a1_from_astro, rel_tol=1e-6), (
        "keystone negative control FAILED: a wrong inclination still matched "
        "the cross-channel a₁ — identity 3 would be vacuous."
    )


# ──────────────────────────────────────────────────────────────────────
# Identity 4 — marginal-evidence reduction: the two Tier-1 cores recover the
# truth linear amplitudes on NOISELESS oracle data, with finite evidence.
# ──────────────────────────────────────────────────────────────────────


class _GaussianBetaPrior:
    """Minimal Gaussian β prior (``.mean`` / ``.cov``) for the linear cores.

    A broad, diagonal prior so the posterior is data-dominated (β̂ ≈ truth)
    yet ``log_evidence`` (which requires an informative prior) is finite.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)


def test_a0_joint_marginal_evidence_reduction_rv(synth_rv_dataset):
    """Identity 4 (RV): ``linear_solve_rv`` recovers the truth ``(γ, C, S)`` on
    NOISELESS oracle RV data and reports a finite ``log_evidence``.
    """
    from orblet.solve.rv import (
        _rv_design_matrix,
        linear_solve_rv,
    )

    truth = _derived_truth()
    prepared = synth_rv_dataset(
        seed=3, n_epochs=40, P_yr=truth["P_yr"], e=truth["e"],
        omega_rad=truth["omega_rad"], tau=truth["tau"], K_kms=truth["K_kms"],
        gamma_kms=truth["gamma_kms"], sigma_kms=0.2,
    )
    t_mjd = np.asarray(prepared["epochs_mjd"], dtype=float)

    # NOISELESS data from the INDEPENDENT oracle RV forward.
    d_clean = oracle_rv(
        t_mjd=t_mjd, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    sigma = np.full_like(d_clean, 0.2)

    X = _rv_design_matrix(
        t_mjd, period_yr=truth["P_yr"], ecc=truth["e"], tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )

    truth_beta = np.array([
        truth["gamma_kms"],
        truth["K_kms"] * math.cos(truth["omega_rad"]),
        truth["K_kms"] * math.sin(truth["omega_rad"]),
    ])
    prior = _GaussianBetaPrior(
        mean=np.zeros(3), cov=np.diag([1e6, 1e6, 1e6]),
    )
    sol = linear_solve_rv(d_clean, sigma, X, beta_prior=prior)

    np.testing.assert_allclose(sol.beta, truth_beta, rtol=1e-8, atol=1e-8)
    assert np.isfinite(sol.log_evidence)


def test_a0_joint_marginal_evidence_reduction_ti(synth_astro_dataset):
    """Identity 4 (astro): ``linear_solve_ti`` recovers the truth 9 amplitudes
    (the 4 truth TI + zero nuisances + truth plx) on NOISELESS oracle
    along-scan data and reports a finite ``log_evidence``.
    """
    from orblet.solve.astrometry import (
        _ti_design_matrix,
        linear_solve_ti,
    )

    truth = _derived_truth()
    # Borrow ONLY the observing geometry (epochs, ψ, parallax factor).
    astro, _ = synth_astro_dataset(seed=9)
    t_mjd = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    plx_factor = np.asarray(astro["parallax_factor_al"], dtype=float)

    # Truth TI amplitudes from the independent oracle (β=0 ⇒ a_phot = a₁).
    A_o, B_o, F_o, G_o = _oracle_thiele_innes(
        a=truth["a_phot_mas"], omega=truth["omega_rad"],
        inc=truth["inc_rad"], Omega=truth["Omega_rad"],
    )

    # NOISELESS along-scan data built from the INDEPENDENT oracle forward:
    #   d_ra = B x + G y ,  d_dec = A x + F y ; project + plx (no PM/offset).
    nu = _oracle_true_anomaly(
        t_mjd, period_yr=truth["P_yr"], ecc=truth["e"], tp_mjd=truth["tp_mjd"],
    )
    r = (1.0 - truth["e"] ** 2) / (1.0 + truth["e"] * np.cos(nu))
    x_orb, y_orb = r * np.cos(nu), r * np.sin(nu)
    d_ra = B_o * x_orb + G_o * y_orb
    d_dec = A_o * x_orb + F_o * y_orb
    d_clean = (
        d_ra * np.sin(psi) + d_dec * np.cos(psi)
        + truth["plx_mas"] * plx_factor
    )
    sigma = np.full_like(d_clean, 0.3)

    X = _ti_design_matrix(
        t_mjd, psi, plx_factor,
        f_per_day=1.0 / truth["P_days"], ecc=truth["e"], tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )

    # Fixed column order: [A, B, F, G, ra_offset, dec_offset, pmra, pmdec, plx].
    truth_beta = np.array([
        A_o, B_o, F_o, G_o, 0.0, 0.0, 0.0, 0.0, truth["plx_mas"],
    ])
    prior = _GaussianBetaPrior(
        mean=np.zeros(9), cov=np.diag(np.full(9, 1e6)),
    )
    sol = linear_solve_ti(d_clean, sigma, X, beta_prior=prior)

    np.testing.assert_allclose(sol.beta, truth_beta, rtol=1e-7, atol=1e-7)
    assert np.isfinite(sol.log_evidence)


# ──────────────────────────────────────────────────────────────────────
# Identity 5 — mirror / RV sign: the node mirror leaves the astro along-scan
# model bit-identical but flips the RV (C, S) sign.
# ──────────────────────────────────────────────────────────────────────


def test_a0_joint_node_mirror_astro_invariant_rv_flips():
    """Identity 5: ``(ω, Ω) → (ω+π, Ω+π)`` leaves the astro along-scan model
    BIT-identical (astrometry cannot break the mirror) but NEGATES the RV
    Cartesian amplitudes ``(C, S)`` (RV does break it).
    """
    truth = _derived_truth()

    # Borrow observing geometry from a deterministic epoch/scan grid (no
    # fixture needed — the mirror is a pure-forward statement).
    t_mjd = truth["epoch_ref_mjd"] + np.linspace(
        -1.5 * truth["P_days"], 1.5 * truth["P_days"], 64,
    )
    rng = np.random.default_rng(0)
    psi = rng.uniform(0.0, 2.0 * math.pi, size=t_mjd.size)
    plx_factor = rng.uniform(-1.0, 1.0, size=t_mjd.size)

    def _oracle_along_scan(omega, Omega):
        A, B, F, G = _oracle_thiele_innes(
            a=truth["a_phot_mas"], omega=omega, inc=truth["inc_rad"],
            Omega=Omega,
        )
        nu = _oracle_true_anomaly(
            t_mjd, period_yr=truth["P_yr"], ecc=truth["e"],
            tp_mjd=truth["tp_mjd"],
        )
        r = (1.0 - truth["e"] ** 2) / (1.0 + truth["e"] * np.cos(nu))
        x_orb, y_orb = r * np.cos(nu), r * np.sin(nu)
        d_ra = B * x_orb + G * y_orb
        d_dec = A * x_orb + F * y_orb
        return (
            d_ra * np.sin(psi) + d_dec * np.cos(psi)
            + truth["plx_mas"] * plx_factor
        )

    model_truth = _oracle_along_scan(truth["omega_rad"], truth["Omega_rad"])
    model_mirror = _oracle_along_scan(
        truth["omega_rad"] + math.pi, truth["Omega_rad"] + math.pi,
    )
    # Astro along-scan model: BIT-identical under the node mirror.
    np.testing.assert_allclose(model_mirror, model_truth, rtol=0.0, atol=1e-9)

    # RV (C, S) under the mirror ω → ω + π : both flip sign.
    C = truth["K_kms"] * math.cos(truth["omega_rad"])
    S = truth["K_kms"] * math.sin(truth["omega_rad"])
    C_mirror = truth["K_kms"] * math.cos(truth["omega_rad"] + math.pi)
    S_mirror = truth["K_kms"] * math.sin(truth["omega_rad"] + math.pi)
    assert math.isclose(C_mirror, -C, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(S_mirror, -S, rel_tol=1e-12, abs_tol=1e-12)
    # And the RV model itself differs (the mirror is physically breakable).
    rv_truth = oracle_rv(
        t_mjd=t_mjd, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    rv_mirror = oracle_rv(
        t_mjd=t_mjd, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"] + math.pi, tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    assert np.max(np.abs(rv_mirror - rv_truth)) > 0.5 * abs(truth["K_kms"]), (
        "the node mirror must change the RV curve (RV breaks the mirror); "
        "if it does not, the RV ω convention is wrong."
    )
