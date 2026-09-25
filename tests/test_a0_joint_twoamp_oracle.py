"""A0 TWO-AMPLITUDE joint oracle — the Tier-2 convention keystone.

Why this test exists
--------------------
Tier 2 (the marginalised-amplitude joint) samples ONE shared photocentre
amplitude ``a_phot`` and ONE spectroscopic amplitude ``K`` at a shared
``(ω, Ω, i)``, and marginalises the linear amplitudes analytically per
channel.  Before that sampler is built, this module pins — with an
INDEPENDENT textbook forward — that the engine's convention layer already
behaves correctly for the two-amplitude reduction: the reduced design
columns, their Klein-4 sign structure, the marginal-evidence invariance
under that group, and the export gauge fold.

Three-layer division of labour
------------------------------
The A0 keystone splits across three complementary layers, each blind to a
different failure mode:

- The engine atom :func:`...astro.forward._ti_constants_unit` + the ``6a``
  reduced-column checks pin the ABSOLUTE SIGNS of the Thiele-Innes /
  RV columns (a global sign flip would fail here).
- The ``6e`` forward-degeneracy tests pin the STRUCTURE of the Klein-4
  forward degeneracy — how the columns transform under ``ω → ω + π`` /
  ``Ω → Ω + π``.  They are STRUCTURALLY BLIND to a global sign error
  (they only compare a column against ± itself), which is exactly why the
  atom / ``6a`` layer is needed alongside them.
- The ``6f`` marginal-evidence tests pin the Klein-4 INVARIANCE of the
  per-channel log-evidence (the marginalised objective must not prefer
  one gauge image over another).
- The ``6g`` tests pin the EXPORT GAUGE FOLD: the unique non-negative
  ``(K > 0, a_phot > 0)`` representative and the map that folds any of the
  four gauge images back onto it.

U5 design (locked)
------------------
Amplitude non-negativity is a GAUGE FIX applied at CHAIN EXPORT — option
(a): the sampler runs with SYMMETRIC diagonal zero-mean Gaussian amplitude
priors (so ``K`` and ``a_phot`` may take either sign; the Klein-4 group
folds the four sign images together), and the non-negative representative
is selected only when the chain is exported.  It is NOT a truncated
(non-negative) prior.  The ``6f`` evidence invariance is the property that
makes this gauge fix legitimate; the ``6g`` fold is the export step.

Independence
------------
The independent oracle side reuses the from-scratch scaffolding of
``test_a0_independent_orbit_oracle.py`` (Kepler + Thiele-Innes + RV, none
of which import an engine forward) and the shared truth of
``test_a0_joint_oracle.py``.  The engine atom ``_ti_constants_unit`` is
exercised ONLY inside the reduced designs under test; every oracle-side
comparison uses the from-scratch ``_ti_constants_unit_oracle`` /
``_oracle_thiele_innes`` / ``oracle_rv``, so the oracle is never routed
through the engine atom.

All inputs synthetic; no data files read.  No engine / source edited.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orblet.constants import AU_M, DAYS_PER_KEPLER_YEAR

# Shared truth + wrap helpers from the A0 joint oracle (NOT re-derived).
# ``_TRUTH`` / ``_ang_diff_mod_pi`` are part of the reuse menu but not needed
# by this batch (angles here compare mod 2π via ``_wrap_to_pi``), so only the
# used names are imported to keep the module lint-clean.
from test_a0_joint_oracle import (  # noqa: E402  (tests/ on sys.path)
    _derived_truth,
    _wrap_to_pi,
)

# Independent from-scratch orbit scaffolding (no engine forward imported).
# ``_oracle_solve_kepler`` is reached transitively through
# ``_oracle_true_anomaly``; only the directly-called names are imported.
from test_a0_independent_orbit_oracle import (  # noqa: E402
    _oracle_thiele_innes,
    _oracle_true_anomaly,
    oracle_rv,
)

_TWO_PI = 2.0 * math.pi

# Convention grid: ω, Ω off the cardinal values spanning all four
# quadrants; i in both hemispheres and off {0, 90, 180}° (matches the
# ``test_campbell_primary_frame_noop`` grid so a single-term sign error
# cannot hide).
_OMEGA_DEG = (23.0, 147.0, 251.0)
_OMEGA_NODE_DEG = (41.0, 199.0, 313.0)
_INC_DEG = (37.0, 63.0, 118.0, 152.0)

# Fixed geometry for the unit-TI probe (same P / masses / plx as the noop
# grid, so ``a_mas`` is a well-conditioned non-unit amplitude to divide out).
_P_YR = 800.0 / DAYS_PER_KEPLER_YEAR
_M_COMP_MSUN = 0.5
_M_TOTAL_MSUN = 1.5
_PLX_MAS = 5.0


def _a_mas_probe() -> float:
    """The signed photocentre amplitude (mas) of the unit-TI probe geometry."""
    a_au = (_P_YR ** 2 * _M_TOTAL_MSUN) ** (1.0 / 3.0)
    return (_M_COMP_MSUN / _M_TOTAL_MSUN) * a_au * _PLX_MAS


# ──────────────────────────────────────────────────────────────────────
# In-file helpers with STABLE names (Batch 2 imports these; a separate
# heavy gate governs any promotion of them to ``src/``).
# ──────────────────────────────────────────────────────────────────────


class _GaussianBetaPrior:
    """Minimal Gaussian β prior (``.mean`` / ``.cov``) for the linear cores.

    A container only — the production ``linear_solve_*`` cores read
    ``.mean`` / ``.cov`` off it.  No convention lives here.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)


def _ti_constants_unit_oracle(
    omega: float, Omega: float, cos_i: float,
) -> tuple[float, float, float, float]:
    """Textbook UNIT-amplitude Thiele-Innes constants, written FROM SCRATCH.

    Independent of the engine atom
    :func:`...astro.forward._ti_constants_unit` (the A0 independence guard):
    a directed test asserts the two agree, so this copy must NOT delegate to
    the engine.  Primary-frame ω, node Ω CCW-from-north, ``cos_i`` passed
    directly, unit amplitude (Hilditch 2001 §5.12).
    """
    cw, sw = math.cos(omega), math.sin(omega)
    cO, sO = math.cos(Omega), math.sin(Omega)
    ci = float(cos_i)
    A = cw * cO - sw * sO * ci
    B = cw * sO + sw * cO * ci
    F = -sw * cO - cw * sO * ci
    G = -sw * sO + cw * cO * ci
    return A, B, F, G


def _reduced_rv_design(
    t_mjd: np.ndarray,
    *,
    P_yr: float,
    e: float,
    omega: float,
    tau: float,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Two-amplitude REDUCED RV design ``(n, 2)`` at a FIXED ``(P, e, ω, τ)``.

    Columns are the partials of ``v = γ + K[cos(ν + ω) + e cosω]`` w.r.t. the
    two reduced amplitudes ``β = (γ, K)`` at fixed ω:

        col 0 = ∂v/∂γ = 1
        col 1 = ∂v/∂K = cos(ν + ω) + e cosω,

    with ``ν`` from the INDEPENDENT ``_oracle_true_anomaly`` and
    ``tp = τ · P_days + epoch_ref_mjd`` (period in Keplerian years).
    """
    t = np.asarray(t_mjd, dtype=float)
    P_days = P_yr * DAYS_PER_KEPLER_YEAR
    tp = tau * P_days + epoch_ref_mjd
    nu = _oracle_true_anomaly(t, period_yr=P_yr, ecc=e, tp_mjd=tp)
    col_gamma = np.ones_like(t)
    col_K = np.cos(nu + omega) + e * math.cos(omega)
    return np.column_stack([col_gamma, col_K])


def _reduced_astro_design(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    plx_factor: np.ndarray,
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
    ``orblet.solve.astrometry._astrometric_columns``:

        col 0 = ∂model/∂a_phot  (the orbit column)
        col 1 = sinψ            (ra_offset)
        col 2 = cosψ            (dec_offset)
        col 3 = sinψ · Δt_yr    (pmra)
        col 4 = cosψ · Δt_yr    (pmdec)
        col 5 = parallax_factor (plx)

    The orbit column is built from the ENGINE atom
    :func:`...astro.forward._ti_constants_unit` (the production side under
    test) times the independent oracle in-plane point ``(x, y)``:

        col0 = (B_u x + G_u y) sinψ + (A_u x + F_u y) cosψ,

    where ``(A_u, B_u, F_u, G_u) = _ti_constants_unit(ω, Ω, cos i)`` so that
    ``a_phot · col0`` reproduces the full photocentre along-scan orbit.
    """
    from orblet.model import _ti_constants_unit

    t = np.asarray(t_mjd, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    pf = np.asarray(plx_factor, dtype=float)

    P_days = P_yr * DAYS_PER_KEPLER_YEAR
    tp = tau * P_days + epoch_ref_mjd
    nu = _oracle_true_anomaly(t, period_yr=P_yr, ecc=e, tp_mjd=tp)
    r = (1.0 - e * e) / (1.0 + e * np.cos(nu))
    x, y = r * np.cos(nu), r * np.sin(nu)

    A_u, B_u, F_u, G_u = _ti_constants_unit(omega, Omega, cos_i)
    d_ra_unit = B_u * x + G_u * y
    d_dec_unit = A_u * x + F_u * y

    sin_psi, cos_psi = np.sin(psi_arr), np.cos(psi_arr)
    col_orbit = d_ra_unit * sin_psi + d_dec_unit * cos_psi
    dt_yr = (t - epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR

    return np.column_stack(
        [col_orbit, sin_psi, cos_psi, sin_psi * dt_yr, cos_psi * dt_yr, pf]
    )


def _gauge_fold(
    omega: float, Omega: float, K: float, a_phot: float,
) -> tuple[float, float, float, float]:
    """Fold any Klein-4 gauge image onto the unique non-negative representative.

    Written FROM SCRATCH.  Given ``(ω, Ω, K, a_phot)`` (any of the four
    gauge images of one physical orbit), returns the representative with
    ``K > 0`` AND ``a_phot > 0``, decided by the AMPLITUDE SIGNS:

        (+, +) → identity
        (−, −) → ω += π,  negate both K and a_phot
        (+, −) → Ω += π,  negate a_phot
        (−, +) → ω += π and Ω += π,  negate K

    where the sign pair is ``(sign K, sign a_phot)``.  The returned angles
    are wrapped into ``[0, 2π)``.
    """
    k_pos = K >= 0.0
    a_pos = a_phot >= 0.0
    if k_pos and a_pos:            # (+, +) — already the representative
        o, big_o, k, a = omega, Omega, K, a_phot
    elif (not k_pos) and (not a_pos):   # (−, −)
        o, big_o, k, a = omega + math.pi, Omega, -K, -a_phot
    elif k_pos and (not a_pos):    # (+, −)
        o, big_o, k, a = omega, Omega + math.pi, K, -a_phot
    else:                          # (−, +)
        o, big_o, k, a = omega + math.pi, Omega + math.pi, -K, a_phot
    return o % _TWO_PI, big_o % _TWO_PI, k, a


def _direct_gls_log_evidence(
    d: np.ndarray,
    sigma: np.ndarray,
    X: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> float:
    """INDEPENDENT direct n×n Gaussian-linear marginal log-evidence.

    ``logE = −½ [ rᵀ C⁻¹ r + log|C| + n log 2π ]`` with
    ``C = X Σ_β Xᵀ + diag(σ²)`` and ``r = d − X μ_β``, using
    ``np.linalg.slogdet`` / ``np.linalg.solve`` on the FULL n×n covariance
    (NOT the reduced Woodbury / matrix-determinant-lemma form the production
    core uses — the whole point is an algebraically-different cross-check).
    """
    d = np.asarray(d, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    X = np.asarray(X, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_cov = np.asarray(prior_cov, dtype=float)

    n = d.shape[0]
    C = X @ prior_cov @ X.T + np.diag(sigma ** 2)
    r = d - X @ prior_mean
    _, slogdet = np.linalg.slogdet(C)
    quad = float(r @ np.linalg.solve(C, r))
    return -0.5 * (quad + slogdet + n * math.log(_TWO_PI))


def _reduced_block_evidences(
    truth: dict,
    *,
    t_rv: np.ndarray,
    t_astro: np.ndarray,
    psi: np.ndarray,
    plx_factor: np.ndarray,
    sigma_rv: float = 0.2,
    sigma_astro: float = 0.3,
):
    """Per-channel reduced two-amplitude blocks on NOISELESS oracle data.

    Batch 2 imports this to assemble the joint marginal objective.  Returns
    a dict ``{"rv": {...}, "astro": {...}}`` where each channel entry carries
    the reduced design ``X``, the noiseless oracle data ``d``, the per-epoch
    ``sigma``, the symmetric diagonal zero-mean reduced-width ``prior`` (the
    U5 gauge-fix prior), and the production-core ``sol`` (with
    ``log_evidence``).  All geometry is the shared truth; no data files read.
    """
    cos_i = math.cos(truth["inc_rad"])

    # ── RV block: reduced (γ, K); noiseless oracle RV. ──
    X_rv = _reduced_rv_design(
        t_rv, P_yr=truth["P_yr"], e=truth["e"], omega=truth["omega_rad"],
        tau=truth["tau"], epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    d_rv = oracle_rv(
        t_mjd=t_rv, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    sig_rv = np.full_like(d_rv, sigma_rv)
    prior_rv = _GaussianBetaPrior(mean=np.zeros(2), cov=np.diag([1e6, 1e6]))

    # ── Astro block: reduced (a_phot, ra, dec, pmra, pmdec, plx). ──
    X_ast = _reduced_astro_design(
        t_astro, psi, plx_factor,
        P_yr=truth["P_yr"], e=truth["e"], omega=truth["omega_rad"],
        Omega=truth["Omega_rad"], cos_i=cos_i, tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    d_ast = _oracle_astro_clean(truth, t_astro, psi, plx_factor)
    sig_ast = np.full_like(d_ast, sigma_astro)
    prior_ast = _GaussianBetaPrior(
        mean=np.zeros(6), cov=np.diag(1e6 * np.ones(6)),
    )

    from orblet.solve.astrometry import (
        linear_solve_ti,
    )
    from orblet.solve.rv import (
        linear_solve_rv,
    )

    sol_rv = linear_solve_rv(d_rv, sig_rv, X_rv, beta_prior=prior_rv)
    sol_ast = linear_solve_ti(d_ast, sig_ast, X_ast, beta_prior=prior_ast)
    return {
        "rv": {
            "X": X_rv, "d": d_rv, "sigma": sig_rv,
            "prior": prior_rv, "sol": sol_rv,
        },
        "astro": {
            "X": X_ast, "d": d_ast, "sigma": sig_ast,
            "prior": prior_ast, "sol": sol_ast,
        },
    }


def _oracle_astro_clean(truth, t_mjd, psi, plx_factor, *, a_phot_mas=None):
    """Noiseless along-scan data from the INDEPENDENT oracle (orbit + plx).

    ``a_phot_mas`` defaults to the truth photocentre amplitude; pass a scaled
    value for the β > 0 deficit test.  No proper motion / offset terms.
    """
    if a_phot_mas is None:
        a_phot_mas = truth["a_phot_mas"]
    A_o, B_o, F_o, G_o = _oracle_thiele_innes(
        a=a_phot_mas, omega=truth["omega_rad"], inc=truth["inc_rad"],
        Omega=truth["Omega_rad"],
    )
    nu = _oracle_true_anomaly(
        t_mjd, period_yr=truth["P_yr"], ecc=truth["e"], tp_mjd=truth["tp_mjd"],
    )
    r = (1.0 - truth["e"] ** 2) / (1.0 + truth["e"] * np.cos(nu))
    x, y = r * np.cos(nu), r * np.sin(nu)
    d_ra = B_o * x + G_o * y
    d_dec = A_o * x + F_o * y
    return (
        d_ra * np.sin(psi) + d_dec * np.cos(psi)
        + truth["plx_mas"] * np.asarray(plx_factor, dtype=float)
    )


def _au_yr_to_kms() -> float:
    """The AU/yr → km/s conversion factor (unit constants only)."""
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    return AU_M / 1000.0 / sec_per_yr


# ──────────────────────────────────────────────────────────────────────
# Tests 1–2 — the engine atom pins the ABSOLUTE Thiele-Innes signs.
# ──────────────────────────────────────────────────────────────────────


def test_ti_constants_unit_matches_campbell_effective_ti():
    """The engine atom equals ``campbell_xy``'s effective (A, B, F, G) / a_mas.

    ``campbell_xy`` does not expose its inline (A, B, F, G) directly; read
    them off via the e=0 unit-point probe — at ``(x_orb, y_orb) = (1, 0)``
    (periastron) the sky mapping gives ``(d_ra, d_dec) = (B, A)`` and at
    ``(0, 1)`` (quarter orbit) ``(G, F)`` — then divide by ``a_mas`` to
    recover the UNIT constants the atom returns.  Guards the deliberately-
    duplicated inline block against drift.
    """
    from orblet.model import (
        _ti_constants_unit,
        campbell_xy,
    )

    a_mas = _a_mas_probe()
    P_days = _P_YR * DAYS_PER_KEPLER_YEAR
    tp = 57388.5
    t_peri = np.array([tp], dtype=float)                     # ν = 0 → (1, 0)
    t_quarter = np.array([tp + 0.25 * P_days], dtype=float)  # ν = π/2 → (0, 1)

    for omega_deg in _OMEGA_DEG:
        for Omega_deg in _OMEGA_NODE_DEG:
            for inc_deg in _INC_DEG:
                omega = math.radians(omega_deg)
                Omega = math.radians(Omega_deg)
                inc = math.radians(inc_deg)

                dra_p, ddec_p = campbell_xy(
                    t_peri, period_yr=_P_YR, ecc=0.0, omega=omega, inc=inc,
                    Omega=Omega, tp_mjd=tp, m_comp_msun=_M_COMP_MSUN,
                    M_total_msun=_M_TOTAL_MSUN, plx_mas=_PLX_MAS,
                )
                dra_q, ddec_q = campbell_xy(
                    t_quarter, period_yr=_P_YR, ecc=0.0, omega=omega, inc=inc,
                    Omega=Omega, tp_mjd=tp, m_comp_msun=_M_COMP_MSUN,
                    M_total_msun=_M_TOTAL_MSUN, plx_mas=_PLX_MAS,
                )
                # (1, 0) → (B, A); (0, 1) → (G, F); divide by a_mas → unit.
                B_eff, A_eff = float(dra_p[0]) / a_mas, float(ddec_p[0]) / a_mas
                G_eff, F_eff = float(dra_q[0]) / a_mas, float(ddec_q[0]) / a_mas

                A_u, B_u, F_u, G_u = _ti_constants_unit(
                    omega, Omega, math.cos(inc),
                )
                msg = (
                    f"unit-TI mismatch at ω={omega_deg} Ω={Omega_deg} "
                    f"i={inc_deg} — the atom must equal campbell_xy's inline "
                    "A,B,F,G / a_mas (duplicated block drifted)."
                )
                np.testing.assert_allclose(A_u, A_eff, rtol=0, atol=1e-12,
                                           err_msg=msg)
                np.testing.assert_allclose(B_u, B_eff, rtol=0, atol=1e-12,
                                           err_msg=msg)
                np.testing.assert_allclose(F_u, F_eff, rtol=0, atol=1e-12,
                                           err_msg=msg)
                np.testing.assert_allclose(G_u, G_eff, rtol=0, atol=1e-12,
                                           err_msg=msg)


def test_ti_constants_unit_equals_independent_textbook():
    """The engine atom equals the from-scratch textbook oracle, per coeff.

    Sweeps the ω/Ω grid and ``cos_i`` over both hemispheres plus the
    cardinal values {0, 1, −1}; the atom (production) must equal
    ``_ti_constants_unit_oracle`` (independent) to machine precision.
    """
    from orblet.model import _ti_constants_unit

    cos_i_vals = [
        math.cos(math.radians(d)) for d in _INC_DEG
    ] + [0.0, 1.0, -1.0]
    for omega_deg in _OMEGA_DEG:
        for Omega_deg in _OMEGA_NODE_DEG:
            for cos_i in cos_i_vals:
                omega = math.radians(omega_deg)
                Omega = math.radians(Omega_deg)
                eng = _ti_constants_unit(omega, Omega, cos_i)
                ora = _ti_constants_unit_oracle(omega, Omega, cos_i)
                np.testing.assert_allclose(eng, ora, rtol=0, atol=1e-13)


# ──────────────────────────────────────────────────────────────────────
# 6a — the reduced columns equal the independent oracle forwards.
# ──────────────────────────────────────────────────────────────────────


def test_6a_reduced_columns_equal_oracle(synth_rv_dataset, synth_astro_dataset):
    """RV col 1 == oracle ``cos(ν+ω)+e cosω``; astro col 0 == oracle col_orbit.

    The astro orbit column is built (inside ``_reduced_astro_design``) from
    the ENGINE atom; here it must equal the col_orbit assembled from the
    from-scratch ``_ti_constants_unit_oracle`` — pinning the reduced design
    to the independent textbook forward.
    """
    truth = _derived_truth()
    cos_i = math.cos(truth["inc_rad"])

    prepared = synth_rv_dataset(
        seed=3, n_epochs=40, P_yr=truth["P_yr"], e=truth["e"],
        omega_rad=truth["omega_rad"], tau=truth["tau"], K_kms=truth["K_kms"],
        gamma_kms=truth["gamma_kms"], sigma_kms=0.2,
    )
    t_rv = np.asarray(prepared["epochs_mjd"], dtype=float)
    astro, _ = synth_astro_dataset(seed=9)
    t_a = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    pf = np.asarray(astro["parallax_factor_al"], dtype=float)

    # RV col 1 vs independent oracle bracket.
    X_rv = _reduced_rv_design(
        t_rv, P_yr=truth["P_yr"], e=truth["e"], omega=truth["omega_rad"],
        tau=truth["tau"], epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    nu_rv = _oracle_true_anomaly(
        t_rv, period_yr=truth["P_yr"], ecc=truth["e"], tp_mjd=truth["tp_mjd"],
    )
    oracle_bracket = np.cos(nu_rv + truth["omega_rad"]) + truth["e"] * math.cos(
        truth["omega_rad"]
    )
    np.testing.assert_allclose(X_rv[:, 1], oracle_bracket, rtol=1e-10,
                               atol=1e-12)
    np.testing.assert_allclose(X_rv[:, 0], np.ones_like(t_rv), rtol=0,
                               atol=0.0)

    # Astro col 0 vs independent oracle col_orbit (from-scratch unit TI).
    X_ast = _reduced_astro_design(
        t_a, psi, pf, P_yr=truth["P_yr"], e=truth["e"],
        omega=truth["omega_rad"], Omega=truth["Omega_rad"], cos_i=cos_i,
        tau=truth["tau"], epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    nu_a = _oracle_true_anomaly(
        t_a, period_yr=truth["P_yr"], ecc=truth["e"], tp_mjd=truth["tp_mjd"],
    )
    r = (1.0 - truth["e"] ** 2) / (1.0 + truth["e"] * np.cos(nu_a))
    x, y = r * np.cos(nu_a), r * np.sin(nu_a)
    A_u, B_u, F_u, G_u = _ti_constants_unit_oracle(
        truth["omega_rad"], truth["Omega_rad"], cos_i,
    )
    col_orbit_oracle = (
        (B_u * x + G_u * y) * np.sin(psi)
        + (A_u * x + F_u * y) * np.cos(psi)
    )
    np.testing.assert_allclose(X_ast[:, 0], col_orbit_oracle, rtol=1e-10,
                               atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# 6b — cross-channel a_spec ↔ (a_phot/ϖ)·sin i link.
# ──────────────────────────────────────────────────────────────────────


def test_6b_cross_channel_a_spec_link_holds():
    """RV ``a_spec = a₁ sin i`` == ``(a_phot / ϖ) · sin i`` at the β=0 truth.

    ``a_spec`` from RV: ``K √(1−e²) P / (2π)`` (AU/yr → km/s inverted);
    the astro side gives ``a₁ = a_phot / ϖ`` (β = 0), so ``a₁ sin i`` must
    agree — the shared amplitude / sin i / parallax bridge.
    """
    truth = _derived_truth()
    a_spec_rv = (
        truth["K_kms"] / _au_yr_to_kms()
        * truth["P_yr"] * math.sqrt(1.0 - truth["e"] ** 2)
        / (2.0 * math.pi)
    )
    a_spec_astro = (
        truth["a_phot_mas"] / truth["plx_mas"] * math.sin(truth["inc_rad"])
    )
    assert math.isclose(a_spec_rv, a_spec_astro, rel_tol=1e-10), (
        f"a_spec cross-channel: RV {a_spec_rv:.12e} != astro {a_spec_astro:.12e}"
        " AU — a SHARED amplitude / sin i / parallax-bridge bug; do NOT loosen."
    )


def test_6b_negative_control_wrong_inc():
    """Teeth for 6b: an inclination off by 15° BREAKS the a_spec link.

    ``a_spec`` from RV is held at the truth (i ≈ 60° base) while the astro
    side uses ``sin(i + 15°)`` — the link must fail (not isclose at rtol
    1e-6) with a relative gap > 0.1, proving the link is not vacuous.
    """
    truth = _derived_truth()
    a_spec_rv = (
        truth["K_kms"] / _au_yr_to_kms()
        * truth["P_yr"] * math.sqrt(1.0 - truth["e"] ** 2)
        / (2.0 * math.pi)
    )
    wrong_inc = truth["inc_rad"] + math.radians(15.0)
    a_spec_astro_wrong = (
        truth["a_phot_mas"] / truth["plx_mas"] * math.sin(wrong_inc)
    )
    rel_gap = abs(a_spec_rv - a_spec_astro_wrong) / abs(a_spec_astro_wrong)
    assert not math.isclose(a_spec_rv, a_spec_astro_wrong, rel_tol=1e-6)
    assert rel_gap > 0.1, (
        "6b negative control FAILED: a 15°-wrong inclination still matched "
        "the a_spec link — the link would be vacuous."
    )


# ──────────────────────────────────────────────────────────────────────
# 6c — reduced marginal evidence == independent direct n×n GLS.
# ──────────────────────────────────────────────────────────────────────


def test_6c_reduced_rv_evidence_matches_direct_gls(synth_rv_dataset):
    """Reduced-RV ``linear_solve_rv.log_evidence`` == direct n×n GLS; β̂ ≈ truth.

    Noiseless oracle RV data, σ = 0.2 km/s, symmetric diagonal zero-mean
    reduced-width prior (the U5 gauge-fix prior).  The production reduced
    (Woodbury) evidence must equal the INDEPENDENT direct n×n form.
    """
    truth = _derived_truth()
    prepared = synth_rv_dataset(
        seed=3, n_epochs=40, P_yr=truth["P_yr"], e=truth["e"],
        omega_rad=truth["omega_rad"], tau=truth["tau"], K_kms=truth["K_kms"],
        gamma_kms=truth["gamma_kms"], sigma_kms=0.2,
    )
    t_rv = np.asarray(prepared["epochs_mjd"], dtype=float)

    blocks = _reduced_block_evidences(
        truth, t_rv=t_rv, t_astro=t_rv, psi=np.zeros_like(t_rv),
        plx_factor=np.zeros_like(t_rv),
    )
    rv = blocks["rv"]
    direct = _direct_gls_log_evidence(
        rv["d"], rv["sigma"], rv["X"], rv["prior"].mean, rv["prior"].cov,
    )
    # rtol 1e-8 target; loosen only with a written cond(C) justification.
    # NOTE: both sides consume the SAME X, so this equality
    # tests Woodbury-vs-direct ALGEBRA only; DESIGN correctness rests on the
    # β̂==truth assertions below (independent-oracle data) — do not weaken them.
    assert math.isclose(rv["sol"].log_evidence, direct, rel_tol=1e-8,
                        abs_tol=1e-6), (
        f"reduced-RV evidence {rv['sol'].log_evidence:.12e} != direct GLS "
        f"{direct:.12e} nats."
    )
    assert np.isfinite(rv["sol"].log_evidence)
    # β̂ = (γ, K) recovered on noiseless data.
    np.testing.assert_allclose(
        rv["sol"].beta, [truth["gamma_kms"], truth["K_kms"]],
        rtol=1e-8, atol=1e-8,
    )


def test_6c_reduced_astro_evidence_matches_direct_gls(synth_astro_dataset):
    """Reduced-astro ``linear_solve_ti.log_evidence`` == direct n×n GLS; β̂ ≈ truth.

    Noiseless oracle along-scan data, σ = 0.3 mas, symmetric diagonal
    zero-mean reduced-width prior.  β̂ = (a_phot, 0, 0, 0, 0, plx).
    """
    truth = _derived_truth()
    astro, _ = synth_astro_dataset(seed=9)
    t_a = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    pf = np.asarray(astro["parallax_factor_al"], dtype=float)

    blocks = _reduced_block_evidences(
        truth, t_rv=t_a, t_astro=t_a, psi=psi, plx_factor=pf,
    )
    ast = blocks["astro"]
    direct = _direct_gls_log_evidence(
        ast["d"], ast["sigma"], ast["X"], ast["prior"].mean, ast["prior"].cov,
    )
    assert math.isclose(ast["sol"].log_evidence, direct, rel_tol=1e-8,
                        abs_tol=1e-6), (
        f"reduced-astro evidence {ast['sol'].log_evidence:.12e} != direct GLS "
        f"{direct:.12e} nats."
    )
    assert np.isfinite(ast["sol"].log_evidence)
    truth_beta = np.array(
        [truth["a_phot_mas"], 0.0, 0.0, 0.0, 0.0, truth["plx_mas"]]
    )
    np.testing.assert_allclose(ast["sol"].beta, truth_beta, rtol=1e-7,
                               atol=1e-7)


# ──────────────────────────────────────────────────────────────────────
# 6d — the photocentre deficit recovers the companion flux fraction.
#
# The RV traces the PRIMARY only (a₁ = a_phot_dark / ϖ); a luminous
# companion with flux fraction ℓ/f pulls the photocentre inward by
# (f − ℓ)/f = 1 − ℓ/f, so the astro-implied a₁ shrinks and the deficit
#     D = 1 − a₁_astro / a₁_rv
# recovers ℓ/f.  ASSUMES the RV is primary-only (no SB2 contamination —
# a different false-positive mode not modelled here).
# ──────────────────────────────────────────────────────────────────────


def _a1_from_rv(truth: dict) -> float:
    """Primary semi-major axis a₁ (AU) implied by the RV K at the truth i."""
    return (
        truth["K_kms"] / _au_yr_to_kms()
        * truth["P_yr"] * math.sqrt(1.0 - truth["e"] ** 2)
        / (2.0 * math.pi * math.sin(truth["inc_rad"]))
    )


def _recover_a_phot_mas(truth, t_a, psi, pf, *, a_phot_mas):
    """Recover a_phot (mas) via the reduced 6-col solve on oracle data.

    β[0] of the reduced astro design is a_phot by construction, so the
    noiseless solve returns the (possibly scaled) photocentre amplitude that
    generated the data.
    """
    from orblet.solve.astrometry import (
        linear_solve_ti,
    )

    cos_i = math.cos(truth["inc_rad"])
    d = _oracle_astro_clean(truth, t_a, psi, pf, a_phot_mas=a_phot_mas)
    X = _reduced_astro_design(
        t_a, psi, pf, P_yr=truth["P_yr"], e=truth["e"],
        omega=truth["omega_rad"], Omega=truth["Omega_rad"], cos_i=cos_i,
        tau=truth["tau"], epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    sigma = np.full_like(d, 0.3)
    sol = linear_solve_ti(d, sigma, X)  # flat prior; β̂ exact on noiseless data
    return float(sol.beta[0])


def test_6d_deficit_zero_at_beta0(synth_astro_dataset):
    """Deficit ``D`` is ~0 for a dark companion (β = 0)."""
    truth = _derived_truth()
    astro, _ = synth_astro_dataset(seed=9)
    t_a = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    pf = np.asarray(astro["parallax_factor_al"], dtype=float)

    a_phot = _recover_a_phot_mas(truth, t_a, psi, pf,
                                 a_phot_mas=truth["a_phot_mas"])
    a1_astro = a_phot / truth["plx_mas"]
    D = 1.0 - a1_astro / _a1_from_rv(truth)
    assert abs(D) < 1e-8, f"β=0 deficit should vanish but D={D:.3e}."


@pytest.mark.parametrize("flux_frac", [0.2, 0.5])
def test_6d_deficit_recovers_flux_fraction_at_betaGt0(
    flux_frac, synth_astro_dataset,
):
    """Deficit ``D`` recovers the companion flux fraction ℓ/f for β > 0.

    The astrometric photocentre amplitude is scaled by ``(f − ℓ)/f``
    (built through the oracle forward); the RV is unchanged; ``D`` must
    equal ℓ/f.
    """
    truth = _derived_truth()
    astro, _ = synth_astro_dataset(seed=9)
    t_a = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    pf = np.asarray(astro["parallax_factor_al"], dtype=float)

    a_phot_scaled = truth["a_phot_mas"] * (1.0 - flux_frac)
    a_phot = _recover_a_phot_mas(truth, t_a, psi, pf, a_phot_mas=a_phot_scaled)
    a1_astro = a_phot / truth["plx_mas"]
    D = 1.0 - a1_astro / _a1_from_rv(truth)
    assert math.isclose(D, flux_frac, rel_tol=1e-8, abs_tol=1e-10), (
        f"deficit D={D:.12e} != flux fraction {flux_frac}."
    )


# ──────────────────────────────────────────────────────────────────────
# 6e — forward degeneracy STRUCTURE (blind to global sign by design).
# ──────────────────────────────────────────────────────────────────────


def _geom_grid():
    """Deterministic epoch / scan / parallax-factor grid (no fixture)."""
    truth = _derived_truth()
    t = truth["epoch_ref_mjd"] + np.linspace(
        -1.5 * truth["P_days"], 1.5 * truth["P_days"], 64,
    )
    rng = np.random.default_rng(0)
    psi = rng.uniform(0.0, _TWO_PI, size=t.size)
    pf = rng.uniform(-1.0, 1.0, size=t.size)
    return truth, t, psi, pf


def _astro_orbit_col(truth, t, psi, pf, *, omega, Omega):
    """The reduced astro orbit column (col 0) at a given (ω, Ω)."""
    cos_i = math.cos(truth["inc_rad"])
    X = _reduced_astro_design(
        t, psi, pf, P_yr=truth["P_yr"], e=truth["e"], omega=omega,
        Omega=Omega, cos_i=cos_i, tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    return X[:, 0]


def _rv_col(truth, t, *, omega):
    """The reduced RV amplitude column (col 1) at a given ω."""
    X = _reduced_rv_design(
        t, P_yr=truth["P_yr"], e=truth["e"], omega=omega, tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    return X[:, 1]


def test_6e_joint_mirror_forward_structure():
    """Node mirror ``(ω, Ω) → (ω+π, Ω+π)``: astro invariant, RV column negates.

    The astro orbit column is bit-identical (unit TI invariant per
    coefficient); the RV amplitude column negates.
    """
    truth, t, psi, pf = _geom_grid()
    omega, Omega = truth["omega_rad"], truth["Omega_rad"]
    cos_i = math.cos(truth["inc_rad"])

    from orblet.model import _ti_constants_unit

    ti0 = _ti_constants_unit(omega, Omega, cos_i)
    ti_mirror = _ti_constants_unit(omega + math.pi, Omega + math.pi, cos_i)
    np.testing.assert_allclose(ti_mirror, ti0, rtol=0, atol=1e-13)

    col0 = _astro_orbit_col(truth, t, psi, pf, omega=omega, Omega=Omega)
    col_mirror = _astro_orbit_col(
        truth, t, psi, pf, omega=omega + math.pi, Omega=Omega + math.pi,
    )
    np.testing.assert_allclose(col_mirror, col0, rtol=0, atol=1e-9)

    rv0 = _rv_col(truth, t, omega=omega)
    rv_mirror = _rv_col(truth, t, omega=omega + math.pi)
    np.testing.assert_allclose(rv_mirror, -rv0, rtol=1e-12, atol=1e-12)


def test_6e_omega_only_flip_negates_both_columns():
    """``ω → ω+π`` (Ω fixed): BOTH the astro col_orbit and the RV column negate.
    """
    truth, t, psi, pf = _geom_grid()
    omega, Omega = truth["omega_rad"], truth["Omega_rad"]
    cos_i = math.cos(truth["inc_rad"])

    from orblet.model import _ti_constants_unit

    ti0 = np.array(_ti_constants_unit(omega, Omega, cos_i))
    ti_flip = np.array(_ti_constants_unit(omega + math.pi, Omega, cos_i))
    np.testing.assert_allclose(ti_flip, -ti0, rtol=1e-12, atol=1e-13)

    col0 = _astro_orbit_col(truth, t, psi, pf, omega=omega, Omega=Omega)
    col_flip = _astro_orbit_col(
        truth, t, psi, pf, omega=omega + math.pi, Omega=Omega,
    )
    np.testing.assert_allclose(col_flip, -col0, rtol=1e-12, atol=1e-12)

    rv0 = _rv_col(truth, t, omega=omega)
    rv_flip = _rv_col(truth, t, omega=omega + math.pi)
    np.testing.assert_allclose(rv_flip, -rv0, rtol=1e-12, atol=1e-12)


def test_6e_Omega_only_flip_negates_astro_rv_invariant():
    """``Ω → Ω+π`` (ω fixed): astro col_orbit negates; RV column bit-identical.
    """
    truth, t, psi, pf = _geom_grid()
    omega, Omega = truth["omega_rad"], truth["Omega_rad"]

    col0 = _astro_orbit_col(truth, t, psi, pf, omega=omega, Omega=Omega)
    col_flip = _astro_orbit_col(
        truth, t, psi, pf, omega=omega, Omega=Omega + math.pi,
    )
    np.testing.assert_allclose(col_flip, -col0, rtol=1e-12, atol=1e-12)

    # DOCUMENTATION, not a check: `_rv_col` takes no Ω argument, so the RV
    # column is Ω-independent BY CONSTRUCTION (the structural fact this
    # identity relies on).  The assertion below cannot fail; it exists to
    # record the convention in executable form.
    rv0 = _rv_col(truth, t, omega=omega)
    rv_same = _rv_col(truth, t, omega=omega)
    np.testing.assert_allclose(rv_same, rv0, rtol=0, atol=0.0)


def test_6e_tau_invariance():
    """τ → τ + 1 (a full period shift of tp) leaves all columns invariant."""
    truth, t, psi, pf = _geom_grid()
    omega, Omega = truth["omega_rad"], truth["Omega_rad"]
    cos_i = math.cos(truth["inc_rad"])

    X0 = _reduced_astro_design(
        t, psi, pf, P_yr=truth["P_yr"], e=truth["e"], omega=omega,
        Omega=Omega, cos_i=cos_i, tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    X1 = _reduced_astro_design(
        t, psi, pf, P_yr=truth["P_yr"], e=truth["e"], omega=omega,
        Omega=Omega, cos_i=cos_i, tau=truth["tau"] + 1.0,
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    np.testing.assert_allclose(X1, X0, rtol=1e-12, atol=1e-12)

    rv0 = _reduced_rv_design(
        t, P_yr=truth["P_yr"], e=truth["e"], omega=omega, tau=truth["tau"],
        epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    rv1 = _reduced_rv_design(
        t, P_yr=truth["P_yr"], e=truth["e"], omega=omega,
        tau=truth["tau"] + 1.0, epoch_ref_mjd=truth["epoch_ref_mjd"],
    )
    np.testing.assert_allclose(rv1, rv0, rtol=1e-12, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# 6f — the marginal evidence is Klein-4 INVARIANT (the U5 gauge-fix key).
# ──────────────────────────────────────────────────────────────────────


def test_6f_evidence_invariant_under_klein4(synth_rv_dataset, synth_astro_dataset):
    """Per-channel ``log_evidence`` is IDENTICAL across the four gauge images.

    With SYMMETRIC diagonal zero-mean priors (the U5 gauge-fix choice) the
    marginal covariance ``X Σ_β Xᵀ + Σ_d`` is invariant under any column
    sign flip (``S Σ_β S = Σ_β`` for diagonal ``Σ_β`` and diagonal sign
    ``S``), so the evidence is the same for {id, ω-flip, Ω-flip, joint}.
    NOTE: requires DIAGONAL ``Σ_β`` and ZERO mean; a non-diagonal prior
    would break the exact invariance.
    """
    from orblet.solve.astrometry import (
        linear_solve_ti,
    )
    from orblet.solve.rv import (
        linear_solve_rv,
    )

    truth = _derived_truth()
    cos_i = math.cos(truth["inc_rad"])
    omega, Omega = truth["omega_rad"], truth["Omega_rad"]

    prepared = synth_rv_dataset(
        seed=3, n_epochs=40, P_yr=truth["P_yr"], e=truth["e"],
        omega_rad=truth["omega_rad"], tau=truth["tau"], K_kms=truth["K_kms"],
        gamma_kms=truth["gamma_kms"], sigma_kms=0.2,
    )
    t_rv = np.asarray(prepared["epochs_mjd"], dtype=float)
    d_rv = oracle_rv(
        t_mjd=t_rv, period_yr=truth["P_yr"], ecc=truth["e"],
        omega_primary=truth["omega_rad"], tp_mjd=truth["tp_mjd"],
        K_kms=truth["K_kms"], gamma_kms=truth["gamma_kms"],
    )
    sig_rv = np.full_like(d_rv, 0.2)
    prior_rv = _GaussianBetaPrior(mean=np.zeros(2), cov=np.diag([1e6, 1e6]))

    astro, _ = synth_astro_dataset(seed=9)
    t_a = np.asarray(astro["obs_time"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    pf = np.asarray(astro["parallax_factor_al"], dtype=float)
    d_ast = _oracle_astro_clean(truth, t_a, psi, pf)
    sig_ast = np.full_like(d_ast, 0.3)
    prior_ast = _GaussianBetaPrior(mean=np.zeros(6), cov=np.diag(1e6 * np.ones(6)))

    # Four Klein-4 gauge images (ω, Ω) offsets.
    gauge = [(0.0, 0.0), (math.pi, 0.0), (0.0, math.pi), (math.pi, math.pi)]

    rv_ev = []
    ast_ev = []
    for dw, dO in gauge:
        X_rv = _reduced_rv_design(
            t_rv, P_yr=truth["P_yr"], e=truth["e"], omega=omega + dw,
            tau=truth["tau"], epoch_ref_mjd=truth["epoch_ref_mjd"],
        )
        rv_ev.append(
            linear_solve_rv(d_rv, sig_rv, X_rv, beta_prior=prior_rv).log_evidence
        )
        X_ast = _reduced_astro_design(
            t_a, psi, pf, P_yr=truth["P_yr"], e=truth["e"], omega=omega + dw,
            Omega=Omega + dO, cos_i=cos_i, tau=truth["tau"],
            epoch_ref_mjd=truth["epoch_ref_mjd"],
        )
        ast_ev.append(
            linear_solve_ti(d_ast, sig_ast, X_ast, beta_prior=prior_ast).log_evidence
        )

    for e in rv_ev[1:]:
        assert math.isclose(e, rv_ev[0], rel_tol=1e-12, abs_tol=1e-9)
    for e in ast_ev[1:]:
        assert math.isclose(e, ast_ev[0], rel_tol=1e-12, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# 6g — the export gauge fold: unique non-negative representative.
# ──────────────────────────────────────────────────────────────────────


def _klein4_images(omega, Omega, K, a_phot):
    """The four gauge images of one physical orbit + induced amplitude signs.

    Physical invariances (RV: ω+π ⇒ K→−K; astro: unit TI negate under a
    lone ω+π or Ω+π, invariant under the joint flip):

        id           : (ω,     Ω,     +K, +a_phot)   signs (+, +)
        ω-flip       : (ω+π,   Ω,     −K, −a_phot)   signs (−, −)
        Ω-flip       : (ω,     Ω+π,   +K, −a_phot)   signs (+, −)
        joint        : (ω+π,   Ω+π,   −K, +a_phot)   signs (−, +)
    """
    return [
        (omega, Omega, K, a_phot),
        (omega + math.pi, Omega, -K, -a_phot),
        (omega, Omega + math.pi, K, -a_phot),
        (omega + math.pi, Omega + math.pi, -K, a_phot),
    ]


def test_6g_klein4_orbit_has_unique_nonneg_representative():
    """The four gauge images carry all four sign pairs; exactly one is (+, +)."""
    truth = _derived_truth()
    images = _klein4_images(
        truth["omega_rad"], truth["Omega_rad"], truth["K_kms"],
        truth["a_phot_mas"],
    )
    sign_pairs = {
        (K >= 0.0, a >= 0.0) for (_, _, K, a) in images
    }
    assert sign_pairs == {(True, True), (False, False), (True, False),
                          (False, True)}
    both_nonneg = [
        img for img in images if img[2] >= 0.0 and img[3] >= 0.0
    ]
    assert len(both_nonneg) == 1


def test_6g_gauge_fold_recovers_truth():
    """``_gauge_fold`` maps EVERY gauge image back to (ω, Ω, K>0, a_phot>0).

    Uses a truth with ω ∈ [π, 2π) AND Ω ∈ [π, 2π) so a naive angle-range
    fold would visibly fail; the fold must recover the true angles (mod 2π)
    and the positive amplitudes from all four images.
    """
    truth = _derived_truth(omega_rad=3.5, Omega_rad=4.0)
    omega, Omega = truth["omega_rad"], truth["Omega_rad"]
    K, a_phot = truth["K_kms"], truth["a_phot_mas"]
    assert math.pi <= omega < _TWO_PI and math.pi <= Omega < _TWO_PI

    for (w, big_o, k, a) in _klein4_images(omega, Omega, K, a_phot):
        w_f, o_f, k_f, a_f = _gauge_fold(w, big_o, k, a)
        # Angles agree modulo 2π; amplitudes are the positive truth.
        assert abs(_wrap_to_pi(w_f - omega)) < 1e-12
        assert abs(_wrap_to_pi(o_f - Omega)) < 1e-12
        assert k_f > 0.0 and a_f > 0.0
        assert math.isclose(k_f, K, rel_tol=1e-12)
        assert math.isclose(a_f, a_phot, rel_tol=1e-12)
