"""Per-coefficient convention guard for ``campbell_xy``'s primary-frame form.

``campbell_xy`` uses a POSITIVE photocenter amplitude and the primary-frame
ω directly.  The companion-frame alternative — a NEGATIVE amplitude
``a_mas = −(m_comp/M_total)·a·plx`` with ``ω_companion = ω + π`` — is
algebraically the same orbit: ``−a·cos(ω+π) = +a·cos ω`` (and likewise for
every sinω/cosω term), so the four Thiele-Innes constants A, B, F, G — hence
``d_ra, d_dec`` — agree to the last bit.

Two checks live here:

1. ``test_campbell_primary_frame_per_coefficient`` — pins EACH of A, B, F, G
   individually against the textbook primary-frame Hilditch oracle
   (``_oracle_thiele_innes`` from ``test_a0_independent_orbit_oracle``).  The
   along-scan-sum agreement tests cannot catch a HALF-removed cancellation
   (one sign flipped but not the other); this per-coefficient check can.
2. ``test_campbell_primary_frame_grid_matches_oracle`` — evaluates the engine
   ``campbell_xy`` over the curated convention grid and asserts the
   ``(d_ra, d_dec)`` output equals the independent primary-frame oracle to
   machine precision.

All inputs synthetic; no data files read.
"""

from __future__ import annotations

import math

import numpy as np

from test_a0_independent_orbit_oracle import _oracle_thiele_innes

from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import campbell_xy


# Curated convention grid (from the Wave-1 plan): ω, Ω off the cardinal
# values and spanning all four quadrants; i in BOTH hemispheres and off
# {0,90,180}°; two eccentricities; fixed masses/plx/period.  Every cos/sin
# term is non-degenerate in several cells, so a single-term sign error
# cannot hide.
_OMEGA_DEG = (23.0, 147.0, 251.0)
_OMEGA_NODE_DEG = (41.0, 199.0, 313.0)
_INC_DEG = (37.0, 63.0, 118.0, 152.0)
_ECC = (0.1, 0.7)

_P_YR = 800.0 / DAYS_PER_KEPLER_YEAR
_M_COMP_MSUN = 0.5
_M_TOTAL_MSUN = 1.5
_PLX_MAS = 5.0
_EPOCH_REF_MJD = 57388.5
# A few epochs spread across the orbit (so x_orb, y_orb are non-trivial).
_T_MJD = np.array(
    [
        _EPOCH_REF_MJD + 0.0,
        _EPOCH_REF_MJD + 90.0,
        _EPOCH_REF_MJD + 250.0,
        _EPOCH_REF_MJD + 410.0,
        _EPOCH_REF_MJD + 650.0,
    ],
    dtype=float,
)
_TP_MJD = 0.4 * 800.0 + _EPOCH_REF_MJD


def _grid_cases():
    """Yield (omega_rad, inc_rad, Omega_rad, ecc) over the convention grid."""
    for omega_deg in _OMEGA_DEG:
        for Omega_deg in _OMEGA_NODE_DEG:
            for inc_deg in _INC_DEG:
                for ecc in _ECC:
                    yield (
                        math.radians(omega_deg),
                        math.radians(inc_deg),
                        math.radians(Omega_deg),
                        ecc,
                    )


def _engine_xy(omega, inc, Omega, ecc):
    """Engine astro forward ``(d_ra, d_dec)`` at one grid cell."""
    return campbell_xy(
        _T_MJD,
        period_yr=_P_YR,
        ecc=ecc,
        omega=omega,
        inc=inc,
        Omega=Omega,
        tp_mjd=_TP_MJD,
        m_comp_msun=_M_COMP_MSUN,
        M_total_msun=_M_TOTAL_MSUN,
        plx_mas=_PLX_MAS,
    )


def _oracle_xy(omega, inc, Omega, ecc):
    """Independent primary-frame oracle ``(d_ra, d_dec)`` at one grid cell.

    Uses POSITIVE photocenter amplitude ``a_phot = (m_comp/M_total)·a·plx``
    and primary-frame ω directly — the clean form the engine must equal.
    The Kepler→(x, y) part is recomputed here from the same constants so the
    comparison isolates the A,B,F,G convention.
    """
    a_au = (_P_YR ** 2 * _M_TOTAL_MSUN) ** (1.0 / 3.0)
    a_phot = (_M_COMP_MSUN / _M_TOTAL_MSUN) * a_au * _PLX_MAS
    A, B, F, G = _oracle_thiele_innes(a=a_phot, omega=omega, inc=inc, Omega=Omega)

    P_days = _P_YR * DAYS_PER_KEPLER_YEAR
    M_anom = (2.0 * math.pi * (_T_MJD - _TP_MJD) / P_days) % (2.0 * math.pi)
    # Independent Newton solve of Kepler's equation (do not import the engine
    # solver, so this remains a genuine cross-check of the projection only).
    E = M_anom.copy()
    for _ in range(100):
        f = E - ecc * np.sin(E) - M_anom
        fp = 1.0 - ecc * np.cos(E)
        E = E - f / fp
        if np.max(np.abs(f / fp)) < 1e-13:
            break
    nu = 2.0 * np.arctan2(
        math.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        math.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )
    r = (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(nu))
    x, y = r * np.cos(nu), r * np.sin(nu)
    d_ra = B * x + G * y
    d_dec = A * x + F * y
    return d_ra, d_dec


def test_campbell_primary_frame_per_coefficient():
    """Each engine A, B, F, G equals the textbook primary-frame Hilditch
    value at every grid cell.

    The engine does not expose A,B,F,G directly, so we read them off via two
    unit-vector probes of ``campbell_xy``: at the in-plane unit points
    (x_orb, y_orb) = (1, 0) and (0, 1) the sky mapping
    ``d_ra = B x + G y``, ``d_dec = A x + F y`` gives
    ``(d_ra, d_dec)|_{x=1,y=0} = (B, A)`` and
    ``(d_ra, d_dec)|_{x=0,y=1} = (G, F)``.  We synthesise epochs that land on
    those in-plane points (periastron x=1,y=0 at e=0; quarter-orbit x=0,y=1
    at e=0) and compare each coefficient individually.  A half-removed
    cancellation (flip the amplitude sign but keep ω+π, or vice versa) negates
    A,B,F,G together and FAILS here, while the along-scan-sum tests can stay
    green by coincidence.
    """
    # At e=0 the in-plane radius is 1 and ν = M (true anomaly = mean anomaly).
    # Pick tp so that at t = tp the point is (x, y) = (1, 0) [ν=0], and at the
    # quarter period the point is (0, 1) [ν=π/2].
    ecc = 0.0
    P_days = _P_YR * DAYS_PER_KEPLER_YEAR
    tp = _EPOCH_REF_MJD
    t_peri = np.array([tp], dtype=float)             # ν = 0  → (x, y) = (1, 0)
    t_quarter = np.array([tp + 0.25 * P_days], dtype=float)  # ν = π/2 → (0, 1)

    a_au = (_P_YR ** 2 * _M_TOTAL_MSUN) ** (1.0 / 3.0)
    a_phot = (_M_COMP_MSUN / _M_TOTAL_MSUN) * a_au * _PLX_MAS

    for omega_deg in _OMEGA_DEG:
        for Omega_deg in _OMEGA_NODE_DEG:
            for inc_deg in _INC_DEG:
                omega = math.radians(omega_deg)
                inc = math.radians(inc_deg)
                Omega = math.radians(Omega_deg)

                A_o, B_o, F_o, G_o = _oracle_thiele_innes(
                    a=a_phot, omega=omega, inc=inc, Omega=Omega,
                )

                dra_p, ddec_p = campbell_xy(
                    t_peri, period_yr=_P_YR, ecc=ecc, omega=omega, inc=inc,
                    Omega=Omega, tp_mjd=tp, m_comp_msun=_M_COMP_MSUN,
                    M_total_msun=_M_TOTAL_MSUN, plx_mas=_PLX_MAS,
                )
                dra_q, ddec_q = campbell_xy(
                    t_quarter, period_yr=_P_YR, ecc=ecc, omega=omega, inc=inc,
                    Omega=Omega, tp_mjd=tp, m_comp_msun=_M_COMP_MSUN,
                    M_total_msun=_M_TOTAL_MSUN, plx_mas=_PLX_MAS,
                )

                # (x=1, y=0) → d_ra = B, d_dec = A.
                B_eng, A_eng = float(dra_p[0]), float(ddec_p[0])
                # (x=0, y=1) → d_ra = G, d_dec = F.
                G_eng, F_eng = float(dra_q[0]), float(ddec_q[0])

                msg = (
                    f"coefficient mismatch at ω={omega_deg} Ω={Omega_deg} "
                    f"i={inc_deg} — the engine A,B,F,G must equal the textbook "
                    "primary-frame Hilditch values (no ω+π / sign half-flip)."
                )
                np.testing.assert_allclose(A_eng, A_o, rtol=0, atol=1e-12, err_msg=msg)
                np.testing.assert_allclose(B_eng, B_o, rtol=0, atol=1e-12, err_msg=msg)
                np.testing.assert_allclose(F_eng, F_o, rtol=0, atol=1e-12, err_msg=msg)
                np.testing.assert_allclose(G_eng, G_o, rtol=0, atol=1e-12, err_msg=msg)


def test_campbell_primary_frame_grid_matches_oracle():
    """Engine ``campbell_xy`` (d_ra, d_dec) == independent primary-frame
    oracle to machine precision over the full convention grid.

    The clean primary-frame form and the documented ω+π/negative-amplitude
    form are algebraically identical, so this convention-pinned equality is
    the no-op confirmation: any divergence beyond float round-off is a real
    convention bug, not the expected no-op.
    """
    for omega, inc, Omega, ecc in _grid_cases():
        dra_eng, ddec_eng = _engine_xy(omega, inc, Omega, ecc)
        dra_orc, ddec_orc = _oracle_xy(omega, inc, Omega, ecc)
        np.testing.assert_allclose(dra_eng, dra_orc, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ddec_eng, ddec_orc, rtol=0, atol=1e-12)
