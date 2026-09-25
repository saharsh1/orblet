"""Cycle RV-C — CORE pure-math companion-mass solver.

These tests exercise the engine-agnostic SB1 mass-function root solve

    fm = (m2 · sin i)^3 / (m1 + m2)^2     [all masses M_sun; fm absorbs G]

solved for the companion mass ``m2`` given ``fm``, ``m1``, ``sin_i``.
No real data, no real source ids — synthetic scalars/arrays only.
"""

from __future__ import annotations

import numpy as np

from orblet.interpret.companion_mass import solve_companion_mass


def _fm_of(m2, m1, sin_i):
    """Reconstruct the mass function from (m2, m1, sin_i)."""
    return (m2 * sin_i) ** 3 / (m1 + m2) ** 2


def test_solves_known_root() -> None:
    # Truth: m1 = 1.0, m2 = 10.0, sin_i = 1.0 -> fm = 1000 / 121.
    fm = np.array([1000.0 / 121.0])
    m2 = solve_companion_mass(fm, np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(m2, [10.0], rtol=1e-9)


def test_residual_reconstructs_fm() -> None:
    rng = np.random.default_rng(0)
    m1 = rng.uniform(0.5, 2.0, size=64)
    sin_i = rng.uniform(0.3, 1.0, size=64)
    m2_true = rng.uniform(0.1, 30.0, size=64)
    fm = _fm_of(m2_true, m1, sin_i)
    m2 = solve_companion_mass(fm, m1, sin_i)
    np.testing.assert_allclose(_fm_of(m2, m1, sin_i), fm, rtol=1e-9)


def test_monotone_in_sin_i_strict() -> None:
    # Higher sin_i (more edge-on) at fixed fm, m1 -> SMALLER m2.
    fm = np.array([5.0, 5.0])
    m1 = np.array([1.0, 1.0])
    m2_low = solve_companion_mass(fm[:1], m1[:1], np.array([0.4]))
    m2_high = solve_companion_mass(fm[:1], m1[:1], np.array([1.0]))
    assert m2_high[0] < m2_low[0]


def test_monotone_in_fm_strict() -> None:
    # Higher fm at fixed m1, sin_i -> larger m2.
    m1 = np.array([1.0])
    sin_i = np.array([1.0])
    m2_small_fm = solve_companion_mass(np.array([1.0]), m1, sin_i)
    m2_large_fm = solve_companion_mass(np.array([50.0]), m1, sin_i)
    assert m2_large_fm[0] > m2_small_fm[0]


def test_nan_on_no_positive_root() -> None:
    # fm <= 0 has no physical positive root.
    m2 = solve_companion_mass(
        np.array([0.0, -3.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])
    )
    assert np.all(np.isnan(m2))


def test_sin_i_zero_is_nan() -> None:
    m2 = solve_companion_mass(np.array([5.0]), np.array([1.0]), np.array([0.0]))
    assert np.isnan(m2[0])


def test_broadcasts_scalar_and_array() -> None:
    fm = np.array([1.0, 5.0, 20.0])
    # m1 and sin_i given as scalars -> broadcast against fm.
    m2 = solve_companion_mass(fm, 1.0, 1.0)
    assert m2.shape == (3,)
    np.testing.assert_allclose(_fm_of(m2, 1.0, 1.0), fm, rtol=1e-9)


def test_single_element() -> None:
    fm = np.array([1000.0 / 121.0])
    m2 = solve_companion_mass(fm, np.array([1.0]), np.array([1.0]))
    assert m2.shape == (1,)
    np.testing.assert_allclose(m2, [10.0], rtol=1e-9)


def test_large_m2_black_hole_regime() -> None:
    # m1 = 1.0, m2 = 100.0 (>> m1), sin_i = 1.0 -> the bracket must reach it.
    m2_true = 100.0
    fm = _fm_of(m2_true, 1.0, 1.0)
    m2 = solve_companion_mass(np.array([fm]), np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(m2, [m2_true], rtol=1e-9)
