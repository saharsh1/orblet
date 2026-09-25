"""
Textbook validation of orblet.kepler.campbell_xy.

The truth recipe uses Murray & Dermott 1999 §2.4 (Kepler's equation,
orbit-plane closed form) and §2.8 (Thiele-Innes rotation matrix).
Independent Newton iteration on Kepler's equation in stdlib math —
does NOT call orblet.kepler.solve_kepler.

Axis-pairing convention (load-bearing):
    Δα* (east)  ← B · x + G · y
    Δδ  (north) ← A · x + F · y

Murray & Dermott §2.8 nominally writes (X, Y) = (north, east) — the
same physical mapping in the opposite tuple order. The test recipe uses
orblet's pairing.

Period-unit convention: orblet.constants.DAYS_PER_KEPLER_YEAR =
365.25 (Julian year). The code under test uses 365.25; the truth recipe
must use 365.25 to match. A change to either constant must update the
corresponding side of this test.
"""

import math

import numpy as np
import pytest

from orblet.kepler import _wrap_to_pi, campbell_xy, solve_kepler


def _newton_solve_kepler(M: float, e: float, tol: float = 1e-13, max_iter: int = 200) -> float:
    """Solve Kepler's equation M = E - e*sin(E) via pure-Python Newton iteration.

    Independent of project's solve_kepler. Uses only stdlib math.
    Convergence criterion: |delta E| < tol (default 1e-13).

    Initial guess E = M + e*sin(M) gives quadratic convergence even at
    e = 0.9 (well within max_iter for the parametrizations used here).
    """
    # Better initial guess than E = M; helps high-e convergence.
    E = M + e * math.sin(M)
    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        delta = f / fp
        E -= delta
        if abs(delta) < tol:
            return E
    raise RuntimeError(f"Newton iteration did not converge: M={M}, e={e}")


def _textbook_thiele_innes(
    M: float, e: float, a: float,
    i: float, omega: float, Omega: float,
) -> tuple[float, float]:
    """Compute (Δα*, Δδ) at mean anomaly M independently of the project.

    Approach (Murray & Dermott 1999):
    1. Solve Kepler's equation M = E - e sin(E) via pure-Python Newton (eq. 2.52)
    2. Orbit-plane (x, y) closed form (eqs. 2.39, 2.40):
         x = a (cos E - e)
         y = a sqrt(1 - e²) sin E
    3. Thiele-Innes rotation (eqs. 2.121-2.124):
         A =  cos(Ω) cos(ω) - sin(Ω) sin(ω) cos(i)
         B =  sin(Ω) cos(ω) + cos(Ω) sin(ω) cos(i)
         F = -cos(Ω) sin(ω) - sin(Ω) cos(ω) cos(i)
         G = -sin(Ω) sin(ω) + cos(Ω) cos(ω) cos(i)

    AXIS PAIRING (load-bearing — must match the implementation):

    orblet's convention:
        Δα* (east)  = B · x + G · y
        Δδ  (north) = A · x + F · y

    Murray & Dermott §2.8 nominally writes (X, Y) = (north, east):
        X = A · x + F · y
        Y = B · x + G · y

    These are the SAME physical mapping written in different tuple
    order (M&D's X = project's Δδ; M&D's Y = project's Δα*). The test
    MUST use the project's pairing — copying M&D verbatim into the
    truth recipe would swap the coordinates and trip every general-
    geometry parametrization. See `_kepler.py:90-91` for the pairing.

    Returns (Δα*, Δδ) in the same units as `a`.
    """
    E = _newton_solve_kepler(M, e)
    x = a * (math.cos(E) - e)
    y = a * math.sqrt(1.0 - e * e) * math.sin(E)
    A = math.cos(Omega) * math.cos(omega) - math.sin(Omega) * math.sin(omega) * math.cos(i)
    B = math.sin(Omega) * math.cos(omega) + math.cos(Omega) * math.sin(omega) * math.cos(i)
    F = -math.cos(Omega) * math.sin(omega) - math.sin(Omega) * math.cos(omega) * math.cos(i)
    G = -math.sin(Omega) * math.sin(omega) + math.cos(Omega) * math.cos(omega) * math.cos(i)
    dra = B * x + G * y   # east — project's Δα*
    ddec = A * x + F * y  # north — project's Δδ
    return dra, ddec


@pytest.mark.parametrize("M_truth", [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
                         ids=["M_0", "M_pi_over_2", "M_pi", "M_3pi_over_2"])
@pytest.mark.parametrize("e_truth", [0.0, 0.5, 0.9],
                         ids=["e_0", "e_0p5", "e_0p9"])
def test_campbell_xy_matches_textbook(M_truth: float, e_truth: float):
    """
    Pin campbell_xy against an independent textbook recipe at 12
    (M, e) parametrizations under one general (i, ω, Ω) geometry.

    See module docstring for axis-pairing and period-unit conventions.
    """
    P_yr = 5.0
    e = e_truth
    omega = math.pi / 6  # 30°
    inc = math.pi / 3    # 60°
    Omega = math.pi / 4  # 45°
    a_mas = 10.0
    tp_mjd = 58000.0

    # Convert M_truth → t_mjd via M = 2π(t - tp)/(P_yr · 365.25)
    P_days = P_yr * 365.25  # Julian year, matches orblet.constants.DAYS_PER_KEPLER_YEAR
    t_mjd = tp_mjd + (M_truth / (2 * math.pi)) * P_days

    dra_fn, ddec_fn = campbell_xy(
        t_mjd=t_mjd,
        period_yr=P_yr,
        ecc=e,
        omega_rad=omega,
        inc_rad=inc,
        Omega_rad=Omega,
        tp_mjd=tp_mjd,
        a_mas=a_mas,
    )

    dra_tb, ddec_tb = _textbook_thiele_innes(
        M=M_truth, e=e, a=a_mas, i=inc, omega=omega, Omega=Omega,
    )

    np.testing.assert_allclose(
        [dra_fn, ddec_fn], [dra_tb, ddec_tb],
        atol=1e-9, rtol=0,
        err_msg=f"Disagreement at M={M_truth}, e={e}",
    )


@pytest.mark.parametrize("e_truth", [0.0, 0.5, 0.9],
                         ids=["e_0", "e_0p5", "e_0p9"])
def test_campbell_xy_periastron_anchor(e_truth: float):
    """
    Periastron anchor case: at t = tp, M = 0 → ν = 0 → (x, y) = (a(1-e), 0).

    Closed-form expectation:
        Δα* = B · a · (1 - e)
        Δδ  = A · a · (1 - e)

    This pins the periastron sign convention WITHOUT depending on the
    Newton solver (M = 0 → E = 0 trivially, no iteration needed).
    Independent failure mode from test_campbell_xy_matches_textbook.
    """
    e = e_truth
    omega = math.pi / 6
    inc = math.pi / 3
    Omega = math.pi / 4
    a_mas = 10.0
    tp_mjd = 58000.0
    P_yr = 5.0

    # At t = tp, mean anomaly is zero; no Newton iteration needed.
    dra_fn, ddec_fn = campbell_xy(
        t_mjd=tp_mjd, period_yr=P_yr, ecc=e,
        omega_rad=omega, inc_rad=inc, Omega_rad=Omega,
        tp_mjd=tp_mjd, a_mas=a_mas,
    )

    # Closed-form Thiele-Innes constants
    A = math.cos(Omega) * math.cos(omega) - math.sin(Omega) * math.sin(omega) * math.cos(inc)
    B = math.sin(Omega) * math.cos(omega) + math.cos(Omega) * math.sin(omega) * math.cos(inc)

    dra_anchor = B * a_mas * (1.0 - e)   # east
    ddec_anchor = A * a_mas * (1.0 - e)  # north

    np.testing.assert_allclose(
        [dra_fn, ddec_fn], [dra_anchor, ddec_anchor],
        atol=1e-9, rtol=0,
        err_msg=f"Periastron anchor disagreement at e={e}",
    )


# ── _wrap_to_pi edge cases ───────────────────────────────────────────────────
#
# ``_wrap_to_pi`` is used at the convention-translation boundary in
# :mod:`orblet.rv_chain` (prior inversion).  Its wrap behaviour
# at the {0, ±π} boundary is load-bearing: any silent bias here would
# leak into the round-trip ω translation.


def _circular_equiv(a: float, b: float) -> float:
    """Smallest absolute angular distance (radians) between two angles."""
    return abs((float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi)


@pytest.mark.parametrize(
    "x",
    [
        0.0,
        math.pi,
        -math.pi,
        math.pi + 1e-12,
        -math.pi + 1e-12,
        10 * math.pi,
        -10 * math.pi,
    ],
    ids=[
        "x_0",
        "x_pi",
        "x_minus_pi",
        "x_pi_plus_eps",
        "x_minus_pi_plus_eps",
        "x_10pi",
        "x_minus_10pi",
    ],
)
def test_wrap_to_pi_edge_cases(x: float) -> None:
    """
    The wrapped value is in ``[−π, π]`` and is circularly equivalent to
    the input within 1e-12 rad.  At the exact boundary points
    ``{0, +π, −π}`` (and integer multiples of π, modulo 2π) the wrap is
    bit-stable — the two equivalent representatives ``+π`` and ``−π``
    map deterministically (Python's ``%`` operator).
    """
    y = float(_wrap_to_pi(x))

    # Range invariant: result is in [-π, π].
    assert -math.pi - 1e-12 <= y <= math.pi + 1e-12

    # Circular-distance invariant: y and x are the same angle mod 2π.
    assert _circular_equiv(x, y) < 1e-12, (
        f"_wrap_to_pi({x}) -> {y}: not circularly equivalent"
    )

    # Exact identity at x == 0.
    if x == 0.0:
        assert y == 0.0


@pytest.mark.parametrize(
    "ecc",
    [0.0, 0.3, 0.6, 0.9, 0.95, 0.99, 0.999],
    ids=["e_0", "e_0p3", "e_0p6", "e_0p9", "e_0p95", "e_0p99", "e_0p999"],
)
def test_solve_kepler_converges_all_anomalies(ecc: float) -> None:
    """
    Regression for the high-e divergence: plain Newton from
    ``E0 = M`` blew up near periastron for ``e ≳ 0.96`` (equation
    residuals up to ~1e20 at e = 0.99), corrupting every forward model
    downstream — visible as phase-jump chords in the dense photocentre
    ellipse.  The bisection repair must leave EVERY entry satisfying the
    Kepler equation, at every eccentricity up to 0.999.
    """
    M = np.linspace(0.0, 2.0 * math.pi, 4001)
    E = solve_kepler(M, ecc)
    resid = np.abs(E - ecc * np.sin(E) - M)
    assert float(resid.max()) < 1e-8, (
        f"e={ecc}: {int(np.sum(resid > 1e-8))} non-converged anomalies, "
        f"max residual {float(resid.max()):.2e}"
    )
    # E(M) is strictly increasing for e < 1 — a wrong-branch solution
    # (the chord signature) breaks monotonicity even when the equation
    # residual is small.
    assert np.all(np.diff(E) > 0.0), f"e={ecc}: E(M) not monotonic"


def test_solve_kepler_unwrapped_branch() -> None:
    """The repair path preserves the caller's 2πk branch: E(M + 2πk) =
    E(M) + 2πk, for positive AND negative k (the astro/RV forward models
    wrap M themselves, but the contract should not silently depend on
    that; negative M exercises the −2πk sign of the branch shift)."""
    ecc = 0.99
    M = np.linspace(0.0, 2.0 * math.pi, 101)
    E0 = solve_kepler(M, ecc)
    E_up = solve_kepler(M + 4.0 * math.pi, ecc)
    E_dn = solve_kepler(M - 4.0 * math.pi, ecc)
    assert float(np.max(np.abs(E_up - E0 - 4.0 * math.pi))) < 1e-7
    assert float(np.max(np.abs(E_dn - E0 + 4.0 * math.pi))) < 1e-7


def test_solve_kepler_converged_path_bit_identical() -> None:
    """Converged-entry byte-identity contract: at moderate e the repair
    branch must never fire, so the result equals a locally re-run plain
    Newton iteration bit-for-bit."""
    ecc = 0.3
    M = np.linspace(0.0, 2.0 * math.pi, 501)
    E_plain = M.copy()
    for _ in range(50):
        dE = (M - E_plain + ecc * np.sin(E_plain)) / (1.0 - ecc * np.cos(E_plain))
        E_plain += dE
        if np.max(np.abs(dE)) < 1e-10:
            break
    E = solve_kepler(M, ecc)
    assert np.array_equal(E, E_plain)
