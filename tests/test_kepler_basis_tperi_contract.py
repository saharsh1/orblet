"""
Kepler-solver precondition for the Kepler-basis periastron conversion.

:func:`orblet.kepler.solve_kepler` must agree with an independent in-test
Newton iteration on Kepler's equation.  Every Kepler-basis conversion
between a phase at the reference epoch and the periastron time rests on
that solve, so it is pinned here on its own.

Citations
---------
- Murray & Dermott 1999, *Solar System Dynamics*, §2.4, eq. 2.52
  (Kepler's equation ``M = E − e sin E``).
"""

from __future__ import annotations

import math

from orblet.kepler import solve_kepler


# ──────────────────────────────────────────────────────────────────────
# D.0 — solve_kepler precondition micro-test
# ──────────────────────────────────────────────────────────────────────


def test_D0_solve_kepler_matches_independent_newton() -> None:
    """
    The project's :func:`orblet.kepler.solve_kepler` must
    agree with an independent, in-test Newton iteration on Kepler's
    equation ``M = E − e sin E`` to within 1e-9 rad.

    The independent Newton loop is spelled out below in pure Python
    (``math`` only) to 1e-12 precision; it does NOT call
    ``solve_kepler``.  Citation: Murray & Dermott 1999, eq. 2.52.
    """
    M_in = 1.0  # rad
    e = 0.5

    # Independent Newton iteration on M = E − e sin E.
    # f(E)  = E − e sin E − M
    # f'(E) = 1 − e cos E
    E = M_in  # standard initial guess for moderate e
    for _ in range(100):
        f = E - e * math.sin(E) - M_in
        fp = 1.0 - e * math.cos(E)
        delta = -f / fp
        E += delta
        if abs(delta) < 1e-12:
            break
    E_truth = E

    # Project's Newton solver on the same input.
    E_project = float(solve_kepler(M_in, e))

    assert abs(E_project - E_truth) < 1e-9, (
        f"D.0: solve_kepler({M_in}, {e}) returned {E_project!r}; "
        f"independent Newton gave {E_truth!r}; diff = "
        f"{E_project - E_truth!r} (tol 1e-9)."
    )
