"""Byte-identity baseline for the three design-column builders.

These three functions build the design matrices that EVERY linear solve
consumes, and the column orders are pinned conventions:

- 5 single-star columns ``[dra*, ddec, pmra*, pmdec, plx]``
- 9 Thiele-Innes columns ``[A, B, F, G, dra*, ddec, pmra*, pmdec, plx]``
- the reduced RV design used by the marginal joint

So the check is byte-identity, not closeness: ``array_equal``, never
``allclose``. A tolerance would let a real change hide under it.

The baseline lives in ``tests/_design_columns_baseline.py`` as literal
floats rather than a binary array file: a tracked ``.py`` is reviewable in a
diff. ``repr(float)`` round-trips exactly in Python 3, so the literals
reproduce the arrays bit-for-bit.

Regenerating means a change to these columns is INTENDED. Say so in the
commit message.
"""

from __future__ import annotations

import numpy as np

from _design_columns_baseline import BASELINE


def _grid():
    """A fixed, deliberately awkward evaluation grid.

    Not a realistic cadence — the point is to exercise the columns, so it
    mixes a wide epoch span, scan angles spanning the full circle
    (including exact multiples of pi/2 where sin/cos hit 0 and +-1), a
    sign-changing parallax factor, and eccentricities from circular to
    high. Deterministic: no RNG.
    """
    t_mjd = np.array(
        [0.0, 1.0, 37.5, 180.25, 365.25, 700.0, 1234.5, 1826.25],
        dtype=float,
    )
    psi = np.array(
        [0.0, np.pi / 2, np.pi, 3 * np.pi / 2,
         0.3, 1.7, 2.9, 5.5],
        dtype=float,
    )
    parallax_factor_al = np.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75, 0.9],
        dtype=float,
    )
    epoch_ref_mjd = 500.0
    return t_mjd, psi, parallax_factor_al, epoch_ref_mjd


# (label, eccentricity, tau, period_days) cases for the orbit-bearing designs.
_SHAPES = [
    ("circular", 0.0, 0.0, 300.0),
    ("mild", 0.2, 0.37, 420.0),
    ("high_e", 0.8, 0.81, 155.0),
    ("tau_wrap", 0.45, 0.99, 1000.0),
]


def _compute() -> dict[str, np.ndarray]:
    """Every array this test pins, keyed by a stable name."""
    from orblet.constants import DAYS_PER_KEPLER_YEAR
    from orblet import ti_design_matrix, rv_design_matrix
    from orblet import astrometric_5param_design_matrix

    t_mjd, psi, pf, epoch_ref = _grid()
    out: dict[str, np.ndarray] = {}

    # The 5 single-star columns (no orbit, no shape dependence).
    out["single_star_5col"] = np.asarray(
        astrometric_5param_design_matrix(
            t_mjd, psi, pf, epoch_ref_mjd=epoch_ref,
        ),
        dtype=float,
    )

    for label, ecc, tau, period_days in _SHAPES:
        out[f"ti_9col__{label}"] = np.asarray(
            ti_design_matrix(
                t_mjd, psi, pf,
                f_per_day=1.0 / period_days, ecc=ecc, tau=tau,
                epoch_ref_mjd=epoch_ref,
            ),
            dtype=float,
        )
        out[f"rv_3col__{label}"] = np.asarray(
            rv_design_matrix(
                t_mjd,
                period_yr=period_days / DAYS_PER_KEPLER_YEAR,
                ecc=ecc, tau=tau, epoch_ref_mjd=epoch_ref,
            ),
            dtype=float,
        )

    return out


def test_design_columns_are_byte_identical_to_the_baseline():
    """Every pinned array matches the committed baseline exactly."""
    current = _compute()

    ref_keys = set(BASELINE)
    cur_keys = set(current)
    assert cur_keys == ref_keys, (
        f"pinned array set changed: only-now={sorted(cur_keys - ref_keys)}, "
        f"only-baseline={sorted(ref_keys - cur_keys)}"
    )
    for key in sorted(ref_keys):
        expected = BASELINE[key]
        got = current[key]
        assert got.shape == expected.shape, (
            f"{key}: shape {got.shape} != baseline {expected.shape}"
        )
        # array_equal, NOT allclose: a relocation must not move a bit.
        assert np.array_equal(got, expected), (
            f"{key}: values differ from the baseline. Max abs delta "
            f"{float(np.max(np.abs(got - expected)))!r}. If this change "
            f"is INTENDED, regenerate the baseline and say so in the "
            f"commit message."
        )


def test_column_counts_are_the_pinned_conventions():
    """The column COUNTS are a documented contract; pin them separately.

    A shape-only check that fails loudly if a column is ever added or
    dropped, independently of the value baseline above.
    """
    current = _compute()
    assert current["single_star_5col"].shape[1] == 5
    for label, _e, _tau, _P in _SHAPES:
        assert current[f"ti_9col__{label}"].shape[1] == 9
        assert current[f"rv_3col__{label}"].shape[1] == 3
