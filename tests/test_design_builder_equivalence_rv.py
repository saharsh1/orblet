"""Byte-identity: RvOrbitBuilder == _rv_design_matrix.

Step 2 of the design-builder refactor. The default RV builder must reproduce
the hardcoded 3-column RV design bit-for-bit (rtol=0, atol=0) across shapes
and reference epochs. Synthetic arrays only.
"""

from __future__ import annotations

import numpy as np

from orblet.design import RvOrbitBuilder
from orblet.solve.rv import _rv_design_matrix


def _times(n: int = 55, seed: int = 5):
    return np.linspace(56900.0, 58300.0, n)


_EPOCHS = (57388.5, 57936.375, 55197.0, 60000.0)
_SHAPES = (
    (500.0 / 365.25, 0.0, 0.25),
    (800.0 / 365.25, 0.6, 0.5),
    (185.0 / 365.25, 0.9, 0.75),
)


def test_rv_builder_names_units_count():
    b = RvOrbitBuilder()
    assert b.column_names == ("gamma", "C", "S")
    assert b.column_units == ("km/s", "km/s", "km/s")
    assert b.n_col == 3


def test_rv_builder_byte_identical():
    b = RvOrbitBuilder()
    t = _times()
    for ref in _EPOCHS:
        for period_yr, ecc, tau in _SHAPES:
            got = b.build(t, period_yr=period_yr, ecc=ecc, tau=tau,
                          epoch_ref_mjd=ref)
            ref_X = _rv_design_matrix(
                t, period_yr=period_yr, ecc=ecc, tau=tau, epoch_ref_mjd=ref,
            )
            np.testing.assert_array_equal(got, ref_X)  # 0/0
            assert got.dtype == np.float64
            assert got.flags["C_CONTIGUOUS"]


def test_rv_builder_leaves_global_rng_untouched():
    b = RvOrbitBuilder()
    t = _times()
    state_before = np.random.get_state()
    b.build(t, period_yr=2.0, ecc=0.3, tau=0.4, epoch_ref_mjd=57388.5)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
