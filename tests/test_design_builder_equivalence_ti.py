"""Byte-identity: TiOrbitBuilder == _ti_design_matrix.

Step 3 of the design-builder refactor (catalog-only: the engine call sites —
scan / samplers — are deliberately NOT rewired; the builder is the labeled
front door to the same single implementation). Equivalence at rtol=0, atol=0
across shapes (incl. e=0 and e=0.9) and reference epochs, plus the
cols-4:9 == single-star-block compose pin. Synthetic arrays only.
"""

from __future__ import annotations

import numpy as np

from orblet.design import (
    SingleStarBlockBuilder,
    TiOrbitBuilder,
)
from orblet.solve.astrometry import _ti_design_matrix


def _geometry(n: int = 60, seed: int = 17):
    rng = np.random.default_rng(seed)
    t_mjd = np.linspace(56800.0, 58400.0, n)
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    pf = rng.uniform(-1.0, 1.0, n)
    return t_mjd, psi, pf


_EPOCHS = (57388.5, 57936.375, 55197.0, 60000.0)
_SHAPES = (
    (1.0 / 500.0, 0.0, 0.25),    # circular
    (1.0 / 800.0, 0.6, 0.5),
    (1.0 / 300.0, 0.9, 0.75),    # near the eccentricity bound
)


def test_ti_builder_names_units_count():
    b = TiOrbitBuilder()
    assert b.column_names == (
        "A", "B", "F", "G",
        "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
    )
    assert b.column_units == (
        "mas", "mas", "mas", "mas", "mas", "mas", "mas/yr", "mas/yr", "mas",
    )
    assert b.n_col == 9


def test_ti_builder_byte_identical():
    b = TiOrbitBuilder()
    t, psi, pf = _geometry()
    for ref in _EPOCHS:
        for f_per_day, ecc, tau in _SHAPES:
            got = b.build(t, psi, pf, f_per_day=f_per_day, ecc=ecc, tau=tau,
                          epoch_ref_mjd=ref)
            ref_X = _ti_design_matrix(
                t, psi, pf, f_per_day=f_per_day, ecc=ecc, tau=tau,
                epoch_ref_mjd=ref,
            )
            np.testing.assert_array_equal(got, ref_X)  # 0/0
            assert got.dtype == np.float64
            assert got.flags["C_CONTIGUOUS"]


def test_ti_builder_cols_4to9_equal_singlestar_builder():
    # Composability pin: the TI design's single-star block IS the 5-column
    # block builder's output, byte-for-byte.
    ti = TiOrbitBuilder()
    ss = SingleStarBlockBuilder()
    t, psi, pf = _geometry(seed=19)
    for ref in _EPOCHS:
        X = ti.build(t, psi, pf, f_per_day=1.0 / 650.0, ecc=0.3, tau=0.4,
                     epoch_ref_mjd=ref)
        np.testing.assert_array_equal(X[:, 4:9],
                                      ss.build(t, psi, pf, epoch_ref_mjd=ref))


def test_ti_builder_leaves_global_rng_untouched():
    b = TiOrbitBuilder()
    t, psi, pf = _geometry()
    state_before = np.random.get_state()
    b.build(t, psi, pf, f_per_day=1.0 / 500.0, ecc=0.2, tau=0.1,
            epoch_ref_mjd=57388.5)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
