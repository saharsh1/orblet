"""Byte-identity: SingleStarBlockBuilder == _astrometric_columns at all sites.

The single-star 5-column block is reused verbatim in THREE engine sites
(_ti_design_matrix cols 4-8, astrometric_5param_design_matrix, and the joint
reduced-astro design cols 1-5).  The builder must be a byte-for-byte drop-in
for all of them, so every equivalence here is at rtol=0, atol=0.  Synthetic
arrays only.
"""

from __future__ import annotations

import numpy as np

from orblet.design import SingleStarBlockBuilder
from orblet.solve.astrometry import (
    _astrometric_columns,
    _ti_design_matrix,
    astrometric_5param_design_matrix,
)


def _geometry(n: int = 50, seed: int = 7):
    rng = np.random.default_rng(seed)
    t_mjd = np.linspace(56800.0, 58200.0, n)
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    pf = rng.uniform(-1.0, 1.0, n)
    return t_mjd, psi, pf


# Reference epochs spanning the DR3/DR4/J2010 surface.
_EPOCHS = (57388.5, 57936.375, 55197.0, 60000.0)


def test_builder_equals_astrometric_columns():
    b = SingleStarBlockBuilder()
    t, psi, pf = _geometry()
    for ref in _EPOCHS:
        got = b.build(t, psi, pf, epoch_ref_mjd=ref)
        ref_block = _astrometric_columns(t, psi, pf, epoch_ref_mjd=ref)
        np.testing.assert_array_equal(got, ref_block)  # 0/0


def test_builder_equals_public_5param_design():
    b = SingleStarBlockBuilder()
    t, psi, pf = _geometry(seed=11)
    for ref in _EPOCHS:
        got = b.build(t, psi, pf, epoch_ref_mjd=ref)
        pub = astrometric_5param_design_matrix(t, psi, pf, epoch_ref_mjd=ref)
        np.testing.assert_array_equal(got, pub)  # 0/0


def test_builder_equals_ti_columns_4to8():
    # The single-star block IS columns 4-8 of the 9-column TI design, at any
    # orbit shape (the block is shape-independent).
    b = SingleStarBlockBuilder()
    t, psi, pf = _geometry(seed=13)
    for ref in _EPOCHS:
        for f_per_day, ecc, tau in ((1.0 / 500.0, 0.0, 0.25),
                                    (1.0 / 800.0, 0.6, 0.5),
                                    (1.0 / 300.0, 0.9, 0.75)):
            X = _ti_design_matrix(
                t, psi, pf, f_per_day=f_per_day, ecc=ecc, tau=tau,
                epoch_ref_mjd=ref,
            )
            got = b.build(t, psi, pf, epoch_ref_mjd=ref)
            np.testing.assert_array_equal(X[:, 4:9], got)  # 0/0
