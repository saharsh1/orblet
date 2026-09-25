"""Contract tests for the columns-only design-builder protocol.

Pins the load-bearing column-order/units/count metadata, the shape/dtype/
C-contiguity guarantee the GLS core relies on, purity + global-RNG
statelessness, and the concat_columns composability primitive.  Synthetic
arrays only.
"""

from __future__ import annotations

import numpy as np
import pytest

from orblet.design import (
    DesignBuilder,
    SingleStarBlockBuilder,
    concat_columns,
)
from orblet.design.builder import (
    _CONCAT_EMPTY_MSG,
    _CONCAT_ROWS_MSG,
)


def _geometry(n: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    t_mjd = np.linspace(57000.0, 58000.0, n)
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    pf = rng.uniform(-1.0, 1.0, n)
    return t_mjd, psi, pf


def test_single_star_builder_satisfies_protocol():
    b = SingleStarBlockBuilder()
    assert isinstance(b, DesignBuilder)


def test_single_star_names_units_count():
    b = SingleStarBlockBuilder()
    assert b.column_names == (
        "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
    )
    # Amplitude units (ϖ amplitude in mas; its column is the parallax factor).
    assert b.column_units == ("mas", "mas", "mas/yr", "mas/yr", "mas")
    assert b.n_col == 5
    assert len(b.column_names) == b.n_col == len(b.column_units)


def test_build_shape_dtype_and_c_contiguity():
    b = SingleStarBlockBuilder()
    t, psi, pf = _geometry()
    X = b.build(t, psi, pf, epoch_ref_mjd=57388.5)
    assert X.shape == (t.size, 5)
    assert X.dtype == np.float64
    # C-contiguity parity with a hand-built np.empty((n, k)) block — a
    # contiguity change would risk BLAS reduction-order drift in XᵀΣ⁻¹X.
    assert X.flags["C_CONTIGUOUS"]


def test_build_is_pure_and_leaves_global_rng_untouched():
    b = SingleStarBlockBuilder()
    t, psi, pf = _geometry()
    t0, psi0, pf0 = t.copy(), psi.copy(), pf.copy()

    state_before = np.random.get_state()
    X1 = b.build(t, psi, pf, epoch_ref_mjd=57388.5)
    X2 = b.build(t, psi, pf, epoch_ref_mjd=57388.5)
    state_after = np.random.get_state()

    np.testing.assert_array_equal(X1, X2)          # deterministic
    np.testing.assert_array_equal(t, t0)           # inputs untouched
    np.testing.assert_array_equal(psi, psi0)
    np.testing.assert_array_equal(pf, pf0)
    # Builder must not advance numpy's GLOBAL RNG stream.
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])


# ── concat_columns composability primitive ────────────────────────────


def test_concat_columns_stacks_and_is_c_contiguous():
    a = np.arange(12.0).reshape(4, 3)
    b = np.arange(8.0).reshape(4, 2)
    out = concat_columns([a, b])
    assert out.shape == (4, 5)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out[:, :3], a)
    np.testing.assert_array_equal(out[:, 3:], b)


def test_concat_columns_matches_hand_built_empty_assign():
    # The composed layout must be byte-identical to the engine's
    # np.empty((n, k)); X[:, a:b] = ... idiom.
    a = np.random.default_rng(0).standard_normal((6, 4))
    b = np.random.default_rng(1).standard_normal((6, 5))
    hand = np.empty((6, 9), dtype=float)
    hand[:, :4] = a
    hand[:, 4:] = b
    np.testing.assert_array_equal(concat_columns([a, b]), hand)


def test_concat_columns_guards():
    with pytest.raises(ValueError) as e1:
        concat_columns([])
    assert str(e1.value) == _CONCAT_EMPTY_MSG
    with pytest.raises(ValueError) as e2:
        concat_columns([np.zeros((3, 2)), np.zeros((4, 2))])
    assert str(e2.value) == _CONCAT_ROWS_MSG
    # value-free: no interpolated value/shape in either message
    assert not any(ch.isdigit() for ch in _CONCAT_EMPTY_MSG + _CONCAT_ROWS_MSG)
