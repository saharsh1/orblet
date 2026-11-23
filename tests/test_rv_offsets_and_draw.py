"""
The RV zero-point columns of several instruments, and the draw of the linear amplitudes.

- ``rv_offset_columns`` / ``RvOffsetBuilder``: one 0/1 column per instrument
  other than the reference (instrument 0); for the full design ``[gamma, C, S]``
  and the reduced design ``[gamma, K]``.
- ``draw_linear_amplitudes``: samples of N(beta, inflate * cov) from a linear
  solution; the draw of ``best_linear_params_rv(draw=N)`` goes through it.

With one instrument both must give the numbers of orblet 0.4.0 bit for bit.
Offline and seeded; synthetic data only.  What is real: the atoms.  What is a
stand-in: the orbit, the epochs, the errors, the noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from orblet import (
    RvOffsetBuilder,
    best_linear_params_rv,
    draw_linear_amplitudes,
    linear_solve_rv,
    rv_offset_columns,
)
from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.design import (
    DesignBuilder,
    JointReducedRvBuilder,
    RvOrbitBuilder,
    concat_columns,
)
from orblet.model import rv_model

EPOCH_REF_MJD = 57000.0
PERIOD_DAYS = 800.0   # the shape of the synthetic orbit
ECC = 0.3
TAU = 0.2
OMEGA_RAD = 1.1
# Three instruments; 0 is the reference.  Zero points and errors in km/s.
INSTRUMENT_NAMES = ("gaia", "B", "C")
ROWS_PER_INSTRUMENT = (15, 20, 10)
TRUE_ZERO_POINT = (5.0, 6.2, 3.9)
ERROR_KMS = (1.0, 0.3, 0.5)


def _make_rows():
    """The synthetic rows: times (MJD), RVs and errors (km/s), the instrument number."""
    rng_times = np.random.default_rng(31)   # seed 31: the epochs
    rng_noise = np.random.default_rng(32)   # seed 32: the noise
    times = []
    rvs = []
    errors = []
    instrument = []
    for number, n_rows in enumerate(ROWS_PER_INSTRUMENT):
        t = np.sort(rng_times.uniform(57000.0, 59000.0, n_rows))
        model = rv_model(t, period_yr=PERIOD_DAYS / DAYS_PER_KEPLER_YEAR, ecc=ECC,
                         omega_rad=OMEGA_RAD, tau=TAU, mass_msun=1.0, M_msun=1.0,
                         offset_kms=TRUE_ZERO_POINT[number], epoch_ref_mjd=EPOCH_REF_MJD)
        times.append(t)
        rvs.append(model + rng_noise.normal(0.0, ERROR_KMS[number], n_rows))
        errors.append(np.full(n_rows, ERROR_KMS[number]))
        instrument.append(np.full(n_rows, number))
    return (np.concatenate(times), np.concatenate(rvs), np.concatenate(errors),
            np.concatenate(instrument))


T_MJD, RV_KMS, ERR_KMS, INSTRUMENT = _make_rows()


def _orbit_design(t):
    """The full RV design [gamma, C, S] at the true shape."""
    return RvOrbitBuilder().build(t, period_yr=PERIOD_DAYS / DAYS_PER_KEPLER_YEAR, ecc=ECC,
                                  tau=TAU, epoch_ref_mjd=EPOCH_REF_MJD)


# ── the columns ───────────────────────────────────────────────────────────

def test_columns_are_the_indicators_of_the_other_instruments():
    block = rv_offset_columns(INSTRUMENT)
    assert block.shape == (T_MJD.size, 2)
    assert block.dtype == np.float64 and block.flags["C_CONTIGUOUS"]
    # column 0: instrument 1 ("B"); column 1: instrument 2 ("C")
    assert np.array_equal(block[:, 0], (INSTRUMENT == 1).astype(float))
    assert np.array_equal(block[:, 1], (INSTRUMENT == 2).astype(float))
    # the reference rows are 0 in every column
    assert np.all(block[INSTRUMENT == 0] == 0.0)


def test_given_count_equals_the_inferred_count():
    assert np.array_equal(rv_offset_columns(INSTRUMENT, n_instruments=3),
                          rv_offset_columns(INSTRUMENT))


def test_one_instrument_gives_no_column():
    block = rv_offset_columns(np.zeros(7, dtype=int))
    assert block.shape == (7, 0)


@pytest.mark.parametrize("bad_index, count", [
    (np.array([0, 0, 2, 2]), None),           # instrument 1 has no row
    (np.array([1, 1, 2, 2]), None),           # the reference has no row
    (np.array([0, 1, 1]), 3),                 # instrument 2 has no row
    (np.array([0, 1, 3]), 3),                 # out of range
    (np.array([0, -1, 1]), None),             # negative
    (np.array([0.0, 1.0]), None),             # not integers
    (np.array([True, False]), None),          # booleans
    (np.zeros((2, 2), dtype=int), None),      # not 1-D
    (np.array([], dtype=int), None),          # no row
    (np.array([0, 1]), 0),                    # count not positive
    (np.array([0, 1]), 2.0),                  # count not an integer
])
def test_bad_index_or_count_raises(bad_index, count):
    with pytest.raises(ValueError):
        rv_offset_columns(bad_index, n_instruments=count)


# ── the builder ───────────────────────────────────────────────────────────

def test_builder_names_units_and_columns():
    builder = RvOffsetBuilder(INSTRUMENT_NAMES)
    assert isinstance(builder, DesignBuilder)
    assert builder.column_names == ("offset_B", "offset_C")
    assert builder.column_units == ("km/s", "km/s")
    assert builder.n_col == 2
    assert np.array_equal(builder.build(INSTRUMENT), rv_offset_columns(INSTRUMENT))


def test_builder_counts_from_its_names():
    # four names, but the rows hold only three instruments: instrument 3 has no row
    builder = RvOffsetBuilder(INSTRUMENT_NAMES + ("D",))
    with pytest.raises(ValueError):
        builder.build(INSTRUMENT)


@pytest.mark.parametrize("bad_names", [(), ("a", "a"), ("a", ""), ("a", 2)])
def test_builder_bad_names_raise(bad_names):
    with pytest.raises(ValueError):
        RvOffsetBuilder(bad_names)


# ── the columns in a solve ────────────────────────────────────────────────

def test_one_instrument_solve_is_bit_identical_to_the_orbit_design_alone():
    reference_rows = INSTRUMENT == 0
    t = T_MJD[reference_rows]
    orbit = _orbit_design(t)
    empty_block = rv_offset_columns(np.zeros(t.size, dtype=int))
    with_block = concat_columns([orbit, empty_block])
    alone = linear_solve_rv(RV_KMS[reference_rows], ERR_KMS[reference_rows], orbit)
    joined = linear_solve_rv(RV_KMS[reference_rows], ERR_KMS[reference_rows], with_block)
    assert alone.beta.tobytes() == joined.beta.tobytes()
    assert alone.cov.tobytes() == joined.cov.tobytes()
    assert alone.chi2 == joined.chi2
    assert alone.logL_marginal == joined.logL_marginal


def test_full_design_recovers_the_zero_points():
    design = concat_columns([_orbit_design(T_MJD), rv_offset_columns(INSTRUMENT)])
    solution = linear_solve_rv(RV_KMS, ERR_KMS, design)
    errors = np.sqrt(np.diag(solution.cov))
    # column 0: the reference zero point; columns 3, 4: the others minus the reference
    assert abs(solution.beta[0] - TRUE_ZERO_POINT[0]) < 4.0 * errors[0]
    for number in (1, 2):
        true_offset = TRUE_ZERO_POINT[number] - TRUE_ZERO_POINT[0]
        assert abs(solution.beta[2 + number] - true_offset) < 4.0 * errors[2 + number]


def test_reduced_design_takes_the_same_block():
    reduced = JointReducedRvBuilder().build(
        T_MJD, P_yr=PERIOD_DAYS / DAYS_PER_KEPLER_YEAR, e=ECC, omega=OMEGA_RAD, tau=TAU,
        epoch_ref_mjd=EPOCH_REF_MJD)
    design = concat_columns([reduced, RvOffsetBuilder(INSTRUMENT_NAMES).build(INSTRUMENT)])
    assert design.shape == (T_MJD.size, 4)   # gamma, K, offset_B, offset_C
    solution = linear_solve_rv(RV_KMS, ERR_KMS, design)
    errors = np.sqrt(np.diag(solution.cov))
    for number in (1, 2):
        true_offset = TRUE_ZERO_POINT[number] - TRUE_ZERO_POINT[0]
        assert abs(solution.beta[1 + number] - true_offset) < 4.0 * errors[1 + number]


# ── the draw ──────────────────────────────────────────────────────────────

def _reference_rows_solution():
    reference_rows = INSTRUMENT == 0
    return linear_solve_rv(RV_KMS[reference_rows], ERR_KMS[reference_rows],
                           _orbit_design(T_MJD[reference_rows]))


@pytest.mark.parametrize("n_draws, seed, inflate", [(1, 0, 1.0), (5, 17, 1.0), (4, 3, 2.5)])
def test_draw_equals_best_linear_params_rv_and_the_040_expression(n_draws, seed, inflate):
    reference_rows = INSTRUMENT == 0
    best = best_linear_params_rv(
        PERIOD_DAYS, ECC, TAU, epochs_mjd=T_MJD[reference_rows], rv=RV_KMS[reference_rows],
        rv_err=ERR_KMS[reference_rows], epoch_ref_mjd=EPOCH_REF_MJD,
        draw=n_draws, inflate=inflate, seed=seed)
    solution = _reference_rows_solution()
    drawn = draw_linear_amplitudes(solution, n_draws, seed, inflate=inflate)
    # the expression of orblet 0.4.0, written out
    rng_040 = np.random.default_rng(seed)
    drawn_040 = rng_040.multivariate_normal(solution.beta, inflate * solution.cov, size=n_draws)
    assert drawn.shape == (n_draws, 3)
    assert drawn.tobytes() == best.draws.tobytes()
    assert drawn.tobytes() == drawn_040.tobytes()


def test_draw_with_a_generator_advances_it():
    solution = _reference_rows_solution()
    rng_draws = np.random.default_rng(8)   # seed 8: the draws below
    first = draw_linear_amplitudes(solution, 2, rng_draws)
    second = draw_linear_amplitudes(solution, 2, rng_draws)
    assert first.tobytes() == draw_linear_amplitudes(solution, 2, 8).tobytes()
    assert first.tobytes() != second.tobytes()


def test_draw_of_several_instruments_keeps_every_column():
    design = concat_columns([_orbit_design(T_MJD), rv_offset_columns(INSTRUMENT)])
    solution = linear_solve_rv(RV_KMS, ERR_KMS, design)
    drawn = draw_linear_amplitudes(solution, 4000, 1)
    assert drawn.shape == (4000, 5)
    # the spread of the draws is the error of the solve, within 5 %
    spread = np.std(drawn, axis=0)   # axis 0: over the draws
    assert np.allclose(spread, np.sqrt(np.diag(solution.cov)), rtol=0.05)


@pytest.mark.parametrize("solution, n_draws, rng, inflate", [
    (SimpleNamespace(beta=np.zeros(2)), 1, 0, 1.0),                         # no cov
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(3)), 1, 0, 1.0),          # wrong cov shape
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(2)), 0, 0, 1.0),          # no draw
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(2)), True, 0, 1.0),       # a boolean count
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(2)), 1, 0.5, 1.0),        # not a seed
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(2)), 1, True, 1.0),       # a boolean seed
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(2)), 1, 0, 0.0),          # inflate not positive
    (SimpleNamespace(beta=np.zeros(2), cov=np.eye(2)), 1, 0, np.inf),       # inflate not finite
])
def test_draw_bad_inputs_raise(solution, n_draws, rng, inflate):
    with pytest.raises(ValueError):
        draw_linear_amplitudes(solution, n_draws, rng, inflate=inflate)
