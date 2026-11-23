"""
A radial-velocity fit with several instruments, from the existing atoms.

Each instrument (a "set" of RVs) has its own zero point and its own extra
noise (jitter).  No function of orblet takes several sets; the pattern is:

- linear layer: the design ``[gamma, C, S]`` of the reference set, plus one
  0/1 column per other set.  The amplitude of a 0/1 column is the zero point
  of that set minus the zero point of the reference set.  The jitter of a
  set enters the per-row error, sqrt(rv_err**2 + jitter_set**2).
- likelihood: one ``loglike`` call per set, added.

Offline and seeded; synthetic data only.  What is real: the atoms.  What is
a stand-in: the orbit, the epochs, the errors, the noise.
"""

from __future__ import annotations

import numpy as np

from orblet.design import RvOrbitBuilder, concat_columns
from orblet.likelihood import loglike
from orblet.model import rv_model
from orblet.solve.rv import linear_solve_rv, recover_K

DAYS_PER_YEAR = 365.25  # only to write the period in days below
EPOCH_REF_MJD = 57000.0
# The true orbit: P = 4000 d, e = 0.70 (keywords of rv_model and loglike).
TRUE_ORBIT = dict(period_yr=4000.0 / DAYS_PER_YEAR, ecc=0.70, omega_rad=1.2, tau=0.35,
                  mass_msun=30.0, M_msun=31.0)
SET_NAMES = ["gaia", "B", "C"]  # "gaia" is the reference set
# Three tables from the set name to a number in km/s.
TRUE_ZERO_POINT = {"gaia": 20.0, "B": 21.5, "C": 19.2}
TRUE_JITTER = {"gaia": 0.0, "B": 0.3, "C": 0.6}
QUOTED_ERROR = {"gaia": 1.5, "B": 0.2, "C": 0.4}


def _make_data():
    """The three synthetic sets: a table from the set name to (t, rv, err)."""
    rng_times = np.random.default_rng(11)   # seed 11: the epochs
    rng_noise = np.random.default_rng(12)   # seed 12: the noise
    times = {
        "gaia": np.sort(rng_times.uniform(57000.0, 58800.0, 17)),   # span 1800 d
        "B": np.sort(rng_times.uniform(58200.0, 61200.0, 25)),      # overlaps "gaia", extends it
        "C": np.sort(rng_times.uniform(60000.0, 61500.0, 12)),      # after "gaia"
    }
    data = {}
    for name in SET_NAMES:
        model = rv_model(times[name], offset_kms=TRUE_ZERO_POINT[name],
                         epoch_ref_mjd=EPOCH_REF_MJD, **TRUE_ORBIT)
        noise_sigma = np.hypot(QUOTED_ERROR[name], TRUE_JITTER[name])
        rv = model + rng_noise.normal(0.0, noise_sigma, model.size)
        err = np.full(model.size, QUOTED_ERROR[name])
        data[name] = (times[name], rv, err)
    return data


DATA = _make_data()


def _stacked():
    """All rows in one table: times, RVs, per-row errors with the jitter, set number."""
    t = np.concatenate([DATA[name][0] for name in SET_NAMES])
    rv = np.concatenate([DATA[name][1] for name in SET_NAMES])
    sigma_parts = []
    set_parts = []
    for number, name in enumerate(SET_NAMES):  # number counts the sets; 0 is the reference
        err = DATA[name][2]
        sigma_parts.append(np.hypot(err, TRUE_JITTER[name]))
        set_parts.append(np.full(err.size, number))
    return t, rv, np.concatenate(sigma_parts), np.concatenate(set_parts)


def _design(t, set_number, own_zero_points):
    """[gamma, C, S] at the true shape; with one 0/1 column per non-reference set."""
    X = RvOrbitBuilder().build(t, period_yr=TRUE_ORBIT["period_yr"], ecc=TRUE_ORBIT["ecc"],
                               tau=TRUE_ORBIT["tau"], epoch_ref_mjd=EPOCH_REF_MJD)
    if not own_zero_points:
        return X
    offset_columns = np.column_stack([
        (set_number == 1).astype(float),   # 1 on the rows of set "B"
        (set_number == 2).astype(float),   # 1 on the rows of set "C"
    ])
    return concat_columns([X, offset_columns])


def test_sum_of_loglike_is_the_several_instrument_likelihood():
    by_orblet = 0.0
    by_hand = 0.0
    for name in SET_NAMES:
        t, rv, err = DATA[name]
        by_orblet += loglike(t, rv, err, offset_kms=TRUE_ZERO_POINT[name],
                             jitter_kms=TRUE_JITTER[name], epoch_ref_mjd=EPOCH_REF_MJD,
                             **TRUE_ORBIT)
        model = rv_model(t, offset_kms=TRUE_ZERO_POINT[name], epoch_ref_mjd=EPOCH_REF_MJD,
                         **TRUE_ORBIT)
        variance = err**2 + TRUE_JITTER[name] ** 2
        by_hand += -0.5 * np.sum((rv - model) ** 2 / variance + np.log(2.0 * np.pi * variance))
    assert abs(by_orblet - by_hand) < 1e-9


def test_zero_points_are_recovered_by_the_linear_solve():
    t, rv, sigma, set_number = _stacked()
    solution = linear_solve_rv(rv, sigma, _design(t, set_number, own_zero_points=True))
    assert solution.beta.shape == (5,)   # gamma, C, S, offset of "B", offset of "C"
    errors = np.sqrt(np.diag(solution.cov))
    # column 0: the zero point of the reference set
    assert abs(solution.beta[0] - TRUE_ZERO_POINT["gaia"]) < 3.0 * errors[0]
    # columns 3 and 4: the zero points of "B" and "C" minus the reference's
    true_offset_b = TRUE_ZERO_POINT["B"] - TRUE_ZERO_POINT["gaia"]
    true_offset_c = TRUE_ZERO_POINT["C"] - TRUE_ZERO_POINT["gaia"]
    assert abs(solution.beta[3] - true_offset_b) < 3.0 * errors[3]
    assert abs(solution.beta[4] - true_offset_c) < 3.0 * errors[4]
    # recover_K reads columns 1 and 2 only; the extra columns do not disturb it
    dense_times = np.linspace(57000.0, 61000.0, 4000)
    true_curve = rv_model(dense_times, offset_kms=0.0, epoch_ref_mjd=EPOCH_REF_MJD, **TRUE_ORBIT)
    true_k = 0.5 * np.ptp(true_curve)    # half the peak-to-peak of one full orbit
    assert abs(recover_K(solution.beta) - true_k) < 0.02 * true_k
    assert solution.chi2 / solution.dof < 1.6


def test_one_zero_point_for_all_sets_fits_clearly_worse():
    t, rv, sigma, set_number = _stacked()
    own = linear_solve_rv(rv, sigma, _design(t, set_number, own_zero_points=True))
    one = linear_solve_rv(rv, sigma, _design(t, set_number, own_zero_points=False))
    # two extra amplitudes; a chi2 drop of 9 is "3 sigma" for one of them
    assert one.chi2 - own.chi2 > 25.0
    assert one.chi2 / one.dof > 1.5 * own.chi2 / own.dof
