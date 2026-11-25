"""
The draw of the linear amplitudes is ``beta + z @ L.T``, whatever linear-algebra library is installed.

``z``: the ``(n_draws, k)`` standard normals of ``default_rng(seed)``.
``L``: the lower Cholesky factor of ``inflate * cov`` (``L @ L.T = inflate * cov``).
``L`` is unique, so these draws do not depend on the BLAS/LAPACK build; the
SVD factor that numpy uses by default has a free sign per column, and two
OpenBLAS versions gave two different draws for one seed.

The test holds the form of the draw, not stored numbers, so it passes with any
library.  One astrometric solution (``best_linear_params_ti``) and one RV
solution (``best_linear_params_rv``), both through ``draw_linear_amplitudes``.
Offline and seeded; synthetic data only.  What is real: the atoms.  What is a
stand-in: the orbits, the epochs, the errors, the noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from orblet import best_linear_params_rv, best_linear_params_ti, draw_linear_amplitudes
from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import rv_model

# The shape both solutions are evaluated at (days, -, periastron-phase fraction).
PERIOD_DAYS = 800.0
ECC = 0.3
TAU = 0.4
# (n_draws, seed, inflate) of each case
DRAW_CASES = [(1, 0, 1.0), (7, 17, 1.0), (400, 4000160, 2.5)]


def _cholesky_draw(beta, cov, n_draws, seed, inflate):
    """The draw written out: ``beta + z @ L.T``, ``L`` the Cholesky factor of ``inflate * cov``."""
    rng_draws = np.random.default_rng(seed)   # the same generator the atom makes from the seed
    z = rng_draws.standard_normal((n_draws, beta.shape[0]))   # one row per draw
    lower_factor = np.linalg.cholesky(inflate * cov)
    return beta + z @ lower_factor.T


def _rv_solution(n_draws, seed, inflate):
    """A synthetic single-instrument RV curve (km/s), solved at the shape, with draws."""
    rng_times = np.random.default_rng(41)   # seed 41: the epochs
    rng_noise = np.random.default_rng(42)   # seed 42: the noise
    times_mjd = np.sort(rng_times.uniform(57000.0, 59000.0, 30))
    clean = rv_model(times_mjd, period_yr=PERIOD_DAYS / DAYS_PER_KEPLER_YEAR, ecc=ECC,
                     omega_rad=1.1, tau=TAU, mass_msun=1.0, M_msun=1.0,
                     offset_kms=5.0, epoch_ref_mjd=57000.0)
    error_kms = np.full(times_mjd.size, 0.5)
    rv_kms = clean + rng_noise.normal(0.0, error_kms)
    return best_linear_params_rv(
        PERIOD_DAYS, ECC, TAU, epochs_mjd=times_mjd, rv=rv_kms, rv_err=error_kms,
        epoch_ref_mjd=57000.0, draw=n_draws, inflate=inflate, seed=seed)


def _ti_solution(astro_data, truth, n_draws, seed, inflate):
    """A synthetic astrometric orbit (mas), solved at the shape, with draws."""
    return best_linear_params_ti(
        PERIOD_DAYS, ECC, TAU,
        epochs_mjd=astro_data["obs_time"], scan_angle=astro_data["scan_angle"],
        parallax_factor_al=astro_data["parallax_factor_al"],
        centroid_pos=astro_data["centroid_pos"], centroid_pos_err=astro_data["centroid_pos_err"],
        epoch_ref_mjd=truth["epoch_ref_mjd"], draw=n_draws, inflate=inflate, seed=seed)


@pytest.mark.parametrize("n_draws, seed, inflate", DRAW_CASES)
def test_rv_draws_are_the_cholesky_form(n_draws, seed, inflate):
    result = _rv_solution(n_draws, seed, inflate)
    expected = _cholesky_draw(result.beta, result.cov, n_draws, seed, inflate)
    assert result.draws.shape == (n_draws, 3)   # gamma, C, S
    assert result.draws.tobytes() == expected.tobytes()
    assert result.draws.tobytes() == draw_linear_amplitudes(
        result, n_draws, seed, inflate=inflate).tobytes()


@pytest.mark.parametrize("n_draws, seed, inflate", DRAW_CASES)
def test_ti_draws_are_the_cholesky_form(synth_astro_dataset, n_draws, seed, inflate):
    astro_data, truth = synth_astro_dataset(seed=3)   # seed 3: one synthetic source
    result = _ti_solution(astro_data, truth, n_draws, seed, inflate)
    expected = _cholesky_draw(result.beta, result.cov, n_draws, seed, inflate)
    assert result.draws.shape == (n_draws, 9)   # the nine linear-TI columns
    assert result.draws.tobytes() == expected.tobytes()
    assert result.draws.tobytes() == draw_linear_amplitudes(
        result, n_draws, seed, inflate=inflate).tobytes()


def test_a_hand_built_covariance_draws_the_cholesky_form():
    # A solution object with only .beta and .cov (no solve behind it) takes
    # the same Cholesky draw.
    solution = SimpleNamespace(beta=np.zeros(3),
                               cov=np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]))
    drawn = draw_linear_amplitudes(solution, 50, 5)
    expected = _cholesky_draw(solution.beta, solution.cov, 50, 5, 1.0)
    assert drawn.tobytes() == expected.tobytes()


@pytest.mark.parametrize("cov", [
    np.diag([1.0, 0.0]),                        # a zero variance: singular
    np.array([[1.0, 2.0], [2.0, 1.0]]),         # a negative eigenvalue
])
def test_a_covariance_that_is_not_positive_definite_raises(cov):
    solution = SimpleNamespace(beta=np.zeros(2), cov=cov)
    with pytest.raises(ValueError, match="positive definite") as caught:
        draw_linear_amplitudes(solution, 3, 0)
    assert caught.value.__cause__ is None   # value-free, not chained to the LinAlgError
