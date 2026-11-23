"""
Tests for the colour term of Gaia DR4 epoch astrometry in the linear solves
(orblet.solve.colour, the colour column, PrecisionPrior).

Offline and seeded; synthetic data only.  The synthetic per-CCD data: the
demo cadence (one row per transit) expanded to 10 CCD rows per transit with
the same time, scan angle and parallax factor; every second transit is
scanned in the opposite direction (see ``_OPPOSITE``).

What is real: the design columns, the solves, the colour column and its sign
constant.  What is a stand-in: the cadence, the colour-factor profile, the
noise, the true parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

import orblet
from orblet import (
    COLOUR_FACTOR_AL_SIGN,
    DPAC_COLOUR_PRIOR_SIGMA,
    ColourBlockBuilder,
    ColourTerm,
    PrecisionPrior,
    astrometric_5param_design_matrix,
    best_linear_params_ti,
    colour_columns,
    fit_astrometric_5param,
    linear_solve_ti,
    scan_ti_frequency,
    ti_design_matrix,
)
from orblet.constants import MJD_J2010_TCB
from orblet.design import DesignBuilder, SingleStarBlockBuilder, concat_columns
from orblet.search import presearch_astrometry_linear_ti
from orblet.simulate.cadence import DemoParallaxConsistentCadence

N_CCD = 10  # CCD rows per transit (the float[10] of the DR4 field)
# Colour factor of each of the 10 rows of a transit, mas·µm: the rounded
# medians of the DR4 pre-release file.  A stand-in profile.
FACTOR_PROFILE = np.array([2.2, 2.9, 2.6, 2.2, 1.8, 1.4, 0.8, 0.0, -0.6, -0.8])
FACTOR_SCATTER = 2.0  # per-row Gaussian scatter added to the profile, mas·µm
SIGMA_ROW_MAS = 0.15  # along-scan error of a row, mas (unless a test says otherwise)
NU_USED = 1.43  # nu_eff_used_in_astrometry of every transit, µm⁻¹
DELTA_NU_TRUE = 0.05  # injected ν − ν_used, µm⁻¹
N_SEEDS = 200  # noise realisations of the pull tests
# True single-star parameters [Δα* mas, Δδ mas, μα* mas/yr, μδ mas/yr, ϖ mas]
SINGLE_STAR_TRUE = np.array([0.3, -0.2, 5.0, -3.0, 2.0])
# True Thiele–Innes amplitudes [A, B, F, G] in mas (photocentre size about 1 mas)
ABFG_TRUE = np.array([0.6, -0.5, 0.4, 0.7])
PERIOD_DAYS = 300.0  # the synthetic orbit
ECC = 0.3
TAU = 0.2

# The cadence is built once: one sky position, one set of epochs (seed 0).
_CADENCE = DemoParallaxConsistentCadence(ra_deg=120.0, dec_deg=-30.0, seed=0)
_EPOCH_REF_MJD = float(np.mean(_CADENCE.astro_obs_time) + MJD_J2010_TCB)
# The demo cadence has scan angles in [0, π) only; Gaia scans in every
# direction.  Every second transit is therefore scanned the opposite way:
# ψ → ψ + π.  The parallax factor F_α* sin ψ + F_δ cos ψ then changes sign,
# exactly.  Without this the colour column (mean factor above zero) is
# correlated with the Δα* column (sin ψ never negative) by a property of the
# stand-in cadence, not of the method.
_OPPOSITE = np.arange(_CADENCE.astro_obs_time.size) % 2 == 1   # the odd transits
_TRANSIT_PSI = np.where(_OPPOSITE, _CADENCE.astro_scan_angle + np.pi,
                        _CADENCE.astro_scan_angle)
_TRANSIT_PARALLAX_FACTOR = np.where(_OPPOSITE, -_CADENCE.astro_parallax_factor_al,
                                    _CADENCE.astro_parallax_factor_al)


def _per_ccd_data(seed, *, delta_nu=DELTA_NU_TRUE, sigma_mas=SIGMA_ROW_MAS, orbit=False,
                  nu_used_per_transit=None):
    """Synthetic per-CCD astrometry dict with an injected colour.

    Inputs: ``seed`` of the noise and of the colour-factor scatter;
    ``delta_nu`` the injected ν − ν_used (µm⁻¹); ``sigma_mas`` the row error;
    ``orbit`` True adds the 300-d orbit; ``nu_used_per_transit`` an optional
    array of ν_used per transit (default: the constant NU_USED).
    Returns: the astrometry dict (obs_time in J2010 days), with the colour keys.
    """
    # generator of this data set; seed = [7 (this test file), the given seed]
    rng_data = np.random.default_rng([7, seed])
    # step 1, expand each transit to N_CCD rows with the same time, ψ, parallax factor
    obs_time = np.repeat(_CADENCE.astro_obs_time, N_CCD)
    psi = np.repeat(_TRANSIT_PSI, N_CCD)
    parallax_factor = np.repeat(_TRANSIT_PARALLAX_FACTOR, N_CCD)
    n_transits = _CADENCE.astro_obs_time.size
    n_rows = obs_time.size
    # step 2, the colour inputs
    factor = np.tile(FACTOR_PROFILE, n_transits) + FACTOR_SCATTER * rng_data.standard_normal(n_rows)
    if nu_used_per_transit is None:
        nu_used = np.full(n_rows, NU_USED)
    else:
        nu_used = np.repeat(np.asarray(nu_used_per_transit, dtype=float), N_CCD)
    # step 3, the noise-free model without colour
    t_mjd = obs_time + MJD_J2010_TCB
    if orbit:
        X = ti_design_matrix(t_mjd, psi, parallax_factor, f_per_day=1.0 / PERIOD_DAYS,
                             ecc=ECC, tau=TAU, epoch_ref_mjd=_EPOCH_REF_MJD)
        model = X @ np.concatenate([ABFG_TRUE, SINGLE_STAR_TRUE])
    else:
        X = astrometric_5param_design_matrix(t_mjd, psi, parallax_factor,
                                             epoch_ref_mjd=_EPOCH_REF_MJD)
        model = X @ SINGLE_STAR_TRUE
    # step 4, the colour displacement s · cf · (ν − ν_used), with the package's
    # own column (so a patched sign constant is honoured), and the noise
    nu_true = NU_USED + delta_nu
    colour_shift = colour_columns(factor)[:, 0] * (nu_true - nu_used)
    noise = sigma_mas * rng_data.standard_normal(n_rows)
    return {
        "obs_time": obs_time,
        "scan_angle": psi,
        "parallax_factor_al": parallax_factor,
        "centroid_pos": model + colour_shift + noise,
        "centroid_pos_err": np.full(n_rows, sigma_mas),
        "colour_factor_al": factor,
        "nu_eff_used": nu_used,
    }


def _without_colour_keys(data):
    """The same dict without the three colour keys."""
    return {key: value for key, value in data.items()
            if key not in ("colour_factor_al", "nu_eff_used", "nu_eff_error")}


def _ti_arrays(data):
    """The array keywords of best_linear_params_ti from an astrometry dict."""
    return {
        "epochs_mjd": data["obs_time"] + MJD_J2010_TCB,
        "scan_angle": data["scan_angle"],
        "parallax_factor_al": data["parallax_factor_al"],
        "centroid_pos": data["centroid_pos"],
        "centroid_pos_err": data["centroid_pos_err"],
        "epoch_ref_mjd": _EPOCH_REF_MJD,
    }


FREE = ColourTerm("free")


# ── T1: nothing changes without a colour term ───────────────────────────────

class TestByteIdentity:
    def test_colour_none_ignores_the_colour_keys_single_star(self):
        data = _per_ccd_data(0)
        with_keys = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD)
        without_keys = fit_astrometric_5param(_without_colour_keys(data),
                                              epoch_ref_mjd=_EPOCH_REF_MJD)
        for name in ("params", "cov", "model", "residuals"):
            assert np.array_equal(getattr(with_keys, name), getattr(without_keys, name))
        assert with_keys.chi2_per_dof == without_keys.chi2_per_dof
        assert with_keys.params.shape == (5,)
        assert with_keys.nu_eff is None and with_keys.colour_mode is None

    def test_colour_none_ignores_the_colour_arrays_orbit(self):
        data = _per_ccd_data(0, orbit=True)
        plain = best_linear_params_ti(PERIOD_DAYS, ECC, TAU, **_ti_arrays(data))
        with_arrays = best_linear_params_ti(
            PERIOD_DAYS, ECC, TAU, **_ti_arrays(data),
            colour_factor_al=data["colour_factor_al"], nu_eff_used=data["nu_eff_used"])
        assert np.array_equal(plain.beta, with_arrays.beta)
        assert np.array_equal(plain.cov, with_arrays.cov)
        assert plain.logL_marginal == with_arrays.logL_marginal
        assert plain.beta.shape == (9,) and plain.colour_mode is None

    def test_precision_prior_class_does_not_change_the_cov_path(self):
        # a covariance prior gives the same numbers as before the precision path
        data = _per_ccd_data(1)
        X = astrometric_5param_design_matrix(
            data["obs_time"] + MJD_J2010_TCB, data["scan_angle"],
            data["parallax_factor_al"], epoch_ref_mjd=_EPOCH_REF_MJD)

        class CovPrior:
            mean = np.zeros(5)
            cov = np.diag([4.0, 4.0, 9.0, 9.0, 1.0])

        solution = linear_solve_ti(data["centroid_pos"], data["centroid_pos_err"], X,
                                   beta_prior=CovPrior())
        assert solution.log_evidence is not None
        precision = PrecisionPrior(mean=np.zeros(5), precision=np.linalg.inv(CovPrior.cov))
        by_precision = linear_solve_ti(data["centroid_pos"], data["centroid_pos_err"], X,
                                       beta_prior=precision)
        np.testing.assert_allclose(by_precision.beta, solution.beta, rtol=1e-12)
        np.testing.assert_allclose(by_precision.cov, solution.cov, rtol=1e-12)
        assert by_precision.logL_marginal == solution.logL_marginal


# ── T2: the column and its builder ──────────────────────────────────────────

class TestColumn:
    def test_builder_protocol_and_values(self):
        builder = ColourBlockBuilder()
        assert isinstance(builder, DesignBuilder)
        assert builder.column_names == ("d_nu_eff",)
        assert builder.column_units == ("1/um",)
        assert builder.n_col == 1
        factor = np.array([2.2, 0.0, -0.8, 1.4])
        block = builder.build(factor)
        assert block.shape == (4, 1)
        assert np.array_equal(block[:, 0], COLOUR_FACTOR_AL_SIGN * factor)
        assert np.array_equal(block, colour_columns(factor))

    def test_sign_constant_is_minus_one(self):
        assert COLOUR_FACTOR_AL_SIGN == -1
        assert orblet.constants.COLOUR_FACTOR_AL_SIGN == -1

    def test_colour_column_goes_after_the_single_star_columns(self):
        data = _per_ccd_data(0)
        t_mjd = data["obs_time"] + MJD_J2010_TCB
        single = SingleStarBlockBuilder().build(
            t_mjd, data["scan_angle"], data["parallax_factor_al"],
            epoch_ref_mjd=_EPOCH_REF_MJD)
        colour = ColourBlockBuilder().build(data["colour_factor_al"])
        stacked = concat_columns([single, colour])
        by_hand = np.empty((t_mjd.size, 6))
        by_hand[:, :5] = single   # columns 0–4: the single-star block
        by_hand[:, 5] = COLOUR_FACTOR_AL_SIGN * data["colour_factor_al"]   # column 5: colour
        assert np.array_equal(stacked, by_hand)

    def test_all_factors_zero_is_a_singular_design(self):
        data = _per_ccd_data(0)
        data["colour_factor_al"] = np.zeros_like(data["colour_factor_al"])
        with pytest.raises(ValueError, match="singular design"):
            fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)

    def test_bad_factor_raises(self):
        with pytest.raises(ValueError, match="colour_factor_al"):
            colour_columns(np.array([1.0, np.nan]))
        with pytest.raises(ValueError, match="colour_factor_al"):
            colour_columns(np.ones((3, 2)))


# ── T3: validation, with value-free messages ────────────────────────────────

SECRET = 123.456789  # a value that must not appear in any error message


def _assert_value_free(excinfo):
    assert "123.45" not in str(excinfo.value)


class TestValidation:
    def test_one_colour_key_without_the_other(self):
        data = _per_ccd_data(0)
        del data["nu_eff_used"]
        with pytest.raises(ValueError, match="both colour_factor_al and nu_eff_used") as excinfo:
            fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        _assert_value_free(excinfo)

    def test_length_mismatch(self):
        data = _per_ccd_data(0)
        data["colour_factor_al"] = np.full(data["colour_factor_al"].size - 1, SECRET)
        with pytest.raises(ValueError, match="one value per row") as excinfo:
            fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        _assert_value_free(excinfo)

    def test_non_finite_factor(self):
        data = _per_ccd_data(0)
        data["colour_factor_al"][3] = np.inf
        with pytest.raises(ValueError, match="finite"):
            fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)

    def test_colour_term_arguments(self):
        with pytest.raises(ValueError, match="needs a finite nu_eff"):
            ColourTerm("fixed")
        with pytest.raises(ValueError, match="needs a finite nu_eff") as excinfo:
            ColourTerm("fixed", nu_eff=1.5, prior_sigma=SECRET)
        _assert_value_free(excinfo)
        with pytest.raises(ValueError, match="takes no nu_eff") as excinfo:
            ColourTerm("free", nu_eff=SECRET)
        _assert_value_free(excinfo)
        with pytest.raises(ValueError, match="prior_sigma"):
            ColourTerm("free", prior_sigma=-0.1)
        with pytest.raises(ValueError, match="prior_sigma"):
            ColourTerm("free", prior_sigma="something")
        with pytest.raises(ValueError, match="'fixed' or 'free'"):
            ColourTerm("both")

    def test_colour_must_be_a_colour_term(self):
        with pytest.raises(ValueError, match="None or a ColourTerm"):
            fit_astrometric_5param(_per_ccd_data(0), epoch_ref_mjd=_EPOCH_REF_MJD,
                                   colour="free")

    def test_prior_from_key_without_the_key(self):
        with pytest.raises(ValueError, match="needs nu_eff_error"):
            fit_astrometric_5param(
                _per_ccd_data(0), epoch_ref_mjd=_EPOCH_REF_MJD,
                colour=ColourTerm("free", prior_sigma="nu_eff_error"))

    def test_a_prior_in_the_scan_raises(self):
        data = _per_ccd_data(0, orbit=True)
        arrays = _ti_arrays(data)
        with pytest.raises(ValueError, match="flat prior"):
            scan_ti_frequency(
                arrays["epochs_mjd"], arrays["scan_angle"], arrays["parallax_factor_al"],
                arrays["centroid_pos"], arrays["centroid_pos_err"],
                f_min_per_day=1 / 400.0, f_max_per_day=1 / 250.0, ecc_grid=[0.3],
                tau_grid=[0.2], epoch_ref_mjd=_EPOCH_REF_MJD, top_k=1,
                colour=ColourTerm("free", prior_sigma=DPAC_COLOUR_PRIOR_SIGMA),
                colour_factor_al=data["colour_factor_al"], nu_eff_used=data["nu_eff_used"])

    def test_free_colour_with_a_beta_prior_raises(self):
        data = _per_ccd_data(0, orbit=True)
        prior = PrecisionPrior(mean=np.zeros(9), precision=np.eye(9))
        with pytest.raises(ValueError, match="cannot be combined with a beta_prior"):
            best_linear_params_ti(
                PERIOD_DAYS, ECC, TAU, **_ti_arrays(data), beta_prior=prior, colour=FREE,
                colour_factor_al=data["colour_factor_al"], nu_eff_used=data["nu_eff_used"])

    def test_prior_with_both_cov_and_precision_raises(self):
        class Both:
            mean = np.zeros(5)
            cov = np.eye(5)
            precision = np.eye(5)

        data = _per_ccd_data(0)
        X = astrometric_5param_design_matrix(
            data["obs_time"] + MJD_J2010_TCB, data["scan_angle"],
            data["parallax_factor_al"], epoch_ref_mjd=_EPOCH_REF_MJD)
        with pytest.raises(ValueError, match="exactly one of"):
            linear_solve_ti(data["centroid_pos"], data["centroid_pos_err"], X, beta_prior=Both())
        wrong_shape = PrecisionPrior(mean=np.zeros(5), precision=np.eye(4))
        with pytest.raises(ValueError, match="precision of beta_prior"):
            linear_solve_ti(data["centroid_pos"], data["centroid_pos_err"], X,
                            beta_prior=wrong_shape)

    def test_precision_prior_repr_shows_no_value(self):
        prior = PrecisionPrior(mean=np.full(3, SECRET), precision=np.eye(3) * SECRET)
        assert "123" not in repr(prior)


# ── T4: recovery of an injected colour ──────────────────────────────────────

def _pulls(orbit):
    """Pulls (estimate − truth)/error of every amplitude, over N_SEEDS noise draws."""
    truth = np.concatenate([SINGLE_STAR_TRUE, [DELTA_NU_TRUE]])
    if orbit:
        truth = np.concatenate([ABFG_TRUE, truth])
    pulls = np.empty((N_SEEDS, truth.size))
    for seed in range(N_SEEDS):  # seed counts the noise realisations
        data = _per_ccd_data(seed, orbit=orbit)
        if orbit:
            result = best_linear_params_ti(
                PERIOD_DAYS, ECC, TAU, **_ti_arrays(data), colour=FREE,
                colour_factor_al=data["colour_factor_al"], nu_eff_used=data["nu_eff_used"])
            estimate, cov = result.beta, result.cov
        else:
            result = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
            estimate, cov = result.params, result.cov
        pulls[seed] = (estimate - truth) / np.sqrt(np.diag(cov))
    return pulls


class TestRecovery:
    def test_single_star_seed_zero(self):
        result = fit_astrometric_5param(_per_ccd_data(0), epoch_ref_mjd=_EPOCH_REF_MJD,
                                        colour=FREE)
        assert result.colour_mode == "free"
        assert result.params.shape == (6,) and result.cov.shape == (6, 6)
        assert abs(result.nu_eff - (NU_USED + DELTA_NU_TRUE)) < 3.0 * result.nu_eff_error
        # nu_eff is ν_ref plus the sixth amplitude
        assert result.nu_eff == NU_USED + result.params[5]

    def test_single_star_pulls(self):
        pulls = _pulls(orbit=False)
        print("T4 single star: pull mean", np.round(pulls.mean(axis=0), 3),
              "pull std", np.round(pulls.std(axis=0, ddof=1), 3))
        assert np.all(np.abs(pulls.mean(axis=0)) < 0.15)
        assert np.all(pulls.std(axis=0, ddof=1) > 0.9)
        assert np.all(pulls.std(axis=0, ddof=1) < 1.1)

    def test_orbit_pulls(self):
        pulls = _pulls(orbit=True)
        print("T4 orbit: pull mean", np.round(pulls.mean(axis=0), 3),
              "pull std", np.round(pulls.std(axis=0, ddof=1), 3))
        assert pulls.shape[1] == 10
        assert np.all(np.abs(pulls.mean(axis=0)) < 0.15)
        assert np.all(pulls.std(axis=0, ddof=1) > 0.9)
        assert np.all(pulls.std(axis=0, ddof=1) < 1.1)

    def test_rows_with_different_nu_used(self):
        # ν_used differs between transits: the data are first moved to ν_ref
        rng_nu = np.random.default_rng([7, 99])  # the per-transit ν_used of this test
        n_transits = _CADENCE.astro_obs_time.size
        nu_used = NU_USED + 0.02 * rng_nu.standard_normal(n_transits)
        data = _per_ccd_data(3, nu_used_per_transit=nu_used)
        result = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        assert abs(result.nu_eff - (NU_USED + DELTA_NU_TRUE)) < 3.0 * result.nu_eff_error

    def test_scan_finds_the_same_peak_with_a_free_colour(self):
        data = _per_ccd_data(0, delta_nu=0.0, orbit=True)   # colour-free data
        arrays = _ti_arrays(data)
        grid = dict(f_min_per_day=1 / 420.0, f_max_per_day=1 / 220.0, oversample=2.0,
                    ecc_grid=[0.1, 0.3, 0.5], tau_grid=[0.0, 0.2, 0.4, 0.6, 0.8],
                    epoch_ref_mjd=_EPOCH_REF_MJD, top_k=1)
        positional = (arrays["epochs_mjd"], arrays["scan_angle"], arrays["parallax_factor_al"],
                      arrays["centroid_pos"], arrays["centroid_pos_err"])
        plain = scan_ti_frequency(*positional, **grid)[0]
        free = scan_ti_frequency(*positional, **grid, colour=FREE,
                                 colour_factor_al=data["colour_factor_al"],
                                 nu_eff_used=data["nu_eff_used"])[0]
        assert free.f_per_day == plain.f_per_day
        assert (free.ecc, free.tau) == (plain.ecc, plain.tau)
        assert plain.beta.shape == (9,) and free.beta.shape == (10,)
        assert free.cov.shape == (10, 10)
        assert free.plx_mas == free.beta[8]   # index 8 is still the parallax

    def test_presearch_passes_the_colour_through(self):
        data = _per_ccd_data(0, orbit=True)
        result = presearch_astrometry_linear_ti(
            data, ra_deg=120.0, dec_deg=-30.0, f_min_per_day=1 / 320.0,
            f_max_per_day=1 / 280.0, oversample=1.0, ecc_grid=[0.3], tau_grid=[0.2],
            abfg_sigma_mas=1.0, system_priors={}, epoch_ref_mjd=_EPOCH_REF_MJD, top_k=1,
            colour=FREE)
        assert result["peaks"][0].beta.shape == (10,)


# ── T5: the fixed mode ──────────────────────────────────────────────────────

class TestFixed:
    def test_fixed_equals_the_colour_moved_into_the_data(self):
        data = _per_ccd_data(2)
        nu_fixed = NU_USED + 0.03
        fixed = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD,
                                       colour=ColourTerm("fixed", nu_eff=nu_fixed))
        # the same solve by hand: subtract column × (ν − ν_used), then 5 parameters
        moved = dict(_without_colour_keys(data))
        column = COLOUR_FACTOR_AL_SIGN * data["colour_factor_al"]
        moved["centroid_pos"] = data["centroid_pos"] - column * (nu_fixed - data["nu_eff_used"])
        by_hand = fit_astrometric_5param(moved, epoch_ref_mjd=_EPOCH_REF_MJD)
        np.testing.assert_allclose(fixed.params, by_hand.params, rtol=1e-12)
        np.testing.assert_allclose(fixed.cov, by_hand.cov, rtol=1e-12)
        assert fixed.params.shape == (5,)
        assert fixed.nu_eff == nu_fixed and fixed.nu_eff_error is None
        assert fixed.colour_mode == "fixed"

    def test_fixed_at_the_true_colour_recovers_the_star(self):
        data = _per_ccd_data(2)
        fixed = fit_astrometric_5param(
            data, epoch_ref_mjd=_EPOCH_REF_MJD,
            colour=ColourTerm("fixed", nu_eff=NU_USED + DELTA_NU_TRUE))
        pull = (fixed.params - SINGLE_STAR_TRUE) / np.sqrt(np.diag(fixed.cov))
        assert np.all(np.abs(pull) < 4.0)

    def test_fixed_at_nu_used_is_exactly_no_colour(self):
        data = _per_ccd_data(2)
        fixed = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD,
                                       colour=ColourTerm("fixed", nu_eff=NU_USED))
        plain = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD)
        assert np.array_equal(fixed.params, plain.params)
        assert np.array_equal(fixed.cov, plain.cov)
        assert np.array_equal(fixed.residuals, plain.residuals)


# ── T6: the prior on the colour amplitude ───────────────────────────────────

class TestPrior:
    def test_weak_data_give_the_prior_width(self):
        data = _per_ccd_data(4, delta_nu=0.0, sigma_mas=50.0)
        flat = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        prior = fit_astrometric_5param(
            data, epoch_ref_mjd=_EPOCH_REF_MJD,
            colour=ColourTerm("free", prior_sigma=DPAC_COLOUR_PRIOR_SIGMA))
        print("T6a: flat width", flat.nu_eff_error, "with prior", prior.nu_eff_error)
        assert flat.nu_eff_error > 5.0 * DPAC_COLOUR_PRIOR_SIGMA
        assert prior.nu_eff_error == pytest.approx(DPAC_COLOUR_PRIOR_SIGMA, rel=0.05)

    def test_strong_data_give_the_data_width(self):
        data = _per_ccd_data(4, sigma_mas=0.1)
        flat = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        prior = fit_astrometric_5param(
            data, epoch_ref_mjd=_EPOCH_REF_MJD,
            colour=ColourTerm("free", prior_sigma=DPAC_COLOUR_PRIOR_SIGMA))
        print("T6b: flat width", flat.nu_eff_error, "with prior", prior.nu_eff_error)
        assert prior.nu_eff_error == pytest.approx(flat.nu_eff_error, rel=0.05)

    def test_precision_prior_equals_the_pseudo_observation_and_the_update(self):
        data = _per_ccd_data(5, sigma_mas=2.0)
        t_mjd = data["obs_time"] + MJD_J2010_TCB
        X = concat_columns([
            astrometric_5param_design_matrix(t_mjd, data["scan_angle"],
                                             data["parallax_factor_al"],
                                             epoch_ref_mjd=_EPOCH_REF_MJD),
            colour_columns(data["colour_factor_al"]),
        ])
        d, sigma = data["centroid_pos"], data["centroid_pos_err"]
        width = DPAC_COLOUR_PRIOR_SIGMA
        precision = np.zeros((6, 6))
        precision[5, 5] = 1.0 / width**2   # the colour amplitude only
        by_precision = linear_solve_ti(d, sigma, X,
                                       beta_prior=PrecisionPrior(np.zeros(6), precision))
        # oracle 1, one pseudo-observation row: value 0, error = the prior
        # width, 1 in the colour column (Lindegren et al. 2021, eq. C.3)
        extra_row = np.zeros((1, 6))
        extra_row[0, 5] = 1.0
        by_row = linear_solve_ti(np.append(d, 0.0), np.append(sigma, width),
                                 np.vstack([X, extra_row]))
        np.testing.assert_allclose(by_precision.beta, by_row.beta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(by_precision.cov, by_row.cov, rtol=1e-10, atol=1e-14)
        # oracle 2, the Gaussian update of the flat solution by one
        # measurement "colour amplitude = 0 ± width"
        flat = linear_solve_ti(d, sigma, X)
        gain = flat.cov[:, 5] / (flat.cov[5, 5] + width**2)
        updated_beta = flat.beta + gain * (0.0 - flat.beta[5])
        updated_cov = flat.cov - np.outer(gain, flat.cov[5, :])
        np.testing.assert_allclose(by_precision.beta, updated_beta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(by_precision.cov, updated_cov, rtol=1e-10, atol=1e-14)
        # the ranking score ignores the prior; no evidence for an improper prior
        assert by_precision.logL_marginal == flat.logL_marginal
        assert by_precision.log_evidence is None
        assert by_precision.dof == flat.dof
        assert by_row.dof == flat.dof + 1

    def test_prior_width_from_the_nu_eff_error_key(self):
        data = _per_ccd_data(4, sigma_mas=50.0)
        data["nu_eff_error"] = np.full(data["centroid_pos"].size, 0.02)
        from_key = fit_astrometric_5param(
            data, epoch_ref_mjd=_EPOCH_REF_MJD,
            colour=ColourTerm("free", prior_sigma="nu_eff_error"))
        by_number = fit_astrometric_5param(
            data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=ColourTerm("free", prior_sigma=0.02))
        assert np.array_equal(from_key.params, by_number.params)
        assert from_key.nu_eff_error == pytest.approx(0.02, rel=0.05)

    def test_prior_in_the_orbit_solve(self):
        data = _per_ccd_data(4, sigma_mas=50.0, orbit=True)
        result = best_linear_params_ti(
            PERIOD_DAYS, ECC, TAU, **_ti_arrays(data),
            colour=ColourTerm("free", prior_sigma=DPAC_COLOUR_PRIOR_SIGMA),
            colour_factor_al=data["colour_factor_al"], nu_eff_used=data["nu_eff_used"],
            draw=3)
        assert result.nu_eff_error == pytest.approx(DPAC_COLOUR_PRIOR_SIGMA, rel=0.05)
        assert result.draws.shape == (3, 10)
        assert result.log_evidence is None and result.logL_marginal is not None


# ── T8: the sign is a convention inside the package ─────────────────────────

class TestSignIsInternal:
    def test_flipped_in_injection_and_fit_recovers_the_same_colour(self, monkeypatch):
        monkeypatch.setattr("orblet.design.columns.COLOUR_FACTOR_AL_SIGN", +1)
        data = _per_ccd_data(0)   # injected with the flipped sign
        result = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        assert abs(result.params[5] - DELTA_NU_TRUE) < 3.0 * result.nu_eff_error

    def test_flipped_in_the_fit_only_flips_the_fitted_colour(self, monkeypatch):
        data = _per_ccd_data(0)   # injected with the package's sign
        monkeypatch.setattr("orblet.design.columns.COLOUR_FACTOR_AL_SIGN", +1)
        result = fit_astrometric_5param(data, epoch_ref_mjd=_EPOCH_REF_MJD, colour=FREE)
        assert abs(result.params[5] + DELTA_NU_TRUE) < 3.0 * result.nu_eff_error


# ── T9: the public surface ──────────────────────────────────────────────────

class TestPublicSurface:
    def test_new_names_are_on_the_front_door(self):
        for name in ("ColourTerm", "PrecisionPrior", "colour_columns", "ColourBlockBuilder",
                     "COLOUR_FACTOR_AL_SIGN", "DPAC_COLOUR_PRIOR_SIGMA"):
            assert name in orblet.__all__
            assert getattr(orblet, name) is not None
        assert DPAC_COLOUR_PRIOR_SIGMA == 0.085
