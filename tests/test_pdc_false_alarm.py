"""
Tests for the grid-wide false-alarm probabilities of the PDC periodogram
(orblet.periodogram.false_alarm).

All tests use synthetic data, offline and seeded.
"""

from __future__ import annotations

import numpy as np
import pytest

from orblet.periodogram import (
    astrometric_segment_distance_matrix,
    compute_pdc_periodogram,
    pdc_false_alarm_probability,
    pdc_grid_false_alarm_probability,
    pdc_scramble_false_alarm_probability,
    scalar_distance_matrix,
    scan_angle_distance_matrix,
)
from orblet.periodogram.dcor import _u_center, _u_inner
from orblet.periodogram.distances import _circular_distance
from orblet.periodogram.false_alarm import _phase_norm

SPAN_DAYS = 1900.0  # time span of the synthetic schedules, days


def _period_grid(span_days):
    """Uniform in frequency, 10 d to twice the span, step 1/(5 span) (bina's grid)."""
    frequencies = np.arange(1.0 / (2.0 * span_days), 1.0 / 10.0, 1.0 / (5.0 * span_days))
    return np.sort(1.0 / frequencies)


def _rv_data(seed, n_epochs=40, amplitude=0.0, period_days=77.0):
    """Times and RVs: white noise of unit width plus an optional sinusoid."""
    rng_data = np.random.default_rng(seed)
    time_days = np.sort(rng_data.uniform(0.0, SPAN_DAYS, n_epochs))
    signal = amplitude * np.sin(2.0 * np.pi * time_days / period_days)
    rv = signal + rng_data.standard_normal(n_epochs)
    return time_days, rv


def _per_ccd_data(seed, n_transits=30):
    """Per-CCD lines: 4 to 9 lines per transit at one time, one scan angle per transit."""
    rng_data = np.random.default_rng(seed)
    lines_per_transit = rng_data.integers(4, 10, n_transits)
    transit_of_line = np.repeat(np.arange(n_transits), lines_per_transit)
    transit_days = np.sort(rng_data.uniform(0.0, SPAN_DAYS, n_transits))
    transit_psi = rng_data.uniform(0.0, np.pi, n_transits)
    time_days = transit_days[transit_of_line]
    psi = transit_psi[transit_of_line]
    residuals = 0.2 * rng_data.standard_normal(time_days.size)
    observation = astrometric_segment_distance_matrix(residuals, psi, L_mas=0.01)
    nuisance = scan_angle_distance_matrix(psi)
    return time_days, observation, nuisance, transit_of_line


# ── Fast form ────────────────────────────────────────────────────────────────

class TestGridFalseAlarmProbability:
    def test_formula(self):
        periods = _period_grid(SPAN_DAYS)
        score = 0.05
        n = 30
        p_point = pdc_false_alarm_probability(score, n)
        n_eff = SPAN_DAYS * (1.0 / periods.min() - 1.0 / periods.max())
        expected = 1.0 - (1.0 - p_point) ** n_eff
        fap = pdc_grid_false_alarm_probability(score, n, periods, SPAN_DAYS)
        assert fap == pytest.approx(expected, rel=1e-12)

    def test_never_below_pointwise(self):
        periods = _period_grid(SPAN_DAYS)
        for score in (0.01, 0.1, 0.3):
            p_point = pdc_false_alarm_probability(score, 50)
            assert pdc_grid_false_alarm_probability(score, 50, periods, SPAN_DAYS) >= p_point

    def test_small_pointwise_value_does_not_round_to_zero(self):
        # a very large score: p ≈ 1e-20; the grid value is ≈ N_eff · p, not 0
        periods = _period_grid(SPAN_DAYS)
        score = 0.9
        n = 100
        p_point = pdc_false_alarm_probability(score, n)
        n_eff = SPAN_DAYS * (1.0 / periods.min() - 1.0 / periods.max())
        fap = pdc_grid_false_alarm_probability(score, n, periods, SPAN_DAYS)
        assert fap > 0.0
        assert fap == pytest.approx(n_eff * p_point, rel=1e-6)

    def test_n_eff_is_at_least_one_and_at_most_the_grid_size(self):
        # one period: N_eff = 1, the grid value is the pointwise one
        assert pdc_grid_false_alarm_probability(0.1, 30, [50.0], SPAN_DAYS) == pytest.approx(
            pdc_false_alarm_probability(0.1, 30))
        # three periods over a wide band: N_eff is clipped to 3
        p_point = pdc_false_alarm_probability(0.1, 30)
        fap = pdc_grid_false_alarm_probability(0.1, 30, [10.0, 100.0, 1000.0], SPAN_DAYS)
        assert fap == pytest.approx(1.0 - (1.0 - p_point) ** 3)

    def test_nan_score_gives_nan(self):
        assert np.isnan(pdc_grid_false_alarm_probability(np.nan, 30, [50.0], SPAN_DAYS))

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError, match="span_days"):
            pdc_grid_false_alarm_probability(0.1, 30, [50.0], 0.0)
        with pytest.raises(ValueError, match="no finite positive period"):
            pdc_grid_false_alarm_probability(0.1, 30, [np.nan, -1.0], SPAN_DAYS)
        with pytest.raises(ValueError, match="n must be >= 4"):
            pdc_grid_false_alarm_probability(0.1, 3, [50.0], SPAN_DAYS)


# ── Scramble null: the algebra ───────────────────────────────────────────────

class TestScrambleAlgebra:
    def test_phase_norm_matches_the_expanded_matrix(self):
        # unequal slot sizes: the slot formula equals ⟨Ũ(B), Ũ(B)⟩ of the n × n matrix
        rng_data = np.random.default_rng(5)
        sizes = rng_data.integers(1, 6, 12).astype(float)
        slot_times = np.sort(rng_data.uniform(0.0, SPAN_DAYS, 12))
        slot_of_row = np.repeat(np.arange(12), sizes.astype(int))
        n = slot_of_row.size
        expanded = _circular_distance(slot_times[slot_of_row], 37.3)
        centred = _u_center(expanded)
        W = _circular_distance(slot_times, 37.3)
        norm = _phase_norm(W, sizes[:, None], n)[0]  # one column: one copy
        assert norm == pytest.approx(_u_inner(centred, centred), rel=1e-12)

    def test_observed_score_equals_the_periodogram_ordinary(self):
        time_days, rv = _rv_data(seed=1, amplitude=0.5)
        periods = _period_grid(SPAN_DAYS)
        observation = scalar_distance_matrix(rv)
        direct = compute_pdc_periodogram(time_days, periods, observation)
        result = pdc_scramble_false_alarm_probability(
            time_days, periods, observation, n_copies=5, rng=0)
        assert result["observed_best_score"] == pytest.approx(direct["best_score"], rel=1e-10)
        assert result["observed_best_period_days"] == direct["best_period_days"]

    def test_observed_score_equals_the_periodogram_semi_partial(self):
        time_days, observation, nuisance, _transits = _per_ccd_data(seed=2)
        periods = _period_grid(SPAN_DAYS)
        direct = compute_pdc_periodogram(
            time_days, periods, observation, nuisance_dist=nuisance, partial_mode="semi")
        for groups in (None, _transits):  # rows alone, then whole transits
            result = pdc_scramble_false_alarm_probability(
                time_days, periods, observation, n_copies=5, rng=0,
                nuisance_dist=nuisance, partial_mode="semi", groups=groups)
            assert result["observed_best_score"] == pytest.approx(
                direct["best_score"], rel=1e-10)

    def test_a_grouped_copy_equals_the_periodogram_of_the_moved_data(self):
        # copy 1 puts transit g at the time of transit permutation[g]; the
        # permutation is the first draw of the same seed
        time_days, observation, nuisance, transits = _per_ccd_data(seed=3)
        periods = _period_grid(SPAN_DAYS)
        result = pdc_scramble_false_alarm_probability(
            time_days, periods, observation, n_copies=3, rng=11,
            nuisance_dist=nuisance, partial_mode="semi", groups=transits)
        rng_permutation = np.random.default_rng(11)
        n_transits = transits.max() + 1
        permutation = rng_permutation.permutation(n_transits)
        transit_days = np.bincount(transits, weights=time_days) / np.bincount(transits)
        moved_days = transit_days[permutation[transits]]
        direct = compute_pdc_periodogram(
            moved_days, periods, observation, nuisance_dist=nuisance, partial_mode="semi")
        assert result["null_best_scores"][0] == pytest.approx(direct["best_score"], rel=1e-10)


# ── Scramble null: the false-alarm probability ───────────────────────────────

class TestScrambleFalseAlarmProbability:
    def test_strong_signal_gets_the_smallest_value(self):
        time_days, rv = _rv_data(seed=4, amplitude=3.0)
        periods = _period_grid(SPAN_DAYS)
        result = pdc_scramble_false_alarm_probability(
            time_days, periods, scalar_distance_matrix(rv), n_copies=99, rng=0)
        assert result["false_alarm_probability"] == pytest.approx(1.0 / 100.0)
        assert result["observed_best_period_days"] == pytest.approx(77.0, rel=0.02)

    def test_noise_is_not_significant(self):
        time_days, rv = _rv_data(seed=6)
        periods = _period_grid(SPAN_DAYS)
        result = pdc_scramble_false_alarm_probability(
            time_days, periods, scalar_distance_matrix(rv), n_copies=99, rng=0)
        assert result["false_alarm_probability"] > 0.01

    def test_same_seed_same_result(self):
        time_days, observation, nuisance, transits = _per_ccd_data(seed=7)
        periods = _period_grid(SPAN_DAYS)
        results = []
        for _repeat in range(2):  # two calls, the same seed
            results.append(pdc_scramble_false_alarm_probability(
                time_days, periods, observation, n_copies=20, rng=3,
                nuisance_dist=nuisance, partial_mode="semi", groups=transits))
        np.testing.assert_array_equal(
            results[0]["null_best_scores"], results[1]["null_best_scores"])
        assert results[0]["n_groups"] == transits.max() + 1

    def test_value_is_a_permutation_p_value(self):
        time_days, rv = _rv_data(seed=8)
        periods = _period_grid(SPAN_DAYS)
        result = pdc_scramble_false_alarm_probability(
            time_days, periods, scalar_distance_matrix(rv), n_copies=40, rng=1)
        n_exceeding = np.sum(result["null_best_scores"] >= result["observed_best_score"])
        assert result["false_alarm_probability"] == (1 + n_exceeding) / 41
        assert result["null_best_scores"].shape == (40,)

    def test_no_nuisance_variance_gives_nan(self):
        time_days, rv = _rv_data(seed=9)
        constant_nuisance = np.zeros((rv.size, rv.size))
        result = pdc_scramble_false_alarm_probability(
            time_days, _period_grid(SPAN_DAYS), scalar_distance_matrix(rv),
            n_copies=5, rng=0, nuisance_dist=constant_nuisance, partial_mode="semi")
        assert np.isnan(result["false_alarm_probability"])
        assert np.all(np.isnan(result["null_best_scores"]))

    def test_bad_inputs_raise(self):
        time_days, rv = _rv_data(seed=10)
        observation = scalar_distance_matrix(rv)
        periods = _period_grid(SPAN_DAYS)
        with pytest.raises(ValueError, match="n_copies"):
            pdc_scramble_false_alarm_probability(time_days, periods, observation, n_copies=0, rng=0)
        with pytest.raises(ValueError, match="requires nuisance_dist"):
            pdc_scramble_false_alarm_probability(
                time_days, periods, observation, n_copies=5, rng=0, partial_mode="semi")
        with pytest.raises(ValueError, match="one label per row"):
            pdc_scramble_false_alarm_probability(
                time_days, periods, observation, n_copies=5, rng=0, groups=[0, 1])
        with pytest.raises(ValueError, match="two groups"):
            pdc_scramble_false_alarm_probability(
                time_days, periods, observation, n_copies=5, rng=0,
                groups=np.zeros(rv.size))
        with pytest.raises(ValueError, match="obs_dist"):
            pdc_scramble_false_alarm_probability(
                time_days, periods, observation[:-1, :-1], n_copies=5, rng=0)
