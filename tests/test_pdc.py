"""
Tests for the PDC (phase distance correlation) periodogram functions
in orblet.periodogram.

All tests use synthetic data — no real Gaia data needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from orblet.periodogram.dcor import _pdc_score, _u_center, _u_inner
from orblet.periodogram.pdc import _phase_distance_matrix
from orblet.periodogram import (
    scalar_distance_matrix,
    astrometric_segment_distance_matrix,
    spectral_distance_matrix,
    scan_angle_distance_matrix,
    compute_pdc_periodogram,
    distance_correlation,
    scan_angle_coupling,
    pdc_false_alarm_probability,
    compute_lomb_scargle_periodogram,
)


# ── U-centering ──────────────────────────────────────────────────────────────

class TestUCenter:
    def test_zero_diagonal(self):
        d = np.random.default_rng(42).uniform(0, 1, (10, 10))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        a = _u_center(d)
        np.testing.assert_array_equal(np.diag(a), 0.0)

    def test_symmetric(self):
        d = np.random.default_rng(42).uniform(0, 1, (10, 10))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        a = _u_center(d)
        np.testing.assert_allclose(a, a.T, atol=1e-12)

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="n >= 4"):
            _u_center(np.zeros((3, 3)))


class TestUInner:
    def test_self_positive(self):
        rng = np.random.default_rng(42)
        d = rng.uniform(0, 1, (10, 10))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        a = _u_center(d)
        assert _u_inner(a, a) > 0


# ── Distance matrices ────────────────────────────────────────────────────────

class TestPhaseDistanceMatrix:
    def test_symmetric(self):
        t = np.array([0, 1, 2, 3, 4], dtype=float)
        d = _phase_distance_matrix(t, 5.0)
        np.testing.assert_allclose(d, d.T, atol=1e-12)

    def test_zero_diagonal(self):
        t = np.array([0, 1, 2, 3, 4], dtype=float)
        d = _phase_distance_matrix(t, 5.0)
        np.testing.assert_array_equal(np.diag(d), 0.0)

    def test_max_at_half_period(self):
        t = np.array([0.0, 2.5])
        d = _phase_distance_matrix(t, 5.0)
        assert d[0, 1] == pytest.approx(6.25)


class TestScalarDistanceMatrix:
    def test_symmetric(self):
        x = np.array([1.0, 3.0, 7.0])
        d = scalar_distance_matrix(x)
        np.testing.assert_allclose(d, d.T)

    def test_values(self):
        x = np.array([0.0, 3.0, 5.0])
        d = scalar_distance_matrix(x)
        assert d[0, 1] == pytest.approx(3.0)
        assert d[0, 2] == pytest.approx(5.0)
        assert d[1, 2] == pytest.approx(2.0)

    def test_zero_diagonal(self):
        d = scalar_distance_matrix(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(np.diag(d), 0.0)


class TestAstrometricSegmentDistance:
    def test_symmetric(self):
        rng = np.random.default_rng(42)
        v = rng.normal(0, 1, 10)
        th = rng.uniform(0, 2 * np.pi, 10)
        d = astrometric_segment_distance_matrix(v, th, L_mas=1.0)
        np.testing.assert_allclose(d, d.T, atol=1e-12)

    def test_zero_diagonal(self):
        rng = np.random.default_rng(42)
        v = rng.normal(0, 1, 10)
        th = rng.uniform(0, 2 * np.pi, 10)
        d = astrometric_segment_distance_matrix(v, th, L_mas=1.0)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        v = rng.normal(0, 1, 10)
        th = rng.uniform(0, 2 * np.pi, 10)
        d = astrometric_segment_distance_matrix(v, th, L_mas=1.0)
        assert np.all(d >= -1e-12)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            astrometric_segment_distance_matrix(
                np.ones(5), np.ones(6), L_mas=1.0
            )


class TestScanAngleDistance:
    def test_pi_periodic(self):
        """Antiparallel scans are the same unoriented line: zero distance."""
        d = scan_angle_distance_matrix(np.array([0.3, 0.3 + np.pi]))
        assert d[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_max_for_perpendicular_scans(self):
        d = scan_angle_distance_matrix(np.array([0.0, np.pi / 2]))
        assert d[0, 1] == pytest.approx(np.pi**2 / 4)

    def test_symmetric_zero_diagonal(self):
        theta = np.random.default_rng(3).uniform(0, 2 * np.pi, 20)
        d = scan_angle_distance_matrix(theta)
        np.testing.assert_allclose(d, d.T, atol=1e-12)
        np.testing.assert_array_equal(np.diag(d), 0.0)

    def test_is_the_phase_distance_with_period_pi(self):
        theta = np.random.default_rng(4).uniform(0, 2 * np.pi, 20)
        np.testing.assert_array_equal(
            scan_angle_distance_matrix(theta),
            _phase_distance_matrix(theta, np.pi),
        )


class TestSpectralDistanceMatrix:
    def test_identical_spectra_zero_distance(self):
        spectra = np.tile(np.sin(np.linspace(0, 2 * np.pi, 100)), (5, 1))
        d = spectral_distance_matrix(spectra)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        spectra = rng.normal(0, 1, (10, 50))
        d = spectral_distance_matrix(spectra)
        np.testing.assert_allclose(d, d.T, atol=1e-12)

    def test_zero_diagonal(self):
        rng = np.random.default_rng(42)
        spectra = rng.normal(0, 1, (10, 50))
        d = spectral_distance_matrix(spectra)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        spectra = rng.normal(0, 1, (10, 50))
        d = spectral_distance_matrix(spectra)
        assert np.all(d >= -1e-12)

    def test_different_spectra_nonzero(self):
        """Distinct spectra should have positive distance."""
        rng = np.random.default_rng(42)
        spectra = rng.normal(0, 1, (5, 50))
        d = spectral_distance_matrix(spectra)
        # Off-diagonal should all be positive.
        off_diag = d[np.triu_indices(5, k=1)]
        assert np.all(off_diag > 0)

    def test_handles_nan(self):
        rng = np.random.default_rng(42)
        spectra = rng.normal(0, 1, (5, 20))
        spectra[0, 5] = np.nan
        d = spectral_distance_matrix(spectra)
        assert np.all(np.isfinite(d))

    def test_2d_required(self):
        with pytest.raises(ValueError, match="2-D"):
            spectral_distance_matrix(np.ones(10))


# ── PDC scores ───────────────────────────────────────────────────────────────

class TestPDCScore:
    def test_identical_matrices_high_score(self):
        rng = np.random.default_rng(42)
        d = rng.uniform(0, 1, (10, 10))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        score = _pdc_score(d, d)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_in_range(self):
        rng = np.random.default_rng(42)
        n = 20
        d1 = rng.uniform(0, 1, (n, n))
        d1 = (d1 + d1.T) / 2
        np.fill_diagonal(d1, 0)
        d2 = rng.uniform(0, 1, (n, n))
        d2 = (d2 + d2.T) / 2
        np.fill_diagonal(d2, 0)
        score = _pdc_score(d1, d2)
        assert -1.0 <= score <= 1.0


# ── Full periodogram ─────────────────────────────────────────────────────────

class TestComputePDCPeriodogram:
    def _make_periodic_data(self, period=10.0, n=50, seed=42):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 100, n))
        rv = 5.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n)
        return t, rv

    def test_returns_expected_keys(self):
        t, rv = self._make_periodic_data()
        obs_dist = scalar_distance_matrix(rv)
        periods = np.geomspace(1, 100, 200)
        result = compute_pdc_periodogram(t, periods, obs_dist)
        assert set(result.keys()) == {
            "periods_days", "scores", "best_period_days", "best_score",
            "partial_mode",
        }

    def test_scores_shape(self):
        t, rv = self._make_periodic_data()
        obs_dist = scalar_distance_matrix(rv)
        periods = np.geomspace(1, 100, 200)
        result = compute_pdc_periodogram(t, periods, obs_dist)
        assert result["scores"].shape == (200,)

    def test_best_period_is_finite(self):
        t, rv = self._make_periodic_data()
        obs_dist = scalar_distance_matrix(rv)
        periods = np.geomspace(1, 100, 200)
        result = compute_pdc_periodogram(t, periods, obs_dist)
        assert np.isfinite(result["best_period_days"])
        assert np.isfinite(result["best_score"])

    def test_semipartial_with_precomputed_nuisance(self):
        t, rv = self._make_periodic_data()
        obs_dist = scalar_distance_matrix(rv)
        # Use a random nuisance matrix.
        rng = np.random.default_rng(99)
        z = rng.normal(0, 1, len(t))
        nuisance_dist = scalar_distance_matrix(z)
        periods = np.geomspace(1, 100, 100)
        result = compute_pdc_periodogram(
            t, periods, obs_dist,
            nuisance_dist=nuisance_dist, partial_mode="semi",
        )
        assert result["partial_mode"] == "semi"
        assert np.any(np.isfinite(result["scores"]))

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            compute_pdc_periodogram(
                np.ones(5), np.ones(10), np.ones((6, 6)),
            )

    def test_astrometric_mode(self):
        """Verify astrometric segment distance works as obs_dist."""
        rng = np.random.default_rng(42)
        n = 30
        t = np.sort(rng.uniform(0, 100, n))
        v = rng.normal(0, 1, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        obs_dist = astrometric_segment_distance_matrix(v, theta)
        periods = np.geomspace(1, 100, 50)
        result = compute_pdc_periodogram(t, periods, obs_dist)
        assert np.isfinite(result["best_score"])


# ── Distance correlation ─────────────────────────────────────────────────────

class TestDistanceCorrelation:
    def test_self_correlation_is_one(self):
        rng = np.random.default_rng(42)
        d = rng.uniform(0, 1, (10, 10))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        assert distance_correlation(d, d) == pytest.approx(1.0, abs=0.01)

    def test_in_range(self):
        rng = np.random.default_rng(42)
        A = scalar_distance_matrix(rng.normal(0, 1, 20))
        B = scan_angle_distance_matrix(rng.uniform(0, 2 * np.pi, 20))
        c = distance_correlation(A, B)
        assert -1.0 <= c <= 1.0

    def test_returns_finite(self):
        rng = np.random.default_rng(42)
        A = scalar_distance_matrix(rng.normal(0, 1, 20))
        B = scalar_distance_matrix(rng.normal(0, 1, 20))
        assert np.isfinite(distance_correlation(A, B))


# ── Scan-angle coupling and the partial-mode coupling key ───────────────────

class TestScanAngleCoupling:
    def _data(self, n=40, seed=5):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 1000, n))
        theta = rng.uniform(0, 2 * np.pi, n)
        v = rng.normal(0, 1.0, n)
        return t, v, theta

    def test_equals_distance_correlation_of_the_two_matrices(self):
        _, v, theta = self._data()
        expected = distance_correlation(
            astrometric_segment_distance_matrix(v, theta, L_mas=0.5),
            scan_angle_distance_matrix(theta),
        )
        assert scan_angle_coupling(v, theta, L_mas=0.5) == expected

    def test_matches_the_periodogram_coupling_key(self):
        t, v, theta = self._data()
        obs = astrometric_segment_distance_matrix(v, theta)
        res = compute_pdc_periodogram(
            t, np.geomspace(5, 500, 30), obs,
            nuisance_dist=scan_angle_distance_matrix(theta),
            partial_mode="semi",
        )
        assert res["coupling"] == scan_angle_coupling(v, theta)

    def test_coupling_key_only_in_partial_mode(self):
        t, v, theta = self._data()
        obs = astrometric_segment_distance_matrix(v, theta)
        res = compute_pdc_periodogram(t, np.geomspace(5, 500, 30), obs)
        assert "coupling" not in res


# ── Pointwise false-alarm probability ───────────────────────────────────────

class TestPDCFalseAlarmProbability:
    def test_chi2_form(self):
        from scipy.stats import chi2
        assert pdc_false_alarm_probability(0.5, 40) == chi2.sf(0.5 * 40 + 1.0, 1)

    def test_decreases_with_score(self):
        fap = pdc_false_alarm_probability(np.array([0.0, 0.1, 0.3, 0.6]), 30)
        assert np.all(np.diff(fap) < 0)

    def test_array_in_array_out_and_nan_passes(self):
        fap = pdc_false_alarm_probability(np.array([0.2, np.nan]), 30)
        assert fap.shape == (2,)
        assert np.isfinite(fap[0]) and np.isnan(fap[1])

    def test_scalar_in_float_out(self):
        assert isinstance(pdc_false_alarm_probability(0.2, 30), float)

    def test_too_few_epochs_raises(self):
        with pytest.raises(ValueError, match="n must be >= 4"):
            pdc_false_alarm_probability(0.5, 3)


# ── Lomb–Scargle input contract ─────────────────────────────────────────────

class TestLombScargleInputs:
    def _data(self):
        rng = np.random.default_rng(8)
        t = np.sort(rng.uniform(0, 300, 30))
        rv = 4.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 0.3, 30)
        return rv, np.full(30, 0.3), t

    def test_non_finite_input_raises(self):
        rv, err, t = self._data()
        for arr in (rv, err, t):
            bad = arr.copy()
            bad[4] = np.nan
            args = [rv, err, t]
            args[[id(a) for a in (rv, err, t)].index(id(arr))] = bad
            with pytest.raises(ValueError, match="must be finite"):
                compute_lomb_scargle_periodogram(*args)

    def test_options_are_keyword_only(self):
        rv, err, t = self._data()
        with pytest.raises(TypeError):
            compute_lomb_scargle_periodogram(rv, err, t, True)

