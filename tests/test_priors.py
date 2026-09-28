"""
Tests for orblet.priors — default prior configurations.

Pure Python tests.
"""

from __future__ import annotations

import math

import pytest

from orblet.priors import (
    EccOmegaDiskPrior,
    UniformCircularPrior,
    default_companion_priors,
    default_system_priors,
)


# ── default_companion_priors ─────────────────────────────────────────────────

class TestDefaultCompanionPriors:
    def test_contains_expected_keys(self):
        priors = default_companion_priors(10.0)
        assert set(priors.keys()) == {"P", "e", "ω", "τ", "mass"}

    def test_period_in_days(self):
        """Period prior should be in days (user-facing units)."""
        priors = default_companion_priors(100.0, period_bounds_factor=2.0)
        dist_name, lo, hi = priors["P"]
        assert dist_name == "LogUniform"
        assert lo == pytest.approx(50.0)
        assert hi == pytest.approx(200.0)

    def test_mass_in_msun(self):
        """Companion mass should be in solar masses (user-facing units)."""
        priors = default_companion_priors(10.0, mass_range_msun=(0.5, 3.0))
        _, lo, hi = priors["mass"]
        assert lo == 0.5
        assert hi == 3.0

    def test_ecc_max(self):
        priors = default_companion_priors(10.0, ecc_max=0.5)
        _, lo, hi = priors["e"]
        assert hi == 0.5

    def test_no_M_in_companion_priors(self):
        """M (total mass) is now at system level, not companion level."""
        priors = default_companion_priors(10.0)
        assert "M" not in priors


# ── default_system_priors ────────────────────────────────────────────────────

class TestDefaultSystemPriors:
    def test_contains_total_mass(self):
        """RV models now have system-level M (total mass)."""
        priors = default_system_priors()
        assert "M" in priors
        dist_name = priors["M"][0]
        assert dist_name == "truncated_Normal"

    def test_custom_stellar_mass(self):
        priors = default_system_priors(
            stellar_mass_msun=1.5, stellar_mass_err=0.3
        )
        _, mu, sigma, lower = priors["M"]
        assert mu == 1.5
        assert sigma == 0.3


# ── The unit-disk priors: the (e, ω) disk is open, the phase disk closed ────

_INSIDE = -math.log(math.pi)


def test_ecc_omega_disk_is_open():
    """The unit circle is e = 1 exactly: the prior rejects it itself."""
    prior = EccOmegaDiskPrior()
    assert prior.logpdf(0.0, 0.0) == _INSIDE
    assert prior.logpdf(math.nextafter(1.0, 0.0), 0.0) == _INSIDE
    assert prior.logpdf(1.0, 0.0) == -math.inf
    assert prior.logpdf(0.0, -1.0) == -math.inf
    assert prior.logpdf(0.6, 0.8) == -math.inf       # 0.36 + 0.64 == 1.0
    assert prior.logpdf(1.0, 1e-6) == -math.inf


def test_phase_disk_is_closed():
    """A phase built as (cos φ, sin φ) sits on the circle and is valid."""
    prior = UniformCircularPrior()
    assert prior.logpdf(1.0, 0.0) == _INSIDE
    assert prior.logpdf(0.6, 0.8) == _INSIDE
    assert prior.logpdf(1.0, 1e-6) == -math.inf
