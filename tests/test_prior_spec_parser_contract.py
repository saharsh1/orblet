"""The prior-spec parser's contract.

Scope: the parser's accepted dialect, the class and parameters it builds
for each spec, and every refusal. It does NOT re-test the prior
distributions themselves — that is ``test_priors*``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orblet.priors import (
    angle_from_xy,
    parse_prior_spec,
    tau_from_xy,
)
from orblet.priors import (
    LogUniformPrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformCircularPrior,
    UniformPrior,
)


# ── the accepted dialect: spec -> (class, attribute expectations) ──────

def test_uniform_spec_builds_a_uniform_prior():
    p = parse_prior_spec(("Uniform", 0.0, 1.0))
    assert isinstance(p, UniformPrior)
    # In-support finite, out-of-support -inf: the behavioural contract.
    assert math.isfinite(float(p.logpdf(0.5)))
    assert float(p.logpdf(-1.0)) == -math.inf
    assert float(p.logpdf(2.0)) == -math.inf


def test_loguniform_spec_builds_a_loguniform_prior():
    p = parse_prior_spec(("LogUniform", 1.0, 100.0))
    assert isinstance(p, LogUniformPrior)
    assert math.isfinite(float(p.logpdf(10.0)))
    assert float(p.logpdf(0.5)) == -math.inf


def test_normal_spec_builds_a_normal_prior():
    p = parse_prior_spec(("Normal", 2.0, 0.5))
    assert isinstance(p, NormalPrior)
    # Peak at mu is the densest point.
    assert float(p.logpdf(2.0)) > float(p.logpdf(3.0))


def test_uniform_circular_spec_takes_no_arguments():
    p = parse_prior_spec(("UniformCircular",))
    assert isinstance(p, UniformCircularPrior)


def test_truncated_normal_three_args_means_no_upper_bound():
    p = parse_prior_spec(("truncated_Normal", 1.0, 0.5, 0.0))
    assert isinstance(p, TruncatedNormalPrior)
    assert float(p.logpdf(-1.0)) == -math.inf       # below `lower`
    assert math.isfinite(float(p.logpdf(50.0)))     # upper is +inf


def test_truncated_normal_four_args_bounds_both_sides():
    p = parse_prior_spec(("truncated_Normal", 1.0, 0.5, 0.0, 2.0))
    assert isinstance(p, TruncatedNormalPrior)
    assert float(p.logpdf(-1.0)) == -math.inf
    assert float(p.logpdf(3.0)) == -math.inf
    assert math.isfinite(float(p.logpdf(1.0)))


def test_arguments_are_coerced_to_float():
    """Integer specs must not produce an integer-typed prior."""
    p = parse_prior_spec(("Uniform", 0, 1))
    assert isinstance(p, UniformPrior)
    assert math.isfinite(float(p.logpdf(0.5)))


# ── the refusals ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "spec",
    [
        ("Uniform", 0.0),                 # too few
        ("Uniform", 0.0, 1.0, 2.0),       # too many
        ("LogUniform", 1.0),
        ("LogUniform", 1.0, 2.0, 3.0),
        ("Normal", 0.0),
        ("Normal", 0.0, 1.0, 2.0),
        ("truncated_Normal", 0.0, 1.0),           # 2 args: neither 3 nor 4
        ("truncated_Normal", 0.0, 1.0, 2.0, 3.0, 4.0),
    ],
)
def test_wrong_argument_count_raises_value_error(spec):
    with pytest.raises(ValueError):
        parse_prior_spec(spec)


def test_unknown_distribution_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown prior distribution name"):
        parse_prior_spec(("Gaussian", 0.0, 1.0))


# ── the two recovery helpers ───────────────────────────────────────────

def test_angle_from_xy_is_atan2_in_radians():
    assert angle_from_xy(1.0, 0.0) == pytest.approx(0.0)
    assert angle_from_xy(0.0, 1.0) == pytest.approx(math.pi / 2)
    assert angle_from_xy(-1.0, 0.0) == pytest.approx(math.pi)
    # Range is (-pi, pi]: the third quadrant comes back NEGATIVE.
    assert angle_from_xy(0.0, -1.0) == pytest.approx(-math.pi / 2)


def test_tau_from_xy_is_a_phase_fraction_in_the_unit_interval():
    assert tau_from_xy(1.0, 0.0) == pytest.approx(0.0)
    assert tau_from_xy(0.0, 1.0) == pytest.approx(0.25)
    assert tau_from_xy(-1.0, 0.0) == pytest.approx(0.5)
    # The wrap is mod 1, so the fourth quadrant lands at 0.75, not -0.25.
    assert tau_from_xy(0.0, -1.0) == pytest.approx(0.75)
    for x, y in [(1.0, 0.0), (0.3, -0.9), (-0.7, 0.2), (0.0, -1.0)]:
        tau = tau_from_xy(x, y)
        assert 0.0 <= tau < 1.0


def test_the_two_helpers_agree_with_each_other():
    """``tau`` is ``angle`` rescaled to turns and wrapped into [0, 1)."""
    rng = np.random.default_rng(20260922)
    for _ in range(50):
        x, y = rng.normal(size=2)
        expected = (angle_from_xy(x, y) / (2 * math.pi)) % 1.0
        assert tau_from_xy(x, y) == pytest.approx(expected)
