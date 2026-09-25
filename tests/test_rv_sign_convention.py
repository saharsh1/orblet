"""The RV sign convention, asserted rather than assumed.

The project uses the standard spectroscopic convention: **radial velocity
is POSITIVE when the star is receding** (moving away from the observer),
and ``omega`` is the PRIMARY's argument of periastron, giving

    v_r = gamma + K [cos(nu + omega) + e cos(omega)]

This is pinned here because a sign-flipped RV input does not
fail — it fits happily and returns ``omega + pi``, silently rotating the
orbit by half a turn.  On an RV-only fit that is nearly invisible; in a
joint fit it fights the astrometric node and degrades the solution.

These tests pin the convention at the forward model, where it originates.

Fixtures are synthetic; no loader, no identifier.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import rv_curve

_EPOCH_REF = 57000.0
_P_YR = 1.0
_P_DAYS = _P_YR * DAYS_PER_KEPLER_YEAR


def _curve(*, omega, ecc=0.0, K=5.0, gamma=0.0, n=2000):
    t = _EPOCH_REF + np.linspace(0.0, _P_DAYS, n, endpoint=False)
    return t, rv_curve(
        t, period_yr=_P_YR, ecc=ecc, omega_rad=omega, tau=0.0,
        K_kms=K, offset_kms=gamma, epoch_ref_mjd=_EPOCH_REF,
    )


def test_K_is_the_semi_amplitude_about_gamma():
    """Peak-to-peak is 2K and the curve is centred on gamma."""
    _, v = _curve(omega=0.0, K=5.0, gamma=12.0)
    assert float(v.max() - v.min()) == pytest.approx(10.0, rel=1e-4)
    assert float(v.mean()) == pytest.approx(12.0, abs=1e-3)


def test_positive_K_means_the_star_recedes_first_after_periastron():
    """The sign convention itself.

    For a circular orbit with omega = 0 and tau = 0 (periastron at the
    reference epoch), the textbook form gives v_r = gamma + K cos(nu), so
    the velocity is at its MAXIMUM at periastron passage — i.e. positive,
    receding.  A codebase that flipped the RV sign would put the minimum
    there instead.
    """
    t, v = _curve(omega=0.0, ecc=0.0, K=5.0, gamma=0.0)
    assert float(v[0]) == pytest.approx(5.0, rel=1e-6), (
        "at periastron with omega = 0 the RV must be +K (receding); a "
        "value of -K means the sign convention has flipped"
    )
    assert float(np.argmax(v)) == 0


def test_a_sign_flipped_input_is_absorbed_as_omega_plus_pi():
    """Why this needs a test: the failure is silent, not loud.

    Negating an RV curve produces exactly the curve of the same orbit with
    omega -> omega + pi (for a circular orbit).  So a sign-flipped input
    fits perfectly well and returns a physically DIFFERENT orientation
    rather than an error.
    """
    _, v = _curve(omega=0.7, ecc=0.0, K=5.0, gamma=0.0)
    _, v_shifted = _curve(omega=0.7 + math.pi, ecc=0.0, K=5.0, gamma=0.0)
    np.testing.assert_allclose(v_shifted, -v, rtol=0, atol=1e-12)


def test_gamma_offsets_do_not_change_the_shape():
    """gamma is an additive zero-point, never a scale."""
    _, a = _curve(omega=0.4, ecc=0.3, gamma=0.0)
    _, b = _curve(omega=0.4, ecc=0.3, gamma=-37.5)
    np.testing.assert_allclose(b - a, -37.5, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ecc", [0.0, 0.3, 0.7])
def test_eccentric_curves_keep_the_e_cos_omega_term(ecc):
    """The textbook form carries the ``+ e cos(omega)`` offset.

    Its presence is what makes the curve asymmetric about gamma for e > 0;
    dropping it is a classic silent error, so pin the mean explicitly:
    the phase-average of cos(nu) over a full period is NOT zero in time,
    but the ``e cos omega`` term shifts the curve by a KNOWN constant.
    """
    omega = 0.6
    _, v = _curve(omega=omega, ecc=ecc, K=5.0, gamma=0.0)
    # At periastron (tau = 0, nu = 0): v = K[cos(omega) + e cos(omega)]
    expected_at_peri = 5.0 * (math.cos(omega) + ecc * math.cos(omega))
    assert float(v[0]) == pytest.approx(expected_at_peri, rel=1e-6)
