"""Parallax factors: defaults, the along-scan projection, and the demo cadence.

The defaults are Gaia at L2 and astropy's built-in ephemeris (no download, no
jplephem). These tests pin what that means, check the demo cadence is built on
the same factor, and — only where a DE432s kernel is already cached — that the
built-in ephemeris agrees with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from orblet.constants import MJD_J2010_TCB, SOLAR_SYSTEM_EPHEMERIS_PIN
from orblet.parallax import (
    L2_OFFSET_AU,
    along_scan_parallax_factor,
    fetch_ephemeris,
    per_direction_parallax_factors,
)
from orblet.simulate.cadence import DemoParallaxConsistentCadence

T = np.linspace(0.0, 5.0 * 365.25, 120) + MJD_J2010_TCB
PSI = np.random.default_rng(0).uniform(0.0, np.pi, T.size)
SKY = ((180.0, -20.0), (294.8, 14.9), (45.0, 60.0), (270.0, -66.5))


def test_defaults_are_l2_and_the_builtin_ephemeris():
    assert SOLAR_SYSTEM_EPHEMERIS_PIN == "builtin"
    ra, dec = SKY[0]
    default = per_direction_parallax_factors(T, ra, dec)
    explicit = per_direction_parallax_factors(T, ra, dec, observer="l2", ephemeris="builtin")
    np.testing.assert_array_equal(default[0], explicit[0])
    np.testing.assert_array_equal(default[1], explicit[1])


def test_l2_is_about_one_percent_beyond_the_geocentre():
    for ra, dec in SKY:
        l2 = along_scan_parallax_factor(T, PSI, ra, dec)
        geo = along_scan_parallax_factor(T, PSI, ra, dec, observer="geocentre")
        slope = float(np.dot(l2, geo) / np.dot(geo, geo))
        assert slope == pytest.approx(1.0 + L2_OFFSET_AU, abs=2e-3)


def test_along_scan_is_the_projection_of_the_two_components():
    ra, dec = SKY[1]
    f_a, f_d = per_direction_parallax_factors(T, ra, dec)
    np.testing.assert_array_equal(
        along_scan_parallax_factor(T, PSI, ra, dec),
        f_a * np.sin(PSI) + f_d * np.cos(PSI),
    )


def test_demo_cadence_uses_the_same_factor():
    cad = DemoParallaxConsistentCadence(ra_deg=45.0, dec_deg=60.0, seed=2)
    expected = along_scan_parallax_factor(
        cad.astro_obs_time + MJD_J2010_TCB, cad.astro_scan_angle, 45.0, 60.0)
    np.testing.assert_array_equal(cad.astro_parallax_factor_al, expected)
    np.testing.assert_array_equal(
        cad.parallax_factor_al_at(cad.astro_obs_time, cad.astro_scan_angle), expected)


def test_unknown_observer_raises():
    with pytest.raises(ValueError, match="observer must be"):
        per_direction_parallax_factors(T, 0.0, 0.0, observer="moon")


def test_fetch_ephemeris_names_the_extra_when_jplephem_is_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_jplephem(name, *args, **kwargs):
        if name.split(".")[0] == "jplephem":
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_jplephem)
    with pytest.raises(ImportError, match=r"orblet\[ephemeris\]"):
        fetch_ephemeris("de432s")


def _de432s_cached() -> bool:
    try:
        import jplephem  # noqa: F401
        from astropy.utils.data import is_url_in_cache
    except ImportError:
        return False
    url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp"
    return is_url_in_cache(url)


@pytest.mark.skipif(not _de432s_cached(), reason="DE432s kernel not cached (no download in tests)")
def test_builtin_agrees_with_de432s():
    for ra, dec in SKY:
        builtin = along_scan_parallax_factor(T, PSI, ra, dec)
        jpl = along_scan_parallax_factor(T, PSI, ra, dec, ephemeris="de432s")
        assert np.max(np.abs(builtin - jpl)) < 1e-6
