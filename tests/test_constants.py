"""The package's constants against ``astropy.constants``, ``astropy.units``
and ``astropy.time``.

Each constant is either bit-identical to astropy's value or differs from it
by a deliberate, documented amount. The identical ones are asserted with
``==``. The deliberate differences are pinned to their size, so a change on
either side (ours, or a new astropy / CODATA / IAU release) turns a test red
instead of silently moving a number.
"""

from __future__ import annotations

import math

import pytest
from astropy import constants as ac
from astropy import units as u
from astropy.time import Time

from orblet import constants as C


# ── Bit-identical to astropy ─────────────────────────────────────────────


def test_c_kms_is_astropy_c() -> None:
    assert C.C_KMS == ac.c.to_value(u.km / u.s)


def test_au_m_is_astropy_au() -> None:
    assert C.AU_M == ac.au.si.value


def test_g_si_is_astropy_g() -> None:
    assert C.G_SI == ac.G.si.value


def test_msun_kg_is_astropy_m_sun() -> None:
    # Both are the IAU 2015 nominal GM_sun divided by the same G.
    assert C.MSUN_KG == ac.M_sun.si.value
    assert C.MSUN_KG * C.G_SI == pytest.approx(ac.GM_sun.si.value, rel=1e-15)


def test_days_per_kepler_year_is_astropy_julian_year() -> None:
    assert C.DAYS_PER_KEPLER_YEAR == u.yr.to(u.day)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("MJD_J2010_TCB", "J2010.0"),
        ("MJD_J2016_TCB", "J2016.0"),
        ("MJD_J2017_5_TCB", "J2017.5"),
    ],
)
def test_epoch_anchors_are_astropy_julian_epochs(name: str, label: str) -> None:
    assert getattr(C, name) == Time(label, scale="tcb").mjd


# ── Deliberate differences, pinned ───────────────────────────────────────

#: ``GM_SUN_SI / GM_sun − 1``: the mass unit 4π²·AU³/yr² (Julian year)
#: against the IAU 2015 nominal solar mass parameter.
_GM_UNIT_OFFSET = 3.7774e-5

#: ``MSUN_IN_MJUP / (M_sun / M_jup) − 1``: the IAU 2009 mass ratio, rounded
#: to 1047.35, against the IAU 2015 nominal ratio GM_sun / GM_jup.
_MJUP_RATIO_OFFSET = -2.0573e-4


def test_gm_sun_si_is_the_unit_system_mass() -> None:
    year_s = C.DAYS_PER_KEPLER_YEAR * 86400.0
    assert C.GM_SUN_SI == 4.0 * math.pi**2 * C.AU_M**3 / year_s**2


def test_gm_sun_si_offset_from_astropy_gm_sun_is_pinned() -> None:
    offset = C.GM_SUN_SI / ac.GM_sun.si.value - 1.0
    assert offset == pytest.approx(_GM_UNIT_OFFSET, rel=1e-4)


def test_msun_in_mjup_offset_from_astropy_ratio_is_pinned() -> None:
    iau2015 = (ac.M_sun / ac.M_jup).decompose().value
    assert C.MSUN_IN_MJUP / iau2015 - 1.0 == pytest.approx(
        _MJUP_RATIO_OFFSET, rel=1e-4
    )
    # And it is the IAU 2009 ratio to the two decimals it is typed with.
    assert C.MSUN_IN_MJUP == round(1047.348644, 2)
