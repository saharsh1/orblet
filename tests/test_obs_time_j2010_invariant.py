"""Boundary invariant: obs_time must be J2010-days at the unconditional-offset
sites (gated SAFE-ADDITIVE assert).

``_assert_obs_time_is_j2010_days`` formalises an ALREADY-TRUE invariant at the
two sites that add ``MJD_J2010_TCB`` unconditionally
(``orbit.prepare`` gaia_obmt branch, ``orblet.search.presearch_astrometry_linear_ti``):
the loader-shaped ``obs_time`` arrives in days-from-J2010-TCB.  These tests pin
that a valid J2010-day array passes and an already-absolute-MJD array raises
(catching a double-applied offset upstream).

The conditional median-guard sites (``resolve_epochs_mjd`` /
``joint.nonlinear_sampler``) are intentionally NOT touched — they self-correct and a
raise there would be behaviour-moving (parked, heavy).

All inputs synthetic; no data files read.
"""

from __future__ import annotations

import numpy as np
import pytest

from orblet.prepare import _assert_obs_time_is_j2010_days


def test_j2010_day_epochs_pass():
    """A few-thousand-day J2010 array (loader convention) passes silently."""
    obs_time = np.linspace(0.0, 2000.0, 50)  # days from J2010
    _assert_obs_time_is_j2010_days(obs_time)  # must not raise


def test_already_mjd_epochs_raise():
    """An absolute-MJD array (median > 30000) raises — the offset would be
    double-applied if this slipped through.
    """
    obs_time = np.linspace(57000.0, 59000.0, 50)  # absolute MJD
    with pytest.raises(ValueError):
        _assert_obs_time_is_j2010_days(obs_time)


def test_empty_obs_time_is_silent():
    """An empty array has no median; the guard is a no-op (defensive)."""
    _assert_obs_time_is_j2010_days(np.array([], dtype=float))
