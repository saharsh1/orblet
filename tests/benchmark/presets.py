"""Test-only truths with the numbers the tests were written against.

orblet ships one preset, :meth:`OrbitSimulator.toy_orbit`. These two stay
here, in the tests, because fixtures and closure tests were built on their
exact values:

- :func:`bh3_like` — orbital numbers roughly matching Gaia BH3 (Panuzzo et al.
  2024), rounded; its sky position is BH3's, so treat it as identifying.
- :func:`circular_toy` — a compact circular orbit (P = 400 d) for quick
  forward-model checks.

Both return a plain :class:`OrbitSimulator`; pass kwargs to override a field.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orblet.simulate.orbit import OrbitSimulator


def bh3_like(**overrides: Any) -> OrbitSimulator:
    """BH3-like truth; ``source_id`` is the synthetic sentinel ``-1``."""
    defaults = dict(
        P_days=4253.1,
        e=0.7291,
        omega_rad=np.deg2rad(157.08),
        Omega_rad=np.deg2rad(195.4),
        i_rad=np.deg2rad(120.94),
        tp_mjd=58620.0,
        m1_msun=0.76,
        m2_msun=33.0,
        ra_deg=294.82716,
        dec_deg=14.93047,
        parallax_mas=1.785,
        pmra_masyr=-28.30,
        pmdec_masyr=-155.18,
        gamma_kms=-333.2,
        rv_sigma_kms=2.0,
        centroid_sigma_mas=0.40,
        source_id=-1,
    )
    defaults.update(overrides)
    return OrbitSimulator(**defaults)


def circular_toy(**overrides: Any) -> OrbitSimulator:
    """Compact circular orbit (P = 400 d, e = 0) for quick smoke tests."""
    defaults = dict(
        P_days=400.0,
        e=0.0,
        omega_rad=0.0,
        Omega_rad=np.deg2rad(45.0),
        i_rad=np.deg2rad(60.0),
        tp_mjd=58000.0,
        m1_msun=1.0,
        m2_msun=2.0,
        ra_deg=180.0,
        dec_deg=0.0,
        parallax_mas=5.0,
        pmra_masyr=0.0,
        pmdec_masyr=0.0,
    )
    defaults.update(overrides)
    return OrbitSimulator(**defaults)
