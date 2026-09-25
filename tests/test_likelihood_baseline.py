"""The three log-likelihoods, byte for byte.

Exact comparison, no tolerance. Three scalars is a small baseline, but the
joint one is not a small test: it runs the full chain — theta split, RV
model, photocentre projection, along-scan projection, both
Gaussian-with-jitter terms — so a change anywhere underneath shows up in it.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _likelihood_baseline import LIKELIHOOD_BASELINE  # noqa: E402


#: Where the likelihoods live.
_SEARCH_PATH = ("orblet.likelihood",)


def _fn(name: str):
    for mod_name in _SEARCH_PATH:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        obj = getattr(mod, name, None)
        if obj is not None:
            return obj
    raise AssertionError(
        f"log-likelihood {name!r} not found in {_SEARCH_PATH}"
    )


# ── the same fixed inputs the generator used ──────────────────────────
EPOCH_REF = 57388.5
T = np.linspace(57388.5, 57388.5 + 5.0 * 365.25, 24)
PSI = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
PF = np.cos(PSI * 1.7) * 0.8
OMEGA, INC, OMEGA_NODE = math.pi / 5.0, math.pi / 3.0, math.pi / 7.0

_rng = np.random.default_rng(20260923)
RV_OBS = 10.0 * np.sin(2 * np.pi * (T - T[0]) / 912.0) + _rng.normal(0, 0.5, T.size)
RV_ERR = np.full(T.size, 0.6)
AL_OBS = _rng.normal(0.0, 0.4, T.size)
AL_ERR = np.full(T.size, 0.45)

THETA = dict(
    P_yr=2.5, e=0.37, omega_rad=OMEGA, tau=0.25, inc_rad=INC,
    Omega_rad=OMEGA_NODE, m2_msun=0.8, M_total_msun=1.9, plx_mas=4.2,
    gamma_kms=-0.25, rv_jitter_kms=0.3, astro_jitter_mas=0.2,
    ra_offset_mas=0.3, dec_offset_mas=-0.4,
    pmra_masyr=1.1, pmdec_masyr=-0.7,
)


def _compute() -> dict[str, float]:
    out: dict[str, float] = {}
    # The RV engine function is named `loglike`; the public re-export calls it
    # `rv_loglike`. Ask for the public name, fall back to the engine one.
    try:
        rv = _fn("rv_loglike")
    except AssertionError:
        rv = _fn("loglike")
    out["rv_loglike"] = rv(
        T, RV_OBS, RV_ERR,
        period_yr=2.5, ecc=0.37, omega_rad=OMEGA, tau=0.25,
        mass_msun=0.8, M_msun=1.9, offset_kms=-0.25, jitter_kms=0.3,
        epoch_ref_mjd=EPOCH_REF,
    )
    out["loglike_along_scan"] = _fn("loglike_along_scan")(
        model_along_scan=np.cos(PSI) * 0.7,
        centroid_pos=AL_OBS, centroid_pos_err=AL_ERR, jitter_mas=0.2,
    )
    out["loglike_joint"] = _fn("loglike_joint")(
        THETA,
        rv_arrays=dict(t_mjd=T, rv_obs=RV_OBS, rv_err=RV_ERR),
        astro_arrays=dict(
            t_mjd=T, psi=PSI, parallax_factor_al=PF,
            centroid_pos=AL_OBS, centroid_pos_err=AL_ERR,
        ),
        epoch_ref_mjd=EPOCH_REF,
    )
    return out


@pytest.mark.parametrize("case", sorted(LIKELIHOOD_BASELINE))
def test_loglike_is_bit_identical_to_the_baseline(case: str) -> None:
    got = float(_compute()[case])
    expected = float(LIKELIHOOD_BASELINE[case])
    assert got == expected, (
        f"{case}: {expected!r} -> {got!r}. A deliberate change re-blesses the "
        "baseline in its own commit."
    )


def test_the_baseline_is_not_empty() -> None:
    assert len(LIKELIHOOD_BASELINE) >= 3, (
        f"baseline holds {len(LIKELIHOOD_BASELINE)} cases; it was captured "
        "with 3."
    )
