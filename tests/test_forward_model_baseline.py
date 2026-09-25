"""Every forward model, byte for byte.

Each forward model evaluated on one fixed orbit and one fixed 32-point grid,
compared against stored values with ``np.array_equal``. Not ``allclose``: a
tolerance would let a real change hide under it, and a refactor has no
licence to move even the last bit.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forward_model_baseline import FORWARD_BASELINE  # noqa: E402


#: Where the forward models live.
_SEARCH_PATH = ("orblet.model",)


def _forward(name: str):
    """The forward function ``name``, wherever it currently lives."""
    for mod_name in _SEARCH_PATH:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        obj = getattr(mod, name, None)
        if obj is not None:
            return obj
    raise AssertionError(
        f"forward function {name!r} not found in {_SEARCH_PATH}"
    )


# ── the fixed orbit and grid, identical to the generator's ────────────
T = np.linspace(57388.5, 57388.5 + 5.0 * 365.25, 32)
PSI = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
PF = np.cos(PSI * 1.7) * 0.8
OMEGA, INC, OMEGA_NODE = math.pi / 5.0, math.pi / 3.0, math.pi / 7.0
EPOCH_REF = 57388.5
ORB = dict(period_yr=2.5, ecc=0.37, tp_mjd=57500.0)


def _compute() -> dict[str, object]:
    """Recompute every baselined case from today's code."""
    out: dict[str, object] = {}

    out["rv_curve"] = _forward("rv_curve")(
        T, period_yr=2.5, ecc=0.37, omega_rad=OMEGA, tau=0.25,
        K_kms=12.5, offset_kms=-3.25, epoch_ref_mjd=EPOCH_REF,
    )
    out["rv_model"] = _forward("rv_model")(
        T, period_yr=2.5, ecc=0.37, omega_rad=OMEGA, tau=0.25,
        mass_msun=0.8, M_msun=1.9, offset_kms=-3.25,
        epoch_ref_mjd=EPOCH_REF,
    )
    out["semi_amplitude_kms"] = _forward("semi_amplitude_kms")(
        mass_msun=0.8, period_yr=2.5, ecc=0.37, M_total_msun=1.9,
    )
    out["_true_anomaly"] = _forward("_true_anomaly")(T, **ORB)
    out["fm_from_K"] = _forward("fm_from_K")(12.5, 2.5, 0.37)

    d_ra, d_dec = _forward("campbell_xy")(
        t_mjd=T, period_yr=2.5, ecc=0.37, omega=OMEGA, inc=INC,
        Omega=OMEGA_NODE, tp_mjd=57500.0,
        m_comp_msun=0.8, M_total_msun=1.9, plx_mas=4.2,
    )
    out["campbell_xy_ra"] = d_ra
    out["campbell_xy_dec"] = d_dec

    ti_ra, ti_dec = _forward("thiele_innes_xy")(
        t_mjd=T, period_yr=2.5, ecc=0.37,
        A_mas=3.1, B_mas=-2.2, F_mas=1.4, G_mas=0.9, tp_mjd=57500.0,
    )
    out["thiele_innes_xy_ra"] = ti_ra
    out["thiele_innes_xy_dec"] = ti_dec

    kx, ky = _forward("kepler_xy_orbit")(T, **ORB)
    out["kepler_xy_orbit_x"] = kx
    out["kepler_xy_orbit_y"] = ky

    out["along_scan_model"] = _forward("along_scan_model")(
        d_ra=d_ra, d_dec=d_dec, psi=PSI, t_mjd=T, epoch_ref_mjd=EPOCH_REF,
        parallax_factor_al=PF, plx_mas=4.2,
        ra_offset_mas=0.3, dec_offset_mas=-0.4,
        pmra_masyr=1.1, pmdec_masyr=-0.7,
    )
    out["_ti_constants_unit"] = np.asarray(
        _forward("_ti_constants_unit")(OMEGA, OMEGA_NODE, math.cos(INC)),
        dtype=float,
    )
    out["_inc_from_u_latent"] = _forward("_inc_from_u_latent")(0.37)
    out["_tau_from_disk_angle"] = _forward("_tau_from_disk_angle")(0.4, -0.9)
    out["tp_from_disk_angle"] = _forward("tp_from_disk_angle")(
        0.4, -0.9, period_days=912.0, epoch_ref_mjd=EPOCH_REF,
    )
    out["rv_model_with_inclination"] = _forward("rv_model_with_inclination")(
        T, period_yr=2.5, ecc=0.37, omega_rad=OMEGA, tau=0.25,
        mass_msun=0.8, M_msun=1.9, offset_kms=-3.25, inc_rad=INC,
        epoch_ref_mjd=EPOCH_REF,
    )
    return out


@pytest.mark.parametrize("case", sorted(FORWARD_BASELINE))
def test_forward_value_is_byte_identical_to_the_baseline(
    case: str,
) -> None:
    got = _compute()[case]
    expected = FORWARD_BASELINE[case]

    if isinstance(expected, list):
        got_arr = np.asarray(got, dtype=float).ravel()
        exp_arr = np.asarray(expected, dtype=float)
        assert got_arr.shape == exp_arr.shape, (
            f"{case}: shape changed {exp_arr.shape} -> {got_arr.shape}"
        )
        assert np.array_equal(got_arr, exp_arr), (
            f"{case}: values moved. Largest difference "
            f"{np.max(np.abs(got_arr - exp_arr)):.3e}. A deliberate change "
            "re-blesses the baseline in its own commit."
        )
    else:
        assert float(got) == float(expected), (
            f"{case}: {float(expected)!r} -> {float(got)!r}"
        )


def test_the_baseline_is_not_empty() -> None:
    """A baseline that silently emptied would pass every case above."""
    assert len(FORWARD_BASELINE) >= 17, (
        f"baseline holds {len(FORWARD_BASELINE)} cases; it was captured with "
        "17. A shrinking baseline is a scan that stopped looking."
    )
