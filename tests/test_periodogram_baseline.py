"""The periodogram sub-package, byte for byte.

Exact comparison, no tolerance. Each case is fingerprinted by the SHA-256 of
its shape and float64 bytes (an N×N kernel is too large to list verbatim),
with the plain sum kept beside it so a failure says roughly how far it moved.

Every function is resolved through the public front door
``orblet.periodogram``. A deliberate change of number re-blesses here in its
own commit and says what moved. The periodogram is a young capability: its
numbers are expected to be re-blessed once more when it is declared stable.

The Lomb–Scargle flagged and NaN-bearing cases pass pre-masked arrays: the
function takes its input as given, and masking is the caller's step.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _periodogram_baseline import PERIODOGRAM_BASELINE  # noqa: E402

import orblet.periodogram as pg  # noqa: E402


# ── fixed synthetic inputs (identical to the generator's) ────────────
_rng = np.random.default_rng(20260925)
N = 40
T = np.sort(_rng.uniform(0.0, 1500.0, N))                   # days
PSI = _rng.uniform(0.0, 2.0 * np.pi, N)                     # rad
AL = (0.6 * np.sin(2 * np.pi * T / 211.0) * np.cos(PSI)
      + 0.4 * np.cos(2 * np.pi * T / 211.0) * np.sin(PSI)
      + _rng.normal(0.0, 0.1, N))                           # mas
RV = 8.0 * np.sin(2 * np.pi * T / 97.0) + _rng.normal(0.0, 0.7, N)  # km/s
RV_ERR = np.full(N, 0.7)
FLAG = np.ones(N, dtype=bool)
FLAG[[3, 11, 27]] = False
SPECTRA = _rng.normal(0.0, 1.0, (N, 64))
SPECTRA[5, 10] = np.nan
PERIODS = 1.0 / np.linspace(1.0 / 1000.0, 1.0 / 20.0, 300)  # days
FREQ = np.linspace(1.0 / 1000.0, 1.0 / 20.0, 300)           # 1/day


def _digest(x) -> dict:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    h = hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()
    return {"sha256": h, "sum": float(np.nansum(a))}


def _ls(out: dict, name: str, res) -> None:
    freq, power, fbest, pbest = res
    out[f"{name}.frequency"] = _digest(freq)
    out[f"{name}.power"] = _digest(power)
    out[f"{name}.best"] = _digest([fbest, pbest])


def _compute() -> dict[str, dict]:
    out: dict[str, dict] = {}

    # Lomb–Scargle: the three grids, a validity-flagged case, a NaN case,
    # and a two-harmonic model on an explicit grid.
    _ls(out, "ls_auto", pg.compute_lomb_scargle_periodogram(RV, RV_ERR, T))
    _ls(out, "ls_lin", pg.compute_lomb_scargle_periodogram(
        RV, RV_ERR, T, spacing="lin", n_freq=500))
    _ls(out, "ls_log", pg.compute_lomb_scargle_periodogram(
        RV, RV_ERR, T, spacing="log", n_freq=500))
    _ls(out, "ls_flagged", pg.compute_lomb_scargle_periodogram(
        RV[FLAG], RV_ERR[FLAG], T[FLAG], frequency=FREQ))
    rv_nan = RV.copy()
    rv_nan[[0, 19]] = np.nan
    err_nan = RV_ERR.copy()
    err_nan[33] = np.nan
    ok = np.isfinite(rv_nan) & np.isfinite(err_nan)
    _ls(out, "ls_nan", pg.compute_lomb_scargle_periodogram(
        rv_nan[ok], err_nan[ok], T[ok], frequency=FREQ))
    _ls(out, "ls_nterms2", pg.compute_lomb_scargle_periodogram(
        RV, RV_ERR, T, frequency=FREQ, nterms=2))

    # Distance kernels.
    d_scalar = pg.scalar_distance_matrix(RV)
    d_seg = pg.astrometric_segment_distance_matrix(AL, PSI, L_mas=1.0)
    out["scalar_distance_matrix"] = _digest(d_scalar)
    out["astrometric_segment_distance_matrix.L1"] = _digest(d_seg)
    out["astrometric_segment_distance_matrix.L0.3"] = _digest(
        pg.astrometric_segment_distance_matrix(AL, PSI, L_mas=0.3))
    out["astrometric_segment_distance_matrix.default_L"] = _digest(
        pg.astrometric_segment_distance_matrix(AL, PSI))
    out["spectral_distance_matrix"] = _digest(
        pg.spectral_distance_matrix(SPECTRA))

    # PDC: ordinary, and partial (the nuisance out of the observation space).
    for name, kwargs, dist in (
        ("pdc_none_rv", {}, d_scalar),
        ("pdc_none_al", {}, d_seg),
    ):
        res = pg.compute_pdc_periodogram(T, PERIODS, dist, **kwargs)
        out[f"{name}.scores"] = _digest(res["scores"])
        out[f"{name}.best"] = _digest(
            [res["best_period_days"], res["best_score"]])

    d_scan = pg.scan_angle_distance_matrix(PSI)
    out["distance_correlation"] = _digest(
        pg.distance_correlation(d_seg, d_scan))
    out["scan_angle_distance_matrix"] = _digest(d_scan)
    res = pg.compute_pdc_periodogram(
        T, PERIODS, d_seg, nuisance_dist=d_scan, partial_mode="semi")
    out["pdc_semi_scan.scores"] = _digest(res["scores"])
    out["pdc_semi_scan.best"] = _digest(
        [res["best_period_days"], res["best_score"], res["coupling"]])
    out["scan_angle_coupling"] = _digest(pg.scan_angle_coupling(AL, PSI))
    out["pdc_false_alarm_probability"] = _digest(
        pg.pdc_false_alarm_probability(res["scores"], N))

    scores = pg.compute_pdc_periodogram(T, PERIODS, d_scalar)["scores"]
    out["peak_fwhm_days.half"] = _digest(pg.peak_fwhm_days(PERIODS, scores))
    out["peak_fwhm_days.0.3"] = _digest(
        pg.peak_fwhm_days(PERIODS, scores, level=0.3))
    return out


@pytest.fixture(scope="module")
def computed() -> dict[str, dict]:
    return _compute()


@pytest.mark.parametrize("case", sorted(PERIODOGRAM_BASELINE))
def test_periodogram_is_bit_identical_to_the_baseline(case, computed) -> None:
    got = computed[case]
    expected = PERIODOGRAM_BASELINE[case]
    assert got["sha256"] == expected["sha256"], (
        f"{case}: bytes changed (sum {expected['sum']!r} -> {got['sum']!r}). "
        "A move is a text operation; a deliberate numeric change re-blesses "
        "in its own commit."
    )


def test_every_computed_case_is_baselined(computed) -> None:
    assert sorted(computed) == sorted(PERIODOGRAM_BASELINE)
