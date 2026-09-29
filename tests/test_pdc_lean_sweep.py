"""The PDC sweep against its frozen reference, byte for byte.

``compute_pdc_periodogram`` is optimised without changing a single bit of
its output. ``tests/_pdc_reference.py`` holds the sweep as it was; every
case here runs both and demands identical bytes: the score curve (NaN
positions included), the best period and score, the coupling, the period
grid echoed back, and — where the reference raises — the same exception
type and message.

Synthetic data only. ``test_periodogram_baseline.py`` pins the absolute
numbers; this file pins that the optimisation moved none of them, over the
inputs the baseline does not reach (both modes at several sizes, invalid
periods, non-contiguous and degenerate inputs, the error paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pdc_reference import reference_pdc_periodogram  # noqa: E402

from orblet.periodogram import (  # noqa: E402
    compute_pdc_periodogram,
    scalar_distance_matrix,
    scan_angle_distance_matrix,
)


def _inputs(n: int, seed: int = 0):
    """Times over ~5 yr, an RV-like signal at 211 d, and scan angles."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1800.0, n))
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    x = 3.0 * np.sin(2.0 * np.pi * t / 211.0) + 0.5 * np.cos(psi) + rng.normal(0.0, 1.0, n)
    return t, scalar_distance_matrix(x), scan_angle_distance_matrix(psi)


def _grid(n_periods: int) -> np.ndarray:
    return np.geomspace(20.0, 1000.0, n_periods)


def _bytes(x) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def _run(fn, args, kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:  # the reference's exceptions are part of the contract
        return "raised", (type(exc), str(exc))


def _assert_identical(args, kwargs) -> None:
    kind_ref, ref = _run(reference_pdc_periodogram, args, kwargs)
    kind_new, new = _run(compute_pdc_periodogram, args, kwargs)
    assert kind_new == kind_ref, (kind_ref, ref, kind_new, new)
    if kind_ref == "raised":
        assert new == ref
        return
    assert new.keys() == ref.keys()
    assert new["partial_mode"] == ref["partial_mode"]
    for key in ("periods_days", "scores", "best_period_days", "best_score", "coupling"):
        if key in ref:
            assert _bytes(new[key]) == _bytes(ref[key]), key
            assert type(new[key]) is type(ref[key]), key


# ── Both modes, three sizes ──────────────────────────────────────────────

_SIZES = [(40, 300), (200, 120), (597, 40)]  # (epochs, trial periods)


@pytest.mark.parametrize("mode", ["none", "semi"])
@pytest.mark.parametrize(("n", "n_periods"), _SIZES, ids=[f"N{n}" for n, _ in _SIZES])
def test_modes_and_sizes(mode: str, n: int, n_periods: int) -> None:
    t, obs, nui = _inputs(n)
    _assert_identical((t, _grid(n_periods), obs), {"nuisance_dist": nui, "partial_mode": mode})


def test_default_mode_without_nuisance() -> None:
    t, obs, _ = _inputs(40)
    _assert_identical((t, _grid(100), obs), {})


def test_minimum_size_four_epochs() -> None:
    t, obs, nui = _inputs(4)
    for mode in ("none", "semi"):
        _assert_identical((t, _grid(50), obs), {"nuisance_dist": nui, "partial_mode": mode})


# ── Invalid trial periods are skipped as NaN ─────────────────────────────


@pytest.mark.parametrize("mode", ["none", "semi"])
def test_invalid_periods_in_grid(mode: str) -> None:
    t, obs, nui = _inputs(40)
    periods = _grid(60)
    periods[[0, 7, 8, 30, 59]] = [np.nan, 0.0, -5.0, np.inf, -np.inf]
    _assert_identical((t, periods, obs), {"nuisance_dist": nui, "partial_mode": mode})


def test_grid_as_list_and_2d() -> None:
    t, obs, _ = _inputs(40)
    _assert_identical((t, list(_grid(30)), obs), {})
    _assert_identical((t, _grid(30).reshape(5, 6), obs), {})


# ── Memory layout: non-contiguous and transposed inputs ──────────────────


@pytest.mark.parametrize("mode", ["none", "semi"])
def test_non_contiguous_inputs(mode: str) -> None:
    t2, _, _ = _inputs(80)
    t = t2[::2]                                   # strided view
    _, obs, nui = _inputs(40, seed=1)
    obs_f = np.asfortranarray(obs)                # column-major copy
    _assert_identical((t, _grid(60)[::-1], obs_f.T), {"nuisance_dist": nui.T, "partial_mode": mode})
    _assert_identical((t, _grid(120)[::2], obs), {"nuisance_dist": obs_f, "partial_mode": mode})


def test_float32_and_int_inputs() -> None:
    t, obs, nui = _inputs(40)
    _assert_identical((t.astype(np.float32), _grid(40).astype(np.float32), obs.astype(np.float32)),
                      {"nuisance_dist": nui.astype(np.float32), "partial_mode": "semi"})
    _assert_identical((np.arange(40), np.arange(20, 60), obs), {})


# ── Degenerate inputs ────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["none", "semi"])
def test_duplicate_epochs(mode: str) -> None:
    t, obs, nui = _inputs(40)
    t = t.copy()
    t[10:14] = t[10]
    _assert_identical((t, _grid(60), obs), {"nuisance_dist": nui, "partial_mode": mode})


@pytest.mark.parametrize("mode", ["none", "semi"])
def test_constant_observations(mode: str) -> None:
    # A zero observation matrix: every score NaN, so nanargmax raises.
    t, obs, nui = _inputs(40)
    _assert_identical((t, _grid(30), np.zeros_like(obs)), {"nuisance_dist": nui, "partial_mode": mode})


def test_constant_nuisance() -> None:
    # zz = 0: the semi-partial score is NaN at every period.
    t, obs, _ = _inputs(40)
    _assert_identical((t, _grid(30), obs), {"nuisance_dist": np.ones_like(obs), "partial_mode": "semi"})


def test_nuisance_equal_to_observations() -> None:
    # E = A − A = 0: a zero residual matrix, NaN everywhere.
    t, obs, _ = _inputs(40)
    _assert_identical((t, _grid(30), obs), {"nuisance_dist": obs, "partial_mode": "semi"})


def test_nan_in_observations() -> None:
    t, obs, nui = _inputs(40)
    obs = obs.copy()
    obs[3, 5] = obs[5, 3] = np.nan
    for mode in ("none", "semi"):
        _assert_identical((t, _grid(30), obs), {"nuisance_dist": nui, "partial_mode": mode})


def test_single_period_and_empty_grid() -> None:
    t, obs, _ = _inputs(40)
    _assert_identical((t, [211.0], obs), {})
    _assert_identical((t, np.array([]), obs), {})


# ── Error paths: same exception, same message ────────────────────────────


def test_unknown_mode_raises_at_first_valid_period() -> None:
    t, obs, nui = _inputs(40)
    _assert_identical((t, _grid(30), obs), {"nuisance_dist": nui, "partial_mode": "bogus"})
    # With no valid period the mode is never checked; nanargmax raises instead.
    _assert_identical((t, [np.nan, -1.0], obs), {"partial_mode": "bogus"})


def test_all_invalid_grid_raises() -> None:
    t, obs, _ = _inputs(40)
    _assert_identical((t, [np.nan, 0.0, -3.0], obs), {})


def test_semi_without_nuisance_raises() -> None:
    t, obs, _ = _inputs(40)
    _assert_identical((t, _grid(30), obs), {"partial_mode": "semi"})


def test_shape_mismatches_raise() -> None:
    t, obs, nui = _inputs(40)
    _assert_identical((t[:-1], _grid(30), obs), {})
    _assert_identical((t, _grid(30), obs), {"nuisance_dist": nui[:-1, :-1], "partial_mode": "semi"})


def test_too_few_epochs_raises() -> None:
    t, obs, _ = _inputs(3)
    _assert_identical((t, _grid(30), obs), {})
