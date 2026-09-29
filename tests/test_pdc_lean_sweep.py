"""The PDC sweep against its frozen reference.

``compute_pdc_periodogram`` is optimised; ``tests/_pdc_reference.py`` holds
the sweep as it was. Every case runs both, under two contracts:

* **Equivalent** (always): the same NaN positions, the same exceptions
  (type and message), the period grid echoed back unchanged, every score
  and the coupling within ``SCORE_ATOL`` of the reference, and the same
  best period — unless the reference's top two scores are themselves
  within ``SCORE_ATOL``, a genuine tie that rounding may break either way.
* **Byte-identical** (while ``EXACT_BYTES`` is true): every float the same
  to the last bit.

``EXACT_BYTES`` is switched off only by a deliberate change of rounding,
which re-blesses ``test_periodogram_baseline.py`` in the same commit and
says what moved; ``MAX_SEEN`` below then records the largest score
difference the equivalence tier measured.

Synthetic data only. The baseline pins the absolute numbers; this file pins
that the optimisation moved none of them (or, if deliberate, how little),
over the inputs the baseline does not reach: both modes at several sizes,
invalid periods, non-contiguous and degenerate inputs, the error paths.
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


#: Scores are correlations, at most 1 in size: an absolute tolerance.
SCORE_ATOL = 1e-12

#: True while the sweep is meant to reproduce the reference bit for bit.
#: False since the phase reduction became P · frac(Δt · (1/P)) instead of
#: np.mod(Δt, P): the largest |score − reference| measured here is 1.3e-13,
#: in the 4-epoch case (few terms, heavy cancellation in the U-centring);
#: from 40 epochs up it is about 2e-15.
EXACT_BYTES = False

#: Largest |score − reference| over all cases, filled as the cases run.
MAX_SEEN = {"score": 0.0}


def _bytes(x) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def _run(fn, args, kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:  # the reference's exceptions are part of the contract
        return "raised", (type(exc), str(exc))


def _assert_equivalent(ref: dict, new: dict) -> None:
    assert new.keys() == ref.keys()
    assert new["partial_mode"] == ref["partial_mode"]
    assert _bytes(new["periods_days"]) == _bytes(ref["periods_days"])
    for key in ("best_period_days", "best_score", "coupling"):
        if key in ref:
            assert type(new[key]) is type(ref[key]), key

    s_ref, s_new = ref["scores"], new["scores"]
    assert np.array_equal(np.isnan(s_new), np.isnan(s_ref)), "NaN positions differ"
    finite = ~np.isnan(s_ref)
    diff = float(np.max(np.abs(s_new[finite] - s_ref[finite]), initial=0.0))
    MAX_SEEN["score"] = max(MAX_SEEN["score"], diff)
    assert diff <= SCORE_ATOL, f"max |score difference| {diff:.3e}"

    if "coupling" in ref:
        if np.isnan(ref["coupling"]):
            assert np.isnan(new["coupling"])
        else:
            assert abs(new["coupling"] - ref["coupling"]) <= SCORE_ATOL
    assert abs(new["best_score"] - ref["best_score"]) <= SCORE_ATOL

    if new["best_period_days"] != ref["best_period_days"]:
        top_two = np.sort(s_ref[finite])[-2:]
        assert top_two[1] - top_two[0] <= SCORE_ATOL, (
            f"best period moved {ref['best_period_days']} -> {new['best_period_days']} "
            "without a tie at the top"
        )


def _assert_identical(args, kwargs, *, ill_conditioned: bool = False) -> None:
    """Both contracts; ``ill_conditioned`` marks a case whose scores are
    rounding noise by construction, which only the byte tier can pin."""
    kind_ref, ref = _run(reference_pdc_periodogram, args, kwargs)
    kind_new, new = _run(compute_pdc_periodogram, args, kwargs)
    assert kind_new == kind_ref, (kind_ref, ref, kind_new, new)
    if kind_ref == "raised":
        assert new == ref
        return
    if ill_conditioned:
        assert np.array_equal(np.isnan(new["scores"]), np.isnan(ref["scores"]))
    else:
        _assert_equivalent(ref, new)
    if EXACT_BYTES:
        for key in ("scores", "best_period_days", "best_score", "coupling"):
            if key in ref:
                assert _bytes(new[key]) == _bytes(ref[key]), key


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
    _assert_identical((t, _grid(120)[::2], obs),
                      {"nuisance_dist": np.asfortranarray(nui), "partial_mode": mode})


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


def test_nuisance_equal_to_observations_other_layout() -> None:
    # The same matrix in column-major order: the two U-centrings sum in a
    # different order, so E = A − (⟨A,Z⟩/⟨Z,Z⟩) Z is rounding residue, not
    # zero, and every score is noise over noise. Any change of rounding
    # moves it arbitrarily; only the byte tier can pin it. (The partial PDC
    # is ill-conditioned whenever the nuisance explains the observations
    # almost completely, i.e. coupling → 1.)
    t, obs, _ = _inputs(40)
    _assert_identical((t, _grid(30), obs),
                      {"nuisance_dist": np.asfortranarray(obs), "partial_mode": "semi"},
                      ill_conditioned=True)


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
