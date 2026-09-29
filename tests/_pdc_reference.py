"""The PDC periodogram sweep exactly as it was before the lean loop — the
reference the optimised sweep must match byte for byte.

A verbatim copy of ``compute_pdc_periodogram`` and of every private helper
it called (``_circular_distance``, ``_u_center``, ``_u_inner``,
``_pdc_score_from_centered``, ``_semipartial_pdc_score_from_centered``,
``_as_1d_float``), taken at commit 2401179. It is self-contained on purpose:
the helpers in ``orblet.periodogram`` may change, this file may not. Only
docstrings are shortened; every executable line is unchanged.

Used by ``tests/test_pdc_lean_sweep.py`` and ``benchmarks/pdc_sweep.py``.
"""

from __future__ import annotations

import numpy as np


def _as_1d_float(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def _circular_distance(x: np.ndarray, period: float) -> np.ndarray:
    dx = x[:, None] - x[None, :]
    phi = np.mod(dx, period)
    return phi * (period - phi)


def _u_center(d: np.ndarray) -> np.ndarray:
    n = d.shape[0]
    if n < 4:
        raise ValueError("Need n >= 4 for unbiased U-centering")

    row_sum = np.sum(d, axis=1, keepdims=True)
    col_sum = np.sum(d, axis=0, keepdims=True)
    total = np.sum(d)

    a = (
        d
        - row_sum / (n - 2)
        - col_sum / (n - 2)
        + total / ((n - 1) * (n - 2))
    )
    np.fill_diagonal(a, 0.0)
    return a


def _u_inner(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    return float(np.sum(a * b) / (n * (n - 3)))


def _pdc_score_from_centered(A, B, aa):
    bb = _u_inner(B, B)
    den = np.sqrt(max(aa, 0.0) * max(bb, 0.0))
    if den == 0.0:
        return np.nan
    return _u_inner(A, B) / den


def _semipartial_pdc_score_from_centered(A, B, Z, zz, az):
    if zz <= 0:
        return np.nan

    # E = A - proj(A onto Z); use precomputed az.
    E = A - (az / zz) * Z
    num = _u_inner(E, B)
    den = np.sqrt(max(_u_inner(E, E), 0.0) * max(_u_inner(B, B), 0.0))

    if den == 0.0:
        return np.nan
    return num / den


def _phase_distance_matrix(t: np.ndarray, period: float) -> np.ndarray:
    return _circular_distance(t, period)


def reference_pdc_periodogram(
    t_days: np.ndarray,
    periods_days: np.ndarray,
    obs_dist: np.ndarray,
    *,
    nuisance_dist: np.ndarray | None = None,
    partial_mode: str = "none",
) -> dict:
    t = _as_1d_float(t_days, "t_days")
    periods = _as_1d_float(periods_days, "periods_days")
    obs_dist = np.asarray(obs_dist, dtype=float)

    n = len(t)
    if obs_dist.shape != (n, n):
        raise ValueError(
            f"obs_dist shape {obs_dist.shape} does not match "
            f"number of epochs ({n})."
        )

    if nuisance_dist is not None:
        nuisance_dist = np.asarray(nuisance_dist, dtype=float)
        if nuisance_dist.shape != (n, n):
            raise ValueError(
                f"nuisance_dist shape {nuisance_dist.shape} does not "
                f"match number of epochs ({n})."
            )

    # ── Pre-center fixed matrices (done once, not per period) ───────────
    A = _u_center(obs_dist)
    aa = _u_inner(A, A)

    Z = None
    zz = az = 0.0
    if partial_mode == "semi":
        if nuisance_dist is None:
            raise ValueError("partial_mode='semi' requires nuisance_dist")
        Z = _u_center(nuisance_dist)
        zz = _u_inner(Z, Z)
        az = _u_inner(A, Z)
        # The same arithmetic as distance_correlation(obs_dist, nuisance_dist).
        coupling = _pdc_score_from_centered(A, Z, aa)

    # ── Sweep over trial periods ──────────────────────────────────────────
    scores = np.full_like(periods, np.nan, dtype=float)

    for i, P in enumerate(periods):
        if not np.isfinite(P) or P <= 0:
            continue

        B = _u_center(_phase_distance_matrix(t, P))

        if partial_mode == "none":
            scores[i] = _pdc_score_from_centered(A, B, aa)
        elif partial_mode == "semi":
            scores[i] = _semipartial_pdc_score_from_centered(
                A, B, Z, zz, az,
            )
        else:
            raise ValueError("partial_mode must be 'none' or 'semi'")

    best_idx = int(np.nanargmax(scores))

    result = {
        "periods_days": periods,
        "scores": scores,
        "best_period_days": float(periods[best_idx]),
        "best_score": float(scores[best_idx]),
        "partial_mode": partial_mode,
    }
    if partial_mode == "semi":
        result["coupling"] = float(coupling)
    return result
