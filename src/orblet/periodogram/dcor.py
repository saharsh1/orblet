"""Unbiased distance correlation: U-centring, the U-centred inner product,
and the ordinary and semi-partial scores built on them.

The statistics core of the PDC periodogram; :func:`distance_correlation` is
the public entry point for a single pair of distance matrices.
"""

from __future__ import annotations

import numpy as np


def _u_center(d: np.ndarray) -> np.ndarray:
    """
    U-center a symmetric distance matrix (unbiased prescription).

    Parameters
    ----------
    d : (N, N) array
        Symmetric distance matrix.

    Returns
    -------
    a : (N, N) array
        U-centered matrix with zero diagonal.
    """
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
    """Inner product for U-centered distance matrices."""
    n = a.shape[0]
    return float(np.sum(a * b) / (n * (n - 3)))


def _pdc_score_from_centered(
    A: np.ndarray,
    B: np.ndarray,
    aa: float,
) -> float:
    """
    Ordinary PDC score from pre-U-centered matrices.

    Parameters
    ----------
    A : (N, N) array
        U-centered observation matrix (precomputed).
    B : (N, N) array
        U-centered phase matrix (computed per period).
    aa : float
        Precomputed _u_inner(A, A).
    """
    bb = _u_inner(B, B)
    den = np.sqrt(max(aa, 0.0) * max(bb, 0.0))
    if den == 0.0:
        return np.nan
    return _u_inner(A, B) / den


def _semipartial_pdc_score_from_centered(
    A: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    zz: float,
    az: float,
) -> float:
    """
    Semi-partial PDC score from pre-U-centered matrices.

    The nuisance is projected out of the observation matrix only:
    E = A − (⟨A,Z⟩/⟨Z,Z⟩) Z, then the score is ⟨E,B⟩ / (‖E‖ ‖B‖).

    Parameters
    ----------
    A : (N, N) array
        U-centered observation matrix (precomputed).
    B : (N, N) array
        U-centered phase matrix (computed per period).
    Z : (N, N) array
        U-centered nuisance matrix (precomputed).
    zz, az : float
        Precomputed inner products: _u_inner(Z,Z), _u_inner(A,Z).
    """
    if zz <= 0:
        return np.nan

    # E = A - proj(A onto Z); use precomputed az.
    E = A - (az / zz) * Z
    num = _u_inner(E, B)
    den = np.sqrt(max(_u_inner(E, E), 0.0) * max(_u_inner(B, B), 0.0))

    if den == 0.0:
        return np.nan
    return num / den


# Convenience wrappers for use outside the hot loop (e.g. distance_correlation).

def _pdc_score(obs_dist: np.ndarray, phase_dist: np.ndarray) -> float:
    """Ordinary unbiased distance-correlation score (non-optimised)."""
    A = _u_center(obs_dist)
    B = _u_center(phase_dist)
    aa = _u_inner(A, A)
    return _pdc_score_from_centered(A, B, aa)


def distance_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """
    Unbiased distance correlation between two distance matrices.

    Useful as a diagnostic: e.g. measuring how strongly the
    observations depend on the scan angle (for that case see
    :func:`orblet.periodogram.scan_angle_coupling`).

    Parameters
    ----------
    A, B : (N, N) arrays
        Symmetric distance (or dissimilarity) matrices of the same size.

    Returns
    -------
    score : float
        Unbiased (U-centred) distance correlation, at most 1.  Near zero
        means no detected dependence; larger positive values mean stronger
        dependence.  Unlike the biased statistic it can be NEGATIVE, and a
        negative value does not mean an opposite dependence.  NaN when
        either norm vanishes.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")
    return _pdc_score(A, B)
