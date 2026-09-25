"""Distance (dissimilarity) matrices for the PDC periodogram, one per data type."""

from __future__ import annotations

import numpy as np

from orblet.periodogram._common import _as_1d_float


def _circular_distance(x: np.ndarray, period: float) -> np.ndarray:
    """
    Circular distance on a coordinate of the given period.

    d_ij = phi_ij × (period − phi_ij),  where phi_ij = (x_i − x_j) mod period.
    Zero at a whole period's separation, maximal (period²/4) at half of one.
    The phase distance (period P, on times) and the scan-angle distance
    (period π, on angles) are both this function.
    """
    dx = x[:, None] - x[None, :]
    phi = np.mod(dx, period)
    return phi * (period - phi)


def scalar_distance_matrix(x: np.ndarray) -> np.ndarray:
    """
    Absolute-difference distance matrix for a scalar variable.

    Suitable for radial velocities or any 1-D observable.

    Parameters
    ----------
    x : 1-D array
        Scalar values (e.g. RV in km/s).

    Returns
    -------
    d : (N, N) array
        d_ij = |x_i - x_j|.
    """
    x = _as_1d_float(x, "x")
    return np.abs(x[:, None] - x[None, :])


def scan_angle_distance_matrix(theta_rad: np.ndarray) -> np.ndarray:
    """
    Circular distance between scan directions, for the partial PDC.

    A scan direction is an unoriented line, so the angle is taken modulo π
    and the distance has the same form as the phase distance:

        phi_ij = (θ_i − θ_j) mod π,     d_ij = phi_ij × (π − phi_ij).

    Zero for parallel scans (including antiparallel ones), maximal (π²/4)
    for perpendicular ones.  This is the nuisance matrix of the partial PDC
    and of :func:`orblet.periodogram.scan_angle_coupling`.

    Parameters
    ----------
    theta_rad : 1-D array
        Scan position angles ψ (radians) — unrelated to the orbital
        ``theta_rad`` chain key, which merely shares the spelling.

    Returns
    -------
    d : (N, N) array
        Scan-angle distance matrix.
    """
    theta = _as_1d_float(theta_rad, "theta_rad")
    return _circular_distance(theta, np.pi)


def astrometric_segment_distance_matrix(
    v_mas: np.ndarray,
    theta_rad: np.ndarray,
    L_mas: float = 0.01,
) -> np.ndarray:
    """
    Dissimilarity matrix for 1-D scanning astrometry.

    Encodes the pairwise dissimilarity between astrometric measurements
    taken at different scan angles, accounting for the finite extent
    of the scan segment.  It depends on both the AL centroid position
    and the scan angle.  It does not in general satisfy the triangle
    inequality, so it is a dissimilarity rather than a metric; the
    distance-correlation statistics are defined for dissimilarities too.

    Parameters
    ----------
    v_mas : 1-D array
        Along-scan centroid positions (mas).
    theta_rad : 1-D array
        Scan position angles (radians).
    L_mas : float, default 0.01
        Across-scan length of the segment that stands in for each 1-D
        measurement (mas).  A scale, not a detector quantity; keep it well
        below the residual amplitude (``L_mas`` ≲ σ/10).  When the
        residuals are much smaller than ``L_mas`` the dissimilarity reduces
        to a function of the scan-angle difference alone and carries no
        information about the data.  Below that scale the results do not
        depend on it; the default suits Gaia residuals (~0.1 mas).

    Returns
    -------
    d : (N, N) array
        Astrometric segment dissimilarity matrix.
    """
    v = _as_1d_float(v_mas, "v_mas")
    th = _as_1d_float(theta_rad, "theta_rad")

    if len(v) != len(th):
        raise ValueError("v_mas and theta_rad must have same length")
    if L_mas <= 0:
        raise ValueError("L_mas must be positive")

    v1 = v[:, None]
    v2 = v[None, :]
    dth = th[:, None] - th[None, :]

    c = np.cos(dth)
    s = np.sin(dth)

    base = v1**2 - 2.0 * v1 * v2 * c + v2**2

    # Four segment-endpoint combinations.
    t1 = base + L_mas * (v1 + v2) * s + 0.5 * L_mas**2 * (1.0 + c)
    t2 = base + L_mas * (v1 - v2) * s + 0.5 * L_mas**2 * (1.0 - c)
    t3 = base - L_mas * (v1 - v2) * s + 0.5 * L_mas**2 * (1.0 - c)
    t4 = base - L_mas * (v1 + v2) * s + 0.5 * L_mas**2 * (1.0 + c)

    _safe_sqrt = lambda x: np.sqrt(np.clip(x, 0.0, None))
    D = 0.5 * (_safe_sqrt(t1) + _safe_sqrt(t2) + _safe_sqrt(t3) + _safe_sqrt(t4) - L_mas)
    np.fill_diagonal(D, 0.0)
    return D


def spectral_distance_matrix(spectra: np.ndarray) -> np.ndarray:
    """
    Chord distance between normalised spectra.

    Each spectrum is normalised to zero mean and unit L2 norm.
    The distance is derived from the cross-correlation function:

        CCF_ij = dot(sp_i, sp_j)   (after normalisation)
        d_ij   = sqrt(2) * sqrt(1 - CCF_ij)

    This equals the Euclidean chord distance between unit vectors
    on the hypersphere.  Identical spectra have d=0; orthogonal
    spectra have d=sqrt(2).

    Parameters
    ----------
    spectra : (N_epochs, N_pixels) array
        All spectra resampled onto a common wavelength grid.
        NaN pixels are set to zero before normalisation.

    Returns
    -------
    d : (N_epochs, N_epochs) array
        Spectral distance matrix.
    """
    S = np.array(spectra, dtype=float)
    if S.ndim != 2:
        raise ValueError(
            f"spectra must be 2-D (N_epochs, N_pixels), got shape {S.shape}"
        )

    # Replace NaN with zero so they don't contribute to the dot product.
    S = np.where(np.isfinite(S), S, 0.0)

    # Normalise each row: subtract mean, divide by L2 norm.
    S = S - S.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(S**2, axis=1, keepdims=True))
    # Guard against zero-norm spectra (all-constant or all-NaN).
    norms = np.where(norms > 0, norms, 1.0)
    S = S / norms

    # Cross-correlation matrix and chord distance.
    ccf = S @ S.T
    # Clip to [−1, 1] for numerical safety before sqrt.
    ccf = np.clip(ccf, -1.0, 1.0)
    D = np.sqrt(2.0) * np.sqrt(1.0 - ccf)
    np.fill_diagonal(D, 0.0)
    return D
