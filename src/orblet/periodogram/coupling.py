"""How strongly a source's astrometry depends on the scan angle.

One number per source, no period scan: the coupling D_cpl between the
astrometric dissimilarity and the scan-angle distance. A quick screen for
sources whose period search the scan geometry is likely to affect, before
any periodogram or orbit is computed.
"""

from __future__ import annotations

import numpy as np

from orblet.periodogram.dcor import distance_correlation
from orblet.periodogram.distances import (
    astrometric_segment_distance_matrix,
    scan_angle_distance_matrix,
)


def scan_angle_coupling(
    v_mas: np.ndarray,
    theta_rad: np.ndarray,
    *,
    L_mas: float = 0.01,
) -> float:
    """
    Scan-angle coupling D_cpl of one source's along-scan data.

    The unbiased distance correlation between the astrometric
    dissimilarity (:func:`astrometric_segment_distance_matrix`) and the
    scan-angle distance (:func:`scan_angle_distance_matrix`):

        D_cpl = ⟨Ã, C̃⟩_U / (‖Ã‖_U ‖C̃‖_U),

    with Ã and C̃ the U-centred matrices.  It is the same number that
    :func:`compute_pdc_periodogram` returns as ``"coupling"`` for a partial
    PDC built from these two matrices.

    Parameters
    ----------
    v_mas : 1-D array
        Along-scan abscissae (mas), as RESIDUALS: remove the 5-parameter
        model first (e.g. ``fit_astrometric_5param(...).residuals``).
        Position, proper motion and parallax are themselves scan-angle
        dependent and would dominate the statistic.
    theta_rad : 1-D array
        Scan position angles ψ (radians).
    L_mas : float, default 0.01
        Segment length of the astrometric dissimilarity (mas); keep it well
        below the residual amplitude (see
        :func:`astrometric_segment_distance_matrix`).  Too large an
        ``L_mas`` drives D_cpl towards 1 whatever the data.

    Returns
    -------
    float
        D_cpl.  Larger positive values mean stronger coupling to the scan
        geometry.  The U-centred statistic can be negative; a negative
        value does not mean an opposite dependence.  NaN when a norm
        vanishes (e.g. every scan angle equal).
    """
    v = np.asarray(v_mas, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    return distance_correlation(
        astrometric_segment_distance_matrix(v, theta, L_mas=L_mas),
        scan_angle_distance_matrix(theta),
    )
