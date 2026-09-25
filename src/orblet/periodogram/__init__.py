"""Period-search atoms: arrays in, a periodogram out.

Two periodograms and what they need:

    lomb_scargle   Lomb–Scargle on epoch radial velocities
    pdc            the phase distance correlation (PDC) periodogram
    distances      the distance matrices PDC compares, one per data type
    dcor           unbiased distance correlation, the statistic under PDC
    coupling       how strongly a source's astrometry depends on scan angle
    peaks          measurements on a periodogram peak

Every public name is importable from here.
"""

from __future__ import annotations

from orblet.periodogram.coupling import scan_angle_coupling
from orblet.periodogram.dcor import distance_correlation
from orblet.periodogram.distances import (
    astrometric_segment_distance_matrix,
    scalar_distance_matrix,
    scan_angle_distance_matrix,
    spectral_distance_matrix,
)
from orblet.periodogram.lomb_scargle import compute_lomb_scargle_periodogram
from orblet.periodogram.pdc import compute_pdc_periodogram, pdc_false_alarm_probability
from orblet.periodogram.peaks import peak_fwhm_days

__all__ = [
    "astrometric_segment_distance_matrix",
    "compute_lomb_scargle_periodogram",
    "compute_pdc_periodogram",
    "distance_correlation",
    "pdc_false_alarm_probability",
    "peak_fwhm_days",
    "scalar_distance_matrix",
    "scan_angle_coupling",
    "scan_angle_distance_matrix",
    "spectral_distance_matrix",
]
