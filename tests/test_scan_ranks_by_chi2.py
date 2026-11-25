"""
The frequency scan ranks its nodes by the profiled chi2, not by logL_marginal.

The profiled chi2 of a node: the weighted residual sum at the best linear
amplitudes for that shape (``sol.chi2`` of ``linear_solve_ti``).
``logL_marginal`` adds ``log det M``, which rewards a shape whose amplitudes are
poorly determined.  The case below makes that happen: an eccentric orbit
(e = 0.6, P = 300 d) whose periastron passage holds no observation.  There the
largest logL_marginal is at e = 0.95, and the smallest chi2 is the true node.

Offline and seeded; synthetic data only.  What is real: the atoms.  What is a
stand-in: the orbit, the epochs, the errors, the noise, and the removed epochs.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import _synthesize_astro
from orblet import scan_ti_frequency
from orblet.solve.astrometry import _ti_design_matrix, linear_solve_ti

PERIOD_DAYS = 300.0   # the true period
TRUE_ECC = 0.6
TRUE_TAU = 0.4        # the conftest's tp_offset: the periastron phase on the grid
ECC_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
TAU_GRID = tuple(np.round(np.arange(0.0, 1.0, 0.05), 2))   # 0.00, 0.05, ..., 0.95
GAP_PHASE = 0.05      # no observation within ±0.05 in phase (±15 d) of periastron
SOURCE_SEED = 16      # a source where the two scores choose different nodes


def _source_without_periastron():
    """The synthetic source, with the epochs near periastron removed: arrays and the reference epoch."""
    astro, truth = _synthesize_astro(seed=SOURCE_SEED, n_epochs=70, P_days=PERIOD_DAYS,
                                     e=TRUE_ECC, sigma_mas=0.4, m_comp_msun=0.5)
    phase = ((astro["obs_time"] - truth["tp_mjd"]) / PERIOD_DAYS) % 1.0   # 0 at periastron
    distance_from_periastron = np.minimum(phase, 1.0 - phase)
    keep = distance_from_periastron > GAP_PHASE
    arrays = {
        "t_mjd": astro["obs_time"][keep],
        "psi": astro["scan_angle"][keep],
        "parallax_factor_al": astro["parallax_factor_al"][keep],
        "d_obs": astro["centroid_pos"][keep],
        "sigma": astro["centroid_pos_err"][keep],
    }
    return arrays, truth["epoch_ref_mjd"]


def _scores_at_true_period(arrays, epoch_ref_mjd):
    """Every (e, tau) node at the true frequency: a list of (ecc, tau, chi2, logL_marginal)."""
    nodes = []
    for ecc in ECC_GRID:
        for tau in TAU_GRID:
            design = _ti_design_matrix(arrays["t_mjd"], arrays["psi"], arrays["parallax_factor_al"],
                                       f_per_day=1.0 / PERIOD_DAYS, ecc=ecc, tau=tau,
                                       epoch_ref_mjd=epoch_ref_mjd)
            solution = linear_solve_ti(arrays["d_obs"], arrays["sigma"], design)
            nodes.append((ecc, tau, solution.chi2, solution.logL_marginal))
    return nodes


def test_the_case_has_no_observation_near_periastron_and_the_scores_disagree():
    arrays, epoch_ref_mjd = _source_without_periastron()
    nodes = _scores_at_true_period(arrays, epoch_ref_mjd)
    smallest_chi2 = min(nodes, key=lambda node: node[2])          # node[2]: chi2
    largest_logl = max(nodes, key=lambda node: node[3])           # node[3]: logL_marginal
    assert (smallest_chi2[0], smallest_chi2[1]) == (TRUE_ECC, TRUE_TAU)
    assert largest_logl[0] == 0.95
    assert largest_logl[2] > smallest_chi2[2]   # the e = 0.95 node fits worse


def test_the_scan_returns_the_node_of_smallest_chi2():
    arrays, epoch_ref_mjd = _source_without_periastron()
    # a narrow band whose first frequency is the true one
    peaks = scan_ti_frequency(*arrays.values(), f_min_per_day=1.0 / PERIOD_DAYS,
                              f_max_per_day=1.02 / PERIOD_DAYS, ecc_grid=ECC_GRID,
                              tau_grid=TAU_GRID, epoch_ref_mjd=epoch_ref_mjd, top_k=1)
    nodes = _scores_at_true_period(arrays, epoch_ref_mjd)
    smallest_chi2 = min(nodes, key=lambda node: node[2])
    peak = peaks[0]
    assert peak.P_days == pytest.approx(PERIOD_DAYS, rel=1e-12)
    assert (peak.ecc, peak.tau) == (TRUE_ECC, TRUE_TAU)
    assert peak.chi2 == smallest_chi2[2]
    # logL_marginal is still reported, and it is not the largest of the frequency
    largest_logl = max(node[3] for node in nodes)
    assert peak.logL_marginal < largest_logl


def test_the_peaks_come_smallest_chi2_first():
    arrays, epoch_ref_mjd = _source_without_periastron()
    peaks = scan_ti_frequency(*arrays.values(), f_min_per_day=1.0 / 1000.0,
                              f_max_per_day=1.0 / 100.0, ecc_grid=(0.0, 0.6, 0.95),
                              tau_grid=(0.0, 0.2, 0.4, 0.6, 0.8), epoch_ref_mjd=epoch_ref_mjd,
                              top_k=5)
    chi2_of_peaks = [peak.chi2 for peak in peaks]
    assert len(peaks) == 5
    assert chi2_of_peaks == sorted(chi2_of_peaks)
