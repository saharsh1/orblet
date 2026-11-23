"""False-alarm probability of the best peak of a PDC periodogram over the
whole trial-period grid.

:func:`orblet.periodogram.pdc_false_alarm_probability` gives the probability
at ONE trial period.  A search reports the best of many periods, and the best
of many is large by chance more often than one.  This module gives two
grid-wide forms:

    pdc_grid_false_alarm_probability       fast: the pointwise value raised
                                           to an effective number of
                                           independent trial periods
    pdc_scramble_false_alarm_probability   calibrated: a permutation null on
                                           the same epochs and the same grid

Symbols
-------
=============  ==========================================================
p              pointwise false-alarm probability of the best score
N_eff          effective number of independent trial periods,
               N_eff = T · (f_max − f_min)
T              time span of the data, days
f = 1/P        trial frequency, cycles/day; P the trial period, days
n              number of data rows (epochs, or per-CCD lines)
G              number of groups (transits); every row of a group shares
               one time, the mean time of the group
s_a            number of rows in time slot a (the group moved to slot a)
A              U-centred observation distance matrix (n × n)
Z              U-centred nuisance distance matrix (n × n)
E              observation matrix with the nuisance projected out,
               E = A − (⟨A,Z⟩/⟨Z,Z⟩) Z; E = A for the ordinary PDC
W              raw phase distance matrix of the G slot times at period P,
               W_ab = φ_ab (P − φ_ab), φ_ab = (T_a − T_b) mod P; W_aa = 0
⟨X,Y⟩          U-centred inner product, Σ_{i≠j} X_ij Y_ij / (n (n − 3))
K              number of scrambled copies
=============  ==========================================================

The scramble null
-----------------
A copy moves every group of rows, with its data AND its nuisance coordinate
(for Gaia, the scan angle), to the time of another group; the set of times
stays the same.  So a copy keeps the joint distribution of the data and the
scan angle and breaks only the link between the data and time.  Under the
null hypothesis "the data do not depend on time" the observed best score is
one more draw of the same distribution as the copies' best scores.

Why it is fast.  U-centring is an orthogonal projection for ⟨·,·⟩, so for
a U-centred E, ⟨E, Ũ(B)⟩ = ⟨E, B⟩ with the RAW phase matrix B.  When the
rows of a group share one time, B is a group-by-group expansion of W, and

    ⟨E, B⟩     = Σ_{a≠b} F_ab W_ab / (n (n − 3)),
    F_ab       = Σ_{i in group at a, j in group at b} E_ij,

    ⟨Ũ(B), Ũ(B)⟩ = [ Σ_ab s_a s_b C_ab² − Σ_a s_a C_aa² ] / (n (n − 3)),
    C_ab       = W_ab − u_a − u_b + v,
    u_a        = (W s)_a / (n − 2),   v = sᵀ W s / ((n − 1)(n − 2)).

Every sum runs over G × G slots, not n × n rows, and the numerators of all
copies at a batch of periods are one matrix product.
"""

from __future__ import annotations

import numpy as np

from orblet.periodogram._common import _as_1d_float
from orblet.periodogram.dcor import _u_center, _u_inner
from orblet.periodogram.distances import _circular_distance
from orblet.periodogram.pdc import pdc_false_alarm_probability

# Memory budget of one work array in the scramble, bytes.  The copies and
# the periods are processed in blocks so that no block array is larger.
_BLOCK_BYTES = 64_000_000

# Bytes per float64 number.
_BYTES_PER_FLOAT = 8


def pdc_grid_false_alarm_probability(score, n: int, periods_days, span_days: float) -> float:
    """
    Fast false-alarm probability of the best PDC score over a period grid.

    The pointwise probability p of the best score (the χ² approximation of
    :func:`pdc_false_alarm_probability`) is corrected for the number of
    trial periods searched:

        FAP_grid = 1 − (1 − p)^N_eff,    N_eff = T · (f_max − f_min),

    with f = 1/P over the finite, positive periods of the grid.  N_eff is
    the number of frequency resolution elements 1/T in the searched band
    (the rule of Horne & Baliunas 1986 for a grid at least as dense as 1/T).
    It is clipped to [1, number of grid periods].

    Inputs
    ------
    score : float
        Best PDC score of the periodogram (``result["best_score"]`` of
        :func:`compute_pdc_periodogram`).  NaN gives NaN.
    n : int
        Number of data rows the score was computed from (n ≥ 4).
    periods_days : 1-D array
        The trial-period grid that was searched, days.
    span_days : float
        Time span of the data, T = t_max − t_min, days (> 0).

    Returns
    -------
    fap : float
        Grid-wide false-alarm probability, in [0, 1].

    Assumptions
    -----------
    - The scores at periods more than one resolution element apart are
      independent.  Harmonics and sub-harmonics of a period are not; they
      make N_eff an overestimate and the result conservative.
    - The pointwise χ² form holds.  It is derived for the ordinary distance
      correlation of independent rows.  On white-noise RVs (ordinary PDC,
      scalar distance, n from 28 to 500) the result follows the scramble
      value and errs on the conservative side.
    - The rows are independent: ``n`` must count independent measurements.
      Per-CCD astrometric lines of one transit share a time and can share
      an error; with a shared error the χ² form is far too optimistic,
      because n counts lines, not transits.  Merge the lines to one value
      per transit first (then the result is valid and conservative), or
      use :func:`pdc_scramble_false_alarm_probability` with the transits
      as ``groups``.

    Notes
    -----
    The correction is the Šidák form of the multiple-testing correction
    (Bonferroni's N_eff · p is its first-order term).  It is the same rule
    as the ``"naive"`` method of astropy's
    ``LombScargle.false_alarm_probability`` (VanderPlas 2018, Sect. 7.4.2),
    there with N_eff = T · f_max.
    """
    # step 1, the inputs
    periods = _as_1d_float(periods_days, "periods_days")
    span = float(span_days)
    if not span > 0:
        raise ValueError("span_days must be positive.")
    good_periods = periods[np.isfinite(periods) & (periods > 0)]
    if good_periods.size == 0:
        raise ValueError("periods_days has no finite positive period.")

    # step 2, the pointwise probability of the best score
    p_point = pdc_false_alarm_probability(float(score), n)
    if not np.isfinite(p_point):
        return float("nan")

    # step 3, N_eff: the resolution elements 1/T in the searched frequency band
    frequencies = 1.0 / good_periods
    n_eff = span * (frequencies.max() - frequencies.min())
    n_eff = min(max(n_eff, 1.0), float(good_periods.size))

    # step 4, 1 − (1 − p)^N_eff, written with log1p and expm1 so that a
    # small p does not round to zero
    fap = -np.expm1(n_eff * np.log1p(-p_point))
    return float(fap)


def pdc_scramble_false_alarm_probability(
    t_days,
    periods_days,
    obs_dist,
    *,
    n_copies: int,
    rng,
    nuisance_dist=None,
    partial_mode: str = "none",
    groups=None,
) -> dict:
    """
    Calibrated false-alarm probability of the best PDC score over a period
    grid, by a scramble (permutation) null.

    Each copy moves every group of rows to the time of another group (a
    random permutation of the groups among the group times) and recomputes
    the PDC periodogram on the same grid.  The data, the nuisance and the
    pairing between them stay; only the link with time is broken.  The
    false-alarm probability is the permutation p-value

        FAP = (1 + #{copies with best score ≥ observed best score}) / (1 + K),

    which is never zero: its smallest value is 1/(1 + K).

    Inputs
    ------
    t_days : 1-D array
        Times of the n rows, days.
    periods_days : 1-D array
        Trial-period grid, days.  Non-finite or non-positive periods give a
        NaN score and are skipped.
    obs_dist : (n, n) array
        Observation distance matrix, as for :func:`compute_pdc_periodogram`.
    n_copies : int
        Number of scrambled copies K (≥ 1).
    rng : numpy.random.Generator or int
        The random generator of the permutations, or a seed for
        ``numpy.random.default_rng``.  It draws K permutations of the groups
        and nothing else.
    nuisance_dist : (n, n) array, optional
        Nuisance distance matrix for ``partial_mode="semi"``.  It moves with
        the data.
    partial_mode : str, default ``"none"``
        ``"none"`` (ordinary PDC) or ``"semi"`` (partial PDC).
    groups : 1-D array of labels, optional
        Rows with the same label form one group (for per-CCD Gaia data, one
        transit) and move together.  Default: every row is its own group.
        Every row of a group is placed at the MEAN time of its group, in the
        observed periodogram as in the copies.

    Returns
    -------
    result : dict
        ``"false_alarm_probability"``    — the permutation p-value, in
        (0, 1]; NaN when the observed best score is NaN.
        ``"observed_best_score"``        — best score of the data, with each
        group at its mean time.  Without ``groups`` (or with groups whose
        rows share one time) it equals ``compute_pdc_periodogram``'s
        ``best_score`` to rounding.
        ``"observed_best_period_days"``  — period of that score, days.
        ``"null_best_scores"``           — (K,) best score of each copy.
        ``"n_copies"``                   — K.
        ``"n_groups"``                   — G, the number of groups.

    Assumptions
    -----------
    The groups are exchangeable under the null: the noise of a group does
    not depend on its time.  A trend or a slow drift in the data breaks
    this; remove it before the periodogram.
    """
    # step 1, the inputs
    t = _as_1d_float(t_days, "t_days")
    periods = _as_1d_float(periods_days, "periods_days")
    obs_dist = np.asarray(obs_dist, dtype=float)
    n = t.size
    if obs_dist.shape != (n, n):
        raise ValueError("obs_dist must be (n, n), n the number of times.")
    n_copies = int(n_copies)
    if n_copies < 1:
        raise ValueError("n_copies must be at least 1.")
    if partial_mode not in ("none", "semi"):
        raise ValueError("partial_mode must be 'none' or 'semi'")
    if partial_mode == "semi" and nuisance_dist is None:
        raise ValueError("partial_mode='semi' requires nuisance_dist")
    rng = np.random.default_rng(rng)

    # step 2, the groups: labels → 0 … G−1, the size of each group and its
    # mean time (the time of its slot in the observed periodogram)
    if groups is None:
        group_index = np.arange(n)
    else:
        labels = np.asarray(groups).ravel()
        if labels.size != n:
            raise ValueError("groups must have one label per row.")
        # the second output of np.unique: the index of each row's label
        group_index = np.unique(labels, return_inverse=True)[1].ravel()
    n_groups = int(group_index.max()) + 1
    group_sizes = np.bincount(group_index, minlength=n_groups).astype(float)
    group_times = np.bincount(group_index, weights=t, minlength=n_groups) / group_sizes
    if n_groups < 2:
        raise ValueError("at least two groups are needed to scramble.")

    # step 3, E and ⟨E,E⟩: the observation matrix with the nuisance projected
    # out (semi-partial), or A itself (ordinary) — as in compute_pdc_periodogram
    A = _u_center(obs_dist)
    if partial_mode == "semi":
        nuisance_dist = np.asarray(nuisance_dist, dtype=float)
        if nuisance_dist.shape != (n, n):
            raise ValueError("nuisance_dist must be (n, n), n the number of times.")
        Z = _u_center(nuisance_dist)
        zz = _u_inner(Z, Z)
        if not zz > 0:
            return _all_nan_result(n_copies, n_groups)
        E = A - (_u_inner(A, Z) / zz) * Z
    else:
        E = A
    ee = _u_inner(E, E)

    # step 4, F: the sums of E over each pair of groups (G × G), as
    # indicatorᵀ · E · indicator; the U-centred diagonal of E is zero, so the
    # pairs i = j add nothing
    indicator = np.zeros((n, n_groups))
    indicator[np.arange(n), group_index] = 1.0
    F_groups = indicator.T @ E @ indicator

    # step 5, the permutations: copy 0 is the data (identity); copy k ≥ 1
    # puts group g at slot permutation[k, g].  In the slot frame copy k has
    # F_k[a, b] = F_groups[g_a, g_b] and slot sizes s_k[a] = size of g_a,
    # where g_a = inverse[k, a] is the group placed at slot a.
    n_total = n_copies + 1
    permutations = np.empty((n_total, n_groups), dtype=int)
    permutations[0] = np.arange(n_groups)
    for k in range(1, n_total):  # k counts the scrambled copies
        permutations[k] = rng.permutation(n_groups)
    inverse = np.argsort(permutations, axis=1)
    slot_sizes = group_sizes[inverse]  # (copies, G)
    all_sizes_equal = bool(np.all(group_sizes == group_sizes[0]))

    # the upper triangle a < b of a G × G matrix: every sum over a ≠ b is
    # twice the sum over it (F and W are symmetric)
    upper_rows, upper_cols = np.triu_indices(n_groups, k=1)
    n_pairs = upper_rows.size

    # step 6, ⟨Ũ(B), Ũ(B)⟩ at every period for every copy (the norm of the
    # centred phase matrix); with equal group sizes it is the same for all
    # copies and is computed once per period
    good = np.isfinite(periods) & (periods > 0)
    n_periods = periods.size
    norm_bb = np.full((n_total, n_periods), np.nan)
    for i in np.flatnonzero(good):  # i counts the trial periods
        W = _circular_distance(group_times, periods[i])
        if all_sizes_equal:
            norm_bb[:, i] = _phase_norm(W, slot_sizes[:1].T, n)[0]
        else:
            norm_bb[:, i] = _phase_norm(W, slot_sizes.T, n)

    # step 7, ⟨E, B⟩ at every period for every copy: blocks of copies times
    # blocks of periods, each block one matrix product of upper triangles
    numerator = np.full((n_total, n_periods), np.nan)
    block = max(1, int(_BLOCK_BYTES // (_BYTES_PER_FLOAT * max(n_pairs, 1))))
    good_index = np.flatnonzero(good)
    for copy_start in range(0, n_total, block):
        copy_stop = min(copy_start + block, n_total)
        F_block = np.empty((copy_stop - copy_start, n_pairs))
        for row, k in enumerate(range(copy_start, copy_stop)):  # k counts copies
            F_slot = F_groups[np.ix_(inverse[k], inverse[k])]
            F_block[row] = F_slot[upper_rows, upper_cols]
        for period_start in range(0, good_index.size, block):
            period_index = good_index[period_start:period_start + block]
            W_block = np.empty((period_index.size, n_pairs))
            for row, i in enumerate(period_index):  # i counts trial periods
                W = _circular_distance(group_times, periods[i])
                W_block[row] = W[upper_rows, upper_cols]
            numerator[copy_start:copy_stop, period_index] = (
                2.0 * (F_block @ W_block.T) / (n * (n - 3))
            )

    # step 8, the scores and the best score of each copy
    denominator = np.sqrt(max(ee, 0.0) * np.clip(norm_bb, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(denominator > 0, numerator / denominator, np.nan)
    best_scores = np.full(n_total, np.nan)
    best_index = np.full(n_total, -1)
    for k in range(n_total):  # k counts copy 0 (the data) and the copies
        if np.any(np.isfinite(scores[k])):
            best_index[k] = int(np.nanargmax(scores[k]))
            best_scores[k] = scores[k, best_index[k]]

    # step 9, the permutation p-value; a NaN copy never counts as exceeding
    observed = best_scores[0]
    null_best = best_scores[1:]
    if np.isfinite(observed):
        n_exceeding = int(np.sum(null_best >= observed))
        fap = (1.0 + n_exceeding) / (1.0 + n_copies)
        observed_period = float(periods[best_index[0]])
    else:
        fap = float("nan")
        observed_period = float("nan")
    return {
        "false_alarm_probability": float(fap),
        "observed_best_score": float(observed),
        "observed_best_period_days": observed_period,
        "null_best_scores": null_best,
        "n_copies": n_copies,
        "n_groups": n_groups,
    }


def _phase_norm(W: np.ndarray, sizes: np.ndarray, n: int) -> np.ndarray:
    """
    ⟨Ũ(B), Ũ(B)⟩ of the row-expanded phase matrix, from slot quantities.

    Inputs
    ------
    W : (G, G) array
        Raw phase distance of the slot times, zero diagonal.
    sizes : (G, M) array
        Column m: the number of rows s_a at each slot a, for one copy.
    n : int
        Number of rows, Σ_a s_a.

    Returns
    -------
    norm : (M,) array
        ⟨Ũ(B), Ũ(B)⟩ for each column of ``sizes``.
    """
    # step 1, the row sums r = W s, u = r/(n−2), and v = sᵀ W s /((n−1)(n−2))
    r = W @ sizes
    total = np.sum(sizes * r, axis=0)
    u = r / (n - 2)
    v = total / ((n - 1) * (n - 2))
    # step 2, Σ_ab s_a s_b C_ab² with C_ab = W_ab − u_a − u_b + v, expanded
    # so that no (G, G, M) array is formed
    q = np.sum(sizes * ((W * W) @ sizes), axis=0)
    su = np.sum(sizes * u, axis=0)
    su2 = np.sum(sizes * u * u, axis=0)
    sur = np.sum(sizes * u * r, axis=0)
    all_pairs = (
        q
        + 2.0 * n * su2
        + n * n * v * v
        - 4.0 * sur
        + 2.0 * v * total
        + 2.0 * su * su
        - 4.0 * n * v * su
    )
    # step 3, remove the n diagonal terms i = j, where C_aa = v − 2 u_a
    diagonal = np.sum(sizes * (v - 2.0 * u) ** 2, axis=0)
    return (all_pairs - diagonal) / (n * (n - 3))


def _all_nan_result(n_copies: int, n_groups: int) -> dict:
    """The result when every score is NaN (no nuisance variance to project)."""
    return {
        "false_alarm_probability": float("nan"),
        "observed_best_score": float("nan"),
        "observed_best_period_days": float("nan"),
        "null_best_scores": np.full(n_copies, np.nan),
        "n_copies": n_copies,
        "n_groups": n_groups,
    }
