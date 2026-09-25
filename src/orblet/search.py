"""Linearised Thiele-Innes astrometric pre-search — frequency scan.

A stage of the staged-convergence route.  This module wraps the
linear core (:mod:`orblet.solve.astrometry`) in a coarse frequency scan
+ a per-peak prune, plus a standalone presearch entry-point that consumes
a loader-shaped ``astro_data`` dict.  The goal is a cheap RANKING of
candidate periods before any MCMC — a global mode-finder, not a fit.

Nothing is logged, printed or persisted
---------------------------------------
The returned peaks are derived from the caller's data: a ranking of
candidate periods, with the photocentre axis each one implies.  This
module hands them back and keeps nothing: no ``print``, no
``logging``, no file write anywhere.  Every error / flag string is a
STATIC, value-free module-level constant — no input value, source
identifier, peak index, or rank ever appears in a message, so nothing
about the data leaks through an exception or a log line either.
Storing or sharing the results is the caller's decision;
nothing here assumes the data are public, or that they are
not.

What ``logL_marginal`` means
----------------------------
``logL_marginal`` is the flat-prior marginal-likelihood RANKING
score: it ranks ``(frequency, e, τ)`` grid points relative to one another
but is NOT an absolute Bayesian evidence.  It is MARGINAL over the nine
linear amplitudes only; over the shape axes ``(e, τ)`` the scan takes
the MAXIMUM per frequency (a PROFILE, not a marginal).  A profile does
not pay for shape freedom, so on average an eccentric node scores at
least as well as a circular one at the same frequency: the reported
``ecc`` of a peak is biased towards the high end of ``ecc_grid`` and
must not be read as a measurement of e (honesty note).
The scan has NO jitter term: the ranking is conditional on the supplied
per-epoch ``sigma`` exactly as given, and no false-alarm probability is
computed anywhere in this module.

Observables only
----------------
A peak carries what the linear solve measured — the shape node, the
score, the nine amplitudes with their covariance, the photocentre axis
``α / ϖ`` and the fitted parallax with its σ — and NO mass.  Masses are
derived downstream with the assumptions stated
(``notebooks/quickstart/04_interpretation_mass.ipynb``).
The scan is the STARTING POINT of the seeded refiner (MAP + emcee,
``compose_ti_seed`` → ``fit_astrometry_orbit``), not a fit: a poor
starting point — e.g. a wrong-period node whose fitted parallax is
small compared with its σ — is for the refiner and the reader to catch;
the peak table shows ``plx_mas ± plx_sigma_mas`` for exactly that reason.
No automation acts on it beyond the two flags below.

1-year alias handling
---------------------
A peak whose frequency lands within a fractional window of 1/365.25
cyc/day (or a low harmonic) is DETECTED AND FLAGGED ONLY — it is not
solved away.  Parallax leakage and the Gaia scanning law make the 1-yr
region intrinsically degenerate with the parallax column; a future
mitigation is a photometric-distance / external-parallax likelihood
constraint that breaks the degeneracy.  Scope of the flag: only the
1-yr line and its 6-month harmonic are
checked.  The ±1/yr sidebands of a CANDIDATE frequency, the 2-yr
subharmonic, and the spectral window of the scan law are NOT flagged;
a user must check those by eye or with a scrambled-time null.

No echoed values
----------------
No θ / amplitude / input value is logged, printed, or persisted here.

Lazy-import discipline
----------------------
``scipy`` is imported INSIDE the function bodies that need it (the
linear solve uses ``scipy.linalg``; :func:`reduced_chi2_band` uses
``scipy.stats.chi2``).  Module scope imports only numpy, orblet and the standard library.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR, MJD_J2010_TCB
from orblet.prepare import _assert_obs_time_is_j2010_days
from orblet.solve.astrometry import (
    _SINGULAR_DESIGN_MSG,
    _ti_design_matrix,
    linear_solve_ti,
)


# ── Static, value-free message / flag constants ──────────────────────
# Every flag string below is a module-level CONSTANT.  They name only the
# REASON class, never a peak index, rank, or numeric value (data-safety).
_PRUNE_MSG_TEMPLATE = (
    "linear-TI presearch peak pruned: the photocenter semi-major axis is "
    "non-positive or degenerate (a non-positive fitted parallax or a "
    "vanishing amplitude); the geometry has collapsed for this peak."
)
_ALIAS_REGION_MSG = (
    "linear-TI presearch peak lies in the one-year alias region "
    "(near the parallax period or a low harmonic); flagged for review, "
    "not solved away."
)
_SCAN_REQUIRES_FLAT_PRIOR_MSG = (
    "scan_ti_frequency requires a flat (None) beta_prior: the "
    "marginal-likelihood ranking is only comparable across frequencies "
    "for a frequency-independent prior."
)

# One-year (parallax) period in cycles/day, plus the fractional window
# and the low harmonics that share the alias.  DETECT-AND-FLAG only.
_F_ONE_YEAR_PER_DAY = 1.0 / 365.25
_ALIAS_FRAC_WINDOW = 0.02  # ±2% in frequency around 1/365.25 and harmonics
_ALIAS_HARMONICS = (1, 2)  # 1-yr and the 6-month harmonic

# Near-singular guard: a coarse node whose normal-matrix minimum Cholesky
# diagonal falls below this tolerance is SKIPPED (treated like a singular
# design) BEFORE the alias frequencies are swept.  A small, documented
# floor — not a tunable knob; the scan is a ranking pre-pass.
_NEAR_SINGULAR_CHOL_TOL = 1e-8

@dataclass(frozen=True)
class TiScanPeak:
    """One ranked candidate from the linearised TI frequency scan.

    A per-peak ranking plus the derived photocentre axis, computed from
    the caller's data; nothing here logs or stores it (see the module
    docstring).

    Attributes
    ----------
    f_per_day : float
        Orbital frequency at the peak (cycles/day).
    P_days : float
        Period ``1 / f_per_day`` (days).
    ecc : float
        Eccentricity of the grid node (dimensionless, ``0 ≤ e < 1``).
    tau : float
        Periastron-phase fraction on the unit disk (``τ ∈ [0, 1)``).
    logL_marginal : float
        Flat-prior marginal-likelihood RANKING score (NOT an absolute
        evidence; see module docstring).
    chi2 : float
        Weighted residual sum at the peak (dimensionless).
    dof : int
        Degrees of freedom ``n − 9``.
    a_phot_au : float or None
        Photocenter semi-major axis (AU) via ``α / plx``; ``None`` when
        the geometry is degenerate (``a_phot ≤ 0``) and the peak is
        flagged.
    plx_mas : float
        The fitted parallax amplitude (mas) at this node — ``beta[8]``,
        a flat-prior linear estimate; at a wrong period it can be far
        from the catalogue value, small, or negative.
    plx_sigma_mas : float or None
        Its formal 1σ uncertainty (mas), ``sqrt(cov[8, 8])`` — the
        uncertainty CONDITIONAL on this node's fixed ``(f, e, τ)`` and
        on the supplied per-epoch σ with no jitter, so a LOWER bound on
        the real parallax uncertainty, not a posterior width.  ``None``
        only when no covariance is attached (hand-built peaks); ``NaN``
        if the covariance diagonal is not positive (a non-PD solve),
        never a misleading 0.  Read ``plx_mas ± plx_sigma_mas`` next to
        a suspicious period before seeding from it (no automation acts
        on it).
    beta : np.ndarray
        Best-fit linear amplitudes, shape ``(9,)``, column order
        ``[A, B, F, G, ra_offset, dec_offset, pmra, pmdec, plx]``.
    cov : np.ndarray or None
        Posterior covariance of ``beta`` (shape ``(9, 9)``); attached
        only to the RETURNED top-k peaks (``None`` is never produced by
        the scan, but the type allows a pruned node carried without cov).
    flagged : bool
        ``True`` when the peak is pruned/flagged for any reason (see
        ``flag_reasons``).
    flag_reasons : tuple[str, ...]
        The static, value-free reason constants that fired (a subset of
        ``{_PRUNE_MSG_TEMPLATE, _ALIAS_REGION_MSG}``).
    """

    f_per_day: float
    P_days: float
    ecc: float
    tau: float
    logL_marginal: float
    chi2: float
    dof: int
    a_phot_au: float | None
    plx_mas: float
    plx_sigma_mas: float | None
    beta: np.ndarray
    cov: np.ndarray | None
    flagged: bool
    flag_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        """Return a shallow dict of the peak fields (caller persistence
        must go to the configured products dir; see module docstring)."""
        return {
            "f_per_day": self.f_per_day,
            "P_days": self.P_days,
            "ecc": self.ecc,
            "tau": self.tau,
            "logL_marginal": self.logL_marginal,
            "chi2": self.chi2,
            "dof": self.dof,
            "a_phot_au": self.a_phot_au,
            "plx_mas": self.plx_mas,
            "plx_sigma_mas": self.plx_sigma_mas,
            "beta": self.beta,
            "cov": self.cov,
            "flagged": self.flagged,
            "flag_reasons": self.flag_reasons,
        }


# ── Seed composer (the quick-look → full-sampler bridge) ──────────────


def compose_ti_seed(
    peak: TiScanPeak,
    *,
    epoch_ref_mjd: float,
    jitter_mas: float | None = None,
) -> dict:
    """Compose a physics-space warm-start seed for the full TI sampler.

    The bridge of the staged-convergence design: the linearised
    quick-look FINDS the solution, ``fit_astrometry_orbit`` REFINES it
    (the full sampler cannot do the finding from a cold start).  This
    function only maps a scan peak into the readable dict
    ``fit_astrometry_orbit`` accepts via ``warmstart_theta=``.

    OPEN BY CONSTRUCTION: the return value is a plain dict of named,
    unit-suffixed physics quantities.  Inspect it, edit any entry, or
    build the whole dict by hand without a ``TiScanPeak`` — the keys are
    exactly the astrometric θ-dict schema
    (``P_yr``, ``e``, ``A_mas``, ``B_mas``, ``F_mas``,
    ``G_mas``, ``plx_mas``, ``ra_offset_mas``, ``dec_offset_mas``,
    ``pmra_masyr``, ``pmdec_masyr``, ``tp_mjd``; ``astro_jitter_mas``
    optional), plus ``epoch_ref_mjd``.

    Contract invariants:

    - **A seed touches the sampler's starting point and NOTHING else** —
      never a prior centre, width, bound or fixed value.  Initialisation
      is not a prior; the posterior is unchanged by where the chains
      start (only by where they manage to go).
    - **The seed carries its epoch.** ``epoch_ref_mjd`` sets both the
      periastron-time zero (τ → ``tp_mjd`` below) and the offset/PM
      zero-point; the sampler RAISES on a mismatch with the fit's own
      reference.
    - **Amplitude source.** ``peak.beta`` holds the scan's own
      ``(A, B, F, G)`` and is the canonical source for a scan-seeded
      fit.  (A fit's EXPORTED ``A_mas…G_mas`` are equally usable: the
      export's Ω fold shifts the ANGLES by ``(ω+π, Ω+π)``, under which
      the amplitudes are exactly invariant, so the exported amplitudes
      are byte-identical to the sampled ones — pinned by
      ``tests/test_ti_export_sign_consistency.py``.)
    - ``jitter_mas`` is the one slot with NO quick-look source (the scan
      holds jitter fixed).  Left ``None``, each chain draws its start
      from the jitter prior — legitimate dispersion in exactly the
      direction the seed does not constrain.

    Parameters
    ----------
    peak : TiScanPeak
        A ranked peak from :func:`scan_ti_frequency`.  The seed built
        from it is returned, never logged; the sampler records only a
        provenance flag.
    epoch_ref_mjd : float
        Reference epoch (MJD, TCB) the seed's τ and offsets/PM refer
        to.  Must equal the ``epoch_ref_mjd`` passed to the fit.
    jitter_mas : float or None, optional
        Explicit jitter start (mas, > 0).  ``None`` (default) → the
        sampler draws per chain from the jitter prior.

    Returns
    -------
    dict
        The physics-space seed described above.
    """
    if jitter_mas is not None and not (float(jitter_mas) > 0.0):
        raise ValueError("jitter_mas must be positive when given")
    if peak.flagged:
        # A flagged (pruned) peak is a legitimate user CHOICE, but not a
        # silent one.  Value-free message (no numbers, no identifiers).
        warnings.warn(
            "compose_ti_seed: the supplied peak is flagged by the scan; "
            "seeding from it is allowed but deliberate — check "
            "flag_reasons.",
            UserWarning,
            stacklevel=2,
        )
    # beta column order (see TiScanPeak.beta):
    # [A, B, F, G, ra_offset, dec_offset, pmra, pmdec, plx]
    beta = np.asarray(peak.beta, dtype=float)
    seed = {
        "P_yr": float(peak.P_days) / DAYS_PER_KEPLER_YEAR,
        "e": float(peak.ecc),
        "A_mas": float(beta[0]),
        "B_mas": float(beta[1]),
        "F_mas": float(beta[2]),
        "G_mas": float(beta[3]),
        "ra_offset_mas": float(beta[4]),
        "dec_offset_mas": float(beta[5]),
        "pmra_masyr": float(beta[6]),
        "pmdec_masyr": float(beta[7]),
        "plx_mas": float(beta[8]),
        # τ (periastron-phase fraction at epoch_ref) → periastron time.
        "tp_mjd": float(epoch_ref_mjd) + float(peak.tau) * float(peak.P_days),
        "epoch_ref_mjd": float(epoch_ref_mjd),
    }
    if jitter_mas is not None:
        seed["astro_jitter_mas"] = float(jitter_mas)
    return seed


# ── Derived-quantity helpers (reuse the chains.py closure) ────────────


def _alpha_mas_from_beta(beta: np.ndarray) -> float:
    """Photocenter amplitude ``α`` (mas) from the TI amplitudes (A,B,F,G).

    Mirrors the ``ti_to_kepler`` inversion in
    :mod:`orblet.elements`:
    ``u = (A²+B²+F²+G²)/2``, ``v = AG − BF``,
    ``α = √(u + √(u² − v²))`` with the discriminant clipped at 0.
    """
    A, B, F, G = float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])
    u = (A * A + B * B + F * F + G * G) / 2.0
    v = A * G - B * F
    disc = max((u + v) * (u - v), 0.0)
    return float(np.sqrt(u + np.sqrt(disc)))


def _a_phot_au_from_alpha(alpha_mas: float, plx_mas: float) -> float:
    """Photocenter semi-major axis (AU) = ``α / plx`` (the ti_to_kepler
    inversion).  ``plx_mas`` must be positive."""
    return float(alpha_mas) / float(plx_mas)


def _is_one_year_alias(f_per_day: float) -> bool:
    """True when ``f_per_day`` is within the fractional window of the
    1-yr parallax frequency or a low harmonic (DETECT-AND-FLAG only)."""
    f = float(f_per_day)
    for k in _ALIAS_HARMONICS:
        f_alias = k * _F_ONE_YEAR_PER_DAY
        if abs(f - f_alias) / f_alias < _ALIAS_FRAC_WINDOW:
            return True
    return False


def _prune_peak_flags(
    *,
    alpha_mas: float,
    plx_mas: float,
    f_per_day: float,
):
    """Screen one peak: derive ``a_phot`` and the two flags.

    Two flags only: the 1-yr alias window, and the
    ``a_phot > 0`` geometry screen — a vanishing fitted amplitude (α = 0,
    no photocentre signal) or a NON-POSITIVE fitted parallax drives
    ``a_phot`` non-positive or non-finite (this is NOT an ``i → 0``
    effect — the TI inversion gives α = a_phot × plx at every
    inclination).  A small-but-positive parallax is NOT flagged: the
    peak carries ``plx_mas ± plx_sigma_mas`` so the reader can see it.

    Returns ``(flagged, flag_reasons, a_phot_au)``; ``a_phot_au`` is
    ``None`` when the geometry screen fires.  All reason strings are
    module-level value-free constants.
    """
    reasons: list[str] = []

    # 1-yr alias: DETECT-AND-FLAG only.
    if _is_one_year_alias(f_per_day):
        reasons.append(_ALIAS_REGION_MSG)

    # Geometry screen: the fitted parallax must be strictly positive and
    # finite (a zero parallax would divide by zero), and so must a_phot.
    if not (float(plx_mas) > 0.0) or not np.isfinite(plx_mas):
        reasons.append(_PRUNE_MSG_TEMPLATE)
        return True, tuple(reasons), None
    a_phot_au = _a_phot_au_from_alpha(alpha_mas, plx_mas)
    if not (a_phot_au > 0.0) or not np.isfinite(a_phot_au):
        reasons.append(_PRUNE_MSG_TEMPLATE)
        return True, tuple(reasons), None

    flagged = len(reasons) > 0
    return flagged, tuple(reasons), float(a_phot_au)


def reduced_chi2_band(dof: int) -> tuple[float, float]:
    """Central-99% reduced-χ² band ``(low, high)`` for ``dof`` d.o.f.

    Returns ``(χ²_0.005 / dof, χ²_0.995 / dof)`` from the central
    chi-square distribution (``scipy.stats.chi2.ppf``).

    This is a measurement-error-only, RELATIVE diagnostic: the band
    assumes the reported per-epoch σ already captures all noise (no
    astrophysical jitter or error inflation), so a reduced χ² outside the
    band signals only a relative mismatch between the model fit and the
    quoted errors — NOT an absolute goodness-of-fit verdict.
    """
    # Lazy scipy import (astro lazy-import contract).
    from scipy.stats import chi2 as _chi2

    n_dof = int(dof)
    lo = float(_chi2.ppf(0.005, n_dof)) / n_dof
    hi = float(_chi2.ppf(0.995, n_dof)) / n_dof
    return lo, hi


def _min_chol_diagonal(M: np.ndarray) -> float:
    """Minimum diagonal of the lower-Cholesky factor of ``M`` (the
    near-singular guard scale), or ``-inf`` when ``M`` is not positive
    definite (so the node is treated as singular and skipped)."""
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return float("-inf")
    return float(np.min(np.diag(L)))


def scan_ti_frequency(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    d_obs: np.ndarray,
    sigma: np.ndarray,
    *,
    f_min_per_day: float,
    f_max_per_day: float,
    oversample: float = 4.0,
    ecc_grid,
    tau_grid,
    epoch_ref_mjd: float,
    top_k: int,
    beta_prior=None,
    return_report: bool = False,
):
    """Coarse 3-D ``(frequency, e, τ)`` linear-TI scan → ranked peaks.

    Returns derived quantities of the caller's data; persists nothing.

    The frequency grid is uniform in FREQUENCY with spacing
    ``Δf = 1 / (oversample · T)``, ``T = t_mjd.max() − t_mjd.min()``.  At
    every ``(f, e, τ)`` node the design matrix is built and the
    GLS solve is run; a node whose design is singular (the
    ``_SINGULAR_DESIGN_MSG`` ValueError) OR near-singular (minimum
    Cholesky diagonal below ``_NEAR_SINGULAR_CHOL_TOL``) is SKIPPED — the
    scan never aborts.  The near-singular guard runs BEFORE the alias
    frequencies are swept.  Nodes are ranked by ``logL_marginal``; the
    top-``top_k`` LOCAL maxima in frequency are returned, with covariance
    attached ONLY to those returned peaks (discarded nodes' covariances
    are dropped — no ``(n_nodes, 9, 9)`` array is ever materialised).

    Parameters
    ----------
    t_mjd, psi, parallax_factor_al : np.ndarray
        Scan geometry, shape ``(n,)`` each (see ``_ti_design_matrix``).
    d_obs, sigma : np.ndarray
        Along-scan observations and per-epoch σ (mas), shape ``(n,)``.
    f_min_per_day, f_max_per_day : float
        Inclusive frequency-grid bounds (cycles/day).
    oversample : float, default 4.0
        Frequency oversampling factor (sets ``Δf``).
    ecc_grid, tau_grid : sequence of float
        Coarse eccentricity and τ-on-the-disk grids.
    epoch_ref_mjd : float
        Reference epoch (MJD), same time scale as ``t_mjd`` (see
        ``_ti_design_matrix``).
    top_k : int
        Number of top-ranked local-maximum peaks to return.
    beta_prior : None
        MUST be ``None``; a non-None prior raises ``ValueError`` with the
        value-free ``_SCAN_REQUIRES_FLAT_PRIOR_MSG`` (cross-frequency
        comparability requires a frequency-independent prior).
    return_report : bool, default False
        When ``True`` return a dict ``{"peaks", "n_flagged"}`` instead of
        the bare peak list.

    Returns
    -------
    list[TiScanPeak]  (or dict when ``return_report`` is True)
    """
    if beta_prior is not None:
        # Cross-frequency comparability only holds for a flat prior.
        raise ValueError(_SCAN_REQUIRES_FLAT_PRIOR_MSG)

    t = np.asarray(t_mjd, dtype=float)
    d = np.asarray(d_obs, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    pf = np.asarray(parallax_factor_al, dtype=float)

    span = float(t.max() - t.min())
    delta_f = 1.0 / (float(oversample) * span)
    n_freq = int(np.floor((f_max_per_day - f_min_per_day) / delta_f)) + 1
    freqs = f_min_per_day + delta_f * np.arange(max(n_freq, 1))

    # Per-frequency best node (over the e × τ sub-grid), holding only the
    # ranking score + the LIGHT fields (no cov yet — cov is attached after
    # the top-k prune so no (n_nodes, 9, 9) array is materialised).
    per_freq: list[dict] = []
    for f in freqs:
        best = None
        for ecc in ecc_grid:
            for tau in tau_grid:
                X = _ti_design_matrix(
                    t, psi_arr, pf,
                    f_per_day=float(f), ecc=float(ecc), tau=float(tau),
                    epoch_ref_mjd=epoch_ref_mjd,
                )
                # Near-singular guard BEFORE the solve (cheap normal matrix
                # Cholesky-diagonal check); skip the node if it is below tol.
                inv_var = 1.0 / sig ** 2
                XtW = X.T * inv_var
                M = XtW @ X
                if _min_chol_diagonal(M) < _NEAR_SINGULAR_CHOL_TOL:
                    continue
                try:
                    sol = linear_solve_ti(d, sig, X, beta_prior=None)
                except ValueError as exc:
                    # Per-node skip on the singular-design error;
                    # the scan continues (no abort).
                    if str(exc) == _SINGULAR_DESIGN_MSG:
                        continue
                    raise
                if best is None or sol.logL_marginal > best["logL_marginal"]:
                    best = {
                        "f_per_day": float(f),
                        "ecc": float(ecc),
                        "tau": float(tau),
                        "logL_marginal": float(sol.logL_marginal),
                        "chi2": float(sol.chi2),
                        "dof": int(sol.dof),
                        "beta": np.asarray(sol.beta, dtype=float),
                        "cov": np.asarray(sol.cov, dtype=float),
                    }
        if best is not None:
            per_freq.append(best)

    if not per_freq:
        return {"peaks": [], "n_flagged": 0} if return_report else []

    # Local maxima in frequency (a node whose score exceeds both
    # frequency neighbours; endpoints compare to their single neighbour).
    scores = np.array([b["logL_marginal"] for b in per_freq], dtype=float)
    local_max_idx = []
    for i in range(len(per_freq)):
        left = scores[i - 1] if i > 0 else -np.inf
        right = scores[i + 1] if i < len(per_freq) - 1 else -np.inf
        if scores[i] >= left and scores[i] >= right:
            local_max_idx.append(i)
    if not local_max_idx:
        local_max_idx = list(range(len(per_freq)))

    # Rank the local maxima by score, take the top_k.
    local_max_idx.sort(key=lambda i: per_freq[i]["logL_marginal"], reverse=True)
    chosen = local_max_idx[: int(top_k)]

    peaks: list[TiScanPeak] = []
    for i in chosen:
        b = per_freq[i]
        alpha_mas = _alpha_mas_from_beta(b["beta"])
        plx_mas = float(b["beta"][8])
        # Formal 1σ on the parallax amplitude from the solve's covariance
        # (the (8, 8) entry is the parallax column, see _BETA_COLUMNS);
        # conditional on this node, no jitter — a lower bound.  A
        # non-positive diagonal (non-PD solve) gives NaN, never 0.
        plx_var = float(b["cov"][8, 8])
        plx_sigma_mas = float(np.sqrt(plx_var)) if plx_var > 0.0 else float("nan")
        flagged, reasons, a_phot_au = _prune_peak_flags(
            alpha_mas=alpha_mas, plx_mas=plx_mas, f_per_day=b["f_per_day"],
        )
        peaks.append(
            TiScanPeak(
                f_per_day=b["f_per_day"],
                P_days=1.0 / b["f_per_day"],
                ecc=b["ecc"],
                tau=b["tau"],
                logL_marginal=b["logL_marginal"],
                chi2=b["chi2"],
                dof=b["dof"],
                a_phot_au=a_phot_au,
                plx_mas=plx_mas,
                plx_sigma_mas=plx_sigma_mas,
                beta=b["beta"],
                cov=b["cov"],  # attached ONLY to the returned top-k peaks
                flagged=flagged,
                flag_reasons=reasons,
            )
        )

    if return_report:
        n_flagged = sum(1 for pk in peaks if pk.flagged)
        return {"peaks": peaks, "n_flagged": n_flagged}
    return peaks


def presearch_astrometry_linear_ti(
    astro_data,
    *,
    ra_deg: float,
    dec_deg: float,
    f_min_per_day: float,
    f_max_per_day: float,
    oversample: float = 4.0,
    ecc_grid,
    tau_grid,
    abfg_sigma_mas: float,
    system_priors,
    epoch_ref_mjd: float,
    top_k: int,
) -> dict:
    """Standalone linearised-TI presearch over a loader-shaped dict.

    Returns the peaks and their photocentre axes, derived from the
    caller's data.  This function PERSISTS NOTHING — no file write, no
    ``print``, no ``logging``; storing or sharing the result is the
    caller's decision, and nothing here assumes either way (see the
    module docstring).

    It reads the synthetic / loader-shaped ``astro_data`` dict
    (``obs_time`` in J2010 days → MJD via ``MJD_J2010_TCB``,
    ``scan_angle``, ``parallax_factor_al``, ``centroid_pos``,
    ``centroid_pos_err``), runs :func:`scan_ti_frequency`, and returns the
    ranked peaks plus the top-ranked NON-flagged peak.

    No mass is derived here.  Derive masses downstream from
    the observables (``a_phot``, ``P``, ``fm_ast``) with the assumptions
    stated.  ``abfg_sigma_mas``, ``system_priors``, ``ra_deg``/``dec_deg``
    are accepted to mirror the engine's presearch call convention and to
    keep the entry shape stable for MCMC seeding; the linear scan itself
    uses a flat β prior.

    Parameters
    ----------
    astro_data : mapping
        Loader-shaped epoch-astrometry dict (synthetic or real; this
        function never inspects identifiers).
    ra_deg, dec_deg : float
        Sky position (deg); accepted for call-shape parity (the linear
        scan works in the along-scan frame and does not consume them).
    f_min_per_day, f_max_per_day, oversample, ecc_grid, tau_grid, top_k :
        Forwarded to :func:`scan_ti_frequency`.
    abfg_sigma_mas : float
        TI-amplitude prior scale (mas); accepted for call-shape parity.
    system_priors : mapping
        System priors dict (accepted for call-shape parity).
    epoch_ref_mjd : float
        Reference epoch (MJD), same time scale as the fit.

    Returns
    -------
    dict
        ``{"peaks": list[TiScanPeak], "best": TiScanPeak | None}`` —
        ``best`` is the top-ranked NON-flagged peak, or ``None`` when
        every returned peak is flagged.
    """
    obs_time = np.asarray(astro_data["obs_time"], dtype=float)
    # Formalise the already-true invariant: this site adds MJD_J2010_TCB
    # unconditionally, so obs_time MUST be in J2010-days (loader convention).
    # A violation (already-MJD epochs) is an upstream contract bug to surface.
    _assert_obs_time_is_j2010_days(obs_time)
    t_mjd = obs_time + MJD_J2010_TCB
    psi = np.asarray(astro_data["scan_angle"], dtype=float)
    pf = np.asarray(astro_data["parallax_factor_al"], dtype=float)
    d_obs = np.asarray(astro_data["centroid_pos"], dtype=float)
    sigma = np.asarray(astro_data["centroid_pos_err"], dtype=float)

    peaks = scan_ti_frequency(
        t_mjd, psi, pf, d_obs, sigma,
        f_min_per_day=f_min_per_day, f_max_per_day=f_max_per_day,
        oversample=oversample, ecc_grid=ecc_grid, tau_grid=tau_grid,
        epoch_ref_mjd=epoch_ref_mjd, top_k=top_k, beta_prior=None,
    )

    best = None
    for pk in peaks:
        if not pk.flagged:
            best = pk
            break

    return {"peaks": peaks, "best": best}
