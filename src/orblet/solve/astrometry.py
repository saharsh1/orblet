"""Linearised Thiele-Innes astrometric pre-search — linear-algebra core.

This module implements the load-bearing linear-algebra core of a
linearised Thiele-Innes (TI) astrometric pre-search.
At a FIXED non-linear shape ``(frequency, e, τ)`` the
Gaia along-scan model is linear in nine amplitudes, so the best-fit
amplitudes and a marginal-likelihood ranking score follow from a single
generalised-least-squares (GLS) solve.  This lets a coarse grid over the
non-linear shape rank candidate periods cheaply before any MCMC.

Conventions (matching ``...astro.forward``)
-------------------------------------------
TI photocenter-amplitude convention
    The nine linear amplitudes are
    ``β = [A, B, F, G, ra_offset, dec_offset, pmra, pmdec, plx]`` in a
    FIXED column order.  ``(A, B, F, G)`` are PHOTOCENTER Thiele-Innes
    amplitudes in mas — already ``× plx`` and ``× m_comp / M_total`` —
    exactly the convention of :func:`...astro.forward.thiele_innes_xy`.
    No further sign flip or mass-ratio scaling is applied here.

    The design columns reproduce the engine's own along-scan model:
    with ``d_ra = B·x_orb + G·y_orb`` and ``d_dec = A·x_orb + F·y_orb``
    projected onto ``model = d_ra·sinψ + d_dec·cosψ`` (see
    :func:`...astro.forward.along_scan_model`), the partial derivatives
    are

        ∂model/∂A = x_orb·cosψ      ∂model/∂B = x_orb·sinψ
        ∂model/∂F = y_orb·cosψ      ∂model/∂G = y_orb·sinψ
        ∂model/∂ra_offset = sinψ    ∂model/∂dec_offset = cosψ
        ∂model/∂pmra = sinψ·Δt_yr   ∂model/∂pmdec = cosψ·Δt_yr
        ∂model/∂plx  = parallax_factor_al

    with ``Δt_yr = (t_mjd − epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR``.

epoch_ref_mjd double-role
    ``epoch_ref_mjd`` is REQUIRED (no default) and plays TWO roles, both
    matching the engine: it is the periastron-time zero (the τ-on-the-
    disk map ``tp = τ·P_days + epoch_ref_mjd`` via the engine atom
    :func:`...astro.forward._tp_from_disk_angle`) AND the proper-motion
    zero-point (the ``Δt_yr`` above).  Changing it shifts BOTH the
    orbit-shape columns (through ``tp``) and the proper-motion columns.

Time scale
    ``t_mjd`` and ``epoch_ref_mjd`` must share the SAME time scale as the
    fit (Gaia DR4: TCB).  Mixing scales silently biases the proper-motion
    and parallax phase.

Phase axis
    The non-linear phase is the engine's τ-on-the-disk fraction
    (``τ ∈ [0, 1)``), NOT a re-derived mean anomaly.  The orbit shape is
    obtained from the engine atoms
    :func:`...astro.forward._kepler_xy_orbit` and
    :func:`...astro.forward._tp_from_disk_angle`, so the linear core and
    the full engine share one source of truth.

Marginal likelihood is a RANKING score
    ``logL_marginal`` is a flat-prior marginal-likelihood RANKING score
    for comparing candidate ``(frequency, e, τ)`` grid points — NOT an
    absolute Bayesian evidence.  It retains the ``log|M|`` Occam term
    (which depends on the non-linear shape, hence matters for ranking)
    but drops the flat prior's constant and the k-dependent
    ``(k/2)·log 2π`` (shape-independent).  One meaning on every path:
    it is computed from the DATA-ONLY normal matrix
    and least-squares amplitudes whether or not a ``beta_prior`` was
    passed; the prior affects only ``beta`` / ``cov`` (the regularised
    posterior) and ``log_evidence`` (the proper prior-weighted integral).
    Sum the SAME score across independent channels at one shape; never
    add a ``logL_marginal`` to a ``log_evidence``; never compare
    ``logL_marginal`` across models with different numbers of amplitudes.

Lazy-import discipline
    ``scipy`` is imported INSIDE the function body that needs it, never
    at module scope (the astro-package lazy-import contract).  This
    module imports only numpy, orblet and the standard library at module scope.

No echoed values
    No θ / amplitude / input value is logged, printed, or persisted in
    this module (no ``print`` / ``logging`` anywhere).  Error messages
    are value-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import (
    _kepler_xy_orbit,
    _tp_from_disk_angle,
)

# The single-star column arithmetic lives in the design package, beside
# the builder that wraps it, and is imported here at MODULE scope,
# one-way: the design package never imports from this module, so no lazy
# import is needed to keep the package importable (see
# :mod:`orblet.design.columns`, "Layering").
#
from orblet.design.columns import astrometric_columns

# The old private spelling, kept so nothing outside this file has to change.
# Same object, so ``SingleStarBlockBuilder.build IS _astrometric_columns``
# remains literally true.
_astrometric_columns = astrometric_columns


# Fixed column order of the linear amplitude vector β.
_BETA_COLUMNS = (
    "A", "B", "F", "G",
    "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
)
_N_BETA = len(_BETA_COLUMNS)

_TI_AMPLITUDE_CHAINS_SHAPE_MSG = (
    "ti_amplitude_chains expects a 2-D (n_draws, 9) beta matrix in the "
    "linear TI column layout (A, B, F, G, ra_offset, dec_offset, pmra, "
    "pmdec, plx); a 1-D vector is ambiguous — pass beta[None, :] for a "
    "single draw."
)


def ti_amplitude_chains(beta: "np.ndarray") -> dict:
    """Name the orbit columns of a linear-TI β draws matrix for ``ti_to_kepler``.

    Purpose: replace the hand-built ``{'A_mas': betas[:, 0], ...,
    'plx_mas': betas[:, 8]}`` mapping (magic column indices, copy-pasted
    across notebooks) with ONE mapping defined next to ``_BETA_COLUMNS``,
    so the chain keys and the engine column order cannot drift apart.

    Parameters
    ----------
    beta : np.ndarray, shape (n_draws, 9)
        Linear-TI amplitude draws in the ``_BETA_COLUMNS`` order — e.g.
        ``best_linear_params_ti(...).draws``.  Units: Thiele-Innes
        photocentre amplitudes A, B, F, G in mas; position offsets in
        mas; proper motion in mas/yr; parallax in mas.

    Returns
    -------
    dict of str → np.ndarray, each shape (n_draws,)
        ``{"A_mas", "B_mas", "F_mas", "G_mas", "plx_mas"}`` — exactly the
        orbit keys ``orblet.elements.ti_to_kepler`` consumes
        (amplitudes and parallax in mas).  Views into ``beta``, not
        copies.  The single-star nuisance columns (offsets, proper
        motion) are deliberately not emitted: they are not orbital
        elements.

    Assumptions / conventions: the amplitudes are PHOTOCENTRE amplitudes
    in the standard NSS axis convention; no
    mass scaling is applied here or downstream in ``ti_to_kepler``'s
    geometry keys.
    """
    arr = np.asarray(beta, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != _N_BETA:
        raise ValueError(_TI_AMPLITUDE_CHAINS_SHAPE_MSG)
    # Indices resolved FROM the column tuple, never hardcoded — this line
    # is the whole point of the function.
    return {
        "A_mas": arr[:, _BETA_COLUMNS.index("A")],
        "B_mas": arr[:, _BETA_COLUMNS.index("B")],
        "F_mas": arr[:, _BETA_COLUMNS.index("F")],
        "G_mas": arr[:, _BETA_COLUMNS.index("G")],
        # KEY is the chain-key spelling; the .index() argument is the
        # design-matrix COLUMN name and deliberately stays "plx".
        "plx_mas": arr[:, _BETA_COLUMNS.index("plx")],
    }

# Value-free error messages (no matrix entries / no input values).
_NON_GAUSSIAN_BETA_PRIOR_MSG = (
    "beta_prior must be a Gaussian prior exposing 'mean' and 'cov'; "
    "non-Gaussian beta priors are not supported in the linear TI solve."
)
_TOO_FEW_EPOCHS_MSG = (
    "linear TI solve needs at least as many epochs as linear "
    "amplitudes; the design matrix has fewer rows than columns."
)
_SINGULAR_DESIGN_MSG = (
    "linear TI normal matrix is singular or not positive definite "
    "(rank-deficient design); cannot solve at this grid point."
)


def _ti_companion_to_total_mass(
    *,
    m_comp: np.ndarray | float,
    P_yr: np.ndarray | float,
    a_phot_au: np.ndarray | float,
):
    """TI Kepler-III closure: total mass from companion mass + period +
    photocenter axis (dark-companion β = 0).

    ``M_total = m_comp · √(m_comp · P_yr² / a_phot³)`` in solar masses.

    This is the ONE source of truth for the TI ``M_total`` closure: a
    sampler's chain post-processing and the frequency scan
    (``orblet.search``) both call it.  It is Kepler III with the
    photocentre amplitude read as the primary's orbit; see the reference
    manual, §3.8 and §7 item 1, for the β = 0 caveat (the result is
    prior-driven through the sampled ``m_comp``, not a measurement).

    The ``a_phot_au ** 3`` denominator is floored at the smallest positive
    float (``np.finfo(float).tiny``) so a degenerate ``a_phot → 0``
    (all four fitted amplitudes vanishing — a no-signal solution; NOT an
    ``i → 0`` effect, since α = a_phot × plx at every inclination)
    yields a finite-but-meaningless value
    rather than a divide-by-zero; downstream consumers MUST screen on
    ``a_phot_au > 0`` first.  Works elementwise for numpy arrays and for
    Python scalars (the chain post-processor passes arrays; the presearch
    passes scalars).

    Parameters
    ----------
    m_comp : array or float
        Companion mass (M_⊙).
    P_yr : array or float
        Orbital period (Julian years).
    a_phot_au : array or float
        Photocenter semi-major axis (AU).

    Returns
    -------
    array or float
        Total system mass (M_⊙).
    """
    safe_a3 = np.maximum(a_phot_au ** 3, np.finfo(float).tiny)
    return m_comp * np.sqrt(m_comp * P_yr * P_yr / safe_a3)


# Fixed column order of the single-star astrometric block (columns 4-8 of
# the TI design matrix): [ra_offset, dec_offset, pmra, pmdec, plx].
_ASTROMETRIC_COLUMNS = (
    "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
)
_N_ASTROMETRIC = len(_ASTROMETRIC_COLUMNS)

# The TI design and the acceleration block live in the design package
# too, beside the builders that wrap them.  Both are PUBLIC names,
# re-exported here so ``orblet.solve.astrometry.{ti_design_matrix,
# acceleration_columns}`` resolve for callers that reach them here as
# well as through orblet's front door.  Their values are pinned by
# ``tests/test_design_columns_baseline.py``.
#
from orblet.design.columns import (  # noqa: E402
    acceleration_columns,
    ti_design_matrix,
)

# Private aliases for the engine-internal callers and the scan; the
# public spellings are the documented ones.
_ti_design_matrix = ti_design_matrix
_acceleration_columns = acceleration_columns


@dataclass(frozen=True)
class LinearTiSolution:
    """Result of the GLS solve at one non-linear grid point.

    All arrays are float64.  No input value / source identifier is
    echoed.

    The solve is GENERIC in the number of design columns ``n_col``:
    ``beta`` has shape ``(n_col,)``, ``cov`` ``(n_col, n_col)``, and
    ``dof = n − n_col``.  The shapes below describe the 9-column TI orbit
    fit; the SAME dataclass is reused by the 5-parameter single-star fit
    (``fit_astrometric_5param``), where ``n_col = 5`` and ``dof = n − 5``.

    Attributes
    ----------
    beta : np.ndarray
        Best-fit linear amplitudes, shape ``(9,)`` (TI orbit fit; column
        order ``[A, B, F, G, ra_offset, dec_offset, pmra, pmdec, plx]``);
        ``(5,)`` ``[ra_offset, dec_offset, pmra, pmdec, plx]`` for the
        single-star fit.
    cov : np.ndarray
        Posterior covariance of ``beta`` (``M⁻¹``), shape ``(9, 9)``
        (or ``(5, 5)`` for the single-star fit).
    logL_marginal : float or None
        Flat-prior marginal-likelihood RANKING score (see module
        docstring); NOT an absolute evidence.  It
        means the SAME thing on every path — the likelihood integrated
        over the amplitudes under a flat prior, from the data-only
        normal matrix and the least-squares amplitudes — whether or not
        a ``beta_prior`` was passed.  ``None`` only when a prior was
        passed AND the data-only normal matrix is singular (no flat
        marginal exists).  Sum it across independent channels at one
        shape; never add it to a ``log_evidence``.
    chi2 : float
        Weighted residual sum ``rᵀ Σ_d⁻¹ r`` at the RETURNED ``beta``
        (the prior-regularised one when a prior was passed).
    dof : int
        Degrees of freedom ``n − n_col`` (``n − 9`` for the TI orbit fit;
        ``n − 5`` for the single-star fit).
    log_evidence : float or None
        Proper Gaussian-linear marginal log-evidence
        ``log N(d | X μ_β, X Σ_β Xᵀ + Σ_d)`` under the INFORMATIVE β
        prior.  Populated ONLY when ``beta_prior is not None``; ``None``
        on the flat (``beta_prior=None``) path, because an improper flat
        prior has no proper evidence.  Unlike ``logL_marginal`` (a
        flat-prior RANKING score that drops the β-prior normalisation),
        ``log_evidence`` retains every constant (``n·log2π``,
        ``Σ log σ²``, ``log|Σ_β|``) and is therefore directly comparable
        to a full fit's absolute log-posterior.  Defaults to ``None`` so
        existing positional / bit-exact anchors are unaffected.
    """

    beta: np.ndarray
    cov: np.ndarray
    logL_marginal: float | None
    chi2: float
    dof: int
    log_evidence: float | None = None


def linear_solve_ti(
    d_obs: np.ndarray,
    sigma: np.ndarray,
    X: np.ndarray,
    *,
    beta_prior=None,
) -> LinearTiSolution:
    """Generalised-least-squares solve for the nine linear amplitudes.

    Solves, with ``Σ_d = diag(σ²)``,

        M  = Xᵀ Σ_d⁻¹ X      ( + Σ_β⁻¹ if a Gaussian prior is given )
        β̂ = M⁻¹ ( Xᵀ Σ_d⁻¹ d_obs  [+ Σ_β⁻¹ μ_β] )
        cov = M⁻¹

    and the flat-prior marginal-likelihood RANKING score, ALWAYS from the
    data-only ingredients ``M_d = Xᵀ Σ_d⁻¹ X`` and the least-squares
    ``β̂_d = M_d⁻¹ Xᵀ Σ_d⁻¹ d_obs`` (the prior never
    enters this number):

        logL_marginal = −0.5 · [ r_dᵀ Σ_d⁻¹ r_d + log|M_d|
                                 + n·log(2π) + Σ log σ²ᵢ ],
        r_d = d_obs − X β̂_d.

    The ``log|M_d|`` term (frequency-dependent Occam factor) is RETAINED;
    the flat prior's constant and the k-dependent ``(k/2)·log 2π`` are
    dropped (shape-independent), so ``logL_marginal`` is a ranking score
    within one model, not an absolute evidence (see module docstring).
    On the flat path ``M = M_d`` and the arithmetic is the historical
    one, byte for byte.  On the prior path the score is ``None`` when
    ``M_d`` is singular (no flat marginal exists — nine columns on few
    transits is exactly when a prior is passed).  A NEAR-singular but
    numerically positive-definite ``M_d`` yields a large score rather
    than ``None``, on both paths alike; the near-singular guard lives in
    the caller (``ti_scan._NEAR_SINGULAR_CHOL_TOL``), not here.

    Proper marginal log-evidence (informative prior only)
    -----------------------------------------------------
    When a Gaussian ``beta_prior`` is supplied, the result also carries
    the PROPER marginal log-evidence (``log_evidence``)

        log_evidence = log N( d | X μ_β , X Σ_β Xᵀ + Σ_d ),

    computed via the numerically-stable REDUCED (9×9) form that reuses
    the precision matrix ``M = Xᵀ Σ_d⁻¹ X + Σ_β⁻¹`` (and its Cholesky)
    already formed for the GLS solve (matrix-determinant lemma +
    Woodbury):

        r      = d − X μ_β
        b      = Xᵀ Σ_d⁻¹ r
        quad   = rᵀ Σ_d⁻¹ r − bᵀ M⁻¹ b
        logdet = Σᵢ log(σ²ᵢ) + log|Σ_β| + log|M|
        log_evidence = −0.5 ( quad + logdet + n·log(2π) ).

    Only ``log|M|`` and ``M⁻¹`` are reused from the GLS solve; the
    ``log|Σ_β|`` term is NOT reused from ``M`` — it needs its own cheap
    9×9 Cholesky of the prior covariance ``Σ_β`` (computed below).

    All constants are retained, so ``log_evidence`` is directly
    comparable to a full fit's absolute log-posterior.  On the flat path
    (``beta_prior=None``) it is ``None`` (an improper prior has no proper
    evidence) and the existing ``beta / cov / logL_marginal / chi2 /
    dof`` are byte-identical to the pre-evidence implementation.

    Parameters
    ----------
    d_obs : np.ndarray
        Along-scan observations (mas), shape ``(n,)``.
    sigma : np.ndarray
        Per-epoch along-scan uncertainties (mas), shape ``(n,)``.
    X : np.ndarray
        Design matrix from :func:`_ti_design_matrix`, shape ``(n, 9)``.
    beta_prior : object or None, optional
        ``None`` ⇒ flat prior (default).  A non-None prior MUST expose
        ``.mean`` (shape ``(9,)``) and ``.cov`` (shape ``(9, 9)``) for a
        Gaussian prior; anything else raises ``ValueError`` (value-free,
        ``from None``).

    Returns
    -------
    LinearTiSolution
    """
    # Lazy scipy import INSIDE the body (astro lazy-import contract).
    import scipy.linalg as sla

    d = np.asarray(d_obs, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    Xmat = np.asarray(X, dtype=float)

    n, n_col = Xmat.shape
    if n < n_col:
        # Value-free guard: not enough epochs to constrain 9 amplitudes.
        raise ValueError(_TOO_FEW_EPOCHS_MSG)

    if beta_prior is not None:
        # A Gaussian prior must expose .mean and .cov; anything else is
        # unsupported.  ``from None`` severs the value-bearing chain.
        if not (hasattr(beta_prior, "mean") and hasattr(beta_prior, "cov")):
            raise ValueError(_NON_GAUSSIAN_BETA_PRIOR_MSG) from None

    # Σ_d⁻¹ is diagonal; carry it as the per-epoch inverse variance.
    inv_var = 1.0 / sig ** 2

    # Data-only normal matrix M_d = Xᵀ Σ_d⁻¹ X and RHS = Xᵀ Σ_d⁻¹ d.
    # These two are the ONLY ingredients of ``logL_marginal``: the
    # ranking score is a function of the data and the design alone, on
    # every path, whether or not a prior was passed.
    XtW = Xmat.T * inv_var  # (9, n): each column scaled by inv_var
    M_data = XtW @ Xmat
    rhs_data = XtW @ d

    if beta_prior is not None:
        prior_mean = np.asarray(beta_prior.mean, dtype=float)
        prior_cov = np.asarray(beta_prior.cov, dtype=float)
        # A singular / non-PD prior covariance cannot be inverted; convert
        # the raw LinAlgError to the project's value-free ValueError so no
        # matrix entry leaks via the numpy message (``from None`` severs
        # the value-bearing chain).
        try:
            prior_prec = np.linalg.inv(prior_cov)
        except np.linalg.LinAlgError:
            raise ValueError(_SINGULAR_DESIGN_MSG) from None
        M = M_data + prior_prec
        rhs = rhs_data + prior_prec @ prior_mean
    else:
        M = M_data
        rhs = rhs_data

    # Cholesky factorisation: reused for the solve, the covariance, AND
    # (on the flat path) log|M| for the ranking score.  A rank-deficient
    # design makes M not positive definite → value-free error.
    try:
        cho = sla.cho_factor(M, lower=True)
    except sla.LinAlgError:
        raise ValueError(_SINGULAR_DESIGN_MSG) from None

    beta = sla.cho_solve(cho, rhs)
    cov = sla.cho_solve(cho, np.eye(n_col))

    r = d - Xmat @ beta
    chi2 = float(r @ (inv_var * r))

    # log|M| = 2 · Σ log(diag(L)), L the lower Cholesky factor.
    chol_factor = cho[0]
    logdet_M = 2.0 * float(np.sum(np.log(np.abs(np.diag(chol_factor)))))

    log_det_sigma2 = float(np.sum(np.log(sig ** 2)))

    # ── The flat-prior ranking score, from the DATA-ONLY ingredients ──
    # Flat path: M is M_data and beta is the least-squares solution, so
    # the score reuses them directly (no second solve).
    # Prior path: re-solve the unregularised system for the score only;
    # ``beta`` / ``cov`` above stay the prior-regularised posterior.  A
    # singular data-only matrix (often the very reason a prior was
    # passed — nine columns on few transits) has no flat marginal — the
    # score is None, never a fudge.
    logL_marginal: float | None
    if beta_prior is None:
        logL_marginal = -0.5 * (
            chi2
            + logdet_M
            + n * np.log(2.0 * np.pi)
            + log_det_sigma2
        )
    else:
        try:
            cho_data = sla.cho_factor(M_data, lower=True)
        except sla.LinAlgError:
            cho_data = None
        if cho_data is None:
            logL_marginal = None
        else:
            beta_flat = sla.cho_solve(cho_data, rhs_data)
            r_flat = d - Xmat @ beta_flat
            chi2_flat = float(r_flat @ (inv_var * r_flat))
            logdet_M_data = 2.0 * float(
                np.sum(np.log(np.abs(np.diag(cho_data[0]))))
            )
            logL_marginal = -0.5 * (
                chi2_flat
                + logdet_M_data
                + n * np.log(2.0 * np.pi)
                + log_det_sigma2
            )

    # Proper marginal log-evidence — informative prior only.  Computed in
    # the reduced (9×9) form that reuses M (here M = Xᵀ Σ_d⁻¹ X + Σ_β⁻¹)
    # and its Cholesky ``cho`` (matrix-determinant lemma + Woodbury).  On
    # the flat path ``log_evidence`` stays None (no proper evidence for an
    # improper prior) — that branch leaves the values above byte-identical.
    log_evidence: float | None = None
    if beta_prior is not None:
        # log|Σ_β| is NOT reused from M — it requires its own (cheap) 9×9
        # Cholesky of the prior covariance; that same factorisation also
        # guards positive-definiteness, so a non-PD Σ_β fails value-free
        # rather than emitting a NaN evidence.
        try:
            prior_cho = sla.cho_factor(prior_cov, lower=True)
        except sla.LinAlgError:
            raise ValueError(_SINGULAR_DESIGN_MSG) from None
        logdet_prior_cov = 2.0 * float(
            np.sum(np.log(np.abs(np.diag(prior_cho[0]))))
        )

        # r = d − X μ_β ; b = Xᵀ Σ_d⁻¹ r ; quad = rᵀ Σ_d⁻¹ r − bᵀ M⁻¹ b.
        # M⁻¹ b reuses the existing Cholesky ``cho`` of M.
        r_prior = d - Xmat @ prior_mean
        b_vec = XtW @ r_prior
        quad = float(r_prior @ (inv_var * r_prior)) - float(
            b_vec @ sla.cho_solve(cho, b_vec)
        )
        logdet_evidence = log_det_sigma2 + logdet_prior_cov + logdet_M
        log_evidence = -0.5 * (
            quad + logdet_evidence + n * np.log(2.0 * np.pi)
        )

    return LinearTiSolution(
        beta=np.asarray(beta, dtype=float),
        cov=np.asarray(cov, dtype=float),
        logL_marginal=logL_marginal,
        chi2=chi2,
        dof=int(n - n_col),
        log_evidence=log_evidence,
    )


# ── Best linear TI params at a shape (+ optional correlated seed draws) ─
#
# Value-free error messages: no input value / source identifier is
# echoed, whatever the caller's data are.
_BLT_NONFINITE_MSG = (
    "epochs_mjd, scan_angle, parallax_factor_al, centroid_pos, and "
    "centroid_pos_err must be 1-D finite arrays of equal length; a "
    "non-finite (NaN/inf) value or a length mismatch was found."
)
_BLT_ECC_MSG = "ecc must satisfy 0 <= ecc < 1."
_BLT_PERIOD_MSG = "period_days must be a positive finite scalar."
_BLT_INFLATE_MSG = (
    "inflate must be a positive finite scalar (the covariance inflation "
    "factor for the seed draws; >= 1 widens the conditional spread)."
)
_BLT_DRAW_MSG = (
    "draw must be None or a positive integer (the number of correlated "
    "seed draws to return)."
)


@dataclass(frozen=True)
class TiBestLinearParams:
    """Best-fit linear Thiele-Innes amplitudes at a FIXED non-linear shape
    ``(P, e, τ)``, with the covariance and optional correlated seed draws.

    The 9-D linear basis is ``β = (A, B, F, G, ra_offset, dec_offset, pmra,
    pmdec, plx)`` — the same basis the marginalised TI sampler draws its
    post-hoc conditional ``β`` from.  All quantities are CONDITIONAL on the
    supplied FIXED ``jitter_mas`` (the full fit samples the jitter; this
    routine holds it fixed, exactly like the quick-look).  No input value /
    source identifier is stored.

    Attributes
    ----------
    beta : np.ndarray
        Best-fit 9-D amplitudes ``(A, B, F, G, ra_offset, dec_offset, pmra,
        pmdec, plx)``, shape ``(9,)``.
    cov : np.ndarray
        Posterior covariance of ``β`` (GLS ``M⁻¹``), shape ``(9, 9)``.
        Generally NON-diagonal — the Thiele-Innes coefficients and the
        astrometric nuisances are correlated through the cadence / scan
        geometry — so a covariance-correct seed MUST draw from the full
        matrix, not from independent per-parameter spreads.
    logL_marginal : float or None
        Flat-prior marginal-likelihood RANKING score (the TI-scan ranking
        statistic across shapes), from the data-only ingredients on every
        path; ``None`` only when a ``beta_prior`` was passed and
        the data-only normal matrix is singular.  NOT an absolute evidence.
    log_evidence : float or None
        Proper Gaussian-linear marginal log-evidence (INFORMATIVE
        ``beta_prior`` only; ``None`` on the flat-prior path).
    chi2 : float
        Weighted residual sum ``rᵀ Σ_d⁻¹ r`` at ``β̂`` (feeds the χ²/dof
        fixed-jitter mis-set check).
    dof : int
        Degrees of freedom ``n − 9``.
    draws : np.ndarray or None
        When ``draw=N`` was requested, ``N`` correlated samples from
        ``N(β̂, inflate · cov)``, shape ``(N, 9)``; else ``None``.  Drawn
        from the FULL covariance (off-diagonal correlations preserved) — a
        covariance-correct seed for a warm-started full fit.  Mass is NOT
        among these parameters: TI measures the photocentre geometry, not
        the mass (see the reference manual, §3.8).
    """

    beta: np.ndarray
    cov: np.ndarray
    logL_marginal: float | None
    log_evidence: float | None
    chi2: float
    dof: int
    draws: np.ndarray | None = None

    def __repr__(self) -> str:  # noqa: D401 - value-free by design
        # Shapes / counts ONLY — never a fitted amplitude, covariance entry,
        # or likelihood score (the default dataclass repr would echo the whole
        # fitted β/cov into any log or traceback that prints the object).
        # Built with str.format, not an f-string, like every message in
        # this module.
        n_draws = 0 if self.draws is None else int(self.draws.shape[0])
        return (
            "TiBestLinearParams(beta={0}, cov={1}, dof={2}, n_draws={3}, "
            "has_evidence={4})"
        ).format(
            tuple(self.beta.shape),
            tuple(self.cov.shape),
            int(self.dof),
            n_draws,
            self.log_evidence is not None,
        )

    __str__ = __repr__


def best_linear_params_ti(
    period_days: float,
    ecc: float,
    tau: float,
    *,
    epochs_mjd: np.ndarray,
    scan_angle: np.ndarray,
    parallax_factor_al: np.ndarray,
    centroid_pos: np.ndarray,
    centroid_pos_err: np.ndarray,
    epoch_ref_mjd: float,
    jitter_mas: float = 0.0,
    beta_prior=None,
    draw: int | None = None,
    inflate: float = 1.0,
    seed: int = 0,
) -> TiBestLinearParams:
    """Best-fit linear Thiele-Innes amplitudes at a FIXED orbit shape
    ``(P, e, τ)`` — one closed-form GLS solve, NO MCMC.

    The astrometric twin of
    :func:`orblet.solve.rv.best_linear_params_rv`: the standalone
    "given the non-linear shape, what are the best 9 linear amplitudes?"
    primitive.  It builds the along-scan design matrix at the shape and runs
    the same :func:`linear_solve_ti` the marginalised sampler uses, so the
    returned ``β̂`` / ``cov`` are IDENTICAL to that sampler's per-shape
    conditional Gaussian.  Uses:

    - overlay the best-fit photocentre orbit at any trial period;
    - rank periods by ``logL_marginal`` (the TI-scan statistic);
    - generate a COVARIANCE-CORRECT seed for a warm-started full TI fit
      (``draw=N`` → ``N`` correlated samples from ``N(β̂, inflate·cov)``).

    Conventions
    -----------
    - Linear basis ``β = (A, B, F, G, ra_offset, dec_offset, pmra, pmdec,
      plx)``.  Convert to Campbell ``(i, Ω, ω, a_phot)`` downstream via the
      project TI inversion if wanted.  Mass is NOT here (TI measures
      geometry, not mass).
    - Period in DAYS (the design matrix uses ``f_per_day = 1 / period_days``).
      Same day convention as the RV twin (project unit-unification note).
    - Jitter is held FIXED at ``jitter_mas``; the noise is
      ``σ_eff = √(centroid_pos_err² + jitter_mas²)`` (mas; matches the
      quick-look).  The returned posterior is conditional on that fixed
      jitter; the draws assume a well-conditioned ``cov`` (a rank-deficient
      design already raises value-free inside ``linear_solve_ti``).

    Parameters
    ----------
    period_days, ecc, tau : float
        Non-linear orbit shape.  ``period_days > 0`` (it is used as
        ``f_per_day = 1/period_days`` — unlike the RV twin, which passes the
        period as a value, this side validates positivity); ``0 ≤ ecc < 1``;
        ``tau`` is the periastron-phase fraction on the unit circle.
    epochs_mjd : np.ndarray
        Observation epochs (MJD, Gaia DR4 TCB), shape ``(n,)``.
    scan_angle : np.ndarray
        Along-scan angle ``ψ`` (radians), one per epoch.
    parallax_factor_al : np.ndarray
        Dimensionless along-scan parallax factor per epoch.
    centroid_pos : np.ndarray
        Along-scan centroid offset (mas), the data ``d``.
    centroid_pos_err : np.ndarray
        Per-epoch along-scan Gaussian error (mas).
    epoch_ref_mjd : float
        Reference epoch (MJD); periastron-time zero AND the PM / parallax
        zero-point.
    jitter_mas : float, optional
        FIXED along-scan jitter (mas), added in quadrature.  Default 0.
    beta_prior : optional
        Gaussian ``β`` prior (object exposing ``.mean`` / ``.cov``, shapes
        ``(9,)`` / ``(9, 9)``) — when given, ``log_evidence`` carries the
        proper marginal evidence.
    draw : int or None, optional
        ``None`` → point estimate only; ``N`` → also return ``N`` correlated
        seed draws.  Default ``None``.
    inflate : float, optional
        Covariance inflation for the draws (``≥ 1`` widens the conditional
        spread to absorb the fixed-jitter linearisation error when seeding a
        jitter-sampling full fit).  Default 1.0 (the exact conditional).  The
        draws assume a well-conditioned ``cov``; a rank-deficient design
        already raises (value-free) inside ``linear_solve_ti`` before any
        draw, but a marginally-singular ``cov`` may surface a NumPy
        "covariance is not positive-semidefinite" RuntimeWarning (no data
        value in it) — harmless for a warm-start seed.
    seed : int, optional
        RNG seed for the draws.  Default 0.

    Returns
    -------
    TiBestLinearParams
    """
    t = np.asarray(epochs_mjd, dtype=float)
    psi = np.asarray(scan_angle, dtype=float)
    pf = np.asarray(parallax_factor_al, dtype=float)
    d = np.asarray(centroid_pos, dtype=float)
    err = np.asarray(centroid_pos_err, dtype=float)

    # Value-free validation (no echoed value / identifier).
    if not (t.ndim == 1 and t.shape == psi.shape == pf.shape == d.shape == err.shape):
        raise ValueError(_BLT_NONFINITE_MSG)
    if not all(np.all(np.isfinite(a)) for a in (t, psi, pf, d, err)):
        raise ValueError(_BLT_NONFINITE_MSG)
    if not (np.isfinite(period_days) and float(period_days) > 0.0):
        raise ValueError(_BLT_PERIOD_MSG)
    if not (0.0 <= float(ecc) < 1.0):
        raise ValueError(_BLT_ECC_MSG)
    if not (np.isfinite(inflate) and float(inflate) > 0.0):
        raise ValueError(_BLT_INFLATE_MSG)
    if draw is not None and (
        isinstance(draw, bool)
        or not isinstance(draw, (int, np.integer))
        or int(draw) <= 0
    ):
        raise ValueError(_BLT_DRAW_MSG)

    # Build X at the shape (frequency = 1/P); jitter-inflated noise.
    f_per_day = 1.0 / float(period_days)
    X = _ti_design_matrix(
        t, psi, pf, f_per_day=f_per_day, ecc=float(ecc), tau=float(tau),
        epoch_ref_mjd=float(epoch_ref_mjd),
    )
    sigma_eff = np.sqrt(err ** 2 + float(jitter_mas) ** 2)

    sol = linear_solve_ti(d, sigma_eff, X, beta_prior=beta_prior)

    # Optional COVARIANCE-CORRECT seed draws from N(β̂, inflate · cov).  The
    # full off-diagonal cov is used (TI-coefficient ↔ nuisance correlations
    # preserved); same multivariate_normal(β̂, cov) construction as the
    # marginalised sampler's post-hoc conditional draw (so inflate=1 is it).
    draws = None
    if draw is not None:
        rng = np.random.default_rng(int(seed))
        draws = rng.multivariate_normal(
            sol.beta, float(inflate) * sol.cov, size=int(draw),
        )

    return TiBestLinearParams(
        beta=sol.beta,
        cov=sol.cov,
        logL_marginal=sol.logL_marginal,
        log_evidence=sol.log_evidence,
        chi2=sol.chi2,
        dof=sol.dof,
        draws=draws,
    )


# ── Public single-star (5-parameter) astrometric fit ─────────────────
#
# A closed-form, orbit-free astrometric solution: the standard Gaia
# 5-parameter single-star model ``(Δα*, Δδ, μα*, μδ, ϖ)``.  No orbit
# shape, no jitter, no priors, no sampler — a thin public surface over
# the single-star design block and the existing GLS solve.


def astrometric_5param_design_matrix(
    t_mjd: np.ndarray,
    psi: np.ndarray,
    parallax_factor_al: np.ndarray,
    *,
    epoch_ref_mjd: float,
) -> np.ndarray:
    """Single-star (5-parameter) along-scan design matrix, shape ``(n, 5)``.

    The standard Gaia 5-parameter astrometric model has NO orbit: the
    along-scan model is linear in the five single-star amplitudes

        model = Δα* · sinψ + Δδ · cosψ
                + (μα* · sinψ + μδ · cosψ) · Δt_yr
                + ϖ · parallax_factor_al,

    with ``Δt_yr = (t_mjd − epoch_ref_mjd) / DAYS_PER_KEPLER_YEAR``.  The
    columns are the partial derivatives w.r.t. those amplitudes, in the
    FIXED order (units in brackets):

        column 0  ∂/∂Δα*   = sinψ                 [Δα* in mas]
        column 1  ∂/∂Δδ    = cosψ                 [Δδ  in mas]
        column 2  ∂/∂μα*   = sinψ · Δt_yr         [μα* in mas/yr]
        column 3  ∂/∂μδ    = cosψ · Δt_yr         [μδ  in mas/yr]
        column 4  ∂/∂ϖ     = parallax_factor_al   [ϖ   in mas]

    This is EXACTLY columns 4-8 of :func:`_ti_design_matrix` (both call the
    shared :func:`_astrometric_columns`), so the single-star convention is
    identical to the orbit fit's.

    Conventions
    -----------
    - ``pmra = μα*`` already includes the ``cos δ`` factor (project
      convention; no ``cos δ`` is applied here — see
      the reference manual, §3.1).
    - Parallax enters ADDITIVELY as ``ϖ · parallax_factor_al`` (the
      parallax factor is precomputed per epoch).
    - Time scale: ``t_mjd`` and ``epoch_ref_mjd`` must share the SAME time
      scale (Gaia DR4: TCB).  Mixing scales silently biases the
      proper-motion and parallax phase.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation epochs (MJD, TCB), shape ``(n,)``.
    psi : np.ndarray
        Along-scan angle ``ψ`` (radians, counterclockwise from north), one
        per epoch.
    parallax_factor_al : np.ndarray
        Dimensionless along-scan parallax factor per epoch.
    epoch_ref_mjd : float
        Proper-motion / parallax zero-point epoch (MJD), REQUIRED.

    Returns
    -------
    np.ndarray
        ``X`` of shape ``(n, 5)``, float64, in the fixed column order
        ``[Δα*, Δδ, μα*, μδ, ϖ]``.
    """
    # Routed through the pluggable single-star block builder (which delegates
    # back to :func:`_astrometric_columns`), so this public surface and every
    # future composed design share ONE column source.  Byte-identical:
    # SingleStarBlockBuilder.build IS _astrometric_columns.  Lazy import keeps
    # the engine off the design/ import graph at module scope.
    from orblet.design import SingleStarBlockBuilder

    return SingleStarBlockBuilder().build(
        t_mjd, psi, parallax_factor_al, epoch_ref_mjd=epoch_ref_mjd,
    )


@dataclass(frozen=True)
class Astrometric5ParamSolution:
    """Closed-form single-star (5-parameter) astrometric fit result.

    All arrays are float64.  No input value / source identifier is
    echoed.

    Attributes
    ----------
    params : np.ndarray
        Best-fit single-star amplitudes, shape ``(5,)``, FIXED column order
        ``[Δα* (mas), Δδ (mas), μα* (mas/yr), μδ (mas/yr), ϖ (mas)]``.
        ``μα* = pmra`` already includes ``cos δ`` (project convention).
    cov : np.ndarray
        Posterior covariance of ``params`` (GLS ``M⁻¹``), shape ``(5, 5)``.
    model : np.ndarray
        Best-fit along-scan model ``X · params`` at each epoch, shape
        ``(n,)``, in mas.
    residuals : np.ndarray
        Along-scan residuals ``d_obs − X · params``, shape ``(n,)``, in mas.
        Computed directly from the 5-parameter model (NOT the loader's
        ra0/dec0 ``compute_astrometric_residuals`` path).
    chi2_per_dof : float
        DESCRIPTIVE goodness-of-fit summary: the weighted χ² ``rᵀ Σ_d⁻¹ r``
        divided by ``dof = n − 5``, under a FIXED diagonal-Gaussian
        likelihood (``Σ_d = diag(σ²)``; no jitter, no error inflation, no
        priors).  This is NOT an inferred astrometric-excess-noise /
        astrometric-jitter / RUWE parameter — it is a fixed-noise fit
        quality diagnostic only.
    rms_mas : float
        DESCRIPTIVE root-mean-square of the along-scan residuals (mas),
        ``sqrt(mean(residuals²))``.  Again a fit-quality summary, NOT an
        inferred excess-noise parameter.

    Shapes / axes
    -------------
    - Epoch axis has length ``n`` (``model`` / ``residuals``).
    - Parameter axis has length 5 (``params`` / both axes of ``cov``).
    """

    params: np.ndarray
    cov: np.ndarray
    model: np.ndarray
    residuals: np.ndarray
    chi2_per_dof: float
    rms_mas: float

    def __repr__(self) -> str:  # noqa: D401 - value-free by design
        # Shapes / counts ONLY — never the fitted 5-parameter amplitudes,
        # covariance, model/residual arrays, or the chi2/rms fit-quality
        # summaries (the object's repr ends up in logs and tracebacks).
        # str.format (not an f-string) to pass the no-f-string AST guard.
        return (
            "Astrometric5ParamSolution(params={0}, cov={1}, n_epochs={2})"
        ).format(
            tuple(self.params.shape),
            tuple(self.cov.shape),
            int(self.model.shape[0]),
        )

    __str__ = __repr__


def fit_astrometric_5param(
    astro_data: dict,
    *,
    epoch_ref_mjd: float | None = None,
) -> Astrometric5ParamSolution:
    """Closed-form single-star (5-parameter) astrometric fit — NO MCMC.

    The standard Gaia 5-parameter astrometric solution
    ``(Δα*, Δδ, μα*, μδ, ϖ)`` with NO orbit, NO jitter, NO priors, and NO
    sampler.  It builds the single-star along-scan design matrix and runs
    the SAME generalised-least-squares solve (:func:`linear_solve_ti`) the
    orbit fit uses, so the closed-form amplitudes / covariance are exact.

    This is the astrometric analogue of "fit a straight line": a fast,
    orbit-free reference fit whose residuals reveal orbital motion, and a
    natural first stage before any orbit search.

    Conventions
    -----------
    - Parameter order / units: ``[Δα* (mas), Δδ (mas), μα* (mas/yr),
      μδ (mas/yr), ϖ (mas)]``.  ``μα* = pmra`` already includes ``cos δ``
      (project convention); parallax enters additively as
      ``ϖ · parallax_factor_al``.
    - Time scale: ``obs_time`` and ``epoch_ref_mjd`` must share ONE time
      scale (Gaia DR4: TCB).  ``obs_time`` is resolved to MJD via the same
      J2010-days→MJD heuristic the orbit fit uses
      (:func:`orblet.prepare.resolve_epochs_mjd`).
    - Likelihood: FIXED diagonal Gaussian ``Σ_d = diag(centroid_pos_err²)``
      (no jitter, no error inflation).  ``chi2_per_dof`` / ``rms_mas`` are
      DESCRIPTIVE fit-quality summaries under that fixed noise model — NOT
      inferred astrometric-excess-noise / RUWE parameters.

    Parameters
    ----------
    astro_data : dict
        Along-scan epoch data.  The SAME fields the orbit fit reads:
        ``obs_time`` (epochs; MJD-TCB or J2010-days-TCB), ``scan_angle``
        (``ψ`` in radians), ``parallax_factor_al`` (dimensionless per
        epoch), ``centroid_pos`` (along-scan offset ``d``, mas),
        ``centroid_pos_err`` (per-epoch Gaussian error, mas).
    epoch_ref_mjd : float
        Proper-motion / parallax zero-point epoch (MJD).  REQUIRED — ``None``
        raises: it is the catalogue reference epoch, which tracks
        the data release (DR3 J2016.0, DR4 J2017.5), so a hardcoded default
        would bias the recovered position / proper motion on DR4 data.

    Returns
    -------
    Astrometric5ParamSolution
        Frozen result (``params`` (5,), ``cov`` (5, 5), ``model`` (n,),
        ``residuals`` (n,), ``chi2_per_dof``, ``rms_mas``).
    """
    from orblet.constants import require_epoch_ref_mjd
    from orblet.prepare import resolve_epochs_mjd

    # The reference epoch is REQUIRED — it is the catalogue PM/parallax zero-point,
    # which tracks the data release (DR3 J2016.0, DR4 J2017.5).  A wrong default would
    # bias the recovered position/proper motion on DR4 data.  Same requirement as the
    # orbit fits; pass it explicitly for a standalone call.
    epoch_ref_mjd = require_epoch_ref_mjd(epoch_ref_mjd, channel="astrometric")

    # Same fields / heuristic as fit_astrometry_orbit_linear_ti (verbatim).
    t_mjd = resolve_epochs_mjd(astro_data)
    psi = np.asarray(astro_data["scan_angle"], dtype=float)
    pf = np.asarray(astro_data["parallax_factor_al"], dtype=float)
    d_obs = np.asarray(astro_data["centroid_pos"], dtype=float)
    sigma = np.asarray(astro_data["centroid_pos_err"], dtype=float)

    X = astrometric_5param_design_matrix(
        t_mjd, psi, pf, epoch_ref_mjd=float(epoch_ref_mjd),
    )

    # Reuse the existing closed-form GLS solve.  With a 5-column X it gives
    # the 5-parameter beta / cov / chi2 / dof = n − 5 automatically, and
    # inherits the value-free under-determination / singular-design guards
    # (n < 5, rank-deficient design, non-positive sigma).
    sol = linear_solve_ti(d_obs, sigma, X)

    params = sol.beta
    model = X @ params
    # Residuals from the 5-parameter model directly (NOT the loader path).
    residuals = d_obs - model

    dof = sol.dof
    chi2_per_dof = float(sol.chi2 / dof)
    rms_mas = float(np.sqrt(np.mean(residuals ** 2)))

    return Astrometric5ParamSolution(
        params=np.asarray(params, dtype=float),
        cov=np.asarray(sol.cov, dtype=float),
        model=np.asarray(model, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        chi2_per_dof=chi2_per_dof,
        rms_mas=rms_mas,
    )
