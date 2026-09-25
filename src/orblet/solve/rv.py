"""Linearised SB1 radial-velocity core (Scheme A — marginalise ω).

This module implements the load-bearing linear-algebra core of a
marginalised SB1 radial-velocity fit.  At a FIXED non-linear shape
``(P, e, τ)`` the Keplerian RV model is LINEAR in three amplitudes, so
the best-fit amplitudes and a marginal-likelihood ranking score follow
from a single 3×3 generalised-least-squares (GLS) solve.

Scheme A linearisation
----------------------
The textbook primary-frame RV model (see ``...rv.forward``)

    v = γ + K [cos(ν + ω) + e cos ω]

expands, using ``cos(ν + ω) = cosν cosω − sinν sinω``, to

    v = γ·1 + (K cosω)·(cosν + e) + (K sinω)·(−sinν),

which is LINEAR in ``β = (γ, C, S) = (γ, K cosω, K sinω)`` once
``(P, e, τ)`` fix the true anomaly ``ν`` via Kepler's equation.  The
design columns are therefore ``[1, cosν + e, −sinν]``.  ω is
"marginalised" in the sense that it is recovered from the linear
amplitudes (``ω = atan2(S, C)``) rather than sampled.

Conventions (matching ``...rv.forward``)
----------------------------------------
RV sign / ω convention
    Primary-frame argument of periastron ω (binary-star convention).
    ``K`` is the PRIMARY semi-amplitude in km/s.  The amplitude split
    ``(C, S) = (K cosω, K sinω)`` and the recovery
    ``K = √(C² + S²)``, ``ω = atan2(S, C)`` follow directly.

Period unit / τ convention
    ``period_yr`` is in Keplerian years (matching ``rv_model`` and the
    chain key).  The periastron time is ``tp = τ · P_days +
    epoch_ref_mjd`` with ``P_days = period_yr · DAYS_PER_KEPLER_YEAR``,
    EXACTLY as ``rv_model`` builds it.

epoch_ref_mjd
    ``epoch_ref_mjd`` is a REQUIRED keyword (no default — the linear core
    does NOT inherit the engine's median-epoch default).  It is the
    periastron-time zero only (RV has no proper-motion term).  Shifting
    it rotates the recovered ω (the gauge that absorbs the orbital
    phase) while leaving ``K`` and ``χ²`` invariant.

True anomaly is the RV-LOCAL atom
    ``ν`` comes from :func:`...rv.forward._true_anomaly` — the SAME atom
    ``rv_model`` uses, so the linear core and the full forward share one
    source of truth.  It is NOT routed through the astro / Campbell
    Kepler projector.

Marginal likelihood is a RANKING score
    ``logL_marginal`` is a flat-prior marginal-likelihood RANKING score
    for comparing candidate ``(P, e, τ)`` grid points — NOT an absolute
    evidence.  It retains the ``log|M|`` Occam term (shape-dependent) but
    drops the β-prior normalisation (shape-independent on the flat path).
    The proper Gaussian-linear marginal ``log_evidence`` is populated
    ONLY under an informative Gaussian β prior; it mirrors the astro
    ``linear_marginal`` evidence form (reduced 3×3 / matrix-determinant
    lemma) and retains ALL constants.

    One meaning on every path.  ``logL_marginal`` is
    computed from the DATA-ONLY normal matrix and least-squares
    amplitudes whether or not a ``beta_prior`` was passed.  The
    prior affects ``beta`` / ``cov`` (the regularised posterior) and
    ``log_evidence`` (the proper prior-weighted integral), never the
    ranking score.  Rules: sum the SAME score across independent channels
    at one shape (both ``logL_marginal`` or both ``log_evidence``); never
    add one kind to the other; the k-dependent constant is dropped, so
    never compare ``logL_marginal`` across models with different numbers
    of amplitudes — use ``log_evidence`` for that.

Lazy-import discipline
    ``scipy`` is imported INSIDE the function body that needs it, never
    at module scope.  This module imports only numpy, orblet and the standard
    library at module scope.

No echoed values
    No θ / amplitude / input value is logged, printed, or persisted in
    this module.  Error messages are value-free module-level Constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.model import _true_anomaly


# Fixed column order of the linear amplitude vector β.
_BETA_COLUMNS = ("gamma", "C", "S")  # (γ, K cosω, K sinω)
_N_BETA = len(_BETA_COLUMNS)

# Value-free error messages (no matrix entries / no input values).
_NON_GAUSSIAN_BETA_PRIOR_MSG = (
    "beta_prior must be a Gaussian prior exposing 'mean' and 'cov'; "
    "non-Gaussian beta priors are not supported in the linear RV solve."
)
_TOO_FEW_EPOCHS_MSG = (
    "linear RV solve needs at least as many epochs as linear "
    "amplitudes; the design matrix has fewer rows than columns."
)
_SINGULAR_DESIGN_MSG = (
    "linear RV normal matrix is singular or not positive definite "
    "(rank-deficient design); cannot solve at this grid point."
)



# The RV design columns live in the design package, beside the builder
# that wraps them.  Imported here at MODULE scope and re-exported, so the
# PUBLIC name ``orblet.rv_design_matrix`` resolves through this module
# as ``docs/open_api.md`` says, while the definition sits one layer down
# where the other channels can reach it without importing across a
# channel boundary.  Its values are pinned by
# ``tests/test_design_columns_baseline.py``.
#
from orblet.design.columns import rv_design_matrix  # noqa: E402

# Private alias for the engine-internal callers and the tests; the
# public spelling above is the one documented on the open surface.
#
_rv_design_matrix = rv_design_matrix


@dataclass(frozen=True)
class RvLinearSolution:
    """Result of the 3×3 GLS solve at one non-linear shape.

    All arrays are float64.  No input value / source identifier is echoed;
    the raw RV / sigma / epoch inputs are NOT stored.

    Attributes
    ----------
    beta : np.ndarray
        Best-fit linear amplitudes ``(γ, C, S) = (γ, K cosω, K sinω)``,
        shape ``(3,)``.
    cov : np.ndarray
        Posterior covariance of ``beta`` (``M⁻¹``), shape ``(3, 3)``.
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
        Degrees of freedom ``n − 3``.
    log_evidence : float or None
        Proper Gaussian-linear marginal log-evidence
        ``log N(d | X μ_β, X Σ_β Xᵀ + Σ_d)`` under the INFORMATIVE β
        prior, with ALL constants retained.  ``None`` on the flat
        (``beta_prior=None``) path (an improper prior has no proper
        evidence).
    """

    beta: np.ndarray
    cov: np.ndarray
    logL_marginal: float | None
    chi2: float
    dof: int
    log_evidence: float | None = None


def linear_solve_rv(
    d_rv: np.ndarray,
    sigma: np.ndarray,
    X: np.ndarray,
    *,
    beta_prior=None,
) -> RvLinearSolution:
    """Generalised-least-squares solve for the three linear RV amplitudes.

    Solves, with ``Σ_d = diag(σ²)``,

        M  = Xᵀ Σ_d⁻¹ X      ( + Σ_β⁻¹ if a Gaussian prior is given )
        β̂ = M⁻¹ ( Xᵀ Σ_d⁻¹ d_rv  [+ Σ_β⁻¹ μ_β] )
        cov = M⁻¹

    and the flat-prior marginal-likelihood RANKING score, ALWAYS from the
    data-only ingredients ``M_d = Xᵀ Σ_d⁻¹ X`` and the least-squares
    ``β̂_d = M_d⁻¹ Xᵀ Σ_d⁻¹ d_rv`` (the prior never enters this
    number):

        logL_marginal = −0.5 · [ r_dᵀ Σ_d⁻¹ r_d + log|M_d|
                                 + n·log(2π) + Σ log σ²ᵢ ],
        r_d = d_rv − X β̂_d.

    On the flat path ``M = M_d`` and ``β̂ = β̂_d``.  On the prior path the
    score is ``None`` when ``M_d`` is singular (no flat marginal exists).

    Proper marginal log-evidence (informative prior only)
    -----------------------------------------------------
    When a Gaussian ``beta_prior`` is supplied, the result also carries
    the PROPER marginal log-evidence (mirrors the astro linear core)

        log_evidence = log N( d | X μ_β , X Σ_β Xᵀ + Σ_d ),

    computed via the reduced (3×3) form that reuses the precision matrix
    ``M = Xᵀ Σ_d⁻¹ X + Σ_β⁻¹`` and its Cholesky (matrix-determinant lemma
    + Woodbury):

        r      = d − X μ_β
        b      = Xᵀ Σ_d⁻¹ r
        quad   = rᵀ Σ_d⁻¹ r − bᵀ M⁻¹ b
        logdet = Σᵢ log(σ²ᵢ) + log|Σ_β| + log|M|
        log_evidence = −0.5 ( quad + logdet + n·log(2π) ).

    The ``log|Σ_β|`` term (including the γ-prior width) is RETAINED — it is
    NOT dropped — so ``log_evidence`` is directly comparable to a full
    fit's absolute log-posterior.  On the flat path ``log_evidence`` is
    ``None``.

    Parameters
    ----------
    d_rv : np.ndarray
        Observed radial velocities (km/s), shape ``(n,)``.
    sigma : np.ndarray
        Per-epoch RV uncertainties (km/s), shape ``(n,)``.
    X : np.ndarray
        Design matrix from :func:`_rv_design_matrix`, shape ``(n, 3)``.
    beta_prior : object or None, optional
        ``None`` ⇒ flat prior (default).  A non-None prior MUST expose
        ``.mean`` (shape ``(3,)``) and ``.cov`` (shape ``(3, 3)``) for a
        Gaussian prior; anything else raises ``ValueError`` (value-free,
        ``from None``).

    Returns
    -------
    RvLinearSolution
    """
    # Lazy scipy import INSIDE the body (lazy-import contract).
    import scipy.linalg as sla

    d = np.asarray(d_rv, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    Xmat = np.asarray(X, dtype=float)

    n, n_col = Xmat.shape
    if n < n_col:
        # Value-free guard: not enough epochs to constrain 3 amplitudes.
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
    XtW = Xmat.T * inv_var  # (3, n): each column scaled by inv_var
    M_data = XtW @ Xmat
    rhs_data = XtW @ d

    if beta_prior is not None:
        prior_mean = np.asarray(beta_prior.mean, dtype=float)
        prior_cov = np.asarray(beta_prior.cov, dtype=float)
        # A singular / non-PD prior covariance cannot be inverted; convert
        # the raw LinAlgError to the value-free ValueError (``from None``
        # severs the value-bearing chain).
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
    # passed) has no flat marginal — the score is None, never a fudge.
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

    # Proper marginal log-evidence — informative prior only.  Reduced 3×3
    # form reusing M and its Cholesky ``cho`` (matrix-determinant lemma +
    # Woodbury).  Flat path leaves ``log_evidence`` None.
    log_evidence: float | None = None
    if beta_prior is not None:
        # log|Σ_β| needs its own (cheap) 3×3 Cholesky of the prior
        # covariance; that factorisation also guards positive-definiteness.
        try:
            prior_cho = sla.cho_factor(prior_cov, lower=True)
        except sla.LinAlgError:
            raise ValueError(_SINGULAR_DESIGN_MSG) from None
        logdet_prior_cov = 2.0 * float(
            np.sum(np.log(np.abs(np.diag(prior_cho[0]))))
        )

        r_prior = d - Xmat @ prior_mean
        b_vec = XtW @ r_prior
        quad = float(r_prior @ (inv_var * r_prior)) - float(
            b_vec @ sla.cho_solve(cho, b_vec)
        )
        logdet_evidence = log_det_sigma2 + logdet_prior_cov + logdet_M
        log_evidence = -0.5 * (
            quad + logdet_evidence + n * np.log(2.0 * np.pi)
        )

    return RvLinearSolution(
        beta=np.asarray(beta, dtype=float),
        cov=np.asarray(cov, dtype=float),
        logL_marginal=logL_marginal,
        chi2=chi2,
        dof=int(n - n_col),
        log_evidence=log_evidence,
    )


def recover_K(beta: np.ndarray) -> float:
    """Recover the primary RV semi-amplitude ``K`` (km/s) from ``β``.

    ``K = √(C² + S²)`` with ``(C, S) = (K cosω, K sinω) = β[1:3]``.

    Units / conventions
    -------------------
    ``K`` is in km/s, the PRIMARY semi-amplitude.  Under the RV inclination
    convention ``sin i ≡ 1`` (RV-only data cannot constrain ``i``), ``K``
    measures the SB1 mass function / the PROJECTED companion mass
    ``m₂ · sin i`` (:data:`orblet.constants.MASS_CONVENTION_PROJECTED`),
    prior-driven on the primary mass ``M₁``.  RV measures ``K`` (the mass
    function), NOT the true companion mass.

    Returns
    -------
    float
        ``K`` in km/s.
    """
    beta = np.asarray(beta, dtype=float)
    return float(np.hypot(beta[1], beta[2]))


def recover_omega(beta: np.ndarray) -> float:
    """Recover the primary-frame argument of periastron ``ω`` (rad) from ``β``.

    ``ω = atan2(S, C)`` with ``(C, S) = (K cosω, K sinω) = β[1:3]``; the
    result is in ``(−π, π]`` (the ``atan2`` range).

    Degeneracy caveat
    -----------------
    At ``e ≈ 0`` the orbit is circular and ``ω`` is GAUGE-DEPENDENT
    (degenerate with the time of periastron / phase zero ``T0``): only
    ``K`` and the orbital phase are physical.  A recovered ``ω`` near
    ``e = 0`` should not be over-interpreted.

    Returns
    -------
    float
        ``ω`` in radians (primary frame).
    """
    beta = np.asarray(beta, dtype=float)
    return float(np.arctan2(beta[2], beta[1]))


# ── Best linear params at a shape (+ optional correlated seed draws) ──
#
# Value-free error messages: no input value / source identifier is
# echoed, whatever the caller's data are.
_BLP_NONFINITE_MSG = (
    "epochs_mjd, rv, and rv_err must be 1-D finite arrays of equal length; "
    "a non-finite (NaN/inf) value or a length mismatch was found."
)
_BLP_ECC_MSG = "ecc must satisfy 0 <= ecc < 1."
_BLP_INFLATE_MSG = (
    "inflate must be a positive finite scalar (the covariance inflation "
    "factor for the seed draws; >= 1 widens the conditional spread)."
)
_BLP_DRAW_MSG = (
    "draw must be None or a positive integer (the number of correlated "
    "seed draws to return)."
)


@dataclass(frozen=True)
class RvBestLinearParams:
    """Best-fit linear RV amplitudes at a FIXED non-linear shape, with the
    covariance and optional correlated seed draws.

    The linear basis is the CARTESIAN amplitude pair
    ``β = (γ, C, S) = (γ, K cosω, K sinω)`` — the SAME basis the
    marginalised RV sampler draws its post-hoc conditional ``β`` from.  All
    quantities are CONDITIONAL on the supplied FIXED ``jitter_kms`` (the
    full fit samples the jitter; this routine holds it fixed, exactly like
    the quick-look).  No input value / source identifier is stored.

    Attributes
    ----------
    beta : np.ndarray
        Best-fit ``(γ, C, S)``, shape ``(3,)``.
    cov : np.ndarray
        Posterior covariance of ``β`` (GLS ``M⁻¹``), shape ``(3, 3)``.
        Generally NON-diagonal — ``C`` and ``S`` are correlated through the
        cadence — so a covariance-correct seed MUST draw from the full
        matrix, not from independent per-parameter spreads.
    logL_marginal : float or None
        Flat-prior marginal-likelihood RANKING score (the periodogram
        ranking statistic across shapes), from the data-only ingredients
        on every path; ``None`` only when a ``beta_prior`` was
        passed and the data-only normal matrix is singular.  NOT an
        absolute evidence.
    log_evidence : float or None
        Proper Gaussian-linear marginal log-evidence (INFORMATIVE
        ``beta_prior`` only; ``None`` on the flat-prior path).
    chi2 : float
        Weighted residual sum ``rᵀ Σ_d⁻¹ r`` at ``β̂`` (feeds the χ²/dof
        fixed-jitter mis-set check).
    dof : int
        Degrees of freedom ``n − 3``.
    draws : np.ndarray or None
        When ``draw=N`` was requested, ``N`` correlated samples from
        ``N(β̂, inflate · cov)``, shape ``(N, 3)``; else ``None``.  Drawn
        from the FULL covariance (off-diagonal correlations preserved) — a
        covariance-correct seed for a warm-started full fit.
    """

    beta: np.ndarray
    cov: np.ndarray
    logL_marginal: float | None
    log_evidence: float | None
    chi2: float
    dof: int
    draws: np.ndarray | None = None


def best_linear_params_rv(
    period_days: float,
    ecc: float,
    tau: float,
    *,
    epochs_mjd: np.ndarray,
    rv: np.ndarray,
    rv_err: np.ndarray,
    epoch_ref_mjd: float,
    jitter_kms: float = 0.0,
    beta_prior=None,
    draw: int | None = None,
    inflate: float = 1.0,
    seed: int = 0,
) -> RvBestLinearParams:
    """Best-fit linear SB1-RV amplitudes ``(γ, C, S)`` at a FIXED orbit
    shape ``(P, e, τ)`` — one closed-form GLS solve, NO MCMC.

    This is the standalone "given the non-linear shape, what are the best
    amplitudes?" primitive.  It builds the design matrix at the shape and
    runs the same :func:`linear_solve_rv` the marginalised sampler uses, so
    the returned ``β̂`` / ``cov`` are IDENTICAL to that sampler's per-shape
    conditional Gaussian.  Uses:

    - overlay the best-fit RV orbit at any trial period on the data;
    - rank periods by ``logL_marginal`` (the RV periodogram statistic);
    - generate a COVARIANCE-CORRECT seed for a warm-started full fit
      (``draw=N`` → ``N`` correlated samples from ``N(β̂, inflate·cov)``).

    Conventions
    -----------
    - Linear basis ``β = (γ, C, S) = (γ, K cosω, K sinω)`` (Cartesian; no
      ``ω`` blow-up at ``e ≈ 0``).  Recover ``K`` / ``ω`` via
      :func:`recover_K` / :func:`recover_omega` if wanted.
    - Period is supplied in DAYS (converted to Keplerian years internally;
      ``P_yr = period_days / DAYS_PER_KEPLER_YEAR``).  NOTE for the future
      astrometry twin: it should ALSO take days — a single period unit
      across engines (see the project note on unit unification).
    - Jitter is held FIXED at ``jitter_kms``; the noise is
      ``σ_eff = √(rv_err² + jitter_kms²)`` (matches the quick-look).  The
      returned posterior is conditional on that fixed jitter.

    Parameters
    ----------
    period_days, ecc, tau : float
        Non-linear orbit shape.  ``0 ≤ ecc < 1``; ``tau`` is the
        periastron-phase fraction (``tp = τ·P_days + epoch_ref_mjd``).
    epochs_mjd, rv, rv_err : np.ndarray
        Observation epochs (MJD) and the RV series with its per-epoch
        Gaussian errors (km/s), equal-length 1-D finite arrays.
    epoch_ref_mjd : float
        Reference epoch (MJD); periastron-time zero only.
    jitter_kms : float, optional
        FIXED extra Gaussian scatter added in quadrature.  Default 0.
    beta_prior : optional
        Gaussian ``β`` prior (object exposing ``.mean`` / ``.cov``) — when
        given, ``log_evidence`` carries the proper marginal evidence.
    draw : int or None, optional
        ``None`` → point estimate only; ``N`` → also return ``N`` correlated
        seed draws.  Default ``None``.
    inflate : float, optional
        Covariance inflation for the draws (``≥ 1`` widens the conditional
        spread to absorb the fixed-jitter linearisation error when seeding a
        jitter-sampling full fit).  Default 1.0 (the exact conditional).  The
        draws assume a well-conditioned ``cov``; a rank-deficient design
        already raises (value-free) inside ``linear_solve_rv`` before any
        draw, but a marginally-singular ``cov`` may surface a NumPy
        "covariance is not positive-semidefinite" RuntimeWarning (no data
        value in it) — harmless for a warm-start seed.
    seed : int, optional
        RNG seed for the draws.  Default 0.

    Returns
    -------
    RvBestLinearParams
    """
    t = np.asarray(epochs_mjd, dtype=float)
    d = np.asarray(rv, dtype=float)
    err = np.asarray(rv_err, dtype=float)

    # Value-free validation (no echoed value / identifier).
    if not (t.ndim == 1 and t.shape == d.shape == err.shape):
        raise ValueError(_BLP_NONFINITE_MSG)
    if not (
        np.all(np.isfinite(t))
        and np.all(np.isfinite(d))
        and np.all(np.isfinite(err))
    ):
        raise ValueError(_BLP_NONFINITE_MSG)
    if not (0.0 <= float(ecc) < 1.0):
        raise ValueError(_BLP_ECC_MSG)
    if not (np.isfinite(inflate) and float(inflate) > 0.0):
        raise ValueError(_BLP_INFLATE_MSG)
    if draw is not None and (
        isinstance(draw, bool)
        or not isinstance(draw, (int, np.integer))
        or int(draw) <= 0
    ):
        raise ValueError(_BLP_DRAW_MSG)

    # Build X at the shape; jitter-inflated noise (matches the quick-look).
    # Routed through the pluggable RV builder (delegates to _rv_design_matrix,
    # byte-identical). This is the warm-start seed primitive — called once, NOT
    # in the emcee hot loop, so the builder indirection is free here. Lazy
    # import keeps the engine off the design/ import graph at module scope.
    from orblet.design import RvOrbitBuilder

    period_yr = float(period_days) / DAYS_PER_KEPLER_YEAR
    X = RvOrbitBuilder().build(
        t, period_yr=period_yr, ecc=float(ecc), tau=float(tau),
        epoch_ref_mjd=float(epoch_ref_mjd),
    )
    sigma_eff = np.sqrt(err ** 2 + float(jitter_kms) ** 2)

    sol = linear_solve_rv(d, sigma_eff, X, beta_prior=beta_prior)

    # Optional COVARIANCE-CORRECT seed draws from N(β̂, inflate · cov).  The
    # full off-diagonal cov is used (C–S correlation preserved); drawn with
    # the SAME multivariate_normal(β̂, cov) construction as the marginalised
    # sampler's post-hoc conditional draw (so inflate=1 is that draw).
    draws = None
    if draw is not None:
        rng = np.random.default_rng(int(seed))
        draws = rng.multivariate_normal(
            sol.beta, float(inflate) * sol.cov, size=int(draw),
        )

    return RvBestLinearParams(
        beta=sol.beta,
        cov=sol.cov,
        logL_marginal=sol.logL_marginal,
        log_evidence=sol.log_evidence,
        chi2=sol.chi2,
        dof=sol.dof,
        draws=draws,
    )


# ── Seed composer (RV twin of astro's compose_ti_seed) ─────────────────
#
# Value-free error messages: no input value / source identifier is
# echoed, whatever the caller's data are.
_CRS_EPOCH_MSG = (
    "compose_rv_seed: the quick-look result's _meta epoch_ref_mjd does "
    "not match epoch_ref_mjd; the quick-look's tau is defined at ITS "
    "epoch."
)
_CRS_JITTER_MSG = "compose_rv_seed: jitter_kms must be > 0 when given."
_CRS_KEYS_MSG = (
    "compose_rv_seed: linear_result must be a fit_rv_orbit_linear result "
    "({'chains': ..., 'summary': ...} with P_days, e, tau and _meta)."
)


def compose_rv_seed(
    linear_result: dict,
    prepared_data: dict,
    *,
    epoch_ref_mjd: float,
    jitter_kms: float | None = None,
) -> dict:
    """Compose a physics-space warm-start seed for the full RV sampler.

    The RV twin of
    :func:`orblet.search.compose_ti_seed`:
    the quick-look → full-fit hand-off of the staged-convergence design
    (the linearised quick-look FINDS the solution;
    a full nonlinear sampler
    REFINES it — the full sampler cannot reliably do the finding from a
    cold start).

    OPEN BY CONSTRUCTION: the return value is a plain dict of named,
    unit-suffixed physics quantities.  Inspect it, edit any entry, or
    build the whole dict by hand — the keys are exactly the RV θ-dict
    schema
    (``P_yr``, ``e``, ``omega_rad``, ``tau``, ``K_kms``,
    ``offset_kms``; ``jitter_kms`` optional), plus ``epoch_ref_mjd``.
    **The seed touches the sampler's starting point and NOTHING else**
    — never a prior centre, width, bound or fixed value.
    Initialisation is not a prior; the posterior is unchanged by where
    the chains start (only by where they manage to go).  A seeded fit is
    MODE-CONDITIONAL: every chain started near this one seed explores
    this basin, and the resulting ``r_hat_*`` certifies within-basin
    agreement only — pass a per-chain LIST of rival seeds to probe
    between basins (see the reference manual, §5.4–5.5).

    Conventions
    -----------
    - ω is the PRIMARY-frame argument of periastron (binary-star
      convention, matching :mod:`...rv.forward`); ``K`` is the PRIMARY
      semi-amplitude, km/s.
    - ``P_yr`` is in Keplerian YEARS — the full sampler's physics key
      for the period latent — whereas the quick-look's own period
      prior and :func:`best_linear_params_rv` both take the period in
      DAYS.
    - ``tau`` is the periastron-phase fraction in ``[0, 1)`` at
      ``epoch_ref_mjd``.

    Epoch contract
    ---------------
    ``linear_result["chains"]["_meta"]["epoch_ref_mjd"]`` MUST equal
    ``epoch_ref_mjd`` to within 1e-6 d — the quick-look's τ is defined
    AT its own epoch, and a mismatch would silently shift it.  A
    mismatch RAISES.

    Shape and amplitudes
    --------------------
    The shape ``(P, e, τ)`` is a CENTRAL point of the quick-look
    posterior: medians of ``P`` and ``e``, and the CIRCULAR mean of the
    wrapped phase ``τ`` (a plain median across the 0/1 seam would land
    half an orbit away, silently).  Not the single best draw: on a broad
    ridge that can sit at a prior edge and be rejected by the full fit.
    The quick-look need not be converged for this to be a usable seed —
    a seed is a starting point, not a measurement; the per-chain nudge
    and the optimizer polish take it the rest of the way.  The three
    linear amplitudes ``(γ, K, ω)`` are then recovered from a SINGLE
    closed-form GLS solve at that shape
    (:func:`best_linear_params_rv`), using the quick-look's OWN fixed
    jitter — so the recovered amplitudes are the ones its posterior is
    conditional on.  ``K`` and ``ω`` are read from the SAME solved
    ``beta`` via :func:`recover_K` / :func:`recover_omega` (single
    source of truth — never inline ``hypot`` / ``atan2``).

    Parameters
    ----------
    linear_result : dict
        A :func:`...rv.linear_sampler.fit_rv_orbit_linear` result
        (``{"chains": ..., "summary": ...}``); ``chains`` must carry
        ``P_days``, ``e``, ``tau`` and ``_meta`` (with ``epoch_ref_mjd``
        and ``rv_jitter_kms``).
    prepared_data : dict
        The SAME prepared-RV dict (``epochs_mjd``, ``rv``, ``rv_err``)
        the quick-look and the full fit both consume.
    epoch_ref_mjd : float
        The full fit's reference epoch; must match the quick-look's.
    jitter_kms : float or None, optional
        Explicit jitter start (km/s, > 0).  ``None`` (default) → the
        key is OMITTED (no quick-look source for jitter); each chain
        then starts its jitter from its own prior draw.

    Returns
    -------
    dict
        The physics-space seed described above.
    """
    chains = linear_result.get("chains", {})
    if not all(k in chains for k in ("P_days", "e", "tau", "_meta")):
        raise ValueError(_CRS_KEYS_MSG)
    meta = chains["_meta"]
    if "epoch_ref_mjd" not in meta or "rv_jitter_kms" not in meta:
        raise ValueError(_CRS_KEYS_MSG)
    if not math.isclose(
        float(meta["epoch_ref_mjd"]), float(epoch_ref_mjd),
        rel_tol=0.0, abs_tol=1e-6,
    ):
        raise ValueError(_CRS_EPOCH_MSG)
    if jitter_kms is not None and not (float(jitter_kms) > 0.0):
        raise ValueError(_CRS_JITTER_MSG)

    # Shape = central point of the quick-look posterior: medians of P and
    # e, and the CIRCULAR mean of τ.  τ is a WRAPPED phase in [0, 1), so
    # a plain median of a posterior straddling the 0/1 seam lands half
    # an orbit away, silently (the closed-form amplitudes then fit the
    # wrong phase and nothing raises).
    # Deliberately NOT the single highest-log-posterior draw: on a broad
    # ridge that draw can sit at a prior edge (measured: e ≈ 0.95 on the
    # eccentric fixture, outside the full fit's e prior → the seed is
    # rejected).  A central point is inside support whenever the
    # posterior is; the per-chain nudge and the optimizer polish do the
    # rest — a seed is a starting point, not a measurement.
    # The quick-look chain key ``P_days`` is in DAYS; the closed-form
    # solve below takes DAYS, and the seed theta takes Keplerian YEARS.
    P_days = float(np.median(np.asarray(chains["P_days"], dtype=float)))
    e = float(np.median(np.asarray(chains["e"], dtype=float)))
    ang = 2.0 * np.pi * np.asarray(chains["tau"], dtype=float)
    tau = float(
        np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
        / (2.0 * np.pi)
    ) % 1.0

    # ONE closed-form solve at the quick-look's shape, held to the
    # quick-look's own FIXED jitter — the amplitudes are the ones its
    # posterior is conditional on.
    sol = best_linear_params_rv(
        P_days, e, tau,
        epochs_mjd=prepared_data["epochs_mjd"],
        rv=prepared_data["rv"],
        rv_err=prepared_data["rv_err"],
        epoch_ref_mjd=float(epoch_ref_mjd),
        jitter_kms=float(meta["rv_jitter_kms"]),
    )
    K = recover_K(sol.beta)
    omega = recover_omega(sol.beta)
    gamma = float(sol.beta[0])

    seed = {
        "P_yr": P_days / DAYS_PER_KEPLER_YEAR,
        "e": e,
        "omega_rad": omega,
        "tau": tau,
        "K_kms": K,
        "gamma_kms": gamma,
        "epoch_ref_mjd": float(epoch_ref_mjd),
    }
    if jitter_kms is not None:
        seed["rv_jitter_kms"] = float(jitter_kms)
    return seed
