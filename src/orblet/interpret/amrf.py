"""AMRF triage — the astrometric mass-ratio function.

Pure, post-posterior / catalogue-level layer: numpy only, no chain
access, no engine import, no I/O, no logging.  It answers ONE question
from three catalogue observables plus an EXTERNAL primary mass: *can a
single main-sequence (MS) companion produce this photocentre orbit?*

Formula provenance: Shahaf, Faigler & Mazeh (2019, MNRAS 487, 5610).

The photocentre factor (one source of truth)
--------------------------------------------
Mass ratio ``q ≡ m₂/m₁``; flux ratio ``β ≡ F₂/F₁`` (β = 0: dark
companion).  Semi-major axes in AU: relative ``a_rel``,
primary-about-barycentre ``a₁ = a_rel·q/(1+q)``, photocentre
``a_phot``.  The photocentre sits at the flux-weighted position, so

    a_phot / a_rel = q/(1+q) − β/(1+β)                         (0.1)
    a_phot / a₁    = 1 − β(1+q) / (q(1+β))                     (0.2)

Limits that this module's tests pin: β = 0 ⇒ ``a_phot = a₁`` exactly;
**β = q ⇒ a_phot = 0 for EVERY q** (the general null, of which
β = q = 1 is a single point); β > q ⇒ the photocentre crosses to the
companion's side and the SIGN flips.  The exported ``a_phot_mas`` is a
FOLDED non-negative magnitude, so that sign is not observable — which
is why every comparison below is made on ``|𝒜|``.

The AMRF
--------
Observables: photocentre angular semi-axis ``α ≡ a_phot·ϖ`` [mas] —
i.e. exactly the chain key ``a_phot_mas`` — parallax ``ϖ`` [mas],
period ``P``, and an EXTERNAL primary mass ``m₁`` [M☉] from
photometry / isochrones, never from the orbit fit:

    𝒜 ≡ (α / ϖ) · m₁^{−1/3} · P_yr^{−2/3}                      (1.1)

Kepler III (``a_rel³ = P_yr² m₁(1+q)``, AU / Keplerian-yr / M☉) turns
(0.1) into the model curve

    𝒜(q, β) = q / (1+q)^{2/3} · [ 1 − β(1+q) / (q(1+β)) ]      (1.2)

implemented in the algebraically identical FACTORED form

    𝒜(q, β) = (q − β) / [ (1+q)^{2/3} (1+β) ]                  (1.2f)

which is finite at q → 0 and never divides by a fitted quantity.  𝒜 is
DIMENSIONLESS in the repo's AU / Keplerian-yr / M☉ unit system, whose
small residual offset from the IAU nominal
``GM_sun`` is a property of that unit system and is pinned by
``tests/test_mass_unit_identity.py`` (no value is restated here — the
test owns the number).

Units and conventions
---------------------
- ``a_phot_mas`` — mas.  The FOLDED, non-negative photocentre semi-axis
  (chain key ``a_phot_mas``).  **Never** pass ``a_phot_au``: both exist
  in the chain and they differ by a factor ϖ, which this layer applies
  itself.
- ``parallax_mas`` — mas.  ``ϖ ≤ 0`` (reachable on the marginalised
  linear path) → NaN, never a negative distance.
- ``period_days`` — DAYS at the public surface (the repo-wide period
  convention, chain key ``P_days``), converted ONCE internally to
  Keplerian years by the named constant
  :data:`orblet.constants.DAYS_PER_KEPLER_YEAR`.  The (1.1) exponent
  is in YEARS; feeding years in at the public surface silently rescales
  𝒜 by 365.25^{2/3} ≈ 51.
- ``m1_msun`` — M☉, REQUIRED (no default), EXTERNAL to the fit.
- β is the flux ratio in the band the ASTROMETRY is measured in — for
  Gaia the broad G band, named once as :data:`AMRF_BETA_BAND`.  A β(q)
  relation calibrated in another band, or an SED-derived β prior, must
  be converted into that band before it is used here.

Triage logic and the CLOSED label set
-------------------------------------
On the main sequence β is an increasing function of q with β < q for
q < 1 and β = 1 at q = 1, so (1.2f) collapses to a one-parameter curve
𝒜_MS(q) = 𝒜(q, β(q, m₁)) that vanishes at both ends of the pinned
domain q ∈ (0, 1] and therefore has an interior MAXIMUM 𝒜̂_MS.  Both
boundaries here are computed NUMERICALLY from the CONFIGURED relation —
never a literal copied from a paper.

- ``|𝒜| ≤ 𝒜̂_MS`` — compatible with a single MS companion;
- ``𝒜̂_MS < |𝒜| ≤ 𝒜̂_triple`` — reachable by a companion that is itself
  an unresolved MS PAIR (Shahaf+2019's second boundary): a
  triple-candidate;
- ``|𝒜| > 𝒜̂_triple`` — beyond both MS constructions.  This has THREE
  explanations, not two: a compact object, a hierarchical companion
  beyond the modelled pair, or an inflated-light
  companion on the CROSSED branch (β > q — stripped star, hot
  subdwarf), where ``|𝒜| → (1+q)^{−2/3}`` as β → ∞ and clears an MS
  boundary easily.  Hence the label
  ``compact-or-crossed-branch-candidate``, never "compact object".
- ``inconclusive`` — 𝒜 undefined, or (when posterior draws are
  supplied) a boundary margin smaller than the row's own 𝒜 spread.

The label set is ENUMERATED and CLOSED (:data:`AMRF_LABELS`).  A label
is a §9/§12-style classification statement, NEVER a probability, and
this module never returns one.

Interpretive semantics
----------------------
- **Lower bound.** A q inverted from 𝒜 at β = 0 is a LOWER bound
  on q — and only on the STANDARD branch (β < q).  Any light from the
  companion shrinks the photocentre orbit, so the same 𝒜 is produced by
  a larger q; on the crossed branch (β > q) the inversion is not a
  bound at all.
- **Censoring.** The catalogue is CENSORED near the 𝒜 = 0 (β ≈ q)
  locus: there the photocentre barely moves, the orbit is not detected,
  and the system never enters an astrometric-binary catalogue.  Absence
  of systems at small |𝒜| is a selection effect, not a physical one.
- **fm_ast inheritance.** ``𝒜³·m₁ ≡ fm_ast`` identically (both are
  ``a_phot_au³ / P_yr²`` times the same external mass bookkeeping), so
  this layer inherits EVERY ``fm_ast`` caveat (reference manual,
  §3.8 and §7): the folded non-negative α gives a
  positive low-S/N noise floor that is CUBED, biasing 𝒜³ — and hence
  the triage — TOWARD the compact-object side; parallax errors enter
  three times; screen on detection significance before quoting a label.
- **m₁ is itself β-conditional.** A photometric / isochrone m₁
  assumes the light is the primary's.  At fixed boundaries an
  OVER-estimated m₁ LOWERS 𝒜 (∝ m₁^{−1/3}) and is therefore
  conservative for the triage label; but the boundaries themselves may
  move with m₁ through β(q, m₁), so the NET direction depends on the
  configured locus and is stated only once the default relation is
  verified.  In the inflated regime (β/(1+β) >
  2·m₂/M) the "primary's" spectroscopic mass is likely the SECONDARY's,
  so the m₁ error and the β error compound.

This module persists nothing, emits no logs, and never receives or
returns a source identifier.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR

# ── Conventions and named constants ───────────────────────────────────

#: The photometric band the flux ratio β refers to.  β is the flux ratio
#: in the band the ASTROMETRY is measured in; for Gaia that is the broad
#: G band (design-note H-5).  Any β(q, m₁) relation plugged in here must
#: be calibrated in this band.
AMRF_BETA_BAND: str = "G"

#: Exponent of the zero-order MS mass–luminosity relation ``L ∝ M^p``
#: used by the SHIPPED default β(q) = q^p.
#:
#: **UNVERIFIED — pending paper check.**  Form, coefficient AND band
#: must be checked against Shahaf, Faigler & Mazeh (2019, MNRAS 487,
#: 5610) before any result derived from the default relation is quoted.
#: This is a documented PLACEHOLDER, deliberately the crudest defensible
#: relation, so that a wrong number is obvious rather than plausible.
MS_FLUX_RATIO_EXPONENT: float = 3.5

#: Name of the unresolved-pair model behind the second boundary (see
#: :func:`amrf_unresolved_pair_locus`).
PAIR_MODEL: str = "equal_split_unresolved_ms_pair"

# The MS locus is DEFINED only on q ∈ (0, 1] (design-note M-9): above
# q = 1 the "primary" is the fainter star and the labels swap meaning.
# The grid lower edge is a numerical stand-in for the open end at 0,
# where 𝒜_MS → 0; it is not a physical cut.
_Q_MIN: float = 1.0e-6
_Q_MAX: float = 1.0

# Grid + refinement for the numerical boundary maxima.  The grid locates
# the maximum, a golden-section refinement polishes it inside the
# bracketing cell; 60 golden steps shrink a 1e-4-wide cell far below
# double precision.
_DEFAULT_N_GRID: int = 4001
_GOLDEN_ITERATIONS: int = 60
_INV_PHI: float = (np.sqrt(5.0) - 1.0) / 2.0

# Contract tolerances for a supplied β(q, m₁) relation.
_BETA_AT_UNITY_TOL: float = 1.0e-9
# Continuity is checked as a JUMP detector on the validation grid: a
# relation whose successive-sample change exceeds this Lipschitz bound
# times the grid step has a step discontinuity (a piecewise relation
# switched branch), not a steep slope.  q^3.5 has max slope 3.5.
_CONTINUITY_MAX_SLOPE: float = 100.0
# Convexity (optional check) is a second-difference test: a genuinely
# convex relation's second differences are >= 0 up to floating-point
# noise; this tolerance absorbs that noise without hiding a real dip
# (the auditor's non-convex counterexample fails it by many orders of
# magnitude, not by a hair).
_CONVEXITY_TOL: float = 1.0e-9

# ── The closed label set ──────────────────────────────────────────────

LABEL_MS_COMPATIBLE: str = "MS-companion-compatible"
LABEL_TRIPLE: str = "triple-candidate"
LABEL_COMPACT_OR_CROSSED: str = "compact-or-crossed-branch-candidate"
LABEL_INCONCLUSIVE: str = "inconclusive"

#: The enumerated, CLOSED label set.  A classification never returns
#: anything outside this tuple, and never a probability.
AMRF_LABELS: tuple[str, ...] = (
    LABEL_MS_COMPATIBLE,
    LABEL_TRIPLE,
    LABEL_COMPACT_OR_CROSSED,
    LABEL_INCONCLUSIVE,
)

# numpy string dtype wide enough for the longest label above.
_LABEL_DTYPE = "<U40"

# ── Value-free guard messages ─────────────────────────────────────────
#
# Static module-level constants: no value, no shape, no identifier ever
# reaches a message from this layer.

_BETA_NOT_VECTORISED_MSG = (
    "the beta relation must be vectorised: called with an array of q it "
    "must return an array of the same shape (it received a 1-D grid and "
    "returned something else)."
)
_BETA_NOT_FINITE_MSG = (
    "the beta relation returned a non-finite value on the q domain "
    "(0, 1]; a flux ratio must be finite everywhere on the domain."
)
_BETA_NEGATIVE_MSG = (
    "the beta relation returned a negative flux ratio; beta >= 0 is "
    "required (design-note M-9)."
)
_BETA_AT_UNITY_MSG = (
    "the beta relation must satisfy beta(1, m1) = 1: an equal-mass "
    "main-sequence pair has equal light. Without it the AMRF maximum "
    "over q is not guaranteed to be interior."
)
_BETA_NOT_BELOW_Q_MSG = (
    "the beta relation must satisfy beta(q, m1) < q for q < 1: on the "
    "main sequence the fainter star's light fraction is always below "
    "its mass fraction. Without it the AMRF triage boundary is not an "
    "exclusion."
)
_BETA_DISCONTINUOUS_MSG = (
    "the beta relation is discontinuous on q in (0, 1] (a jump larger "
    "than the allowed Lipschitz bound between adjacent grid samples); a "
    "numerically located maximum would not be trustworthy."
)
_BETA_NOT_CONVEX_MSG = (
    "the beta relation is not numerically convex on q in (0, 1] (a "
    "second difference on the validation grid went negative beyond "
    "tolerance). Convexity, not merely a steep or high-exponent shape, "
    "is the precondition for the unresolved-pair boundary to sit above "
    "the single-MS boundary; triple_amrf_boundary carries its own "
    "runtime guard for a relation that fails this, but this check lets "
    "the property be asserted up front."
)
_N_GRID_TOO_SMALL_MSG = (
    "n_grid must be at least 2: a one-point grid has no step size and "
    "no interior to maximise over."
)
_DRAWS_NOT_MAPPING_MSG = (
    "draws must be a mapping of joint per-draw inputs with keys drawn "
    "from a_phot_mas / parallax_mas / period_days / m1_msun. Independent "
    "per-input sigmas are deliberately NOT accepted: alpha and the "
    "parallax come from the SAME astrometric solution and are "
    "correlated, so quadrature over independent errors understates the "
    "AMRF uncertainty (design-note L-6)."
)
_DRAWS_UNKNOWN_KEY_MSG = (
    "draws contained a key outside a_phot_mas / parallax_mas / "
    "period_days / m1_msun; only joint per-draw inputs for those four "
    "observables are accepted."
)
_DRAWS_SHAPE_MSG = (
    "every supplied draws array must broadcast to rows + (n_draws,): "
    "one dimension per row of the point inputs, plus a trailing draw "
    "axis. A bare (n_draws,) array supplied for a multi-row input "
    "carries no row axis and would otherwise silently broadcast the "
    "draw axis against the row axis instead of raising."
)
_DRAWS_NO_AXIS_MSG = (
    "draws arrays must carry a trailing draw axis; a scalar entry "
    "carries no spread and cannot define a sigma margin."
)
_PLACEHOLDER_BOUNDARY_MSG = (
    "boundary computed from the UNVERIFIED placeholder MS relation "
    "(orblet.interpret.amrf.ms_flux_ratio); form, coefficient and band "
    "remain to be checked against Shahaf, Faigler & Mazeh (2019) before "
    "a label derived from this boundary is quoted."
)
_TRIPLE_BELOW_MS_CLAMPED_MSG = (
    "unresolved-pair boundary fell below the single-MS boundary for the "
    "configured relation; clamped to the single-MS boundary -- the "
    "relation is likely non-convex on q in (0, 1], which breaks the "
    "ordering guarantee triple_amrf_boundary otherwise relies on. "
    "Labels derived from this boundary are unreliable."
)

BetaRelation = Callable[[np.ndarray, float], np.ndarray]


# ── The default (PLACEHOLDER) main-sequence flux-ratio relation ───────


def ms_flux_ratio(q: np.ndarray | float, m1_msun: float) -> np.ndarray:
    """Zero-order main-sequence flux ratio ``β(q, m₁) = q^p`` (G band).

    **UNVERIFIED — pending paper check.**  This is a documented
    PLACEHOLDER, not a calibrated relation: it is the zero-order
    mass–luminosity scaling ``L ∝ M^p`` with
    ``p =`` :data:`MS_FLUX_RATIO_EXPONENT` ``= 3.5``, so that
    ``β = L₂/L₁ = q^p``.  Its functional FORM, its COEFFICIENT and its
    BAND all remain to be checked by the user against Shahaf, Faigler &
    Mazeh (2019, MNRAS 487, 5610) before any number derived from it is
    quoted.  Swapping it is the intended operation: every boundary and
    classification entry point below takes a ``beta_relation`` callable.

    It satisfies the four contract properties
    (:func:`validate_beta_relation`) exactly: ``β(1, m₁) = 1``,
    ``β(q, m₁) < q`` for ``q < 1``, ``β ≥ 0``, and continuity on
    ``q ∈ (0, 1]``.

    Parameters
    ----------
    q : array or float
        Mass ratio ``m₂/m₁`` (dimensionless), on the pinned MS domain
        ``(0, 1]``.
    m1_msun : float
        Primary mass (M☉).  DELIBERATELY IGNORED by this zero-order
        default while the two-argument signature is kept: β depends on
        where the primary sits on the main sequence, not on ``q``
        alone, and a calibrated replacement will use it.

    Returns
    -------
    numpy.ndarray
        Flux ratio ``β = F₂/F₁`` in the :data:`AMRF_BETA_BAND` band
        (dimensionless).  ``NaN`` where ``q < 0`` (unphysical).
    """
    del m1_msun  # see H-4 in the parameter docs: signature, not usage
    q_arr = np.asarray(q, dtype=float)
    with np.errstate(invalid="ignore"):
        beta = q_arr ** MS_FLUX_RATIO_EXPONENT
    return np.where(q_arr >= 0.0, beta, np.nan)


def validate_beta_relation(
    beta_relation: BetaRelation,
    m1_msun: float,
    *,
    n_grid: int = _DEFAULT_N_GRID,
    check_convexity: bool = False,
) -> None:
    """Assert the contract every β(q, m₁) relation must satisfy.

    Checked numerically on a uniform grid over the pinned domain
    ``q ∈ (0, 1]``:

    1. the relation is VECTORISED (array in → same-shape array out);
    2. ``β`` is finite on the domain;
    3. ``β ≥ 0``;
    4. ``β(1, m₁) = 1`` (equal masses ⇒ equal light);
    5. ``β(q, m₁) < q`` for ``q < 1`` (the fainter star's light fraction
       is below its mass fraction);
    6. ``β`` is continuous on the domain — enforced as a jump detector,
       so a steep but continuous relation passes and a piecewise
       relation that switches branch fails.

    Properties 4 and 5 ALONE guarantee that ``𝒜_MS(q)`` vanishes at both
    ends of the domain and therefore has an interior maximum: without
    them the "boundary" would not be an exclusion.  They do NOT
    guarantee the unresolved-pair boundary sits above the single-MS
    boundary — that additionally needs ``β`` CONVEX on ``(0, 1]``
    (see :func:`amrf_unresolved_pair_locus`); ``check_convexity``
    below is an optional, stronger check for a relation the caller
    wants to certify up front rather than rely on
    :func:`triple_amrf_boundary`'s runtime clamp-and-warn guard.

    Parameters
    ----------
    beta_relation : callable
        ``beta_relation(q_array, m1_msun) -> beta_array``.
    m1_msun : float
        Primary mass (M☉) at which the relation is checked.  The
        contract is m₁-conditional because the relation is.
    n_grid : int, optional
        Number of grid samples across ``(0, 1]``.  Must be ``>= 2``.
    check_convexity : bool, optional
        When ``True``, additionally assert that ``β`` is numerically
        CONVEX on the grid (second differences ``>= -tol``).  Default
        ``False``: convexity is not part of the base four-property
        contract, and :func:`triple_amrf_boundary` guards the
        consequence of a non-convex relation at runtime regardless of
        whether this stronger check was requested.

    Raises
    ------
    ValueError
        With a value-free message naming which contract property
        failed.  No input value, shape or identifier appears in it.
    """
    if int(n_grid) < 2:
        raise ValueError(_N_GRID_TOO_SMALL_MSG)
    q = np.linspace(_Q_MIN, _Q_MAX, int(n_grid))
    beta = np.asarray(beta_relation(q, m1_msun), dtype=float)
    if beta.shape != q.shape:
        raise ValueError(_BETA_NOT_VECTORISED_MSG)
    if not np.all(np.isfinite(beta)):
        raise ValueError(_BETA_NOT_FINITE_MSG)
    if np.any(beta < 0.0):
        raise ValueError(_BETA_NEGATIVE_MSG)

    beta_at_one = float(np.asarray(beta_relation(1.0, m1_msun), dtype=float))
    if not abs(beta_at_one - 1.0) <= _BETA_AT_UNITY_TOL:
        raise ValueError(_BETA_AT_UNITY_MSG)

    interior = q < 1.0
    if not np.all(beta[interior] < q[interior]):
        raise ValueError(_BETA_NOT_BELOW_Q_MSG)

    # Jump detector: |Δβ| between adjacent samples, against a Lipschitz
    # bound scaled by the grid step.  A discontinuity survives grid
    # refinement (the jump stays finite while the allowance shrinks);
    # a steep slope does not.
    dq = float(q[1] - q[0])
    if np.any(np.abs(np.diff(beta)) > _CONTINUITY_MAX_SLOPE * dq):
        raise ValueError(_BETA_DISCONTINUOUS_MSG)

    if check_convexity:
        second_diff = np.diff(beta, n=2)
        if np.any(second_diff < -_CONVEXITY_TOL):
            raise ValueError(_BETA_NOT_CONVEX_MSG)


# ── The AMRF: observable and model curve ──────────────────────────────


def amrf(
    a_phot_mas: np.ndarray | float,
    parallax_mas: np.ndarray | float,
    period_days: np.ndarray | float,
    m1_msun: np.ndarray | float,
) -> np.ndarray:
    """Astrometric mass-ratio function 𝒜 from catalogue observables.

    Implements (1.1) of the module docstring:

        𝒜 = (α / ϖ) · m₁^{−1/3} · P_yr^{−2/3}

    with ``α ≡ a_phot·ϖ`` the photocentre angular semi-axis in mas
    (i.e. the ``a_phot_mas`` chain key), so ``α/ϖ`` is ``a_phot`` in AU
    (1 arcsec × 1 pc ≡ 1 AU).  Dimensionless in the repo's
    AU / Keplerian-yr / M☉ unit system.

    Elementwise over broadcast inputs; never raises on unphysical
    input.

    Parameters
    ----------
    a_phot_mas : array or float
        Photocentre angular semi-major axis α (mas) — the FOLDED,
        non-negative magnitude (chain key ``a_phot_mas``).  **Never
        pass the AU-valued photocentre axis**: both live in the chain
        and this function applies the parallax itself.
    parallax_mas : array or float
        Parallax ϖ (mas).  ``ϖ ≤ 0`` → NaN (no physical distance).
    period_days : array or float
        Orbital period in DAYS (chain key ``P_days``), converted once
        internally to Keplerian years by
        :data:`orblet.constants.DAYS_PER_KEPLER_YEAR`.
    m1_msun : array or float
        EXTERNAL primary mass (M☉) — photometry / isochrones, never
        this fit.  Required; ``m₁ ≤ 0`` → NaN.  See the module
        docstring: a photometric m₁ is itself β-conditional, and an
        over-estimated m₁ LOWERS 𝒜.

    Returns
    -------
    numpy.ndarray
        𝒜 (dimensionless), broadcast shape of the inputs.  NaN where
        ``ϖ ≤ 0``, ``P ≤ 0``, ``m₁ ≤ 0``, or any input is NaN
        (NaN-in → NaN-out).  A COUNTED report of those screens is
        available from :func:`classify_amrf` (``_meta``); this bare
        transform returns the array only.

    Notes
    -----
    ``𝒜³·m₁ ≡ fm_ast`` identically, so every ``fm_ast`` caveat (reference
    manual, §3.8 and §7) applies here unchanged — in
    particular the CUBED low-S/N noise floor of the folded α, which
    biases the triage toward the compact-object side.
    """
    alpha, plx, p_days, m1 = np.broadcast_arrays(
        np.asarray(a_phot_mas, dtype=float),
        np.asarray(parallax_mas, dtype=float),
        np.asarray(period_days, dtype=float),
        np.asarray(m1_msun, dtype=float),
    )
    # ONE conversion, by the named constant: the (1.1) exponent is in
    # Keplerian years while the public surface takes days.
    p_yr = p_days / DAYS_PER_KEPLER_YEAR
    physical = (plx > 0.0) & (p_yr > 0.0) & (m1 > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (alpha / plx) * m1 ** (-1.0 / 3.0) * p_yr ** (-2.0 / 3.0)
    return np.where(physical, raw, np.nan)


def amrf_curve(
    q: np.ndarray | float, beta: np.ndarray | float
) -> np.ndarray:
    """Model AMRF curve 𝒜(q, β) — the FACTORED form (1.2f).

        𝒜(q, β) = (q − β) / [ (1+q)^{2/3} (1+β) ]

    Algebraically identical to the provenance identity

        𝒜(q, β) = q/(1+q)^{2/3} · [ 1 − β(1+q)/(q(1+β)) ]      (1.2)

    but finite at ``q → 0`` and free of a division by a fitted
    quantity.

    SIGNED by construction, and that sign is physical: ``β > q`` puts
    the photocentre on the COMPANION's side of the barycentre.  The
    observable ``a_phot_mas`` is a folded non-negative magnitude, so
    the sign is not measurable and every triage comparison is made on
    ``|𝒜|``.

    Limits pinned by the tests: ``𝒜(q, 0) = q(1+q)^{−2/3}`` (dark
    companion, monotone increasing in q); ``𝒜(q, q) = 0`` for EVERY q
    (the general null — the photocentre does not move when light and
    mass are shared in the same proportion); ``|𝒜| → (1+q)^{−2/3}`` as
    ``β → ∞`` (the crossed branch's ceiling).

    Parameters
    ----------
    q : array or float
        Mass ratio ``m₂/m₁`` (dimensionless), assumed ``≥ 0``.
    beta : array or float
        Flux ratio ``F₂/F₁`` in the :data:`AMRF_BETA_BAND` band
        (dimensionless), assumed ``≥ 0``.

    Returns
    -------
    numpy.ndarray
        Signed 𝒜 (dimensionless), broadcast shape of the inputs.  The
        function is a pure formula: it does not screen its domain, and
        NaN inputs propagate.
    """
    q_arr = np.asarray(q, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    return (q_arr - beta_arr) / (
        (1.0 + q_arr) ** (2.0 / 3.0) * (1.0 + beta_arr)
    )


def _on_ms_domain(q_arr: np.ndarray) -> np.ndarray:
    """Boolean mask of the pinned MS domain ``q ∈ (0, 1]`` (M-9)."""
    return (q_arr > 0.0) & (q_arr <= _Q_MAX)


def amrf_ms_locus(
    q: np.ndarray | float,
    m1_msun: float,
    *,
    beta_relation: BetaRelation = ms_flux_ratio,
) -> np.ndarray:
    """𝒜 along the main-sequence locus: ``𝒜(q, β(q, m₁))``.

    Collapses the two-parameter curve (1.2f) to one parameter by
    substituting a configurable mass–luminosity relation.  The
    ``beta_relation`` callable is the injection point — the shipped
    default :func:`ms_flux_ratio` is an UNVERIFIED placeholder.

    Parameters
    ----------
    q : array or float
        Mass ratio ``m₂/m₁``.  Values outside the PINNED domain
        ``(0, 1]`` return NaN: above ``q = 1`` the MS locus is
        undefined here because the labels swap meaning.
    m1_msun : float
        Primary mass (M☉), passed through to the relation.  The
        two-argument relation signature is deliberate even when the
        default ignores it.
    beta_relation : callable, optional
        ``beta_relation(q_array, m1_msun) -> beta_array``, calibrated
        in the :data:`AMRF_BETA_BAND` band.  NOT validated here — this
        is the maximiser's hot path; call
        :func:`validate_beta_relation` once (the boundary and
        classification entry points do).

    Returns
    -------
    numpy.ndarray
        𝒜 on the MS locus (dimensionless), NaN off the domain.
    """
    q_arr = np.asarray(q, dtype=float)
    beta = np.asarray(beta_relation(q_arr, m1_msun), dtype=float)
    return np.where(_on_ms_domain(q_arr), amrf_curve(q_arr, beta), np.nan)


def amrf_unresolved_pair_locus(
    q: np.ndarray | float,
    m1_msun: float,
    *,
    beta_relation: BetaRelation = ms_flux_ratio,
) -> np.ndarray:
    """𝒜 for a companion that is itself an unresolved EQUAL-MASS MS pair.

    MODEL CHOICE, stated explicitly because the unresolved-pair model
    is a modelling assumption, not an observable.  The companion of
    total mass ``m₂ = q·m₁`` is modelled as TWO main-sequence stars of
    ``m₂/2`` each, unresolved by Gaia and close enough that the pair's
    internal orbit is not the signal.  Consequences:

    - each pair member sits on the SAME supplied β relation at mass
      ratio ``q/2``, so the companion's total light is
      ``β_pair = 2·β(q/2, m₁)``;
    - an equal-mass pair's light centroid coincides with its own
      barycentre, so the pair enters (0.1) exactly as a single point of
      mass ``q·m₁`` and flux ``β_pair`` — no extra photocentre term;
    - the equal split IS the split that MAXIMISES ``|𝒜|`` at fixed
      total companion mass whenever the mass–luminosity relation is
      CONVEX on ``q ∈ (0, 1]`` — that is the true precondition, NOT
      merely "exponent > 1"; the shipped default ``q^3.5`` happens to
      be convex, which is why exponent > 1 has looked like a sufficient
      proxy for it here.  A convex ``L(M)`` summed at fixed total mass
      is MINIMISED by the equal split, minimum companion light gives
      the largest photocentre wobble, hence the largest 𝒜.  So "equal
      split" and "maximised over the pair split" coincide here rather
      than conflicting; with a non-convex relation they need not, and
      this function still evaluates the EQUAL split — check convexity
      (:func:`validate_beta_relation`'s ``check_convexity``) before
      swapping in a replacement relation.
    - the degenerate split (all mass in one member) reduces the pair
      locus to the single-star locus only in that limit; it does NOT
      by itself guarantee the pair boundary sits above the MS boundary
      for a NON-convex relation.  :func:`triple_amrf_boundary` carries
      its own runtime ordering guard — it CLAMPS the pair boundary to
      the MS boundary (and warns) whenever the numerically computed
      pair boundary comes out below it — so a non-convex replacement
      relation cannot silently invert the two boundaries.

    Parameters
    ----------
    q : array or float
        TOTAL companion mass ratio ``m₂/m₁``, pinned to ``(0, 1]`` as
        for the single-star locus; NaN off the domain.
    m1_msun : float
        Primary mass (M☉).
    beta_relation : callable, optional
        As for :func:`amrf_ms_locus`.  Evaluated at ``q/2``, which is
        inside ``(0, 1]`` whenever ``q`` is.

    Returns
    -------
    numpy.ndarray
        𝒜 along the unresolved-pair locus (dimensionless).
    """
    q_arr = np.asarray(q, dtype=float)
    # Two members of mass ratio q/2 each; their light ADDS (unresolved).
    beta_pair = 2.0 * np.asarray(
        beta_relation(0.5 * q_arr, m1_msun), dtype=float
    )
    return np.where(
        _on_ms_domain(q_arr), amrf_curve(q_arr, beta_pair), np.nan
    )


# ── Numerically computed boundaries ───────────────────────────────────


def _max_abs_on_domain(locus, n_grid: int) -> float:
    """Maximum of ``|locus(q)|`` over ``q ∈ (0, 1]``, numerically.

    Coarse grid scan to locate the cell, golden-section refinement
    inside the bracketing cell.  Boundary maxima are handled (the
    bracket is clamped to the domain), which the unresolved-pair locus
    needs: unlike the single-star locus it does NOT vanish at ``q = 1``.
    Returns a float; never a literal from a paper.
    """
    q_grid = np.linspace(_Q_MIN, _Q_MAX, int(n_grid))
    values = np.abs(np.asarray(locus(q_grid), dtype=float))
    best = int(np.nanargmax(values))
    grid_max = float(values[best])

    lo = float(q_grid[max(best - 1, 0)])
    hi = float(q_grid[min(best + 1, int(n_grid) - 1)])

    def scalar_abs(x: float) -> float:
        return float(np.abs(np.asarray(locus(x), dtype=float)))

    # Golden-section MAXIMISATION on [lo, hi].
    c = hi - _INV_PHI * (hi - lo)
    d = lo + _INV_PHI * (hi - lo)
    f_c, f_d = scalar_abs(c), scalar_abs(d)
    for _ in range(_GOLDEN_ITERATIONS):
        if f_c > f_d:
            hi, d, f_d = d, c, f_c
            c = hi - _INV_PHI * (hi - lo)
            f_c = scalar_abs(c)
        else:
            lo, c, f_c = c, d, f_d
            d = lo + _INV_PHI * (hi - lo)
            f_d = scalar_abs(d)

    # The refinement can never do worse than the grid it started from.
    return max(grid_max, f_c, f_d)


def ms_amrf_boundary(
    m1_msun: float,
    *,
    beta_relation: BetaRelation = ms_flux_ratio,
    n_grid: int = _DEFAULT_N_GRID,
    validate: bool = True,
) -> float:
    """𝒜̂_MS — the largest 𝒜 a single MS companion can produce.

    ``𝒜̂_MS = max_{q ∈ (0, 1]} |𝒜(q, β(q, m₁))|``, computed NUMERICALLY
    from the CONFIGURED relation — never a literal threshold from a
    paper.  The relation's contract is validated first
    (:func:`validate_beta_relation`), and that is what guarantees the
    maximum exists and that exceeding it is an EXCLUSION: on the main
    sequence ``β < q`` always.  When the relation in use is the shipped
    :func:`ms_flux_ratio` placeholder, a value-free ``UserWarning`` is
    emitted: that relation's form, coefficient and band are UNVERIFIED.

    Parameters
    ----------
    m1_msun : float
        EXTERNAL primary mass (M☉).  The boundary is m₁-conditional
        whenever the relation is.
    beta_relation : callable, optional
        β(q, m₁) in the :data:`AMRF_BETA_BAND` band.
    n_grid : int, optional
        Grid resolution for the maximisation.
    validate : bool, optional
        When ``True`` (default), run :func:`validate_beta_relation`
        first.  Callers that already validated the SAME relation once
        (e.g. :func:`classify_amrf`'s per-m₁ loop) may pass ``False``
        to skip the repeated grid evaluation; the placeholder warning
        above still fires regardless of this flag.

    Returns
    -------
    float
        𝒜̂_MS (dimensionless).

    Raises
    ------
    ValueError
        If the supplied relation violates the contract (value-free
        message).

    Warns
    -----
    UserWarning
        Value-free, when ``beta_relation`` is the shipped UNVERIFIED
        placeholder.
    """
    if beta_relation is ms_flux_ratio:
        warnings.warn(_PLACEHOLDER_BOUNDARY_MSG, UserWarning, stacklevel=2)
    if validate:
        validate_beta_relation(beta_relation, m1_msun, n_grid=n_grid)
    return _max_abs_on_domain(
        lambda q: amrf_ms_locus(q, m1_msun, beta_relation=beta_relation),
        n_grid,
    )


def triple_amrf_boundary(
    m1_msun: float,
    *,
    beta_relation: BetaRelation = ms_flux_ratio,
    n_grid: int = _DEFAULT_N_GRID,
    validate: bool = True,
) -> float:
    """𝒜̂_triple — the largest 𝒜 an unresolved MS PAIR companion gives.

    ``𝒜̂_triple = max_{q ∈ (0, 1]} |𝒜(q, 2·β(q/2, m₁))|``, computed
    numerically from the configured relation under the equal-split
    unresolved-pair model documented at
    :func:`amrf_unresolved_pair_locus` (:data:`PAIR_MODEL`).  It is the
    boundary that separates triple-candidates from the rest.

    **Ordering guard.**  The boundary is never below
    :func:`ms_amrf_boundary` ONLY when the configured relation is
    CONVEX on ``q ∈ (0, 1]`` (see :func:`amrf_unresolved_pair_locus`).
    This function computes the MS boundary too (same relation, m₁ and
    grid) and, if the numerically located pair boundary comes out
    BELOW it — the signature of a non-convex relation — CLAMPS the
    result to the MS boundary and emits a value-free ``UserWarning``.
    Labels derived from a clamped boundary are unreliable: the clamp
    only prevents an inverted (and therefore nonsensical) exclusion
    window, it does not recover the intended triple boundary.  When
    the relation in use is the shipped :func:`ms_flux_ratio`
    placeholder, a SEPARATE value-free ``UserWarning`` is also emitted
    for the same UNVERIFIED-relation reason as :func:`ms_amrf_boundary`.

    Parameters
    ----------
    m1_msun : float
        EXTERNAL primary mass (M☉).
    beta_relation : callable, optional
        β(q, m₁) in the :data:`AMRF_BETA_BAND` band, evaluated at
        ``q/2`` for each pair member.
    n_grid : int, optional
        Grid resolution for the maximisation.
    validate : bool, optional
        When ``True`` (default), run :func:`validate_beta_relation`
        first.  See :func:`ms_amrf_boundary` for the intended caller
        (the per-m₁ loop in :func:`classify_amrf`) that passes
        ``False``.  The ordering guard and the placeholder warning
        both still run regardless of this flag.

    Returns
    -------
    float
        𝒜̂_triple (dimensionless); clamped to 𝒜̂_MS when the numerical
        maximisation would otherwise invert the two boundaries.

    Raises
    ------
    ValueError
        If the supplied relation violates the contract (value-free
        message).

    Warns
    -----
    UserWarning
        Value-free: once when ``beta_relation`` is the shipped
        UNVERIFIED placeholder, and again if the ordering guard had to
        clamp the boundary.
    """
    if beta_relation is ms_flux_ratio:
        warnings.warn(_PLACEHOLDER_BOUNDARY_MSG, UserWarning, stacklevel=2)
    if validate:
        validate_beta_relation(beta_relation, m1_msun, n_grid=n_grid)
    triple = _max_abs_on_domain(
        lambda q: amrf_unresolved_pair_locus(
            q, m1_msun, beta_relation=beta_relation
        ),
        n_grid,
    )
    # Ordering guard (design-note M-1 rider): the pair boundary is only
    # PROVABLY >= the single-MS boundary when beta(q) is convex on
    # (0, 1] (see amrf_unresolved_pair_locus).  A non-convex relation
    # can invert the two numerically; treating that inverted gap as a
    # real exclusion window would mislabel MS-compatible rows as
    # triple/compact candidates, so it is clamped instead.
    ms = _max_abs_on_domain(
        lambda q: amrf_ms_locus(q, m1_msun, beta_relation=beta_relation),
        n_grid,
    )
    if triple < ms:
        warnings.warn(_TRIPLE_BELOW_MS_CLAMPED_MSG, UserWarning, stacklevel=2)
        return ms
    return triple


# ── Classification ────────────────────────────────────────────────────

_DRAW_KEYS = ("a_phot_mas", "parallax_mas", "period_days", "m1_msun")


def _boundaries_per_row(
    m1_flat: np.ndarray,
    beta_relation: BetaRelation,
    n_grid: int,
    *,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (𝒜̂_MS, 𝒜̂_triple), computed once per UNIQUE m₁.

    The boundaries depend on m₁ only through β(q, m₁), so rows sharing
    an m₁ share a boundary; the default relation makes them all equal.
    Rows with a non-finite or non-positive m₁ get NaN boundaries.

    ``validate`` is forwarded to :func:`ms_amrf_boundary` /
    :func:`triple_amrf_boundary` unchanged.  :func:`classify_amrf`
    passes ``False`` here after validating the relation ONCE itself, so
    this loop does not re-run the full contract grid evaluation for
    every unique m₁ (A8 performance note); the ordering guard and the
    placeholder warning in :func:`triple_amrf_boundary` still run
    either way.
    """
    ms = np.full(m1_flat.shape, np.nan, dtype=float)
    triple = np.full(m1_flat.shape, np.nan, dtype=float)
    usable = np.isfinite(m1_flat) & (m1_flat > 0.0)
    for value in np.unique(m1_flat[usable]):
        selected = usable & (m1_flat == value)
        ms[selected] = ms_amrf_boundary(
            float(value), beta_relation=beta_relation, n_grid=n_grid,
            validate=validate,
        )
        triple[selected] = triple_amrf_boundary(
            float(value), beta_relation=beta_relation, n_grid=n_grid,
            validate=validate,
        )
    return ms, triple


def _amrf_draw_sigma(
    draws,
    point_inputs: tuple[np.ndarray, ...],
    rows_shape: tuple[int, ...],
) -> np.ndarray:
    """Per-row σ(𝒜) from JOINT posterior/uncertainty draws.

    ``draws`` is a mapping over a subset of
    ``(a_phot_mas, parallax_mas, period_days, m1_msun)``; every entry
    must broadcast to ``rows_shape + (n_draws,)``.  Keys not supplied
    are held at their point value.  Joint draws are REQUIRED rather
    than independent per-input sigmas because α and ϖ come from the
    same astrometric solution and are correlated (design-note L-6).

    Each supplied array's ``ndim`` must equal ``len(rows_shape) + 1``
    exactly (one dimension per row axis, plus the trailing draw axis):
    for a multi-row call a bare ``(n_draws,)`` array carries no row
    axis and would otherwise silently broadcast the draw axis against
    the row axis (:data:`_DRAWS_SHAPE_MSG`); the scalar-row case
    (``rows_shape == ()``, so a plain 1-D ``(n_draws,)`` array) keeps
    working exactly as before.
    """
    if not hasattr(draws, "keys") or not hasattr(draws, "__getitem__"):
        raise ValueError(_DRAWS_NOT_MAPPING_MSG)
    if any(key not in _DRAW_KEYS for key in draws.keys()):
        raise ValueError(_DRAWS_UNKNOWN_KEY_MSG)

    supplied = {
        key: np.asarray(draws[key], dtype=float)
        for key in _DRAW_KEYS
        if key in draws
    }
    if not supplied:
        raise ValueError(_DRAWS_NOT_MAPPING_MSG)
    if any(arr.ndim == 0 for arr in supplied.values()):
        raise ValueError(_DRAWS_NO_AXIS_MSG)
    expected_ndim = len(rows_shape) + 1
    if any(arr.ndim != expected_ndim for arr in supplied.values()):
        raise ValueError(_DRAWS_SHAPE_MSG)

    # Rows on the leading axes, draws on the trailing axis: the point
    # values are broadcast against a length-1 draw axis.
    columns = []
    for key, point in zip(_DRAW_KEYS, point_inputs):
        if key in supplied:
            columns.append(supplied[key])
        else:
            columns.append(point.reshape(rows_shape + (1,)))
    try:
        broadcast = np.broadcast_arrays(*columns)
    except ValueError:
        raise ValueError(_DRAWS_SHAPE_MSG) from None

    amrf_draws = amrf(*broadcast)
    # Collapse leading axes to rows; the trailing axis holds the draws.
    n_draws = amrf_draws.shape[-1]
    flat = amrf_draws.reshape(-1, n_draws)
    if flat.shape[0] != int(np.prod(rows_shape, dtype=int)):
        raise ValueError(_DRAWS_SHAPE_MSG)

    # amrf_sigma is the per-row population std (np.nanstd, ddof=0) over
    # the finite draws of that row.  A row with exactly ONE finite draw
    # therefore gets sigma = 0.0 (not NaN): its sigma-margins below
    # divide by zero and go to +/-inf rather than NaN, so such a row is
    # NEVER marked inconclusive by the near-boundary sigma check.
    sigma = np.full(flat.shape[0], np.nan, dtype=float)
    have = np.sum(np.isfinite(flat), axis=1) > 0
    if np.any(have):
        sigma[have] = np.nanstd(flat[have], axis=1)
    return sigma


def classify_amrf(
    a_phot_mas: np.ndarray | float,
    parallax_mas: np.ndarray | float,
    period_days: np.ndarray | float,
    m1_msun: np.ndarray | float,
    *,
    beta_relation: BetaRelation = ms_flux_ratio,
    draws=None,
    sigma_threshold: float = 1.0,
    n_grid: int = _DEFAULT_N_GRID,
) -> dict:
    """Label each row against the two numerically computed boundaries.

    Returns a LABEL from the closed set :data:`AMRF_LABELS` plus the
    MARGIN to both boundaries — **never a probability**.  The label is
    a statement about which main-sequence constructions are excluded;
    it is not a detection, not a compact-object
    confirmation, and not a posterior weight.

    All comparisons are on ``|𝒜|``: the observable α is a folded
    non-negative magnitude, so the sign of 𝒜 — the side of
    the barycentre the photocentre sits on — is not measurable.

    **Point labels are allowed but are UNCERTAINTY-FREE.**  With
    ``draws=None`` a row sitting a hair above 𝒜̂_MS is labelled
    ``triple-candidate`` with no statement of significance whatsoever.
    Supply ``draws`` to get the margins in σ and to let near-boundary
    rows fall to ``inconclusive``.  Independent per-input σ's are
    deliberately NOT accepted: α and ϖ come from the SAME astrometric
    solution and are correlated, so quadrature over independent errors
    understates σ(𝒜).

    Parameters
    ----------
    a_phot_mas, parallax_mas, period_days, m1_msun : array or float
        As for :func:`amrf` — mas, mas, DAYS, M☉.  ``m1_msun`` is
        EXTERNAL and required; see the module docstring for its
        β-conditionality.
    beta_relation : callable, optional
        β(q, m₁) in the :data:`AMRF_BETA_BAND` band; the contract is
        validated ONCE (at one representative usable m₁, not once per
        unique m₁) rather than inside the per-m₁ boundary loop.  The
        shipped default :func:`ms_flux_ratio` is an UNVERIFIED
        placeholder.
    draws : mapping, optional
        JOINT per-draw inputs, keys from ``a_phot_mas`` /
        ``parallax_mas`` / ``period_days`` / ``m1_msun``, each
        broadcastable to ``rows + (n_draws,)``.  Keys omitted are held
        at their point value.  Used only to report margins in σ and to
        mark near-boundary rows inconclusive.
    sigma_threshold : float, optional
        A row whose margin to EITHER boundary is smaller than
        ``sigma_threshold × σ(𝒜)`` is labelled ``inconclusive``.
        Default 1.0.  Ignored when ``draws`` is None.
    n_grid : int, optional
        Grid resolution for the boundary maximisation.

    Returns
    -------
    dict
        ``{"amrf", "label", "boundary_ms", "boundary_triple",
        "margin_ms", "margin_triple", "amrf_sigma", "margin_ms_sigma",
        "margin_triple_sigma", "_meta"}``.  Arrays carry the broadcast
        shape of the inputs.  Margins are ``|𝒜| − boundary``, so
        POSITIVE means the boundary is exceeded (the MS construction is
        excluded).  ``amrf_sigma`` is ``np.nanstd`` (population,
        ``ddof=0``) of the per-row 𝒜 draws; a row with exactly ONE
        finite draw therefore has ``amrf_sigma == 0`` (never NaN), which
        sends that row's σ-margins to ``+/-inf`` rather than leaving it
        ``inconclusive``.  σ-margins are the margin divided by σ(𝒜), and
        are NaN without ``draws``.  ``_meta`` carries the conventions
        and value-free COUNTS only: ``n_rows``; ``n_amrf_nan`` (rows
        where 𝒜 is non-finite); ``n_nonpositive_parallax`` (rows whose
        parallax is non-finite OR non-positive — a NaN parallax fails
        ``> 0`` and is counted here too, not only in
        ``n_nonfinite_input``); ``n_nonfinite_input`` (rows where any of
        the four inputs is non-finite); plus ``beta_band``,
        ``pair_model``, ``q_domain``, ``labels``, ``uncertainty_mode``,
        ``sigma_threshold``, ``beta_relation_name``, and
        ``beta_relation_is_shipped_placeholder`` (``True`` while the
        UNVERIFIED default relation is in use).

    Raises
    ------
    ValueError
        If the β relation violates its contract, ``n_grid < 2``, or
        ``draws`` is not a mapping of joint per-draw inputs broadcastable
        to ``rows + (n_draws,)`` (value-free messages).

    Warns
    -----
    UserWarning
        Value-free, forwarded from the per-m₁ boundary computation: once
        when ``beta_relation`` is the shipped UNVERIFIED placeholder,
        and again for any m₁ whose numerically computed triple boundary
        had to be clamped to the MS boundary by the ordering guard.
    """
    alpha, plx, p_days, m1 = np.broadcast_arrays(
        np.asarray(a_phot_mas, dtype=float),
        np.asarray(parallax_mas, dtype=float),
        np.asarray(period_days, dtype=float),
        np.asarray(m1_msun, dtype=float),
    )
    rows_shape = alpha.shape
    amrf_values = amrf(alpha, plx, p_days, m1)

    # Work on a flat row axis: 0-d (scalar) inputs ravel to length 1 and
    # are reshaped back at the end, so masked assignment stays legal.
    flat_amrf = np.ravel(amrf_values)
    flat_m1 = np.ravel(m1)

    # A8 performance: validate the relation's contract ONCE here (at one
    # representative usable m1) rather than once per unique m1 inside
    # the boundary loop below.  The placeholder warning and the runtime
    # ordering guard in triple_amrf_boundary still run per m1 regardless
    # of this flag -- only the (expensive) grid re-evaluation of the
    # four-property contract is skipped in the loop.
    usable_m1 = flat_m1[np.isfinite(flat_m1) & (flat_m1 > 0.0)]
    if usable_m1.size > 0:
        validate_beta_relation(
            beta_relation, float(usable_m1[0]), n_grid=n_grid,
        )
    boundary_ms, boundary_triple = _boundaries_per_row(
        flat_m1, beta_relation, n_grid, validate=False,
    )

    abs_amrf = np.abs(flat_amrf)
    margin_ms = abs_amrf - boundary_ms
    margin_triple = abs_amrf - boundary_triple

    if draws is None:
        sigma = np.full(flat_amrf.shape, np.nan, dtype=float)
    else:
        sigma = _amrf_draw_sigma(draws, (alpha, plx, p_days, m1), rows_shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        margin_ms_sigma = margin_ms / sigma
        margin_triple_sigma = margin_triple / sigma

    # A3: mutually EXCLUSIVE conditions evaluated once (np.select), so no
    # later assignment can silently overwrite an earlier one.  In
    # particular cond_compact REQUIRES margin_ms > 0.0 explicitly: a row
    # with margin_ms <= 0.0 can never land in cond_triple or cond_compact,
    # even if a non-convex (mis-ordered) boundary pair were ever to slip
    # through the triple_amrf_boundary clamp (belt-and-suspenders on top
    # of that runtime guard).
    decidable = np.isfinite(margin_ms) & np.isfinite(margin_triple)
    cond_ms = decidable & (margin_ms <= 0.0)
    cond_triple = decidable & (margin_ms > 0.0) & (margin_triple <= 0.0)
    cond_compact = decidable & (margin_ms > 0.0) & (margin_triple > 0.0)
    label = np.select(
        [cond_ms, cond_triple, cond_compact],
        [LABEL_MS_COMPATIBLE, LABEL_TRIPLE, LABEL_COMPACT_OR_CROSSED],
        default=LABEL_INCONCLUSIVE,
    ).astype(_LABEL_DTYPE)

    if draws is not None:
        # A margin smaller than the row's own AMRF spread does not
        # separate the labels: say so rather than pick a side.
        near = np.isfinite(sigma) & (
            (np.abs(margin_ms) < sigma_threshold * sigma)
            | (np.abs(margin_triple) < sigma_threshold * sigma)
        )
        label[decidable & near] = LABEL_INCONCLUSIVE

    meta = {
        "beta_band": AMRF_BETA_BAND,
        "beta_relation_name": getattr(
            beta_relation, "__name__", type(beta_relation).__name__
        ),
        # True when the UNVERIFIED shipped placeholder is in use; a
        # swapped-in relation is not thereby "verified", so the flag
        # names the placeholder rather than claiming verification.
        "beta_relation_is_shipped_placeholder": (
            beta_relation is ms_flux_ratio
        ),
        "pair_model": PAIR_MODEL,
        "q_domain": (0.0, _Q_MAX),
        "labels": AMRF_LABELS,
        "uncertainty_mode": "none" if draws is None else "draws",
        "sigma_threshold": float(sigma_threshold),
        "n_rows": int(flat_amrf.size),
        "n_amrf_nan": int(np.sum(~np.isfinite(flat_amrf))),
        "n_nonpositive_parallax": int(np.sum(~(np.ravel(plx) > 0.0))),
        "n_nonfinite_input": int(
            np.sum(
                ~(
                    np.isfinite(np.ravel(alpha))
                    & np.isfinite(np.ravel(plx))
                    & np.isfinite(np.ravel(p_days))
                    & np.isfinite(flat_m1)
                )
            )
        ),
    }

    return {
        "amrf": amrf_values,
        "label": label.reshape(rows_shape),
        "boundary_ms": boundary_ms.reshape(rows_shape),
        "boundary_triple": boundary_triple.reshape(rows_shape),
        "margin_ms": margin_ms.reshape(rows_shape),
        "margin_triple": margin_triple.reshape(rows_shape),
        "amrf_sigma": sigma.reshape(rows_shape),
        "margin_ms_sigma": margin_ms_sigma.reshape(rows_shape),
        "margin_triple_sigma": margin_triple_sigma.reshape(rows_shape),
        "_meta": meta,
    }
