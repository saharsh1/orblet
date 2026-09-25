"""Engine-agnostic post-fit companion-mass layer (RV and astrometric).

This module inverts a **mass function** for a candidate companion mass
``m2``, given an *external* primary mass ``m1``.  Two entry points, one
shared solver:

- :func:`companion_mass_from_rv_posterior` — spectroscopic ``fm`` from
  an SB1 RV chain, plus an assumed / marginalised ``sin i``;
- :func:`companion_mass_from_astrometric_posterior` — astrometric
  ``fm_ast_msun`` from an observables-first Thiele-Innes chain.  NO ``sin i``
  enters: the amplitudes measured the inclination, so ``fm_ast`` is
  projection-free by construction, and passing ``sin_i = 1`` into the
  shared solver is exact — it simply selects the projection-free form
  ``fm_ast = m2^3 / (m1 + m2)^2`` — not an edge-on assumption.

Mass-function relations (the one equation, two readings)
--------------------------------------------------------
    RV:     fm     = (m2 · sin i)^3 / (m1 + m2)^2     [M_sun]
    astro:  fm_ast =  m2^3 / (m1 + m2)^2              [M_sun, β = 0]

Both arrive pre-reduced in solar masses (no ``G`` / ``MSUN`` at this
layer): ``fm`` absorbs ``G`` through ``P K^3 (1 - e^2)^{3/2} / (2πG)``;
``fm_ast = a_phot^3 / P^2`` with ``a_phot`` in AU and ``P`` in years is
a total mass by Kepler III, scaled by the cubed photocentre fraction.

The astrometric reading is conditional on a DARK companion (β = 0).
With light, ``fm_ast = |m2/M − β/(1+β)|^3 · M`` — one equation in THREE
quantities (m1, m2, β): supply any two, infer the third.  This module
implements the β = 0 corner; the ``beta`` keyword on the astrometric
adapter is RESERVED (raises until the β > 0 inversion lands), and
the m1 + m2 → β mode (the luminous-companion consistency test) is a
planned SIBLING function, not an argument here.

Ensemble mechanics (both adapters)
----------------------------------
Monte-Carlo error propagation by draw pairing: posterior draw ``k`` of
the mass function is inverted with draw ``k`` of the external inputs.
Pairing independent draws samples the joint distribution exactly, so
the output ``m2`` array is a posterior sample carrying BOTH the
measurement spread and the input-ignorance spread, with within-chain
correlations (m2 vs P, e, ...) preserved row by row.  External draws
come from dedicated seeded RNG streams (offsets below), reproducible
and decorrelated from the samplers' own streams.

Relation to ``companion_mass_bracket`` (astrometric_upper_limit.py)
-------------------------------------------------------------------
``companion_mass_bracket`` REUSES :func:`solve_companion_mass` to turn
two point estimates (a measured ``a1·sin i`` and a non-detection upper
limit) into a scalar ``[M2_min, M2_max]`` bracket — a quick-look bound.
The adapters here answer the calibrated-posterior version of the same
question, draw by draw.  Same solver, same conventions, different
granularity; neither re-derives the other's formula.

The ``M_total`` formed here is literally ``m1 + m2`` AFTER inverting a
mass function — never Kepler III on a photocentre axis with a sampled
mass, which would be a prior-driven construction.

Scientific conventions (see the reference manual, §6.1)
-------------------------------------------------------
- Only the mass function is MEASURED; ``m1`` is an ASSUMED input on
  BOTH paths, and ``sin i`` is an assumed input on the RV path ONLY.
- RV: ``sin_i = 1`` (default) yields the MINIMUM companion mass
  (:data:`orblet.constants.MASS_CONVENTION_PROJECTED`); a
  CALLER-SUPPLIED ``sin i`` (one number, or one value per draw) yields
  the mass for that ``sin i``
  (:data:`orblet.constants.MASS_CONVENTION_TRUE`).  Two modes only:
  this layer never draws an inclination itself.  A caller who wants an
  orientation-averaged mass draws ``sin i`` themselves (ISOTROPIC, i.e.
  uniform in ``cos i``, NOT uniform in ``sin i``) and passes the array.
- Astrometric: the sentinel is
  :data:`orblet.constants.MASS_CONVENTION_ASTROMETRIC_EXTERNAL_M1`
  (inclination measured by the amplitudes; external ``m1``; β = 0).
- ``β`` is the COMPANION-TO-PRIMARY FLUX RATIO ``L2 / L1`` in the band
  that sets the photocentre — for Gaia astrometry the broad ``G`` band,
  NOT RVS and not bolometric.  The photocentre light fraction is
  ``β/(1+β)``.
- Both paths assume a DARK (β = 0) companion; luminous-companion / SB2 /
  blend / triple scenarios can break them — and on the astrometric path
  the bias is TWO-SIDED: ``fm_ast`` is inflated when
  ``β/(1+β) > 2·m2/M``, deflated for ``0 < β/(1+β) < 2·m2/M``, and
  EXACTLY UNBIASED at the boundary (and trivially at β = 0, where there
  is no contamination to bias it).  So the output is a CANDIDATE
  compact-companion mass, never a confirmed dark companion.  In the
  inflated regime the secondary DOMINATES the light, so the external
  ``m1`` — normally read off "the primary's" spectrum — is likely wrong
  too: a compounded error, not a single bias.
- The mass function and ``m1`` are assumed statistically INDEPENDENT.
  This holds when ``m1`` comes from information that does NOT reuse the
  astrometric solution — e.g. spectroscopic ``log g`` + ``T_eff``.
  It does NOT strictly hold for an isochrone / spectro-photometric mass
  built on the SAME parallax that scales ``fm_ast ∝ α³/ϖ³``: a parallax
  error then moves ``fm_ast`` and ``m1`` together, and pairing them as
  independent UNDERSTATES the ``m2`` spread — in the direction of a more
  massive, more compact-object-like companion.  Modest when
  ``σ_ϖ/ϖ`` is small, material when it is not.  Carrying that
  correlation needs row-aligned ``(m1, ϖ)`` draws, which this signature
  cannot express yet (an ndarray raises today).

This module persists nothing and emits no logs: it is a pure transform
over arrays already in memory.
"""

from __future__ import annotations

import numpy as np

from orblet.constants import (
    MASS_CONVENTION_ASTROMETRIC_EXTERNAL_M1,
    MASS_CONVENTION_PROJECTED,
    MASS_CONVENTION_TRUE,
)

# Cycle RV-C (Option B, self-contained): this layer imports only the
# PUBLIC prior CLASSES from the composable surface and builds the right
# one from a spec tuple with a SMALL LOCAL dispatch (``_prior_from_spec``
# below).  It deliberately does NOT import ``parse_prior_spec`` from the
# prior module: the mass layer needs only the few families it
# dispatches below, and keeping the dispatch local keeps this public
# adapter independent of the sampler-facing spec parser and its
# conventions.  The classes defer their scipy
# load (in ``logpdf``/``sample``), so this MODULE BODY adds no top-level
# scipy/emcee import (test-backed by
# ``tests/test_orblet_optional_dependencies.py``).
from orblet.priors import (
    LogUniformPrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)


# ── Guard messages ────────────────────────────────────────────────────
#
# The messages below are static module-level constants and stay that way
# (callers identity-check several of them).  That is a choice, not a
# prohibition: this is the post-fit layer a user debugs interactively,
# and an unhelpful message costs real time, so a NEW exception or warning
# here MAY name the offending shape or value.  What this module never
# does is write persistent output — print / logging / traceback — whose
# streams outlive the session.  Never put a source identifier in any
# message.
#
_MISSING_CHAINS_MSG = (
    "rv_chain_result must contain a 'chains' mapping (the "
    "fit_rv_orbit_linear schema)."
)
_MISSING_FM_MSG = (
    "rv_chain_result['chains'] must contain the mass-function array "
    "'fm_spec_msun'."
)
_FM_NOT_1D_MSG = (
    "rv_chain_result['chains']['fm_spec_msun'] must be a 1-D array of "
    "posterior draws."
)
_BAD_PRIMARY_MASS_PRIOR_MSG = (
    "primary_mass_prior must be a bare float or a recognised prior-spec "
    "tuple (e.g. ('Normal', mu, sigma) / ('truncated_Normal', mu, sigma, "
    "lo) / ('Uniform', lo, hi))."
)
_BAD_SIN_I_MSG = (
    "sin_i must be a number in (0, 1], or one value per mass-function "
    "draw (a 1-D array of matching length, entries in [0, 1]). Prior "
    "specs and prior objects are not accepted: this layer gives the "
    "minimum mass (sin_i = 1) or the mass for the sin_i you supply."
)

# Astrometric-adapter messages (astrometric-worded; the RV constants
# above must NOT be reused here — a wrong-path message is itself a
# defect).  The static ones are identity-checked by tests; the shape
# message is a PREFIX completed with the offending shape (permissive
# tier — informative, never an identifier).
_AST_MISSING_CHAINS_MSG = (
    "result must contain a 'chains' mapping (the "
    "fit_astrometry_orbit_linear_ti schema)."
)
_AST_MISSING_FM_AST_MSG = (
    "result['chains'] must contain the astrometric mass-function array "
    "'fm_ast_msun' — an observables-first Thiele-Innes chain. "
    "Campbell-basis chains carry no fm_ast and are not supported here."
)
_AST_FM_AST_BAD_SHAPE_MSG = (
    "result['chains']['fm_ast_msun'] must be a non-empty 1-D array of "
    "posterior draws; got shape "
)
_AST_JOINT_OR_RV_REJECTED_MSG = (
    "this chain carries spectroscopic keys, so it is a joint or RV "
    "chain, not an astrometry-only TI chain. A joint fit constrains the "
    "mass split WITHOUT an external primary mass — feeding one in would "
    "silently override measured information. Offending keys: "
)
_AST_NOT_TI_BASIS_MSG = (
    "result['chains'] is not recognisably a Thiele-Innes observables "
    "chain: it needs _meta['basis'] == 'thiele_innes' or the A_mas, B_mas, "
    "F_mas, G_mas amplitude keys (and a present basis sentinel must not "
    "contradict them). Failing closed rather than guessing the basis."
)
_AST_NONPOSITIVE_FIXED_M1_MSG = (
    "primary_mass_prior must be a positive mass in M_sun when supplied "
    "as a fixed float; a non-positive primary has no physical root and "
    "would return an all-NaN block that reads like a data problem. Got "
)
_AST_BETA_NOT_IMPLEMENTED_MSG = (
    "beta is a RESERVED argument: the beta > 0 inversion "
    "(fm_ast = |m2/M - beta/(1+beta)|^3 * M, two roots + a cancellation "
    "blind spot) is engine-track A2 and is not implemented yet. Pass "
    "beta=None; beta = 0 (dark companion) is the built-in, named "
    "assumption (_meta['beta_assumption'])."
)


# ── RNG stream offset (Cycle RV-C) ───────────────────────────────────
#
# The mass layer draws one auxiliary random quantity: ``m1`` (``sin i``
# is always caller-supplied, never drawn here).  The offset MUST stay
# distinct from every stream the RV sampler itself consumes so the mass
# layer is both reproducible and statistically independent of the fit's
# draws.  The RV linear sampler uses ``seed + c`` per chain (small
# ``c``) and ``seed + 100003`` for its post-hoc β-draw stream
# (``rv/linear_sampler.py``); the large prime below avoids all of those.
_M1_DRAW_STREAM_OFFSET = 700019


def solve_companion_mass(
    fm: np.ndarray | float,
    m1: np.ndarray | float,
    sin_i: np.ndarray | float,
) -> np.ndarray:
    """Solve the SB1 mass function for the companion mass ``m2``.

    Inverts

        fm = (m2 · sin i)^3 / (m1 + m2)^2

    for ``m2`` (solar masses), elementwise over broadcast inputs.

    The function ``h(m2) = (m2 · sin_i)^3 / (m1 + m2)^2`` is STRICTLY
    MONOTONE INCREASING in ``m2`` on ``m2 > 0`` whenever ``fm > 0``,
    ``m1 > 0`` and ``sin_i ∈ (0, 1]``, so the positive root is unique.
    A vectorised bisection on this proven-monotone function finds it
    (numpy-only; no scipy).  The upper bracket is grown by doubling from
    a generous seed until ``h(upper) ≥ fm`` (capped to avoid a runaway
    loop), so black-hole-mass companions with ``m2 ≫ m1`` are bracketed:
    as ``m2 → ∞``, ``h ≈ sin_i^3 · m2``, hence the root scales like
    ``fm / sin_i^3`` and the doubling search reaches it in a few steps.

    Parameters
    ----------
    fm : array or float
        Mass function (M_sun): the spectroscopic ``fm`` OR the
        astrometric ``fm_ast_msun``.  Already absorbs ``G`` (no ``G`` /
        ``MSUN`` is applied here).
    m1 : array or float
        Assumed primary mass (M_sun), broadcastable against ``fm``.
    sin_i : array or float
        ``sin i`` (dimensionless), broadcastable against ``fm``.  An
        ASSUMED input on the RV path only; on the astrometric path the
        caller passes exactly ``1.0``, which is EXACT (``fm_ast`` is
        projection-free — the inclination is already folded into the
        photocentre amplitudes), not an assumption.

    Returns
    -------
    numpy.ndarray
        Companion mass ``m2`` (M_sun), same broadcast shape as the
        inputs.  Elements with no physical positive root
        (``fm ≤ 0`` or ``sin_i = 0``) are returned as ``numpy.nan``.
        Never raises on unphysical input.
    """
    fm_a, m1_a, sin_a = np.broadcast_arrays(
        np.asarray(fm, dtype=float),
        np.asarray(m1, dtype=float),
        np.asarray(sin_i, dtype=float),
    )
    out = np.full(fm_a.shape, np.nan, dtype=float)

    # A positive root exists only for fm > 0 and sin_i > 0 (and m1 > 0).
    valid = (fm_a > 0.0) & (sin_a > 0.0) & (m1_a > 0.0)
    if not np.any(valid):
        return out

    fm_v = fm_a[valid]
    m1_v = m1_a[valid]
    sin_v = sin_a[valid]

    def h(m2):
        return (m2 * sin_v) ** 3 / (m1_v + m2) ** 2

    # Lower bracket: 0 (h(0) = 0 < fm). Upper bracket: grow by doubling
    # until h(upper) >= fm.  Seed from the large-m2 asymptote root guess
    # fm / sin^3 (a lower bound on the true root, since the (m1+m2)^2
    # denominator only makes h smaller than the asymptote), padded.
    lo = np.zeros_like(fm_v)
    hi = np.maximum(fm_v / sin_v**3, m1_v) + 1.0
    # Cap the doubling count generously; for double precision the root is
    # always bracketed well within this many doublings.
    for _ in range(200):
        need = h(hi) < fm_v
        if not np.any(need):
            break
        hi = np.where(need, hi * 2.0, hi)

    # Vectorised bisection.  ~60 iterations drives the bracket below
    # machine precision relative to the root for all realistic masses.
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        too_small = h(mid) < fm_v
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)

    out[valid] = 0.5 * (lo + hi)
    return out


def _prior_from_spec(spec):
    """Build a public prior CLASS instance from a ``(name, *args)`` spec.

    Self-contained Option-B dispatch (Cycle RV-C): the only prior forms a
    primary-mass or marginalised-``sin i`` prior needs.  No dependency on
    the RV engine's ``parse_prior_spec``.  The constructed object exposes
    ``.sample(rng, size)``.

    Supported specs:

    - ``("Normal", mu, sigma)``
    - ``("truncated_Normal", mu, sigma, lo)``           (one-sided)
    - ``("Uniform", lo, hi)``
    - ``("LogUniform", lo, hi)``

    Raises ``ValueError`` / ``TypeError`` (the prior-class constructors'
    own value-interpolating errors, or a bare ``ValueError`` on an
    unknown name) — the CALLERS catch these and re-raise the layer's own
    value-free constant ``from None``.
    """
    name = spec[0]
    args = spec[1:]
    if name == "Normal":
        mu, sigma = args
        return NormalPrior(float(mu), float(sigma))
    if name == "truncated_Normal":
        mu, sigma, lo = args
        return TruncatedNormalPrior(
            float(mu), float(sigma), float(lo), float("inf")
        )
    if name == "Uniform":
        lo, hi = args
        return UniformPrior(float(lo), float(hi))
    if name == "LogUniform":
        lo, hi = args
        return LogUniformPrior(float(lo), float(hi))
    raise ValueError


def _draw_primary_mass(primary_mass_prior, n: int, seed: int):
    """Return an (n,) array of primary masses.

    A bare float becomes a constant array (``m1_mode = "fixed"``); a
    prior-spec tuple is dispatched (``_prior_from_spec``) and sampled on a
    dedicated RNG stream (``m1_mode = "drawn"``).  Raises the layer's own
    value-free ``ValueError`` (``from None``) if the spec is malformed.
    """
    if isinstance(primary_mass_prior, (int, float)) and not isinstance(
        primary_mass_prior, bool
    ):
        return np.full(n, float(primary_mass_prior), dtype=float), "fixed"
    try:
        prior = _prior_from_spec(primary_mass_prior)
    except (ValueError, TypeError):
        raise ValueError(_BAD_PRIMARY_MASS_PRIOR_MSG) from None
    rng = np.random.default_rng(int(seed) + _M1_DRAW_STREAM_OFFSET)
    return np.asarray(prior.sample(rng, n), dtype=float), "drawn"


def _resolve_sin_i(sin_i, n: int):
    """Return ``(sin_i_array, sin_i_mode)`` from a CALLER-SUPPLIED ``sin i``.

    Two forms only — the caller takes the minimum mass or provides
    ``sin i``; this layer never draws an inclination itself:

    - a number in ``(0, 1]`` → constant array; ``sin_i_mode`` is that float;
    - one value per draw: a 1-D array-like of length ``n``, or a carrier
      exposing it as ``.values`` (e.g.
      :class:`orblet.interpret.flux_ratio.MeasuredSinI`), entries in
      ``[0, 1]`` (NaN tolerated; a ``0`` or NaN entry gives a NaN mass on
      that row) → ``sin_i_mode`` is ``"per_draw"``.

    Anything else — a prior-spec tuple, a prior object — raises
    ``ValueError``.  Dimensionless.
    """
    if isinstance(sin_i, (bool, tuple, str)):
        raise ValueError(_BAD_SIN_I_MSG)

    if isinstance(sin_i, (int, float, np.integer, np.floating)):
        value = float(sin_i)
        if not (0.0 < value <= 1.0):
            raise ValueError(_BAD_SIN_I_MSG)
        return np.full(n, value, dtype=float), value

    values = getattr(sin_i, "values", sin_i)
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(_BAD_SIN_I_MSG) from None
    # Row alignment is the whole point of the per-draw form: shape (n,)
    # exactly — no broadcasting, no truncation.
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(_BAD_SIN_I_MSG)
    finite = arr[np.isfinite(arr)]
    if np.any((finite < 0.0) | (finite > 1.0)):
        raise ValueError(_BAD_SIN_I_MSG)
    return arr, "per_draw"


def _summarise(arr: np.ndarray) -> dict:
    """Per-array summary block, IDENTICAL in form to the
    ``fit_rv_orbit_linear`` summary block (median / std / q16 / q84).

    NaN draws (rows with no physical root) are dropped before
    summarising so the percentiles describe the physical sub-population.
    """
    finite = arr[np.isfinite(arr)]
    # All draws unphysical (e.g. an fm chain entirely <= 0, or fixed
    # sin_i = 0): the finite subset is empty, where np.percentile([], 16)
    # raises IndexError and np.median([]) warns.  Return an all-NaN block
    # rather than raise — the per-draw NaN masses + finite_fraction == 0.0
    # already communicate the degenerate input honestly.
    if finite.size == 0:
        nan = float("nan")
        return {"median": nan, "std": nan, "q16": nan, "q84": nan}
    return {
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "q16": float(np.percentile(finite, 16)),
        "q84": float(np.percentile(finite, 84)),
    }


def companion_mass_from_rv_posterior(
    rv_chain_result,
    *,
    primary_mass_prior,
    sin_i=1.0,
    seed: int = 0,
):
    """Derive companion-mass posterior draws from an RV-fit posterior.

    Engine-agnostic adapter: it consumes only the mass-function draws
    ``rv_chain_result["chains"]["fm_spec_msun"]`` (the ``fit_rv_orbit_linear``
    schema), combines them with an assumed primary mass and inclination,
    and returns companion-mass draws plus a summary.  See the module
    docstring for the mass-function equation and the (load-bearing)
    scientific conventions.

    Parameters
    ----------
    rv_chain_result : mapping
        An RV-fit result dict.  Must expose ``["chains"]["fm_spec_msun"]`` as a
        1-D array of mass-function draws (M_sun).
    primary_mass_prior : float or tuple
        Either a bare primary mass in M_sun (held fixed across draws) or
        a prior-spec tuple (e.g. ``("Normal", mu, sigma)``) sampled once
        per ``fm`` draw on a dedicated RNG stream.
    sin_i : float or 1-D array-like, default 1.0
        CALLER-SUPPLIED ``sin i`` (dimensionless).  Two modes only:

        - ``1.0`` (default) → the MINIMUM companion mass, ``m2 sin i``
          (``MASS_CONVENTION_PROJECTED``);
        - your own ``sin i`` → the mass for that ``sin i``
          (``MASS_CONVENTION_TRUE``): one number in ``(0, 1]``, or one
          value per ``fm`` draw (a 1-D array of matching length, or a
          carrier exposing it as ``.values`` such as
          :class:`orblet.interpret.flux_ratio.MeasuredSinI`), entries in
          ``[0, 1]``.

        The mass is only as good as the ``sin i`` supplied: the function
        cannot tell a measured inclination from a guessed one, and the
        sentinel does not try to.  Out-of-range values, prior-spec
        tuples and prior objects raise ``ValueError`` — this layer never
        draws an inclination.  A caller who wants an orientation-averaged
        mass draws ``sin i`` themselves (isotropic: ``cos i`` uniform,
        NOT ``sin i`` uniform) and passes the array.
    seed : int, default 0
        Base seed for the ``m1`` draw (used only when
        ``primary_mass_prior`` is a prior spec).  An offset keeps that
        stream distinct from the RV sampler's own streams.

    Returns
    -------
    dict
        ``{"mass": {"m2", "m2_sini", "M_total"},
           "summary": {<same keys>: {median, std, q16, q84}},
           "m1_draws_msun": ndarray,
           "_meta": {scalars / sentinels only}}``.
        Masses are in M_sun.  ``_meta`` carries
        ``mass_convention`` / ``m1_mode`` / ``sin_i_mode`` / ``seed`` /
        ``finite_fraction`` (the fraction of draws with a physical
        positive root).

        ``m1_draws_msun`` is the (n,) array of primary masses this call
        actually USED, one row per ``fm`` draw (constant when
        ``primary_mass_prior`` is a float).  It is surfaced so a
        downstream layer can pair ``m1`` with ``m2`` on identical rows
        by VALUE instead of re-drawing on the shared seeded stream and
        assuming the two agree — "same stream" is only "same draws" when
        the draw count, prior spec and call order all match.  The first
        consumer is
        :func:`orblet.interpret.flux_ratio.beta_from_joint_posterior`.

    Raises
    ------
    ValueError
        With a value-free message (a module constant) if the input dict
        lacks ``chains`` / ``fm_spec_msun``, that array is not 1-D, or the
        prior specs
        are malformed.  Malformed prior specs are re-raised
        ``from None`` so no underlying value-bearing message leaks.
    """
    if not isinstance(rv_chain_result, dict) or "chains" not in rv_chain_result:
        raise ValueError(_MISSING_CHAINS_MSG)
    chains = rv_chain_result["chains"]
    if not isinstance(chains, dict) or "fm_spec_msun" not in chains:
        raise ValueError(_MISSING_FM_MSG)

    fm = np.asarray(chains["fm_spec_msun"], dtype=float)
    if fm.ndim != 1:
        raise ValueError(_FM_NOT_1D_MSG)

    n = fm.shape[0]
    m1, m1_mode = _draw_primary_mass(primary_mass_prior, n, seed)
    sin_arr, sin_i_mode = _resolve_sin_i(sin_i, n)

    m2 = solve_companion_mass(fm, m1, sin_arr)
    m2_sini = m2 * sin_arr
    m_total = m1 + m2

    mass = {"m2": m2, "m2_sini": m2_sini, "M_total": m_total}
    summary = {name: _summarise(arr) for name, arr in mass.items()}

    # Two labels only: PROJECTED (minimum mass) when sin_i is the number
    # 1.0; otherwise the mass for the caller-supplied sin i.
    is_projected = sin_i_mode == 1.0
    mass_convention = (
        MASS_CONVENTION_PROJECTED if is_projected else MASS_CONVENTION_TRUE
    )

    meta = {
        "mass_convention": mass_convention,
        "m1_mode": m1_mode,
        "sin_i_mode": sin_i_mode,
        "seed": int(seed),
        "finite_fraction": float(np.mean(np.isfinite(m2))),
    }

    # ``m1_draws_msun`` is ADDITIVE: every other key
    # keeps its exact contents, and the m1 array is exposed alongside
    # them rather than inside ``mass`` (which would also change the
    # ``summary`` key set).  It is the primary mass, not a companion
    # mass, so it does not belong in the companion-mass block.
    return {
        "mass": mass,
        "summary": summary,
        "m1_draws_msun": m1,
        "_meta": meta,
    }


# Spectroscopic key names whose presence marks a joint or RV chain.  A
# joint fit constrains the mass split itself (fm_spec + fm_ast + i
# together), so the external-m1 inversion below must refuse it.
#
# NAME-BASED and therefore fail-OPEN under a future key rename: an
# exporter that renamed its RV keys would slip past this guard and be
# silently over-constrained by an external m1.  The nuisance keys below
# widen the net beyond the four mass/amplitude names, and
# ``test_astrometric_mass_adapter.py`` carries a contract test that
# walks the DECLARED key sets in ``_chain_schema.py`` and fails
# if any RV-bearing schema stops intersecting this tuple.  The
# fail-CLOSED form would be a positive ``_meta['channel']`` sentinel on
# every exporter; filed, not built here.
_SPECTROSCOPIC_KEYS = (
    "fm_spec_msun",
    "rv_K_kms",
    "K_kms",
    "gamma_kms",
    "rv_jitter_kms",
    "a_spec_au",
)

# TI amplitude signature used as the basis fallback when the chain
# carries no ``_meta['basis']`` sentinel (bare ``vectors_to_chains_ti``
# exports have none — the sampler adds ``_meta`` later).
_TI_AMPLITUDE_KEYS = ("A_mas", "B_mas", "F_mas", "G_mas")


def companion_mass_from_astrometric_posterior(
    result: dict,
    *,
    primary_mass_prior: float | tuple,
    beta: None = None,
    seed: int = 0,
) -> dict:
    """Derive companion-mass draws from a TI astrometric-fit posterior.

    Engine-agnostic adapter, mirroring
    :func:`companion_mass_from_rv_posterior`: it consumes only the
    astrometric mass-function draws ``result["chains"]["fm_ast_msun"]``
    (the observables-first Thiele-Innes schema), pairs them
    draw-by-draw with an EXTERNAL primary mass, and inverts

        fm_ast = m2^3 / (m1 + m2)^2        [M_sun, beta = 0]

    for ``m2`` via the shared solver at ``sin_i = 1`` — which is EXACT
    here, not an assumption: ``fm_ast = a_phot^3 / P^2`` is
    projection-free because the inclination is already folded into the
    measured photocentre amplitudes.  There is deliberately NO ``sin_i``
    parameter.

    Parameters
    ----------
    result : mapping
        A TI fit-result dict.  Must expose ``["chains"]["fm_ast_msun"]`` as a
        non-empty 1-D array of draws (M_sun) and be recognisably a
        Thiele-Innes observables chain (``_meta['basis']`` sentinel or
        the ``A_mas..G_mas`` amplitude keys).  Joint and RV chains are
        REJECTED: a joint fit constrains the mass split without an
        external ``m1``, so imposing one would override measured
        information.
    primary_mass_prior : float or tuple
        External primary mass (M_sun): a bare float > 0 (held fixed
        across draws) or a prior-spec tuple (e.g.
        ``("Normal", mu, sigma)``) sampled once per ``fm_ast`` draw on
        the shared dedicated m1 RNG stream
        (``seed + _M1_DRAW_STREAM_OFFSET``).  Must come from information
        EXTERNAL to the astrometric fit — see the module docstring for
        the parallax-correlation caveat that makes an isochrone mass
        only APPROXIMATELY independent.

        A prior with support below zero is effectively TRUNCATED at
        zero: draws with ``m1 <= 0`` have no physical root, come back
        NaN, and are excluded from ``summary``, so the reported
        percentiles are conditional on ``m1 > 0``.  Pass
        ``("truncated_Normal", mu, sigma, lo)`` to state that
        truncation explicitly rather than inheriting it silently.
    beta : None
        RESERVED for the A2 beta > 0 inversion (photocentre light
        ratio).  Any supplied value — including an explicit ``0.0`` —
        raises :class:`NotImplementedError` today, so a beta = 0 answer
        can never be returned under a caller-stated beta.  When that
        inversion lands, this argument (and a draw-aligned array form, for
        isochrone-correlated ``(m1, beta)`` ensembles) activates without
        changing the call shape.
    seed : int, default 0
        Base seed for the m1 draws (offset stream; reproducible).  In a
        BATCH run pass a per-target seed: leaving the default makes
        every target draw the identical m1 sequence, which is harmless
        per target but perfectly correlates the m1 nuisance across the
        population, under-dispersing any ensemble statistic.

    Returns
    -------
    dict
        ``{"mass": {"m2", "M_total"},
           "summary": {<same keys>: {median, std, q16, q84}},
           "_meta": {mass_convention, beta_assumption, m1_mode,
                     m1_input, n_draws, finite_fraction, seed}}``.
        ``m1_input`` records the ASSUMED primary mass itself (the float,
        or the prior spec rendered as a string) — the single most
        load-bearing input, so the result is self-describing about the
        condition it is conditional on rather than requiring the reader
        to reconstruct it from ``M_total - m2``.
        Masses in M_sun; ``M_total = m1 + m2`` per draw.  There is NO
        ``m2_sini`` key (that name belongs to the RV path and would
        carry the opposite meaning here).  Draws where ``fm_ast`` is
        NaN (the Stage-1 non-positive-parallax screen), non-positive,
        or where a drawn ``m1 <= 0``, come back NaN and are counted out
        of ``finite_fraction`` — never silently dropped.

    Raises
    ------
    ValueError
        On a malformed / non-TI / joint input (messages above; the
        shape message names the offending shape — never an identifier).
    NotImplementedError
        When ``beta`` is not ``None`` (reserved).
    """
    if not isinstance(result, dict) or "chains" not in result:
        raise ValueError(_AST_MISSING_CHAINS_MSG)
    chains = result["chains"]
    if not isinstance(chains, dict):
        raise ValueError(_AST_MISSING_CHAINS_MSG)

    # Joint / RV rejection FIRST: those chains may legitimately carry
    # fm_ast too, so this check must precede the fm_ast lookup.
    spectroscopic = [k for k in _SPECTROSCOPIC_KEYS if k in chains]
    if spectroscopic:
        raise ValueError(
            _AST_JOINT_OR_RV_REJECTED_MSG + ", ".join(spectroscopic)
        )

    if "fm_ast_msun" not in chains:
        raise ValueError(_AST_MISSING_FM_AST_MSG)

    # Basis: a present sentinel must SAY thiele_innes; with no sentinel,
    # the amplitude signature is required.  Fail closed, never guess.
    meta_in = chains.get("_meta")
    basis = meta_in.get("basis") if isinstance(meta_in, dict) else None
    has_amplitudes = all(k in chains for k in _TI_AMPLITUDE_KEYS)
    if basis is not None:
        if basis != "thiele_innes":
            raise ValueError(_AST_NOT_TI_BASIS_MSG)
    elif not has_amplitudes:
        raise ValueError(_AST_NOT_TI_BASIS_MSG)

    if beta is not None:
        raise NotImplementedError(_AST_BETA_NOT_IMPLEMENTED_MSG)

    fm_ast = np.asarray(chains["fm_ast_msun"], dtype=float)
    if fm_ast.ndim != 1 or fm_ast.size == 0:
        raise ValueError(
            _AST_FM_AST_BAD_SHAPE_MSG + repr(fm_ast.shape)
        )

    n = fm_ast.shape[0]
    m1, m1_mode = _draw_primary_mass(primary_mass_prior, n, seed)

    # A FIXED non-positive m1 is a caller blunder, not a prior tail: it
    # would return an all-NaN block that reads like a data problem.
    # (A drawn m1 <= 0 IS a prior tail and is screened per draw below.)
    if m1_mode == "fixed" and not float(primary_mass_prior) > 0.0:
        raise ValueError(
            _AST_NONPOSITIVE_FIXED_M1_MSG + repr(float(primary_mass_prior))
        )

    # sin_i = 1.0 selects the projection-free form of the shared solver
    # (see the module docstring) — exact for fm_ast, not an assumption.
    m2 = solve_companion_mass(fm_ast, m1, 1.0)
    m_total = m1 + m2  # NaN wherever m2 is NaN (screened draws).

    mass = {"m2": m2, "M_total": m_total}
    summary = {name: _summarise(arr) for name, arr in mass.items()}

    meta = {
        "mass_convention": MASS_CONVENTION_ASTROMETRIC_EXTERNAL_M1,
        "beta_assumption": 0.0,
        "m1_mode": m1_mode,
        # The assumed m1 itself: a float when fixed, else the spec
        # rendered as a string (kept a plain scalar/str so the block
        # stays JSON-serialisable for batch products).
        "m1_input": (
            float(primary_mass_prior)
            if m1_mode == "fixed"
            else str(primary_mass_prior)
        ),
        "n_draws": int(n),
        "finite_fraction": float(np.mean(np.isfinite(m2))),
        "seed": int(seed),
    }

    return {"mass": mass, "summary": summary, "_meta": meta}
