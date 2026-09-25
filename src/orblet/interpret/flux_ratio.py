"""Photocentre flux ratio ``beta`` from a JOINT orbit.

Post-posterior layer: this module reads posterior draws that a joint
RV + astrometric fit already produced and derives the photocentre light
ratio from them.  It contacts no sampler, fits nothing, persists
nothing and emits no logs — a pure transform over arrays in memory.

The one identity
----------------
With ``q = m2/m1`` the mass ratio and ``beta = F2/F1`` the companion /
primary flux ratio, the photocentre sits at the flux-weighted position:

    a_phot / a_rel = q/(1+q) - beta/(1+beta)
    a_phot / a1    = 1 - beta (1+q) / (q (1+beta)) = (q - beta) / (q (1+beta))

with ``a1 = a_rel q/(1+q)`` the primary's orbit about the barycentre.
Reading the second line off the data gives beta.  Three limits fix the
signs: ``beta = 0`` gives ``a_phot = a1`` exactly (the deficit identity);
``beta = q`` gives ``a_phot = 0`` for EVERY q (the general null, of which
``beta = q = 1`` is one point); ``beta > q`` puts the photocentre on the
COMPANION's side, flipping the sign of ``a_phot``.

The observable, and why there are two answers
---------------------------------------------
The joint fit exports ``a_phot_mas`` as a FOLDED non-negative magnitude,
so the sign above is not observable and the measurable quantity is

    r = |a_phot| / a1 = (a_phot_mas / plx_mas) * sin i / a_spec_au

computed by MULTIPLYING by ``sin i``.  ``a_spec_au`` is
the joint engine's own ``a1 sin i`` export — reused, never re-derived.
Inverting with the sign lost yields TWO roots:

    standard branch  beta  = q (1 - r) / (1 + q r)   valid r <= 1,  beta in [0, q]
    mirror branch    beta' = q (1 + r) / (1 - q r)   valid r < 1/q, beta' in [q, inf)

Both are returned, each with a validity mask; a single-branch summary is
FORBIDDEN.  The ambiguity is not an export-folding artefact that better
sampling could remove: it is the exact Klein-4
``(Omega, a_phot) -> (Omega + pi, -a_phot)`` invariance of the Tier-2
target (manual Sections 4.3/4.4), so ONLY information external to this
fit — an SED flux ratio, a spectroscopic secondary, resolved imaging —
can break it.  Choosing a branch here and "confirming" it with an
externally derived SED beta later would double-count that evidence.

The mirror branch is QUALITATIVE.  Read it as a flag
for a luminous / inflated companion, never as a quantitative flux ratio:
in that regime the light is dominated by the secondary, so the external
primary mass — normally read off "the primary's" spectrum — is likely
the secondary's, and the ``q`` fed into ``beta'`` inherits a wrong m1
(the reference manual, §6.5).  Because it is
qualitative-only, ``summary`` carries NO median/std/q16/q84 block for
``beta_mirror`` — only ``_meta["frac_mirror_valid"]`` (what fraction of
draws admit the branch at all) and ``_meta["mirror_branch_admitted"]``
(whether ANY draw does), with ``_meta["mirror_is_qualitative"] = True``
marking the convention.

Physical domain of r
--------------------
``r`` lives on ``[0, max(1, 1/q)]``.  For ``q < 1`` the range
``1 < r < 1/q`` is PHYSICAL — the photocentre has crossed to the
companion's side (the inflated / stripped-secondary regime,
``beta/(1+beta) > 2 m2/M``), and it is the strongest luminous-companion
signature the orbit alone provides.  Only ``r > max(1, 1/q)`` is a
closure failure: no ``beta >= 0`` reproduces it.  Such draws are
reported (both branches NaN, counted in ``_meta``), NEVER clipped into
the domain.

Noise-floor bias
----------------
Because ``|a_phot|`` is folded, a weakly detected photocentre carries a
positive low-S/N floor: ``r`` is biased UP, and the standard-branch beta
is therefore biased DOWN, toward 0 — toward the dark-companion reading,
the very conclusion a compact-object search wants to be conservative
about.  A DETECTION-SIGNIFICANCE SCREEN on ``a_phot`` is REQUIRED before
quoting beta, mirroring the ``fm_ast`` rule.  This module does not
invent the threshold: pass the externally computed significance through
``a_phot_detection_significance`` so the result records it, and read
``_meta["frac_r_ge_one"]`` — posterior mass piling at the ``beta = 0``
boundary is the signature of exactly this bias.

beta and D are ONE IDENTITY APART
---------------------------------
The deficit ``D = 1 - r`` carries the SAME information as beta; the map
between them is algebra, not evidence.  Nobody may "confirm" one with
the other.  The joint chain exports its own ``D`` computed in the
DIVIDE form (``1 - a_phot_au / (a_spec_au / sin i)``); away from face-on
it agrees with the multiply form used here to float tolerance, and near
face-on (``sin i -> 0``) the two forms diverge numerically — this module
pins the multiply form and never mixes them.

Conditionality (state it wherever beta is quoted)
-------------------------------------------------
- the PRIMARY-MASS PRIOR, through ``q``.  m1 is itself beta-conditional:
  a photometric / isochrone mass assumes the light is the primary's.  An
  external primary-mass source must therefore deliver row-aligned
  ``(m1, beta)`` pairs, not beta alone; and a catalogue ``fluxratio``
  column may descend from the SAME astrometric solution, so its
  provenance must be verified before it is used as external evidence.
- BAND HOMOGENEITY: beta is the flux ratio in the band the astrometry is
  measured in — for Gaia the broad G band (:data:`AMRF_BETA_BAND`).  An
  SED-derived or MS-relation beta must be converted into that band
  first; bolometric and RVS ratios are different numbers.
- the JOINT FIT's own conditioning: the period alias it settled on, its
  fixed-jitter tier, and every quality cut upstream of it.

Units carried through unchanged (reference manual, §1):
masses M_sun, ``a_spec_au`` / ``a_phot_au`` in AU, ``a_phot_mas`` /
``plx_mas`` in mas, ``inc_rad`` in radians, period days at the public
surface.  ``q`` comes from the SPECTROSCOPIC mass function only (see
:func:`beta_from_joint_posterior`).
"""

from __future__ import annotations

import warnings

import numpy as np

# The mass layer is the ONLY q source (design note H-6) and owns the m1
# draw stream.  The two private helpers are imported rather than
# re-implemented so the fallback path uses the IDENTICAL stream and the
# summary blocks keep one shape across the post-fit layers
# (known_traps #12: a re-derived twin is how these drift).
from orblet.interpret.companion_mass import (
    _draw_primary_mass,
    _summarise,
    companion_mass_from_rv_posterior,
)


# The band beta is defined in.  Defined ONCE here (design note H-5); the
# A1 AMRF module must IMPORT this name rather than re-declare it, or the
# MS beta(q, m1) relation and this inversion can silently disagree about
# what "flux ratio" means.
from orblet.interpret.amrf import AMRF_BETA_BAND  # noqa: E402  (single band authority — design-note H-5; amrf.py owns it)

# Chain keys this layer consumes: the four observables that build r, plus
# the spectroscopic mass function that builds q.
_REQUIRED_CHAIN_KEYS = (
    "fm_spec_msun",
    "a_phot_mas",
    "plx_mas",
    "inc_rad",
    "a_spec_au",
)

# Astrometry-only (Thiele-Innes) signature, used to give the H-6 refusal
# its own message instead of a generic missing-key one.
_TI_SIGNATURE_KEYS = ("A_mas", "B_mas", "F_mas", "G_mas")

# Full-joint Campbell mass signature (sci-F5): these quantities are
# solved under a beta = 0 (dark-companion) LOCK, so any m1/q built from
# them would already assume the answer this layer exists to infer.
_CAMPBELL_SIGNATURE_KEYS = ("m2_msun", "M_total_msun", "a_rel_au")


# ── Guard messages ────────────────────────────────────────────────────
#
# STRICT tier: every message below is a module-level constant with NO
# dynamic composition — no f-string, no ``.format``, no ``%``, no
# concatenation with a runtime value.  Nothing derived from the chain can
# reach a message, so no fitted amplitude, mass or identifier can leak
# through a traceback or a captured warning.
_MISSING_CHAINS_MSG = (
    "joint_marginal_result must contain a 'chains' mapping (the "
    "fit_joint_orbit_marginal schema)."
)
_TI_ONLY_REFUSED_MSG = (
    "this looks like an astrometry-only Thiele-Innes chain: it carries "
    "no spectroscopic mass function. The mass ratio q MUST come from "
    "fm_spec_msun, which is beta-free (beta scales the photocentre "
    "amplitude, never the ellipse shape or sin i). Inverting the "
    "astrometric fm_ast instead would assume beta = 0 in order to infer "
    "beta. Supply a JOINT posterior."
)
_MISSING_KEYS_MSG = (
    "joint_marginal_result['chains'] must contain fm_spec_msun, "
    "a_phot_mas, plx_mas, inc_rad and a_spec_au (the "
    "fit_joint_orbit_marginal schema)."
)
_BAD_DRAW_SHAPE_MSG = (
    "every consumed chain array must be a non-empty 1-D array of "
    "posterior draws, all of the same length: beta is a PER-DRAW "
    "quantity and the rows must align."
)
_M1_ROW_MISMATCH_MSG = (
    "the primary-mass draw array returned by the mass layer does not "
    "have one row per posterior draw, so beta could not be formed on "
    "aligned rows. Row alignment is a correctness requirement here, not "
    "a convenience."
)
_NO_DETECTION_SCREEN_MSG = (
    "a_phot_detection_significance was not supplied: the folded "
    "|a_phot| carries a positive low-S/N floor that biases r up and the "
    "standard-branch beta DOWN toward the dark-companion reading, while "
    "the mirror-branch beta RISES with r and is therefore biased toward "
    "a MORE extreme luminous companion. Screen the photocentre detection "
    "before quoting beta; see _meta['frac_r_ge_one'] for the boundary "
    "pile-up."
)
_M1_FALLBACK_MSG = (
    "the mass layer returned no m1 draw array, so the primary masses "
    "were re-drawn on the shared seeded stream. 'Same stream' is not "
    "'same draws' unless the draw count, prior spec and call order all "
    "match; row alignment is assumed, not proven. Recorded in "
    "_meta['m1_draws_source']."
)
_CAMPBELL_LOCKED_REFUSED_MSG = (
    "this chain carries a full-joint Campbell mass signature (m2_msun / "
    "M_total_msun / a_rel_au): those quantities are solved under a "
    "beta = 0 (dark-companion) lock, so any m1 or q built from them "
    "would already assume the answer this layer exists to infer. Supply "
    "the fm_spec_msun-based marginal chain schema instead."
)
_MEASURED_SIN_I_SIZE_MISMATCH_MSG = (
    "MeasuredSinI.sample was asked for a different number of draws than "
    "it holds measured sin i values for; truncating or broadcasting "
    "would silently break the row alignment this shim exists to "
    "guarantee."
)


class MeasuredSinI:
    """Row-aligned ``sin i`` carrier for the mass layer's ``sin_i`` hook.

    :func:`companion_mass_from_rv_posterior` accepts one ``sin i``
    number or one value per draw.  A joint fit MEASURES the inclination,
    so the right input is this posterior's own per-draw ``sin i``, kept
    row-aligned with the mass function.  The mass layer reads
    ``.values`` directly (a plain 1-D array works just as well); this
    carrier is kept so existing notebooks keep running, and ``.sample``
    remains only for that compatibility (``rng`` unused by design).

    Values are dimensionless, in ``[0, 1]``.
    """

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float)

    def sample(self, rng, size):
        # rng is deliberately unused: these are MEASURED draws, not a
        # prior sample, and re-randomising them would break row alignment.
        # size must match exactly: truncating or broadcasting a mismatch
        # would silently break the row alignment this shim guarantees.
        if int(size) != self.values.shape[0]:
            raise ValueError(_MEASURED_SIN_I_SIZE_MISMATCH_MSG)
        return self.values


# Private alias: the leading-underscore spelling of the same class, for
# internal callers that use it.
_MeasuredSinI = MeasuredSinI


def signed_photocentre_axis_ratio(q, beta) -> np.ndarray:
    """Signed ``a_phot / a1`` for a given mass ratio and flux ratio.

    The FORWARD map of the photocentre identity, in its
    factored form

        s = (q - beta) / (q (1 + beta))

    which is the same quantity as ``1 - beta (1+q) / (q (1+beta))`` but
    stays finite as ``beta -> q`` and needs no ``(1+q)`` division.

    Parameters
    ----------
    q : array or float
        Mass ratio ``m2 / m1`` (dimensionless, > 0).
    beta : array or float
        Flux ratio ``F2 / F1`` in the astrometric band
        (:data:`AMRF_BETA_BAND`; dimensionless, >= 0).

    Returns
    -------
    numpy.ndarray
        ``s = a_phot / a1``, dimensionless and SIGNED: ``s = 1`` at
        ``beta = 0``, ``s = 0`` at ``beta = q`` (the general null),
        ``s < 0`` for ``beta > q`` (the photocentre has crossed to the
        companion's side).  ``s`` is bounded above by 1 and tends to
        ``-1/q`` as ``beta -> inf``, which is exactly why the folded
        observable ``r = |s|`` lives on ``[0, max(1, 1/q)]``.
        Non-positive / non-finite ``q`` or negative ``beta`` gives NaN.
    """
    q_a, beta_a = np.broadcast_arrays(
        np.asarray(q, dtype=float), np.asarray(beta, dtype=float),
    )
    out = np.full(q_a.shape, np.nan, dtype=float)
    ok = np.isfinite(q_a) & np.isfinite(beta_a) & (q_a > 0.0) & (beta_a >= 0.0)
    np.divide(q_a - beta_a, q_a * (1.0 + beta_a), out=out, where=ok)
    return out


def beta_branches_from_axis_ratio(r, q) -> dict:
    """Invert the folded axis ratio into BOTH flux-ratio branches.

    The sign of ``a_phot`` is not observable (the export folds it), so
    inverting :func:`signed_photocentre_axis_ratio` has two roots:

        standard  beta  = q (1 - r) / (1 + q r)    valid r <= 1
        mirror    beta' = q (1 + r) / (1 - q r)    valid r < 1/q

    Both are returned WITH their validity masks.  Outside a branch's
    validity the entry is NaN — never clipped to a boundary value, which
    would manufacture a dark-companion (or an infinite-beta) reading out
    of a closure failure.

    Parameters
    ----------
    r : array or float
        Folded photocentre-to-primary axis ratio ``|a_phot| / a1``
        (dimensionless, >= 0).
    q : array or float
        Mass ratio ``m2 / m1`` (dimensionless, > 0), broadcastable
        against ``r``.

    Returns
    -------
    dict
        ``"standard"`` / ``"mirror"``: flux-ratio arrays (dimensionless,
        band :data:`AMRF_BETA_BAND`), NaN where that branch is invalid.
        ``"valid_standard"`` / ``"valid_mirror"``: boolean masks.
        ``"in_domain"``: boolean mask for the PHYSICAL domain
        ``r <= max(1, 1/q)``; ``False`` marks a closure failure that no
        ``beta >= 0`` can produce.

    Notes
    -----
    The two branches BRACKET the truth: the standard root lies in
    ``[0, q]`` and the mirror root in ``[q, inf)``, meeting at
    ``beta = q`` (where ``r = 0``).  Which one is real is not decidable
    from the orbit — see the module docstring on the Klein-4 invariance.
    """
    r_a, q_a = np.broadcast_arrays(
        np.asarray(r, dtype=float), np.asarray(q, dtype=float),
    )
    usable = np.isfinite(r_a) & np.isfinite(q_a) & (q_a > 0.0) & (r_a >= 0.0)

    inv_q = np.full(r_a.shape, np.nan, dtype=float)
    np.divide(1.0, q_a, out=inv_q, where=usable)

    valid_standard = usable & (r_a <= 1.0)
    # Strict: at r = 1/q the mirror denominator vanishes and beta' -> inf.
    valid_mirror = usable & (r_a < inv_q)
    in_domain = usable & (r_a <= np.maximum(1.0, inv_q))

    standard = np.full(r_a.shape, np.nan, dtype=float)
    # 1 + q r > 0 wherever q, r >= 0, so this denominator never vanishes.
    np.divide(
        q_a * (1.0 - r_a), 1.0 + q_a * r_a, out=standard, where=valid_standard,
    )
    mirror = np.full(r_a.shape, np.nan, dtype=float)
    # 1 - q r > 0 is exactly the mirror validity condition r < 1/q.
    np.divide(
        q_a * (1.0 + r_a), 1.0 - q_a * r_a, out=mirror, where=valid_mirror,
    )

    return {
        "standard": standard,
        "mirror": mirror,
        "valid_standard": valid_standard,
        "valid_mirror": valid_mirror,
        "in_domain": in_domain,
    }


def _mode_summary(beta_standard, r, q, mirror_valid, mask) -> dict:
    """Summary block for one posterior mode (a boolean row selection).

    The mirror branch stays QUALITATIVE-only per mode, exactly as in
    the top-level summary (sci-F6 / design-note M-8): a boolean and an
    admitted fraction, never a percentile block.  ``mirror_valid`` is
    the caller's validity array already intersected with the usable
    mask, so the fraction is over the mode's draws.
    """
    n_mode = int(np.count_nonzero(mask))
    if n_mode > 0:
        frac_mirror = float(np.count_nonzero(mirror_valid & mask) / n_mode)
    else:
        frac_mirror = float("nan")
    return {
        "beta_standard": _summarise(beta_standard[mask]),
        "r": _summarise(r[mask]),
        "q": _summarise(q[mask]),
        "mirror_branch_admitted": bool(np.any(mirror_valid & mask)),
        "frac_mirror_valid": frac_mirror,
        "n_draws": n_mode,
    }


def beta_from_joint_posterior(
    joint_marginal_result,
    *,
    primary_mass_prior,
    seed: int = 0,
    a_phot_detection_significance: float | None = None,
) -> dict:
    """Derive per-draw photocentre flux-ratio branches from a joint fit.

    Reads a joint (RV + astrometry) posterior, forms the folded axis
    ratio ``r`` per draw, and inverts it into BOTH flux-ratio branches
    (see the module docstring for the identity, the two roots, the
    Klein-4 reason the branch cannot be chosen here, the r-domain, and
    the noise-floor bias).  Nothing is fitted and no sampler is touched.

    The mass ratio ``q``
    --------------------
    ``q`` comes from the SPECTROSCOPIC mass function ``fm_spec_msun`` via
    :func:`orblet.interpret.companion_mass.companion_mass_from_rv_posterior`,
    paired with this posterior's MEASURED per-draw ``sin i`` and the
    external primary-mass prior.  That route is mandatory:
    ``fm_spec`` is beta-free, because beta scales the photocentre
    amplitude but not the ellipse shape or the inclination.  The
    astrometric adapter is FORBIDDEN as a q source — it inverts
    ``fm_ast`` under ``beta = 0``, so using it would assume beta = 0 in
    order to infer beta.  An astrometry-only chain is refused.

    Parameters
    ----------
    joint_marginal_result : mapping
        A joint fit-result dict exposing ``["chains"]`` with
        ``fm_spec_msun``, ``a_phot_mas``, ``plx_mas``, ``inc_rad`` and
        ``a_spec_au`` as 1-D arrays of EQUAL length (one row per
        posterior draw).  ``cos_i`` and ``_meta["bimodality_flag"]``,
        when present, drive the per-mode summaries.
    primary_mass_prior : float or tuple
        External primary mass in M_sun: a bare float (held fixed across
        draws) or a prior-spec tuple such as ``("Normal", mu, sigma)``.
        Must come from information external to this fit; it is itself
        beta-conditional (module docstring, conditionality list).
    seed : int, default 0
        Base seed handed to the mass layer for its auxiliary m1 draws.
    a_phot_detection_significance : float or None, default None
        The EXTERNALLY computed detection significance of the photocentre
        amplitude (dimensionless — a S/N, or a delta-chi-squared-based
        sigma).  This layer does not compute it and imposes no threshold;
        it records the value so a quoted beta carries its screen.  When
        omitted a value-free ``UserWarning`` is raised, because an
        unscreened beta is biased toward the dark-companion reading.

    Returns
    -------
    dict
        ``"beta"``: ``{"standard", "mirror"}`` — per-draw flux ratios in
        the :data:`AMRF_BETA_BAND` band, NaN off-branch.

        ``"valid"``: ``{"standard", "mirror", "in_domain"}`` boolean
        masks (an r-domain closure failure is ``~in_domain``).

        ``"derived"``: ``{"r", "D", "q", "m1_msun", "m2_msun", "sin_i"}``
        — ``r`` dimensionless, ``D = 1 - r`` (the SAME information as
        beta, one identity apart), masses M_sun, ``sin_i``
        dimensionless.

        ``"summary"``: median / std / q16 / q84 blocks for
        ``beta_standard``, ``r`` and ``q``, over the finite draws of
        each.  ``beta_mirror`` deliberately carries NO percentile block
        here — the mirror branch is QUALITATIVE-only (module docstring);
        its presence is instead recorded in ``_meta`` (see below).

        ``"summary_per_mode"``: the same blocks (with the mirror branch
        again QUALITATIVE-only — a per-mode admitted boolean and
        fraction, no percentile block) split by the sign of ``cos_i``
        when the input chain flags an unresolved inclination mirror,
        else ``None``.  ``r`` itself is invariant under
        ``i -> pi - i`` (it contains no ``cos i``), so beta gives no
        leverage on that mirror either — but the two modes need not
        carry the same fitted ``a_phot`` / ``K``, so they are summarised
        separately.

        ``"provenance"``: scalars recording the q source, the m1-draw
        source, the ``sin i`` source, the r form, the primary-mass prior
        and the seed.

        ``"_meta"``: scalars — band, draw counts, mass convention,
        ``bimodality_flag``, ``m1_draws_source``, the detection-screen
        record (``detection_screen`` is ``"recorded"`` /
        ``"not_recorded"``, and ``screen_threshold_applied`` is always
        ``None`` — this layer records a significance value but never
        enforces a floor on it), ``a_phot_posterior_snr`` (a
        posterior-SPREAD diagnostic — median over std of the
        ``a_phot_mas`` draws — that says how TIGHT the posterior is,
        NOT whether the orbit is detected against the no-orbit null; it
        is not a substitute for ``a_phot_detection_significance``), the
        branch / domain fractions (``frac_standard_valid``,
        ``frac_mirror_valid``, ``frac_r_outside_domain``,
        ``frac_r_ge_one``, each taken over the draws with a finite
        ``r`` and ``q``), ``mirror_branch_admitted`` (whether ANY draw
        admits the mirror branch), and ``mirror_is_qualitative``
        (always ``True`` — see the module docstring).

    Raises
    ------
    ValueError
        With a value-free module-constant message when the input is not
        a joint chain (including the astrometry-only refusal and the
        full-joint-Campbell-signature refusal), when a required key is
        missing, when the consumed arrays are not equal-length
        non-empty 1-D draws, or when the mass layer's m1 draws do not
        align row-for-row.

    Warns
    -----
    UserWarning
        Value-free, when no detection significance is supplied, and when
        the m1 draws had to be re-drawn on the shared stream instead of
        being passed through from the mass layer.
    """
    if not isinstance(joint_marginal_result, dict) or (
        "chains" not in joint_marginal_result
    ):
        raise ValueError(_MISSING_CHAINS_MSG)
    chains = joint_marginal_result["chains"]
    if not isinstance(chains, dict):
        raise ValueError(_MISSING_CHAINS_MSG)

    # sci-F5 refusal: a full-joint Campbell chain solves m2/M_total/a_rel
    # under a beta = 0 lock, so building m1 or q from them would already
    # assume the answer this layer exists to infer.  Checked unconditionally
    # (before the H-6 check below), since such a chain may also carry
    # fm_spec_msun and would otherwise slip past it.
    if any(k in chains for k in _CAMPBELL_SIGNATURE_KEYS):
        raise ValueError(_CAMPBELL_LOCKED_REFUSED_MSG)

    # H-6 refusal FIRST, so an astrometry-only chain gets the message that
    # explains the circularity rather than a generic missing-key one.
    if "fm_spec_msun" not in chains:
        ti_like = ("fm_ast_msun" in chains) or all(
            k in chains for k in _TI_SIGNATURE_KEYS
        )
        raise ValueError(_TI_ONLY_REFUSED_MSG if ti_like else _MISSING_KEYS_MSG)
    if any(k not in chains for k in _REQUIRED_CHAIN_KEYS):
        raise ValueError(_MISSING_KEYS_MSG)

    arrays = {
        key: np.asarray(chains[key], dtype=float)
        for key in _REQUIRED_CHAIN_KEYS
    }
    if any(a.ndim != 1 or a.size == 0 for a in arrays.values()):
        raise ValueError(_BAD_DRAW_SHAPE_MSG)
    n_draws = arrays["fm_spec_msun"].shape[0]
    if any(a.shape[0] != n_draws for a in arrays.values()):
        raise ValueError(_BAD_DRAW_SHAPE_MSG)

    if a_phot_detection_significance is None:
        warnings.warn(_NO_DETECTION_SCREEN_MSG, UserWarning, stacklevel=2)

    # ── r, the folded axis ratio, in the MULTIPLY form (design note L-1) ─
    # a_phot_au = a_phot_mas / plx_mas (mas / mas -> AU); then
    # r = a_phot_au * sin i / a_spec_au, with a_spec_au = a1 sin i the
    # engine's own export.  Multiplying by sin i (rather than dividing
    # a_spec by it) keeps the near-face-on limit numerically tame; D is
    # defined from THIS form and never mixed with the chain's divide-form
    # export.  Non-positive parallax and non-positive a_spec draws are
    # NaN, not dropped.
    sin_i = np.sin(arrays["inc_rad"])
    plx_ok = np.isfinite(arrays["plx_mas"]) & (arrays["plx_mas"] > 0.0)
    a_phot_au = np.full(n_draws, np.nan, dtype=float)
    np.divide(
        arrays["a_phot_mas"], arrays["plx_mas"], out=a_phot_au, where=plx_ok,
    )
    numerator = a_phot_au * sin_i
    spec_ok = (
        np.isfinite(arrays["a_spec_au"])
        & (arrays["a_spec_au"] > 0.0)
        & np.isfinite(numerator)
    )
    r = np.full(n_draws, np.nan, dtype=float)
    np.divide(numerator, arrays["a_spec_au"], out=r, where=spec_ok)
    deficit = 1.0 - r

    # ── q from the SPECTROSCOPIC mass function only (H-6) ────────────────
    mass = companion_mass_from_rv_posterior(
        joint_marginal_result,
        primary_mass_prior=primary_mass_prior,
        sin_i=_MeasuredSinI(sin_i),
        seed=seed,
    )
    m2 = np.asarray(mass["mass"]["m2"], dtype=float)

    # M-10: take the m1 draws the mass layer USED, by value.  The shared
    # seeded stream is a FALLBACK only — re-drawing reproduces the same
    # numbers solely when count, prior spec and call order all match, and
    # that is an assumption, not a guarantee.
    m1_from_layer = mass.get("m1_draws_msun")
    if m1_from_layer is None:
        warnings.warn(_M1_FALLBACK_MSG, UserWarning, stacklevel=2)
        m1, _ = _draw_primary_mass(primary_mass_prior, n_draws, seed)
        m1_source = "shared_stream_fallback"
    else:
        m1 = np.asarray(m1_from_layer, dtype=float)
        m1_source = "mass_layer"
    if m1.ndim != 1 or m1.shape[0] != n_draws:
        raise ValueError(_M1_ROW_MISMATCH_MSG)

    q = np.full(n_draws, np.nan, dtype=float)
    m1_ok = np.isfinite(m1) & (m1 > 0.0) & np.isfinite(m2)
    np.divide(m2, m1, out=q, where=m1_ok)

    branches = beta_branches_from_axis_ratio(r, q)

    # Fractions are conditional on a usable draw (finite r AND q); an
    # all-NaN denominator would otherwise read as "no boundary mass".
    usable = np.isfinite(r) & np.isfinite(q) & (q > 0.0)
    n_usable = int(np.count_nonzero(usable))

    def _frac(mask) -> float:
        if n_usable == 0:
            return float("nan")
        return float(np.count_nonzero(mask & usable) / n_usable)

    # sci-F6: the mirror branch is QUALITATIVE-only (module docstring,
    # design-note M-8), so it gets NO median/std/q16/q84 block here --
    # only the boolean/fraction recorded in _meta below.
    summary = {
        "beta_standard": _summarise(branches["standard"]),
        "r": _summarise(r),
        "q": _summarise(q),
    }
    mirror_branch_admitted = bool(
        np.any(branches["valid_mirror"] & usable)
    )

    meta_in = chains.get("_meta")
    bimodal = bool(
        isinstance(meta_in, dict) and meta_in.get("bimodality_flag", False)
    )
    summary_per_mode = None
    if bimodal and "cos_i" in chains:
        cos_i = np.asarray(chains["cos_i"], dtype=float)
        if cos_i.shape == (n_draws,):
            summary_per_mode = {
                "cos_i_positive": _mode_summary(
                    branches["standard"], r, q,
                    branches["valid_mirror"] & usable,
                    cos_i > 0.0,
                ),
                "cos_i_negative": _mode_summary(
                    branches["standard"], r, q,
                    branches["valid_mirror"] & usable,
                    cos_i <= 0.0,
                ),
            }

    # A posterior-SPREAD diagnostic, NOT a detection test: it says how
    # tight the a_phot posterior is, not whether the orbit is detected
    # against the no-orbit null.  Kept because the screen the caller must
    # apply needs a number to look at.
    a_phot_finite = arrays["a_phot_mas"][np.isfinite(arrays["a_phot_mas"])]
    if a_phot_finite.size > 0 and float(np.std(a_phot_finite)) > 0.0:
        a_phot_posterior_snr = float(
            np.median(a_phot_finite) / np.std(a_phot_finite)
        )
    else:
        a_phot_posterior_snr = float("nan")

    provenance = {
        "q_source": "fm_spec_msun_via_companion_mass_from_rv_posterior",
        "m1_draws_source": m1_source,
        "sin_i_source": "joint_chain_inc_rad",
        "r_form": "multiply_by_sin_i",
        "primary_mass_prior": (
            float(primary_mass_prior)
            if isinstance(primary_mass_prior, (int, float))
            and not isinstance(primary_mass_prior, bool)
            else str(primary_mass_prior)
        ),
        "seed": int(seed),
    }

    meta = {
        "beta_band": AMRF_BETA_BAND,
        "n_draws": int(n_draws),
        "n_draws_usable": n_usable,
        "mass_convention": mass["_meta"]["mass_convention"],
        "m1_draws_source": m1_source,
        "bimodality_flag": bimodal,
        "detection_screen": (
            "not_recorded" if a_phot_detection_significance is None
            else "recorded"
        ),
        # This layer never enforces a floor on the significance it
        # records -- it only carries the caller-supplied value through.
        "screen_threshold_applied": None,
        "a_phot_detection_significance": (
            None if a_phot_detection_significance is None
            else float(a_phot_detection_significance)
        ),
        "a_phot_posterior_snr": a_phot_posterior_snr,
        "finite_fraction_r": float(np.mean(np.isfinite(r))),
        "frac_standard_valid": _frac(branches["valid_standard"]),
        "frac_mirror_valid": _frac(branches["valid_mirror"]),
        "frac_r_outside_domain": _frac(~branches["in_domain"]),
        # The beta = 0 boundary pile-up: the noise-floor bias signature.
        "frac_r_ge_one": _frac(r >= 1.0),
        "mirror_branch_admitted": mirror_branch_admitted,
        "mirror_is_qualitative": True,
        "seed": int(seed),
    }

    return {
        "beta": {
            "standard": branches["standard"],
            "mirror": branches["mirror"],
        },
        "valid": {
            "standard": branches["valid_standard"],
            "mirror": branches["valid_mirror"],
            "in_domain": branches["in_domain"],
        },
        "derived": {
            "r": r,
            "D": deficit,
            "q": q,
            "m1_msun": m1,
            "m2_msun": m2,
            "sin_i": sin_i,
        },
        "summary": summary,
        "summary_per_mode": summary_per_mode,
        "provenance": provenance,
        "_meta": meta,
    }
