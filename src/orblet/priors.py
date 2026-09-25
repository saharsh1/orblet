"""
Prior definitions for Keplerian orbital fitting (Python engines).

This module provides default prior configurations for the Python
samplers.

All priors are specified as plain Python dictionaries mapping parameter
names to ``(distribution_name, *args)`` tuples, parsed by
``parse_prior_spec`` into the engines' prior objects.

User-facing units
-----------------
- Period (P)      : **days**
- Companion mass  : **solar masses**
- Stellar mass M  : solar masses
- Eccentricity    : dimensionless  [0, 1)
- ω (arg. peri.)  : radians  (wrapped via UniformCircular)
- τ (phase frac.) : dimensionless  [0, 1]

The engines convert internally where needed (the period latent is in
Keplerian years; the exported ``P_days`` chain column is days).

Prior behavioral contract (normative)
-------------------------------------
This prose is the authoritative behavioral contract for ANY object
used as a prior in the composable surface — the two ``typing.Protocol``
classes below (``LogPriorCallable``, ``PriorDistribution``) only pin
the call *shape*; this docstring pins the *behavior*:

This contract concerns FINITE REAL inputs only.  NaN / non-finite
input is UNDEFINED here: the built-ins make no guarantee for it (in
particular they do NOT promise ``-inf`` for NaN).  The sampler does
not rely on the prior to reject non-finite input: ``e`` / ``u_inc``
are pre-screened by explicit support guards, and every other prior
contribution is followed by a ``math.isfinite`` check that returns
``-inf`` on any non-finite accumulation (emcee then rejects the
proposal).

- A prior returns a plain Python ``float`` (a finite log-density or
  ``-inf``), never an array, ``None``, or a tuple.
- Outside the support, a prior returns ``-inf`` (``-numpy.inf``) for
  finite real input; it does NOT raise on (finite) out-of-support
  input.  emcee handles ``-inf`` posterior values cleanly (the
  proposal is simply rejected), so an exception there would crash an
  otherwise-valid sampler step.
- The built-in classes in this module (``UniformPrior``,
  ``NormalPrior``, ``TruncatedNormalPrior``, ``LogUniformPrior``,
  ``UniformCircularPrior``, ``CosUniformInclinationPrior``) are
  convenience HELPERS, NOT a mandatory framework: nothing in the
  composable path requires a prior to be an instance of them or to
  subclass them.
- A user-supplied ``def my_log_prior(theta) -> float`` (the
  ``LogPriorCallable`` shape, ``theta`` a flat ``Mapping[str, float]``)
  and any duck-typed object exposing ``.logpdf(...) -> float`` (the
  ``PriorDistribution`` shape) are FIRST-CLASS WITHOUT SUBCLASSING any
  class here.  Scoped guarantee: for **prior log-density evaluation on
  the composable surface** (``orbit.model`` / ``orbit.likelihood`` /
  ``orbit.posterior``) such priors are accepted purely on structural
  shape, never via ``isinstance``.  Accordingly neither Protocol is
  ``@runtime_checkable`` and no isinstance-gated control flow keys off
  them in the log-density path.
- Caveat (engine-internal, not the composable surface): the joint
  Python sampler's warm-start initial-spread heuristic
  (a sampler's ``_warmstart_sigma_from_prior``-style heuristic)
  special-cases the six built-in helper classes by ``isinstance`` to
  pick a tailored initial-spread width.  A user-supplied custom prior
  (callable or duck-typed) therefore stays CORRECT for the posterior
  but falls back to the DEFAULT initial spread (only the walker init
  scatter is affected, never the sampled density).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

# Re-exported for backward compatibility — callers may import these names
# from ``orblet.priors`` directly.  The canonical definitions live in
# :mod:`orblet.constants`.
from orblet.constants import MSUN_IN_MJUP, DAYS_PER_KEPLER_YEAR  # noqa: F401


# ``__all__`` declares the module's INTENDED public API: the prior
# classes, the prior-builder helpers, the re-exported constants, and
# the two prior Protocols.  It deliberately does NOT re-export the
# incidentally-imported names (``np``, ``math``, ``dataclass``,
# ``annotations``) that a no-``__all__`` ``import *`` would have leaked
# at base — those were never public API, and no consumer star-imports
# this module in orblet itself.
__all__ = [
    # Concrete prior helper classes.
    "UniformPrior",
    "NormalPrior",
    "TruncatedNormalPrior",
    "LogUniformPrior",
    "UniformInFrequencyPeriodPrior",
    "UniformCircularPrior",
    "EccOmegaDiskPrior",
    "CosUniformInclinationPrior",
    # Public builder / helper functions.
    "default_companion_priors",
    "uniform_in_frequency_period_prior",
    "default_system_priors",
    # Re-exported constants from orblet.constants.
    "MSUN_IN_MJUP",
    "DAYS_PER_KEPLER_YEAR",
    # ── C3c additions ──
    "LogPriorCallable",
    "PriorDistribution",
]


# ── Minimal prior protocol (two DISTINCT structural concepts) ────────────────
#
# These name the two shapes a prior may take on the composable surface.
# They are intentionally TWO separate Protocols, not one merged one:
# the composable path accepts a plain callable, while the built-in
# helpers expose ``.logpdf``.  The normative *behavior* (float return,
# ``-inf`` outside support, no raise) lives in the module docstring
# above — these Protocols pin only the call *shape*.
#
# Deliberately NOT ``@runtime_checkable``: a user prior is accepted on
# structural shape alone, never via ``isinstance``.  Making either
# ``@runtime_checkable`` would re-enable isinstance-gated control flow,
# which the contract forbids.  ``typing`` is stdlib — importing it here
# keeps ``import orblet.priors`` free of scipy / emcee.


class LogPriorCallable(Protocol):
    """Structural shape of a composable user log-prior.

    A plain ``def my_log_prior(theta) -> float`` satisfies this without
    subclassing.  ``theta`` is a flat mapping of parameter name to
    value.  See the module docstring for the normative behavioral
    contract (returns a float; ``-inf`` outside support; does not raise
    on out-of-support input).
    """

    def __call__(self, theta: Mapping[str, float]) -> float: ...


class PriorDistribution(Protocol):
    """Structural shape of the built-in prior helpers (and any
    duck-typed object exposing ``.logpdf``).

    Any object with a ``logpdf(*args) -> float`` method satisfies this
    without subclassing.  See the module docstring for the normative
    behavioral contract.
    """

    def logpdf(self, *args: float) -> float: ...


# ── Concrete prior classes ───────────────────────────────────────────────────
#
# The canonical prior classes.  They live in the public composable
# surface (:mod:`orblet.priors`) rather than in any channel's sampler, so
# every channel uses the same classes and a user's custom prior plugs in
# beside them.
#
# Lazy-import policy (load-bearing — verified by
# ``tests/test_python_rv_lazy_imports.py``): ``scipy`` is imported INSIDE the
# ``NormalPrior.logpdf`` / ``TruncatedNormalPrior`` method bodies, NEVER at
# module scope.  This keeps ``import orblet.priors`` free of scipy
# loads.  Do NOT hoist ``scipy`` to the module top.
#
# All ``logpdf`` methods return ``-numpy.inf`` for arguments outside the
# support — emcee handles ``-inf`` posterior values cleanly (proposals are
# simply rejected).


@dataclass(frozen=True)
class UniformPrior:
    """Uniform prior on ``[lo, hi]``.  ``logpdf`` returns ``-inf``
    outside the bounds, ``-log(hi - lo)`` inside.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (self.hi > self.lo):
            raise ValueError(
                f"UniformPrior requires hi > lo; got lo={self.lo}, hi={self.hi}"
            )

    def logpdf(self, x: float) -> float:
        if x < self.lo or x > self.hi:
            return -np.inf
        return -math.log(self.hi - self.lo)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=size)


@dataclass(frozen=True)
class NormalPrior:
    """Gaussian prior with mean ``mu`` and standard deviation ``sigma``.

    Units are those of the parameter the prior is attached to (the
    engines attach it in physics units: days, km/s, mas, M☉, radians).
    Unbounded support; ``logpdf`` includes the normalisation constant.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise ValueError(
                f"NormalPrior requires sigma > 0; got sigma={self.sigma}"
            )

    def logpdf(self, x: float) -> float:
        # Lazy scipy import keeps module-import cheap.
        from scipy.stats import norm
        return float(norm.logpdf(x, self.mu, self.sigma))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, size=size)


@dataclass(frozen=True)
class TruncatedNormalPrior:
    """Truncated Gaussian on ``[lower, upper]``.

    Both bounds are required.  For one-sided truncation, pass a finite
    bound on the active side and ``+inf`` / ``-inf`` on the other.
    """

    mu: float
    sigma: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise ValueError(
                f"TruncatedNormalPrior requires sigma > 0; got "
                f"sigma={self.sigma}"
            )
        if not (self.upper > self.lower):
            raise ValueError(
                f"TruncatedNormalPrior requires upper > lower; got "
                f"lower={self.lower}, upper={self.upper}"
            )

    def logpdf(self, x: float) -> float:
        if x < self.lower or x > self.upper:
            return -np.inf
        from scipy.stats import truncnorm
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        return float(truncnorm.logpdf(x, a, b, loc=self.mu, scale=self.sigma))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        from scipy.stats import truncnorm
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        return truncnorm.rvs(
            a, b, loc=self.mu, scale=self.sigma, size=size, random_state=rng,
        )


@dataclass(frozen=True)
class LogUniformPrior:
    """Log-uniform (Jeffreys) prior on ``[lo, hi]`` with ``lo > 0``.

    PDF is ``p(x) = 1 / (x * log(hi / lo))`` on the support, so
    ``logpdf(x) = -log(x) - log(log(hi) - log(lo))``.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (self.lo > 0 and self.hi > self.lo):
            raise ValueError(
                f"LogUniformPrior requires 0 < lo < hi; got "
                f"lo={self.lo}, hi={self.hi}"
            )

    def logpdf(self, x: float) -> float:
        if x <= 0 or x < self.lo or x > self.hi:
            return -np.inf
        return -math.log(x) - math.log(math.log(self.hi) - math.log(self.lo))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=size)
        return self.lo * (self.hi / self.lo) ** u


@dataclass(frozen=True)
class UniformInFrequencyPeriodPrior:
    """Period prior that is UNIFORM IN FREQUENCY ``f = 1/P`` on ``[lo, hi]``.

    This is a PHYSICS-SPACE prior on the period ``P`` (DAYS) whose density
    is ``∝ 1/P²`` — the change-of-variables image of a Uniform density on
    ``f = 1/P`` (``df = -dP/P²``).  On the support ``[lo, hi]`` (DAYS) the
    pdf is

        ``p(P) = (1/P²) / (1/lo − 1/hi)``,

    so ``logpdf(P) = -2·log P − log(1/lo − 1/hi)`` inside, ``-inf`` outside.

    NOT uninformative — it FAVORS SHORT PERIODS (``∝ 1/P²``)
    --------------------------------------------------------
    A flat-in-frequency prior is the proper "uniform" prior in the
    FREQUENCY representation, but in PERIOD space it puts dramatically more
    mass on short periods (the ``1/P²`` weighting).  It is therefore NOT
    the uninformative / scale-invariant choice (that is the log-uniform
    ``∝ 1/P`` Jeffreys prior).  Choose this prior deliberately when a
    flat-in-frequency belief is intended; do not reach for it as a default.

    Independent of the SAMPLING representation
    ------------------------------------------
    This is a statement about the PRIOR DENSITY on the physical period,
    NOT about the latent ``transform`` (``"linear"`` / ``"log"`` /
    ``"freq"``).  The same physics-space prior can be sampled in ANY
    representation; the engine adds the correct change-of-variables
    Jacobian per representation, so the posterior on ``P`` is identical
    whichever sampling representation is used.  In particular, using the
    ``param_transform={"P": "freq"}`` SAMPLING representation does NOT
    imply this prior, and using this prior does NOT require the ``"freq"``
    sampling representation.

    The ``logpdf`` convention follows the rest of this module: a plain
    Python ``float``; ``-inf`` for finite real input outside the support;
    no raise on out-of-support input.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not (self.lo > 0 and self.hi > self.lo):
            raise ValueError(
                "UniformInFrequencyPeriodPrior requires 0 < lo < hi; got "
                f"lo={self.lo}, hi={self.hi}"
            )

    def logpdf(self, x: float) -> float:
        if x <= 0 or x < self.lo or x > self.hi:
            return -np.inf
        # Normalisation: ∫_lo^hi P^-2 dP = 1/lo − 1/hi.
        norm = 1.0 / self.lo - 1.0 / self.hi
        return -2.0 * math.log(x) - math.log(norm)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        # Uniform in frequency f ∈ [1/hi, 1/lo], then P = 1/f.
        f_lo = 1.0 / self.hi
        f_hi = 1.0 / self.lo
        f = rng.uniform(f_lo, f_hi, size=size)
        return 1.0 / f


def uniform_in_frequency_period_prior(
    P_lo: float, P_hi: float,
) -> UniformInFrequencyPeriodPrior:
    """Build a uniform-in-frequency period prior on ``[P_lo, P_hi]`` (DAYS).

    Convenience constructor for :class:`UniformInFrequencyPeriodPrior`.
    The returned prior is a PHYSICS-SPACE density on the period ``P`` that
    is uniform in ``f = 1/P`` — i.e. ``∝ 1/P²``.

    IMPORTANT: this prior FAVORS SHORT PERIODS (``∝ 1/P²``); it is NOT
    uninformative (the scale-invariant / uninformative choice is the
    log-uniform ``∝ 1/P`` prior).  It is INDEPENDENT of the sampling
    representation: it states a belief about the physical period and can
    be sampled in the ``"linear"``, ``"log"`` or ``"freq"`` latent
    representation alike (the engine supplies the correct
    change-of-variables Jacobian per representation).

    Parameters
    ----------
    P_lo, P_hi : float
        Period support bounds in DAYS, ``0 < P_lo < P_hi``.

    Returns
    -------
    UniformInFrequencyPeriodPrior
        The ``∝ 1/P²`` period prior.
    """
    return UniformInFrequencyPeriodPrior(lo=float(P_lo), hi=float(P_hi))


@dataclass(frozen=True)
class UniformCircularPrior:
    """Uniform-on-the-unit-disk reparametrization.

    Used for ``ω`` (and τ-phase, which we draw on the same disk).  The
    angle is recovered via
    :func:`angle_from_xy`
    (or :func:`...tau_from_xy` for τ).

    The pdf is ``1/π`` on the unit disk and zero outside; ``logpdf``
    returns ``-log(π)`` inside, ``-inf`` outside.
    """

    def logpdf(self, x: float, y: float) -> float:
        if x * x + y * y > 1.0:
            return -np.inf
        return -math.log(math.pi)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        # Uniform on the unit disk via inverse-CDF on r and uniform θ.
        r = np.sqrt(rng.uniform(0.0, 1.0, size=size))
        theta = rng.uniform(0.0, 2 * math.pi, size=size)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


@dataclass(frozen=True)
class EccOmegaDiskPrior:
    """Joint (e, ω) reparametrization on the unit disk via ``(h, k)``.

    The RadVel / exoplanet field-standard √e–ω disk.  The
    two latents are

        ``h = √e · cos ω``,   ``k = √e · sin ω``

    so that the eccentricity and argument of periastron are recovered as

        ``e = h² + k²``,   ``ω = atan2(k, h)``.

    The disk RADIUS therefore carries ``√e`` — this DISTINGUISHES this
    prior from :class:`UniformCircularPrior` (used for ω and τ), where
    the radius is an unidentifiable phase magnitude with no physical
    meaning.  Here the radius is load-bearing: it encodes ``√e``.

    Density / Jacobian
    ------------------
    A point sampled UNIFORMLY on the unit disk in ``(h, k)`` induces,
    via the change of variables ``dh dk = ½ de dω`` (the Jacobian of
    ``(h, k) → (e, ω)``), a density that is **uniform in e on [0, 1] AND
    uniform in ω on [0, 2π]**.  This is exactly the ``Uniform(0, 1)``
    eccentricity prior the engine intends — the ``√e`` enters ONLY
    through the ``e = h² + k²`` decode downstream, never as an extra
    density factor here.

    Because the uniform-on-the-disk density in ``(h, k)`` is the SAME as
    :class:`UniformCircularPrior` (``1/π`` on the unit disk, zero
    outside), the ``logpdf`` is IDENTICAL: ``-log(π)`` inside the unit
    disk, ``-inf`` on/outside it.  The eccentricity-prior Jacobian is
    absorbed by this uniform-disk density together with the downstream
    ``e = h² + k²`` decode; no separate Jacobian term is added.

    The hard support ``h² + k² < 1`` enforces ``e < 1`` automatically
    (the open unit disk), so the eccentricity bound is carried by the
    disk geometry rather than a scalar support guard.
    """

    def logpdf(self, x: float, y: float) -> float:
        # Uniform on the open unit disk: -log(π) inside, -inf on/outside.
        # IDENTICAL to UniformCircularPrior (the √e enters only via the
        # downstream e = x² + y² decode, not the density).
        if x * x + y * y > 1.0:
            return -np.inf
        return -math.log(math.pi)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        # Uniform on the unit disk via inverse-CDF on r and uniform θ
        # (mirrors UniformCircularPrior.sample); the resulting (h, k)
        # decode to e uniform on [0, 1] and ω uniform on [0, 2π].
        r = np.sqrt(rng.uniform(0.0, 1.0, size=size))
        theta = rng.uniform(0.0, 2 * math.pi, size=size)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


class CosUniformInclinationPrior:
    """Isotropic full-sphere inclination prior (``Sine`` on ``[0, π]``).

    The latent ``u`` is sampled uniformly on ``[0, 1]``; the
    corresponding inclination is

        i = arccos(1 − 2 u) ∈ [0, π]

    so that ``cos i = 1 − 2u`` is uniform on ``[-1, +1]`` (isotropic
    orientation).  The implied density on the original ``i`` axis is

        p(i) = (1/2) · sin(i)        on ``[0, π]``

    equivalent to a Sine distribution on ``[0, π]`` (the standard
    isotropic choice).  The prior on the latent ``u`` is
    uniform-on-``[0, 1]`` (``logpdf = 0``) within the support and
    ``-inf`` outside.

    The full-sphere mapping matters: an upper-hemisphere mapping
    ``i = arccos(1 − u) ∈ [0, π/2]`` would truncate the sphere and
    silently select one side of the (i, ω, Ω) ↔ (π−i, π−ω, π−Ω)
    photocenter mirror.  The ``_meta["i_domain"]`` chain sentinel
    reports ``I_DOMAIN_FULL``.  See the reference manual, §4.3 and §7
    item 9, for the mirror.

    This class is intentionally minimal: ``logpdf`` operates on the
    latent ``u`` and includes neither the data-likelihood nor the
    Jacobian (callers handle the ``u → i`` transform once at the
    forward-model boundary, not at every prior evaluation).

    Examples
    --------
    >>> prior = CosUniformInclinationPrior()
    >>> prior.logpdf(0.5)            # in support (u = 0.5 ⇒ i = π/2)
    0.0
    >>> import math
    >>> math.isinf(prior.logpdf(1.5))  # out of support
    True
    """

    def logpdf(self, u: float) -> float:
        if u < 0.0 or u > 1.0:
            return -np.inf
        return 0.0

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=size)

    @staticmethod
    def u_to_inc(u: float) -> float:
        """Map latent ``u`` to inclination ``i = arccos(1 − 2 u)`` in
        radians, ``i ∈ [0, π]``.  Inverse of :func:`inc_to_u`.
        Full-sphere mapping.
        """
        return float(math.acos(1.0 - 2.0 * u))

    @staticmethod
    def inc_to_u(inc: float) -> float:
        """Map inclination ``i ∈ [0, π]`` to latent
        ``u = (1 − cos(i)) / 2 ∈ [0, 1]``.  Inverse of :func:`u_to_inc`.
        Full-sphere mapping.
        """
        return float((1.0 - math.cos(inc)) / 2.0)


def default_companion_priors(
    period_days: float,
    *,
    period_bounds_factor: float = 3.0,
    ecc_max: float = 0.99,
    mass_range_msun: tuple[float, float] = (0.0001, 5.0),
) -> dict:
    """
    Return default priors for a single companion (planet "b").

    All values are in **user-facing units**: period in days, masses in
    solar masses.  The engines convert internally (the period latent
    is in Keplerian years; the exported ``P_days`` column is days).

    Parameters
    ----------
    period_days : float
        Initial period estimate in days (e.g. from Lomb-Scargle).
    period_bounds_factor : float, default 3.0
        The period prior spans
        ``[period_days / factor, period_days * factor]``.
    ecc_max : float, default 0.99
        Upper bound on eccentricity.  Must be < 1.
    mass_range_msun : tuple of float, default (0.0001, 5.0)
        (min, max) companion mass in **solar masses**.
        The default ``(0.0001, 5.0)`` covers sub-stellar to
        ordinary stellar companions; it EXCLUDES stellar-mass black
        holes (and any companion above 5 M☉).  Override it explicitly
        for compact-object work.  Nothing in orblet flags a posterior
        piled against the 5 M☉ edge: check the draws against the bound
        yourself.  Draws sitting at it mean the prior, not the data,
        set the mass, and a mildly truncated posterior is easy to
        miss by eye.

    Returns
    -------
    priors : dict
        Keys are engine parameter names; values are tuples of
        ``(distribution_name, *args)``.  Period is in **days**,
        companion mass is in **solar masses**.
    """
    p_lo = period_days / period_bounds_factor
    p_hi = period_days * period_bounds_factor

    return {
        "P":    ("LogUniform", p_lo, p_hi),
        "e":    ("Uniform", 0.0, ecc_max),
        "ω":    ("UniformCircular",),
        "τ":    ("Uniform", 0.0, 1.0),
        "mass": ("LogUniform", mass_range_msun[0], mass_range_msun[1]),
    }


def default_system_priors(
    stellar_mass_msun: float = 1.0,
    stellar_mass_err: float = 0.2,
) -> dict:
    """
    Return default system-level priors for RV orbit fitting.

    The system mass ``M`` is the **total** mass (M_star + M_comp),
    matching the astrometric convention used by ``Visual{KepOrbit}``.
    This ensures correct Kepler III and RV semi-amplitude computation
    for massive companions (e.g. black holes), where M_comp is not
    negligible relative to M_star.

    The default prior is a truncated Normal centred on the stellar
    mass estimate.  For massive companions, the user should widen
    this prior (e.g. ``LogUniform(0.5, 50)``).

    Parameters
    ----------
    stellar_mass_msun : float, default 1.0
        Central value of the total-mass prior (solar masses).
        For low-mass companions this is approximately the stellar mass.
    stellar_mass_err : float, default 0.2
        1-σ width of the truncated-normal total-mass prior.

    Returns
    -------
    priors : dict
        Keys are engine parameter names; values are tuples of
        ``(distribution_name, *args)``.
    """
    return {
        "M": ("truncated_Normal", stellar_mass_msun, stellar_mass_err, 0.1),
    }


# ── Spec parsing and the unit-disk recovery helpers ──────────────────────────
#
# Three helpers every channel shares:
#
# - ``angle_from_xy`` and ``tau_from_xy`` recover an angle, and the
#   periastron phase τ ∈ [0, 1), from a point on the unit disk — the latent
#   pair the samplers use so that no angle has a wrap-around boundary;
# - ``parse_prior_spec`` turns a prior tuple such as
#   ``("truncated_Normal", mu, sigma, low)`` into one of the prior classes
#   defined ABOVE in this module.
#
# Why they belong here: ``parse_prior_spec`` is shared by every channel
# (RV, astrometric, joint) and constructs this module's classes.  Living
# inside one channel would force the others to import across a channel
# boundary, which is how import cycles start.  This module is a leaf (its
# only project import is ``orblet.constants``), so importing it cannot
# create a cycle.  Contract: ``tests/test_prior_spec_parser_contract.py``.
#


def angle_from_xy(x: float, y: float) -> float:
    """Return ``atan2(y, x)`` in radians (range ``[-π, π]``)."""
    return float(math.atan2(y, x))


def tau_from_xy(x: float, y: float) -> float:
    """Return ``atan2(y, x) / (2π) mod 1``, the phase fraction in ``[0, 1)``."""
    return float((math.atan2(y, x) / (2 * math.pi)) % 1.0)


def parse_prior_spec(spec: tuple) -> Any:
    """Parse a ``(distribution_name, *args)`` tuple into a prior object.

    Accepts the same dialect as :func:`default_companion_priors`:

    - ``("Uniform", lo, hi)``
    - ``("LogUniform", lo, hi)``
    - ``("Normal", mu, sigma)``
    - ``("UniformCircular",)``
    - ``("truncated_Normal", mu, sigma, lower)``
      or ``("truncated_Normal", mu, sigma, lower, upper)``

    Returns
    -------
    A prior object exposing a ``logpdf`` method.

    Raises
    ------
    ValueError
        If the distribution name is unrecognised or the argument count
        does not match.
    """
    name = spec[0]
    args = spec[1:]
    if name == "Uniform":
        if len(args) != 2:
            raise ValueError(
                f"Uniform expects 2 args (lo, hi); got {args!r}"
            )
        return UniformPrior(float(args[0]), float(args[1]))
    if name == "LogUniform":
        if len(args) != 2:
            raise ValueError(
                f"LogUniform expects 2 args (lo, hi); got {args!r}"
            )
        return LogUniformPrior(float(args[0]), float(args[1]))
    if name == "Normal":
        if len(args) != 2:
            raise ValueError(
                f"Normal expects 2 args (mu, sigma); got {args!r}"
            )
        return NormalPrior(float(args[0]), float(args[1]))
    if name == "UniformCircular":
        return UniformCircularPrior()
    if name == "truncated_Normal":
        if len(args) == 3:
            mu, sigma, lower = args
            return TruncatedNormalPrior(
                float(mu), float(sigma), float(lower), float("inf"),
            )
        if len(args) == 4:
            mu, sigma, lower, upper = args
            return TruncatedNormalPrior(
                float(mu), float(sigma), float(lower), float(upper),
            )
        raise ValueError(
            "truncated_Normal expects 3 or 4 args "
            f"(mu, sigma, lower [, upper]); got {args!r}"
        )
    raise ValueError(f"Unknown prior distribution name: {name!r}")
