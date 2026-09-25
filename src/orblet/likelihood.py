"""Pre-built log-likelihood shortcuts for the common Gaussian-with-jitter case.

The functions re-exported here are **one specific choice** of likelihood
(Gaussian residuals with jitter added in quadrature), not the canonical
likelihood for this package.  The actual building blocks are the pure
forward-model functions in :mod:`orblet.model`; users are expected
to write their own ``log_lik(theta)`` when they need anything other than
the Gaussian-jitter form (Student-t, Huber, custom outlier model, ...).

Typical custom-likelihood pattern in a notebook::

    from orblet.model import rv_model
    import numpy as np

    def my_log_lik(theta):
        rv_pred = rv_model(t_mjd, **theta_kwargs, epoch_ref_mjd=...)
        resid = rv_obs - rv_pred
        # whatever you want: chi², Student-t, robust, ...
        return -0.5 * np.sum((resid / rv_err) ** 2)

The convenience shortcuts here exist so that the common Gaussian case
does not require boilerplate, and so that the ``fit_*_orbit`` wrappers
have a default likelihood to compose.

Calling convention
------------------
All re-exported functions use flat keyword arguments.  See each engine
function's docstring for parameter names and units.

Lazy-import contract
--------------------
Imports of this module must stay cheap (no scipy / no emcee pulled at
module load), matching the PEP 562 ``__getattr__`` template in
:mod:`orblet.priors` and verified by
``tests/test_public_composable_api.py``.
"""

from __future__ import annotations

import math
from typing import Mapping, Protocol

import numpy as np

from orblet.model import (
    _along_scan_for_theta_campbell,
    joint_theta_view,
    rv_model,
)

# ── Minimal log-likelihood protocol (one structural concept) ─────────────────
#
# This pins ONLY the call *shape* of a composable user log-likelihood —
# the LIKELIHOOD-side mirror of ``LogPriorCallable`` in
# :mod:`orblet.priors`.  Deliberately NOT ``@runtime_checkable``:
# a user likelihood is accepted on structural shape alone, never via
# ``isinstance``.  ``typing`` is stdlib, so importing it here keeps
# ``import orblet.likelihood`` free of scipy / emcee (lazy-import
# contract above is preserved).
#
# Exactly ONE protocol is added here (no ``PriorDistribution`` mirror):
# the composable likelihood path accepts a plain callable only; there is
# no ``.logpdf``-style built-in likelihood object to mirror.


class LogLikelihoodCallable(Protocol):
    """Structural shape of a composable user log-likelihood.

    A plain ``def my_log_lik(theta) -> float`` satisfies this without
    subclassing.  ``theta`` is a flat ``Mapping[str, float]`` of
    physical parameter name to value (e.g. the period under ``P_days``
    in DAYS).  Returns a float log-likelihood.

    Contrast with :class:`orblet.priors.LogPriorCallable`: a
    custom likelihood carries NO change-of-variables Jacobian — it is a
    function of the PHYSICAL parameters, and the per-log-slot ``+raw``
    latent-space Jacobian lives entirely on the prior side.
    """

    def __call__(self, theta: Mapping[str, float]) -> float: ...


# PEP 562 lazy re-exports.  See the note in :mod:`orblet.model`
# and the template in :mod:`orblet.priors`: the
# likelihood modules transitively pull scipy through the forward
# model, so we defer the import until first attribute access.

# ══════════════════════════════════════════════════════════════════════
# RADIAL VELOCITY
# ══════════════════════════════════════════════════════════════════════


def loglike(
    t_mjd: np.ndarray,
    rv_obs: np.ndarray,
    rv_err: np.ndarray,
    *,
    period_yr: float,
    ecc: float,
    omega_rad: float,
    tau: float,
    mass_msun: float,
    M_msun: float,
    offset_kms: float,
    jitter_kms: float,
    epoch_ref_mjd: float,
) -> float:
    """Return the log-likelihood for the RV dataset under the given orbit.

    Parameters
    ----------
    t_mjd : np.ndarray
        Observation times (MJD).
    rv_obs : np.ndarray
        Observed RVs (km/s).
    rv_err : np.ndarray
        Per-epoch RV uncertainties (km/s).  Combined in quadrature with
        ``jitter_kms``.
    period_yr, ecc, omega_rad, tau, mass_msun, M_msun, offset_kms : float
        Orbital parameters; see
        :func:`orblet.model.rv_model`.
    jitter_kms : float
        RV jitter in km/s, combined in quadrature with ``rv_err``.
    epoch_ref_mjd : float
        Reference MJD for the τ → tp conversion.

    Returns
    -------
    float
        The Gaussian log-likelihood.  Returns ``-inf`` if ``ecc`` is
        outside ``[0, 1)`` (defensive guard; should be screened by the
        prior, but emcee proposals can wander out-of-support), if the
        jitter is negative, or if the result is not finite (a NaN
        parameter or model): a sampler must see a rejected step, never
        a NaN.  Finite results are unchanged.
    """
    e = float(ecc)
    if not (0.0 <= e < 1.0):
        return -math.inf
    jitter = float(jitter_kms)
    if jitter < 0.0:
        return -math.inf

    rv_pred = rv_model(
        t_mjd,
        period_yr=period_yr,
        ecc=ecc,
        omega_rad=omega_rad,
        tau=tau,
        mass_msun=mass_msun,
        M_msun=M_msun,
        offset_kms=offset_kms,
        epoch_ref_mjd=epoch_ref_mjd,
    )
    rv_err = np.asarray(rv_err, dtype=float)
    rv_obs = np.asarray(rv_obs, dtype=float)
    sigma_eff_sq = rv_err * rv_err + jitter * jitter
    resid = rv_obs - rv_pred
    # Standard Gaussian log-likelihood with per-epoch uncertainty:
    #   logL = -1/2 [ Σ resid²/σ_eff² + Σ log(2π σ_eff²) ].
    quad = float(np.sum(resid * resid / sigma_eff_sq))
    norm_term = float(np.sum(np.log(2.0 * np.pi * sigma_eff_sq)))
    loglike_value = -0.5 * (quad + norm_term)
    # A NaN parameter or model propagates to a NaN sum; emcee raises on
    # NaN, so hand back a rejected step instead.
    if not math.isfinite(loglike_value):
        return -math.inf
    return loglike_value


#: The public spelling of the RV log-likelihood. The engine function keeps
#: its bare name `loglike`; `rv_loglike` is the symmetric public name, as it
#: was when this module re-exported it lazily.
rv_loglike = loglike


# ══════════════════════════════════════════════════════════════════════
# ASTROMETRY
# ══════════════════════════════════════════════════════════════════════


def loglike_along_scan(
    *,
    model_along_scan: np.ndarray,
    centroid_pos: np.ndarray,
    centroid_pos_err: np.ndarray,
    jitter_mas: float,
) -> float:
    """Return the Gaussian along-scan log-likelihood.

    Parameters
    ----------
    model_along_scan : np.ndarray
        Predicted along-scan offset in mas.  Shape ``(n_epochs,)``.
    centroid_pos : np.ndarray
        Observed along-scan centroid positions in mas.
    centroid_pos_err : np.ndarray
        Per-epoch centroid uncertainties in mas.
    jitter_mas : float
        Astrometric jitter in mas, combined in quadrature with
        ``centroid_pos_err``.

    Returns
    -------
    float
        Sum of per-epoch Gaussian log-likelihoods (nats).  Returns
        ``-inf`` for negative jitter (defensive guard; the prior should
        also block it), and whenever the result is not finite (a NaN in
        the model, e.g. from a non-finite parallax factor upstream, or a
        NaN jitter): a sampler must see a rejected step, never a NaN.
        Finite results are unchanged.
    """
    if jitter_mas < 0.0:
        return -math.inf
    centroid_pos = np.asarray(centroid_pos, dtype=float)
    centroid_pos_err = np.asarray(centroid_pos_err, dtype=float)
    model = np.asarray(model_along_scan, dtype=float)

    sigma_eff_sq = centroid_pos_err * centroid_pos_err + jitter_mas * jitter_mas
    resid = centroid_pos - model
    quad = float(np.sum(resid * resid / sigma_eff_sq))
    norm_term = float(np.sum(np.log(2.0 * np.pi * sigma_eff_sq)))
    loglike_value = -0.5 * (quad + norm_term)
    # A NaN model or jitter propagates to a NaN sum; emcee raises on NaN,
    # so hand back a rejected step instead.
    if not math.isfinite(loglike_value):
        return -math.inf
    return loglike_value

# ══════════════════════════════════════════════════════════════════════
# JOINT RV + ASTROMETRY
#
# The joint log-likelihood is the algebraic SUM of the two above, with
# the inclination shared: the RV side scales K by sin i, the astrometric
# side projects through cos i.
# ══════════════════════════════════════════════════════════════════════


def loglike_joint(
    theta: dict,
    *,
    rv_arrays: dict,
    astro_arrays: dict,
    epoch_ref_mjd: float,
) -> float:
    """Return the joint RV+astrometric log-likelihood at ``theta``.

    Dark companion assumed (β = 0).  The astrometric channel goes through
    the Campbell forward (``_along_scan_for_theta_campbell`` →
    ``campbell_xy``), whose amplitude is the PRIMARY's orbit ``a_1``
    taken as the photocentre orbit.  The full joint fit therefore
    hard-wires ``a_phot = a_1``: on a luminous pair the recovered
    ``m2_msun`` is biased low, and the result cannot by itself establish
    that the companion is dark.

    Parameters
    ----------
    theta : dict
        Joint-fit theta dict (see :func:`...joint.forward.joint_theta_view`
        for the schema).  ``inc_rad`` is the shared inclination — the
        RV side scales K by ``sin(inc_rad)`` and the astrometric side
        projects via ``cos(inc_rad)`` inside the Thiele-Innes
        constants.
    rv_arrays : dict
        Must contain ``"t_mjd"`` (np.ndarray of MJD), ``"rv_obs"``
        (km/s), ``"rv_err"`` (km/s).  Shape-compatible with the inputs
        of :func:`...rv.likelihood.loglike`.
    astro_arrays : dict
        Must contain ``"t_mjd"``, ``"psi"`` (rad), ``"parallax_factor_al"``,
        ``"centroid_pos"`` (mas), ``"centroid_pos_err"`` (mas).
        Shape-compatible with the inputs consumed by
        :func:`...astro.forward._along_scan_for_theta_campbell`.
    epoch_ref_mjd : float
        Shared reference MJD for the τ → tp conversion in both engines.

    Returns
    -------
    float
        Sum of the RV and astrometric Gaussian log-likelihoods (nats).
        Returns ``-inf`` if either channel returns ``-inf`` (the
        downstream sampler short-circuits on non-finite posteriors).
    """
    rv_theta, astro_theta = joint_theta_view(theta, epoch_ref_mjd=epoch_ref_mjd)

    # Unpack the narrow RV-view dict at the call site (C2 flat-kwarg API).
    # ``joint_theta_view`` still returns dicts so chain-export consumers
    # are unaffected; the unpack lives only here.
    ll_rv = rv_loglike(
        np.asarray(rv_arrays["t_mjd"], dtype=float),
        np.asarray(rv_arrays["rv_obs"], dtype=float),
        np.asarray(rv_arrays["rv_err"], dtype=float),
        period_yr=rv_theta["P_yr"],
        ecc=rv_theta["e"],
        omega_rad=rv_theta["omega_rad"],
        tau=rv_theta["tau"],
        mass_msun=rv_theta["m2_msun"],
        M_msun=rv_theta["M_total_msun"],
        offset_kms=rv_theta["gamma_kms"],
        jitter_kms=rv_theta["rv_jitter_kms"],
        epoch_ref_mjd=float(epoch_ref_mjd),
    )
    if not math.isfinite(ll_rv):
        return -math.inf

    model_along = _along_scan_for_theta_campbell(
        astro_theta,
        t_mjd=np.asarray(astro_arrays["t_mjd"], dtype=float),
        psi=np.asarray(astro_arrays["psi"], dtype=float),
        parallax_factor_al=np.asarray(
            astro_arrays["parallax_factor_al"], dtype=float,
        ),
        epoch_ref_mjd=float(epoch_ref_mjd),
    )
    ll_astro = loglike_along_scan(
        model_along_scan=model_along,
        centroid_pos=np.asarray(astro_arrays["centroid_pos"], dtype=float),
        centroid_pos_err=np.asarray(
            astro_arrays["centroid_pos_err"], dtype=float,
        ),
        jitter_mas=float(astro_theta["astro_jitter_mas"]),
    )
    if not math.isfinite(ll_astro):
        return -math.inf

    return float(ll_rv) + float(ll_astro)


#: The names this module advertises. `loglike` and `loglike_joint` are
#: defined above and importable by name; `rv_loglike` is the advertised
#: spelling of the first.
__all__ = [
    "rv_loglike",
    "loglike_along_scan",
    "LogLikelihoodCallable",
]
