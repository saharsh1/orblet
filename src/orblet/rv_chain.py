"""RV-chain postprocessing helpers for the joint RV+astrometric workflow.

This module consumes RV posterior chains (``dict[str, np.ndarray]``
with keys ``P_days`` [days], ``e``, ``omega_rad`` [radians],
``tp_mjd`` [MJD], ``K_kms`` [km/s]) and produces:

- :func:`a1_sini_from_rv_chain`: projected primary semi-major axis
  ``a1·sin(i)`` in AU (and mas, given a parallax).
- :func:`astrometric_priors_from_rv_chain`: 5-tuple ``truncated_Normal``
  priors on ``(P, e, ω)`` keyed for the joint-fit prior surface.  ``P``
  is returned in **days** (the user-facing unit).  ``τ`` is intentionally
  omitted (Kepler-basis incompatibility — see the function docstring).

All helpers are pure NumPy.
Pinned by ``tests/test_rv_chain.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from orblet.constants import (
    AU_M,
    OMEGA_CONVENTION_PRIMARY,
)
from orblet.kepler import _wrap_to_pi


# AU per (km/s · day): 86400 s/day × 1000 m/km / AU_M [m].
# Bit-equivalent to the notebook's literal 86400.0 / 1.495978707e8 —
# see ``tests/test_rv_chain.py::test_au_per_kms_day_bit_equivalent``.
_AU_PER_KMS_DAY: float = 86400.0 / (AU_M / 1000.0)


def _med_sigma(arr: np.ndarray, sigma_widen: float) -> tuple[float, float]:
    """Median + IQR-half-width estimator: σ = (q84 − q16) / 2 × ``sigma_widen``."""
    arr = np.asarray(arr, dtype=float)
    median = float(np.median(arr))
    sigma = 0.5 * float(np.quantile(arr, 0.84) - np.quantile(arr, 0.16))
    sigma *= float(sigma_widen)
    return median, sigma


def _trunc(
    median: float,
    sigma: float,
    n_sigma: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[str, float, float, float, float]:
    """Build a 5-tuple ``("truncated_Normal", μ, σ, lower, upper)``.

    ``lo`` / ``hi`` clip the ±n_sigma bounds to physically-allowed ranges
    (e.g. ``lo=0`` for a non-negative parameter).
    """
    lower = median - n_sigma * sigma
    upper = median + n_sigma * sigma
    if lo is not None:
        lower = max(lower, lo)
    if hi is not None:
        upper = min(upper, hi)
    return ("truncated_Normal", median, sigma, lower, upper)


def a1_sini_from_rv_chain(
    chain: dict[str, np.ndarray],
    parallax_mas: float | None = None,
) -> dict[str, Any]:
    """Compute ``a1·sin(i)`` [AU] (and optionally [mas]) from an RV chain.

    Formula
    -------
        a1 sin(i) [AU] = K · P · sqrt(1 − e²) / (2π)        # K [km/s] × P [days]
        a1 sin(i) [mas] = a1 sin(i) [AU] × parallax [mas]

    Parameters
    ----------
    chain : dict of np.ndarray
        Output of ``fit_rv_orbit(...)["chains"]``.  Required keys:
        ``"P_days"`` (days),
        ``"e"`` (dimensionless), ``"K_kms"`` (km/s).
    parallax_mas : float, optional
        If given, the returned dict additionally contains the
        ``a1_sini_mas`` array.

    Returns
    -------
    dict
        ``{"a1_sini_au": ndarray, "summary_au": {median, q16, q84}}``
        plus, if parallax given:
        ``{"a1_sini_mas": ndarray, "summary_mas": {...}}``.

    Notes
    -----
    The ``sqrt(1 − e²)`` factor returns NaN for ``e > 1`` (unphysical
    samples) and exactly 0 at ``e = 1.0``.  No clamping is applied —
    callers should filter their chain before calling if unphysical
    samples are present.
    """
    P_days = np.asarray(chain["P_days"], dtype=float)   # days
    K_kms = np.asarray(chain["K_kms"], dtype=float)  # already km/s in the chain
    e = np.asarray(chain["e"], dtype=float)

    a1_sini_au = K_kms * P_days * np.sqrt(1.0 - e ** 2) / (2.0 * math.pi) * _AU_PER_KMS_DAY

    def _q(arr: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(arr)),
            "q16": float(np.quantile(arr, 0.16)),
            "q84": float(np.quantile(arr, 0.84)),
        }

    out: dict[str, Any] = {"a1_sini_au": a1_sini_au, "summary_au": _q(a1_sini_au)}
    if parallax_mas is not None:
        a1_sini_mas = a1_sini_au * float(parallax_mas)
        out["a1_sini_mas"] = a1_sini_mas
        out["summary_mas"] = _q(a1_sini_mas)
    return out


def predicted_k_kms_from_astrometric_chain(
    chain: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Predicted primary RV semi-amplitude ``K`` [km/s] from a TI chain.

    The exact inverse of :func:`a1_sini_from_rv_chain`, under the
    **dark-companion (β = 0) assumption**.  This is the astrometry→RV
    cross-check: given an astrometric photocentre orbit, what primary RV
    curve does it predict?  Overlaying the measured RVs on that prediction
    validates the astrometric orbit spectroscopically (the mirror of using
    astrometry to validate a spectroscopic orbit).

    Physics
    -------
    For a DARK companion the photocentre coincides with the primary, so the
    photocentre semi-major axis ``a_phot`` equals the primary's barycentric
    ``a₁``.  Inverting the RV projected-size relation
    (``a₁ sin i = K P √(1−e²) / 2π``)::

        a₁ sin i [AU] = a_phot [AU] × sin(i)            # β = 0: a_phot = a₁
        K [km/s]      = a₁ sin i × 2π / (P [days] × √(1−e²) × _AU_PER_KMS_DAY)

    with ``_AU_PER_KMS_DAY`` the same constant :func:`a1_sini_from_rv_chain`
    uses, so ``K → a₁ sin i → K`` round-trips to machine precision.

    **β > 0 is NOT supported here.**  For a LUMINOUS companion
    ``a_phot = a₁ |1 − β/B|`` (``β`` the companion LIGHT fraction, ``B`` its
    MASS fraction), so recovering ``a₁`` needs the mass fraction — which
    astrometry alone does not provide.  In the **faint-secondary regime**
    ``β < B`` (which includes the compact-object case and typical dim
    MS/WD companions) the photocentre is suppressed, ``a_phot ≤ a₁``, and
    the predicted ``K`` is a *lower bound*: a measured RV AMPLITUDE larger
    than the prediction is the expected signature of such a blend, not a
    contradiction.  For a BRIGHT secondary (``β > B``) the photocentre
    motion can vanish (``β = B``) or reverse, so ``a_phot`` may exceed
    ``a₁`` and the bound direction flips — but that SB2 / obvious-blend
    regime is flagged separately by spectroscopy.

    **Node degeneracy (caller beware).**  The along-scan Thiele-Innes
    amplitudes are invariant under ``(ω, Ω) → (ω + π, Ω + π)``, so an
    astrometry-only chain fixes ``ω`` on only ONE of two branches.  This
    predicted ``K`` (a magnitude) is unaffected, but any RV CURVE the caller
    builds from the chain's ``ω`` flips sign about ``γ`` under the mirror
    (``cos(ν + ω) → −cos(ν + ω)``).  The measured RVs are what BREAK the
    degeneracy — the overlay should show both branches, or note that a
    vertical mirror is the expected node-degeneracy signature.

    Conventions
    -----------
    Consumes the Python astrometric-TI chain keys ``"P_days"`` (days),
    ``"e"``, ``"a_phot_au"`` (photocentre semi-major axis, AU), and
    ``"inc_rad"`` (inclination, radians).  The ``ω`` / ``τ`` needed to build the
    actual RV *curve* are taken from the SAME chain draws by the caller and
    are already primary-frame (:data:`OMEGA_CONVENTION_PRIMARY`), matching
    the RV forward model's ``omega_rad`` convention — so the predicted curve
    and the RV data live in one convention with no sign flip.

    Parameters
    ----------
    chain : dict of np.ndarray
        Astrometric-TI chain dict.  Required keys ``"P_days"``, ``"e"``,
        ``"a_phot_au"``, ``"inc_rad"``.

    Returns
    -------
    dict
        ``{"k_kms": ndarray, "summary_kms": {median, q16, q84}}`` — one
        predicted ``K`` per posterior draw (km/s), β = 0.

    Notes
    -----
    ``√(1 − e²)`` is NaN for ``e > 1`` and 0 at ``e = 1`` (no clamping);
    filter unphysical draws upstream if present.  Face-on draws
    (``sin i → 0``) give ``K → 0`` — physical, not an error.
    """
    P_days = np.asarray(chain["P_days"], dtype=float)
    e = np.asarray(chain["e"], dtype=float)
    a_phot_au = np.asarray(chain["a_phot_au"], dtype=float)
    sin_i = np.sin(np.asarray(chain["inc_rad"], dtype=float))

    a1_sini_au = a_phot_au * sin_i          # β = 0: a_phot = a₁
    k_kms = (
        a1_sini_au * (2.0 * math.pi)
        / (P_days * np.sqrt(1.0 - e ** 2) * _AU_PER_KMS_DAY)
    )

    def _q(arr: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(arr)),
            "q16": float(np.quantile(arr, 0.16)),
            "q84": float(np.quantile(arr, 0.84)),
        }

    return {"k_kms": k_kms, "summary_kms": _q(k_kms)}


def astrometric_priors_from_rv_chain(
    chain: dict[str, np.ndarray],
    *,
    n_sigma: float = 3.0,
    sigma_widen: float = 1.0,
) -> dict[str, tuple[str, float, float, float, float]]:
    """Build ``truncated_Normal`` priors on ``(P, e, ω)`` from an RV chain.

    Parameters carried over from the RV fit (and so frozen by the
    spectroscopic posterior): ``P`` (days), ``e``, ``ω`` (rad).
    Parameters left free for the astrometric fit (NOT included here):
    ``i``, ``Ω``, ``mass`` — sampled from their default astrometric
    priors.

    Parameters
    ----------
    chain : dict of np.ndarray
        Required keys: ``"P_days"`` (days), ``"e"`` (dimensionless),
        ``"omega_rad"`` (radians, any wrap convention — pre-wrapped to
        ``[0, 2π)`` internally).
    n_sigma : float, default 3.0
        Half-width of the truncation in σ units.
    sigma_widen : float, default 1.0
        Multiplicative widening factor for σ (1.0 = raw IQR-half-width).

    Returns
    -------
    dict
        Keys exactly ``{"P", "e", "ω"}``.  Each value is a 5-tuple
        ``("truncated_Normal", μ, σ, lower, upper)``:

        - ``P`` is in **days** (μ = median of the chain's ``P_days``).
        - ``e`` is dimensionless and clipped to ``[0, 0.99]``.
        - ``ω`` is in **radians**, wrapped to ``[0, 2π)``.

    Notes
    -----
    σ is estimated as ``(q84 − q16) / 2`` from the chain — robust to
    mild non-Gaussianity in the posterior and matches the
    ``truncated_Normal`` interpretation.

    ω wrap: take the σ on ``(cos ω, sin ω)`` and back-project.  Robust
    to posteriors that straddle ``2π → 0``.  The median is the circular
    mean (atan2 of the mean unit vector).  For strongly bimodal or
    near-uniform ω posteriors the truncated_Normal at ±n_sigma may be
    looser than appropriate.

    ω convention
    ------------
    The chain's ``omega_rad`` is in the PRIMARY frame (binary-star
    convention) and the emitted prior centre is in the PRIMARY frame
    too, wrapped to ``(-π, π]`` — the engines' own ω domain (a unit-disk
    latent exported as ``atan2``).  No π shift is applied here; a prior
    built by an older version of this function, which shifted the centre
    by −π for a companion-frame consumer, is off by π.

    What the engines actually honour: the ``P`` and ``e`` entries.  The
    ``ω`` entry is ADVISORY today —
    both the astrometric Campbell resolver and the joint resolver replace
    any user prior on ``ω`` / ``Ω`` / ``θ`` with ``UniformCircular`` and
    emit a warning (``docs/model_and_likelihoods.md`` §5.2).  It
    is emitted in the right frame so that it becomes correct the day
    non-uniform circular priors are supported, and so a user reading it
    sees the spectroscopic ω constraint in the engine's convention.

    The function asserts the chain carries
    ``_meta["omega_convention"] == OMEGA_CONVENTION_PRIMARY``; chains
    without the sentinel are rejected (a mismatched convention would
    silently shift the joint-fit posterior by π).

    ``τ`` is intentionally NOT emitted: the joint fit uses the Kepler
    astrometric basis, which samples ``θ`` (position angle at the
    reference epoch) instead of ``τ`` (phase fraction).
    """
    # ── ω convention sentinel guard ───────────────────────────────────
    # Reject chains that lack the sentinel or carry a non-primary
    # convention.  A silent π offset on ω would mis-locate the joint-fit
    # posterior; we surface the mismatch rather than guess.  The error
    # message intentionally carries no chain values (logging-hygiene).
    meta = chain.get("_meta") if hasattr(chain, "get") else None
    omega_conv = None
    if isinstance(meta, dict):
        omega_conv = meta.get("omega_convention")
    if omega_conv != OMEGA_CONVENTION_PRIMARY:
        raise ValueError(
            "RV chain has no primary-frame omega_convention sentinel; add "
            "_meta['omega_convention'] if ω is the primary's (manual §1.4)."
        )

    P_days = np.asarray(chain["P_days"], dtype=float)
    e_arr = np.asarray(chain["e"], dtype=float)
    omega_arr = np.mod(np.asarray(chain["omega_rad"], dtype=float), 2.0 * np.pi)

    P_med, P_sig = _med_sigma(P_days, sigma_widen)
    e_med, e_sig = _med_sigma(e_arr, sigma_widen)

    # ω wrap: σ taken on (cos ω, sin ω); circular-mean median via atan2.
    cos_w = np.cos(omega_arr)
    sin_w = np.sin(omega_arr)
    omega_med = float(np.arctan2(np.mean(sin_w), np.mean(cos_w))) % (2.0 * np.pi)
    omega_sig = float(np.hypot(np.std(cos_w), np.std(sin_w))) * float(sigma_widen)

    # PRIMARY frame in, PRIMARY frame out.  The RV chain's ω and the
    # astrometric fit's ω are both the primary's (reference manual, §1.4),
    # so the prior is centred on the RV median as is: shifting it by π
    # (converting to the companion's frame, as a planet code would want)
    # would put the prior on the wrong side of the orbit here.
    # The centre is wrapped into (-π, π]; the width is the circular spread
    # of (cos ω, sin ω) times ``sigma_widen``, so a broad RV posterior
    # gives a broad prior.
    omega_center = float(_wrap_to_pi(omega_med))

    return {
        "P": _trunc(P_med, P_sig, n_sigma, lo=0.0),
        "e": _trunc(e_med, e_sig, n_sigma, lo=0.0, hi=0.99),
        # ω: rad, primary frame, bounds at ±n_sigma.  ADVISORY today:
        # the engines override angular priors with UniformCircular and
        # warn (see the docstring); P and e are the entries they honour.
        "ω": (
            "truncated_Normal",
            omega_center,
            omega_sig,
            omega_center - n_sigma * omega_sig,
            omega_center + n_sigma * omega_sig,
        ),
    }
