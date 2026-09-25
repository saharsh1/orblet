"""Subtract the 5-parameter astrometric model; keep what is left.

It opens no file and contacts nothing: it is model arithmetic on arrays the
caller already holds (epochs, scan angles, parallax factors, abscissae), with
the 5-parameter model of the reference manual, §3.1, subtracted.

Two public levels, and they are a LAYER, not a duplicate:

- :func:`orblet.astrometric_5param_design_matrix` — the five columns;
- :func:`astrometric_residuals` — build the model from given parameters and
  subtract it.

The second does real work the first does not: the local-plane origin offset
``(ra − ra0)·cos δ``, which converts a catalogue position into the epoch
table's own frame.

That it is wanted in three places at once — plotting (to draw the orbit
alone, with the parallax wobble and the proper-motion drift removed), the
PDC pre-search, and the blend screens — is why it is a routine rather than
six lines repeated at each call site.

The two modes are scientifically different
------------------------------------------
**catalog** — the caller supplies the five parameters, so the subtracted
model is EXTERNAL to the data being fit and no part of the orbit is absorbed.

**fitted** — no catalogue position, so the position offset is solved as a
2-parameter nuisance in ``sin ψ`` / ``cos ψ``. That absorbs whatever part of
the orbit looks like a constant offset, and biases a subsequent period
search toward weaker peaks.

Prefer catalogue values when you have them. Subtracting NOTHING is worse
than either: an un-removed proper-motion drift is a secular trend that
phases coherently at long periods and drives a periodogram to the range edge.

Because the difference matters, :func:`astrometric_residuals` REPORTS which
mode ran, rather than leaving it to be inferred from which arguments the
caller happened to pass.

Conventions
-----------
- O−C: residual is observed minus model. A positive residual means the
  source sits in the +AL direction relative to the 5-parameter prediction.
- ``t_ref`` and ``astro["obs_time"]`` share one clock (TCB days from
  J2010.0). The reference epoch is never defaulted: DR3 is J2016.0 and DR4
  is J2017.5, and assuming the wrong one injects a spurious scan-modulated
  signal of ``|pm| × 1.5 yr``.
- Proper motion is linear in Julian years (365.25 d).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR

#: How the position offset was obtained. Load-bearing — see the module
#: docstring on why the two are not interchangeable.
OFFSET_FROM_CATALOG = "catalog"
OFFSET_FITTED = "fitted"


@dataclass(frozen=True, repr=False)
class AstrometricResiduals:
    """Along-scan residuals plus the provenance of the position offset.

    Attributes
    ----------
    residuals_mas : np.ndarray
        O−C along-scan residuals in mas, one per epoch.
    offset_mode : str
        :data:`OFFSET_FROM_CATALOG` when the caller supplied a position and
        the local-plane origin was known, so the offset is external;
        :data:`OFFSET_FITTED` when it was solved as a sin/cos nuisance and
        has therefore absorbed part of any orbit present.
    """

    residuals_mas: np.ndarray
    offset_mode: str

    def __repr__(self) -> str:  # noqa: D401 - shape-only by design
        return "AstrometricResiduals(n={0}, offset_mode={1!r})".format(
            int(np.size(self.residuals_mas)), self.offset_mode
        )


def astrometric_residuals(
    astro: dict,
    *,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    parallax_mas: float = 0.0,
    pmra_masyr: float = 0.0,
    pmdec_masyr: float = 0.0,
    t_ref: float | None = None,
) -> AstrometricResiduals:
    """Subtract a 5-parameter astrometric model; return the remainder.

    Returns an :class:`AstrometricResiduals`: the residual array
    (``residuals_mas``) and which offset mode produced it.

    Parameters
    ----------
    astro : dict
        Epoch astrometry. Requires ``centroid_pos``, ``scan_angle`` and
        ``obs_time``; uses ``parallax_factor_al``, and ``ra0_deg`` /
        ``dec0_deg`` for the explicit-offset branch. ``ref_epoch_days`` is
        consulted when ``t_ref`` is not given.
    ra_deg, dec_deg : float, optional
        Catalogue position. Supply BOTH — and have ``ra0_deg`` / ``dec0_deg``
        in the bundle — to take the catalog branch; otherwise the offset is
        fitted.
    parallax_mas, pmra_masyr, pmdec_masyr : float
        The remaining three parameters. ``pmra`` is μ_α*, already including
        cos δ.
    t_ref : float, optional
        Proper-motion reference epoch, same clock as ``obs_time``. Falls back
        to ``astro["ref_epoch_days"]``; raises if neither is present.

    Returns
    -------
    AstrometricResiduals
        The residuals, and which offset mode produced them.

    Raises
    ------
    ValueError
        If the measurement or epochs are missing, if the reference epoch is
        unknown, or if a parallax was supplied that cannot be subtracted for
        want of a parallax factor.
    """
    centroid = astro["centroid_pos"]
    psi = astro["scan_angle"]
    t = astro["obs_time"]
    f_pi = astro["parallax_factor_al"]

    if centroid is None or t is None:
        raise ValueError("centroid_pos and obs_time are required")

    # ── The reference epoch must be KNOWN, never assumed ──────────────────────
    # DR3's epoch is J2016.0, DR4's is J2017.5.  Silently defaulting to the wrong one
    # leaves a spurious scan-modulated residual of |pm| x 1.5 yr — and, on the branch
    # that computes the position offset explicitly, nothing absorbs it.  Precedence:
    # an explicit t_ref, then the epoch the loader recorded, then refuse.
    if t_ref is None:
        t_ref = astro.get("ref_epoch_days")
    if t_ref is None:
        raise ValueError(
            "the astrometric reference epoch is unknown: pass t_ref, or use a bundle "
            "carrying 'ref_epoch_days' (a loader-shaped bundle does).  Refusing to "
            "guess — DR3 is J2016.0 and DR4 is J2017.5, and assuming the wrong one "
            "injects a spurious signal of |pm| x 1.5 yr."
        )

    # ── The parallax must not be silently skipped ─────────────────────────────
    # Quietly skipping the parallax when its factor is missing would leave a
    # coherent ~1.7 mas one-year signal in data the caller believes is
    # clean — and manufactures the very one-year alias the period search must guard
    # against (reference manual, §3.1 and §3.6).
    if parallax_mas != 0.0 and f_pi is None:
        raise ValueError(
            "parallax_mas was given but 'parallax_factor_al' is missing, so the "
            "parallax cannot be subtracted.  Refusing to return residuals that still "
            "contain a full one-year parallax signal."
        )

    # Time difference in Julian years (365.25 days).
    dt_yr = (t - t_ref) / DAYS_PER_KEPLER_YEAR

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # ── Build model ──────────────────────────────────────────────────────
    model = np.zeros_like(centroid)

    # 1. Position offset: Δα₀* sin(ψ) + Δδ₀ cos(ψ).
    #    centroid_pos_al is measured from (ra0, dec0), which may differ
    #    from the catalog (ra, dec).  This offset is large and
    #    scan-angle-dependent.
    ra0 = astro.get("ra0_deg")
    dec0 = astro.get("dec0_deg")
    _offset_computed = False

    if ra_deg is not None and dec_deg is not None and ra0 is not None and dec0 is not None:
        # Compute offset explicitly from the catalog vs. local-plane
        # reference positions.  Convert degree differences to mas.
        # Δα* includes the cos(dec) projection.
        dec_rad = np.deg2rad(dec_deg)
        d_ra_mas = (ra_deg - ra0) * np.cos(dec_rad) * 3_600_000.0
        d_dec_mas = (dec_deg - dec0) * 3_600_000.0
        model += d_ra_mas * sin_psi + d_dec_mas * cos_psi
        _offset_computed = True

    # 2. Proper motion: (μ_α* sin(ψ) + μ_δ cos(ψ)) × Δt.
    if pmra_masyr != 0.0 or pmdec_masyr != 0.0:
        model += (pmra_masyr * sin_psi + pmdec_masyr * cos_psi) * dt_yr

    # 3. Parallax: π × f_π.
    if f_pi is not None and parallax_mas != 0.0:
        model += parallax_mas * f_pi

    # O−C convention (observed minus model): positive residual means
    # the source is displaced in the +AL direction relative to the
    # 5-parameter prediction.
    residuals = centroid - model

    # ── Fit position offset if not computed explicitly ────────────────────
    # The offset is a 2-parameter linear model: a·sin(ψ) + b·cos(ψ).
    # We solve for (a, b) via ordinary least squares and subtract.
    if not _offset_computed:
        # Design matrix: columns are sin(ψ) and cos(ψ).
        A = np.column_stack([sin_psi, cos_psi])
        # Least-squares fit for the position offset.
        coeffs, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None)
        residuals = residuals - A @ coeffs

    return AstrometricResiduals(
        residuals_mas=residuals,
        offset_mode=(
            OFFSET_FROM_CATALOG if _offset_computed else OFFSET_FITTED
        ),
    )
