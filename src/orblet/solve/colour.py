"""The colour term of Gaia DR4 epoch astrometry in the linear astrometric solves.

The published along-scan position of a CCD observation was measured with an
assumed effective wavenumber ν_used.  If the true effective wavenumber of
the source is ν, the position is displaced:

    w_pub = model + s · cf · (ν − ν_used),

with cf the per-CCD ``colour_factor_al`` and s the sign constant.  The
solves can use this in two ways (:class:`ColourTerm`):

    fixed    the caller gives ν; the data are moved to that colour before
             the solve; the design does not change
    free     ν is one more linear amplitude, with a flat prior or a
             Gaussian prior of a given width (the six-parameter solution
             of Gaia's own astrometric pipeline)

One colour per source: the same ν at every epoch.

Symbols
-------
===========  ===============================================================
w_pub        published along-scan position of a row, mas (``centroid_pos``)
cf           ``colour_factor_al`` of the row, mas·µm (Gaia DR4, per CCD)
ν_used       ``nu_eff_used_in_astrometry`` of the row's transit, µm⁻¹
ν            effective wavenumber of the source, µm⁻¹ (fitted or given)
ν_ref        reference colour of a free fit: the median of ν_used over the
             rows, µm⁻¹
δν           the fitted amplitude, ν − ν_ref, µm⁻¹ (column ``d_nu_eff``)
s            ``orblet.constants.COLOUR_FACTOR_AL_SIGN``
column       s · cf, from :func:`orblet.design.columns.colour_columns`, the
             only reader of s
σ_prior      width of the Gaussian prior on δν, µm⁻¹
===========  ===============================================================

Input keys of the astrometry dict (all optional; one value per row of
``centroid_pos``): ``colour_factor_al`` (mas·µm), ``nu_eff_used`` (µm⁻¹;
the transit's value repeated on its CCD rows), ``nu_eff_error`` (µm⁻¹;
read only when the prior width is ``"nu_eff_error"``).  Nothing is clipped
here: the caller masks gross colour factors.

No input value is echoed in an error message.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orblet.design.columns import colour_columns

#: Width (µm⁻¹) of the Gaussian prior on the effective wavenumber with which
#: a free colour fit reproduces the ``pseudocolour`` and its error of Gaia's
#: six-parameter solutions (prior mean: ν_used).  It is NOT the per-source
#: ``nu_eff_error`` of the DR4 epoch file, which is a different quantity.
DPAC_COLOUR_PRIOR_SIGMA = 0.085

#: The two modes of :class:`ColourTerm`.
_COLOUR_MODES = ("fixed", "free")

#: The text that selects the ``nu_eff_error`` key as the prior width.
_PRIOR_FROM_KEY = "nu_eff_error"

_COLOUR_MODE_MSG = "ColourTerm.mode must be 'fixed' or 'free'."
_COLOUR_FIXED_MSG = (
    "ColourTerm(mode='fixed') needs a finite nu_eff (the colour to correct "
    "to) and takes no prior_sigma."
)
_COLOUR_FREE_MSG = (
    "ColourTerm(mode='free') takes no nu_eff; prior_sigma must be None "
    "(flat), a positive finite number, or the text 'nu_eff_error'."
)
_COLOUR_KEYS_MSG = (
    "a colour term needs both colour_factor_al and nu_eff_used: 1-D finite "
    "arrays with one value per row of the data."
)
_COLOUR_ERROR_KEY_MSG = (
    "prior_sigma='nu_eff_error' needs nu_eff_error: a 1-D array of finite "
    "positive values with one value per row of the data."
)
_COLOUR_ZERO_FACTOR_MSG = (
    "every colour_factor_al is zero: the colour column is empty and a free "
    "colour cannot be fitted (singular design)."
)
_COLOUR_NOT_A_TERM_MSG = "colour must be None or a ColourTerm."


@dataclass(frozen=True)
class ColourTerm:
    """How a linear astrometric solve treats the colour of the source.

    Attributes
    ----------
    mode : str
        ``"fixed"``: move the data to the colour ``nu_eff`` before the
        solve.  ``"free"``: fit the colour as one more linear amplitude.
    nu_eff : float or None
        ``"fixed"`` only, required there: the effective wavenumber to
        correct to, µm⁻¹.  The caller computes it (from a colour index, a
        spectrum or a model); this package does not.
    prior_sigma : float, str or None
        ``"free"`` only.  ``None``: flat prior.  A positive number: the
        width (µm⁻¹) of a Gaussian prior centred on ν_ref, the median of
        ``nu_eff_used``; :data:`DPAC_COLOUR_PRIOR_SIGMA` reproduces Gaia's
        six-parameter solutions.  ``"nu_eff_error"``: the width is the
        median of the ``nu_eff_error`` input.
    """

    mode: str
    nu_eff: float | None = None
    prior_sigma: float | str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _COLOUR_MODES:
            raise ValueError(_COLOUR_MODE_MSG)
        if self.mode == "fixed":
            nu_ok = _is_finite_number(self.nu_eff)
            if not nu_ok or self.prior_sigma is not None:
                raise ValueError(_COLOUR_FIXED_MSG)
            return
        # mode "free"
        sigma = self.prior_sigma
        sigma_ok = (
            sigma is None
            or (isinstance(sigma, str) and sigma == _PRIOR_FROM_KEY)
            or (_is_finite_number(sigma) and float(sigma) > 0.0)
        )
        if self.nu_eff is not None or not sigma_ok:
            raise ValueError(_COLOUR_FREE_MSG)


@dataclass(frozen=True)
class _ColourSetup:
    """What a solve needs from a :class:`ColourTerm` and the colour inputs.

    Attributes
    ----------
    mode : str
        ``"fixed"`` or ``"free"``.
    data : np.ndarray
        The along-scan data to solve, mas, shape ``(n,)``.  ``"fixed"``:
        moved to the colour ν.  ``"free"``: moved to the one reference
        colour ν_ref when ν_used differs between rows; otherwise the input
        array itself, untouched.
    column : np.ndarray or None
        ``"free"``: the ``(n, 1)`` colour column to append to the design.
        ``"fixed"``: ``None`` (the design does not change).
    nu_ref : float
        ``"free"``: the reference colour ν_ref, µm⁻¹; the fitted colour is
        ν_ref plus the colour amplitude.  ``"fixed"``: the given ν.
    prior_sigma : float or None
        ``"free"``: the width of the Gaussian prior on the colour
        amplitude, µm⁻¹, or ``None`` for a flat prior.
    """

    mode: str
    data: np.ndarray
    column: np.ndarray | None
    nu_ref: float
    prior_sigma: float | None


def _is_finite_number(value) -> bool:
    """True for a real, finite number (bool and text are not numbers here)."""
    if isinstance(value, (bool, str)) or value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _prepare_colour(
    colour,
    d_obs: np.ndarray,
    colour_factor_al,
    nu_eff_used,
    nu_eff_error,
) -> _ColourSetup | None:
    """Turn a colour request and the colour inputs into what a solve needs.

    Inputs
    ------
    colour : ColourTerm or None
        The request.  ``None``: no colour term; the other colour inputs are
        not read at all.
    d_obs : np.ndarray
        Along-scan data w_pub, mas, shape ``(n,)``.
    colour_factor_al, nu_eff_used : array or None
        The colour inputs, one value per row (mas·µm and µm⁻¹).
    nu_eff_error : array or None
        Per-row ``nu_eff_error`` (µm⁻¹); read only for
        ``prior_sigma="nu_eff_error"``.

    Returns
    -------
    _ColourSetup or None
        ``None`` when ``colour`` is ``None``; otherwise the setup (see
        :class:`_ColourSetup`).
    """
    if colour is None:
        return None
    if not isinstance(colour, ColourTerm):
        raise ValueError(_COLOUR_NOT_A_TERM_MSG)

    # step 1, the two colour inputs: both present, one finite value per row
    if colour_factor_al is None or nu_eff_used is None:
        raise ValueError(_COLOUR_KEYS_MSG)
    factor = np.asarray(colour_factor_al, dtype=float)
    nu_used = np.asarray(nu_eff_used, dtype=float)
    n_rows = d_obs.shape[0]
    if factor.shape != (n_rows,) or nu_used.shape != (n_rows,):
        raise ValueError(_COLOUR_KEYS_MSG)
    if not (np.all(np.isfinite(factor)) and np.all(np.isfinite(nu_used))):
        raise ValueError(_COLOUR_KEYS_MSG)

    # step 2, the column s · cf (the only place the sign enters)
    column = colour_columns(factor)
    column_values = column[:, 0]   # the one column as a 1-D array

    # step 3, fixed: move the data to the given colour ν,
    # d_corr = w_pub − s · cf · (ν − ν_used); the design does not change
    if colour.mode == "fixed":
        nu_fixed = float(colour.nu_eff)
        data = d_obs - column_values * (nu_fixed - nu_used)
        return _ColourSetup(
            mode="fixed", data=data, column=None, nu_ref=nu_fixed,
            prior_sigma=None,
        )

    # step 4, free: the reference colour ν_ref (median of ν_used).  When
    # ν_used is the same on every row (the normal case) the data are not
    # touched; otherwise they are first moved to ν_ref, so that the one
    # amplitude δν = ν − ν_ref describes every row.
    if not np.any(column_values != 0.0):
        raise ValueError(_COLOUR_ZERO_FACTOR_MSG)
    nu_ref = float(np.median(nu_used))
    if np.all(nu_used == nu_used[0]):
        data = d_obs
    else:
        data = d_obs - column_values * (nu_ref - nu_used)

    # step 5, free: the width of the prior on δν (None: flat)
    prior_sigma = colour.prior_sigma
    if isinstance(prior_sigma, str):
        # the text "nu_eff_error": the median of the per-row input
        if nu_eff_error is None:
            raise ValueError(_COLOUR_ERROR_KEY_MSG)
        nu_error = np.asarray(nu_eff_error, dtype=float)
        if (
            nu_error.shape != (n_rows,)
            or not np.all(np.isfinite(nu_error))
            or not np.all(nu_error > 0.0)
        ):
            raise ValueError(_COLOUR_ERROR_KEY_MSG)
        prior_sigma = float(np.median(nu_error))
    elif prior_sigma is not None:
        prior_sigma = float(prior_sigma)

    return _ColourSetup(
        mode="free", data=data, column=column, nu_ref=nu_ref,
        prior_sigma=prior_sigma,
    )
