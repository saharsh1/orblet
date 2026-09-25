"""
Prepare epoch radial-velocity data for orbital fitting.

This module takes an epoch-RV dictionary — the loader shape
:class:`~orblet.simulate.orbit.OrbitSimulator` documents, which any reader of
epoch RVs can produce — and returns a clean, validated dictionary ready for
an orbital fit.

Responsibilities
----------------
- Validate required fields and array shapes.
- Apply an optional user-supplied boolean mask.
- Remove non-finite radial-velocity epochs.
- Convert observation times to MJD (if not already).
- Enforce a minimum number of usable epochs.
- Resolve astrometric ``obs_time`` (MJD or days-from-J2010) to MJD for
  every engine and pre-fit tool (:func:`resolve_epochs_mjd`).

The output dictionary uses the same units as the input (km/s for RV,
days for time).  Unit conversion to whatever the engine needs is
deferred to the engine.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time

from orblet.constants import MJD_J2010_TCB


# ── Time-system conversion helpers ────────────────────────────────────────────

# Gaia On-Board Mission Time (OBMT) reference epoch:
#   OBMT = 0  corresponds to  2010-01-01T00:00:00 TCB  =  JD 2455197.5 TCB
#   MJD = JD - 2400000.5, so MJD_REF = 55197.0
# Use ``MJD_J2010_TCB`` from ``orblet.constants`` directly.


# Threshold (days) separating the two Gaia epoch regimes: J2010-days-TCB
# (days from MJD_J2010_TCB = 55197.0; typically a few thousand) vs absolute
# MJD-TCB (Gaia era, > 55000).  One constant for the conditional resolver
# below and the two one-sided guards that follow it.
_J2010_DAYS_MAX = 30000.0


def resolve_epochs_mjd(astro_data: dict) -> np.ndarray:
    """Return ``astro_data["obs_time"]`` as MJD (float64, days).

    Gaia epoch-astrometry times arrive in one of two regimes: absolute
    MJD-TCB (Gaia era, above 55000) or days from J2010.0 TCB (the loader
    convention; typically a few thousand).  The median of the array
    decides: below ``_J2010_DAYS_MAX`` it is J2010-days and the
    ``MJD_J2010_TCB`` offset is added; otherwise the array is returned
    as MJD unchanged.  Every engine and the pre-fit tools call this one
    function, so a seed composed from user arrays converts by the SAME
    rule the fit uses.

    Parameters
    ----------
    astro_data : dict
        Loader-shaped astrometric dict; only ``"obs_time"`` is read
        (1-D array of epochs in TCB, either MJD or days from J2010.0).

    Returns
    -------
    np.ndarray
        ``obs_time`` as float64 MJD (TCB, days), same shape as the input.
        An empty or all-NaN array is returned unchanged.
    """
    obs_time_raw = np.asarray(astro_data["obs_time"], dtype=float)
    if np.nanmedian(obs_time_raw) < _J2010_DAYS_MAX:
        return obs_time_raw + MJD_J2010_TCB
    return obs_time_raw


def _assert_obs_time_is_j2010_days(obs_time: np.ndarray) -> None:
    """Assert ``obs_time`` is in J2010-days (NOT already absolute MJD).

    Several call sites convert epochs unconditionally via
    ``obs_time + MJD_J2010_TCB`` because their contract is that ``obs_time``
    arrives in days-from-J2010-TCB (the loader convention).  This guard
    FORMALISES that already-true invariant: it raises if the median epoch
    looks like an absolute MJD (≥ 30000), which would mean the offset is
    being double-applied.  It does NOT silently fix the value (that is the
    job of the conditional median-guard at the sampler boundary); a violation
    here is a real upstream contract bug to surface, not paper over.

    Value-free: the message names no epoch value (data-safety).
    """
    arr = np.asarray(obs_time, dtype=float)
    if arr.size == 0:
        return
    if float(np.nanmedian(arr)) >= _J2010_DAYS_MAX:
        raise ValueError(
            "obs_time looks like absolute MJD (median above the J2010-day "
            "regime); this site expects days-from-J2010-TCB before the "
            "+MJD_J2010_TCB offset. Pass J2010-day epochs, or route through "
            "the conditional epoch-resolver instead."
        )


# Mirror of ``_assert_obs_time_is_j2010_days``, for sites whose contract is
# the OTHER regime: absolute MJD.  Kept here (next to its mirror and the
# shared threshold) rather than in an engine, because all three joint engines
# import it and the two regimes must be defined in one place.
_NOT_ABSOLUTE_MJD_MSG = (
    "RV epochs are not on the absolute-MJD scale: their median sits in the "
    "J2010-day regime (days from MJD_J2010_TCB), which is what the RV loader "
    "and the demo simulator emit. The joint engines resolve the ASTROMETRIC "
    "channel to absolute MJD, so J2010-day RV epochs would sit "
    "MJD_J2010_TCB days away from it while sharing one time of periastron — "
    "the RV phase would be silently wrong. Prepare the RV data with "
    'prepare_rv_for_orbit(..., time_scale="gaia_obmt").'
)


def assert_rv_epochs_are_absolute_mjd(epochs_mjd: np.ndarray) -> None:
    """Raise unless ``epochs_mjd`` is on the absolute-MJD scale.

    Used by the joint engines, whose two channels must share one clock.  The
    test is the epoch REGIME (median below ``_J2010_DAYS_MAX``), not a
    distance between the channels: any real MJD epoch — Gaia-era or archival —
    lies far above the threshold, so an RV set that legitimately predates the
    astrometric epochs by years is not flagged, while the J2010-day/MJD
    mix-up always is.

    Parameters
    ----------
    epochs_mjd : np.ndarray
        RV epochs, days, expected on the absolute-MJD scale.

    Raises
    ------
    ValueError
        If the median epoch lies in the J2010-day regime.  The message names
        no epoch value (data-safety).
    """
    arr = np.asarray(epochs_mjd, dtype=float)
    if arr.size == 0:
        return
    median = float(np.nanmedian(arr))
    if not np.isfinite(median):
        return          # non-finite input is the NaN guards' business
    if median < _J2010_DAYS_MAX:
        raise ValueError(_NOT_ABSOLUTE_MJD_MSG)


def _convert_time_to_mjd(
    times: np.ndarray,
    time_scale: str,
) -> np.ndarray:
    """
    Convert an array of observation times to MJD (float64, days).

    Parameters
    ----------
    times : np.ndarray
        Observation times as a 1-D float array.
    time_scale : str
        One of:
        - ``"mjd"``        — already MJD; returned as-is.
        - ``"gaia_obmt"``  — Gaia on-board mission time in days since
          J2010.0 TCB.  Converted by adding the reference offset.
        - ``"jd"``         — Julian Date; converted to MJD by subtracting
          2 400 000.5.

    Returns
    -------
    mjd : np.ndarray
        Times in MJD (days).

    Raises
    ------
    ValueError
        If ``time_scale`` is not recognised.
    """
    if time_scale == "mjd":
        # "mjd" is a pass-through, so a caller who picks it for loader-shaped
        # (J2010-day) epochs gets no conversion and no error — the L-1 bug.
        # Assert the branch's own contract, the mirror of the "gaia_obmt"
        # guard below.
        assert_rv_epochs_are_absolute_mjd(times)
        return times.copy()

    if time_scale == "gaia_obmt":
        # OBMT is days-from-J2010-TCB by definition; assert the invariant the
        # unconditional offset relies on (catches a double-applied offset).
        _assert_obs_time_is_j2010_days(times)
        return times + MJD_J2010_TCB

    if time_scale == "jd":
        # JD inputs are interpreted as TCB-JD (the project's native scale,
        # consistent with Gaia obs_time_tcb).  astropy.Time provides
        # input-validation and handles the JD/MJD conversion via its
        # internal two-float arithmetic.
        return Time(times, format="jd", scale="tcb").mjd

    raise ValueError(
        f"Unknown time_scale '{time_scale}'. "
        f"Use 'mjd', 'gaia_obmt', or 'jd'."
    )


# ── Required keys in the input rv_data dictionary ────────────────────────────

_REQUIRED_KEYS = ("radial_velocity", "radial_velocity_err", "obs_time_rv")


# ── Main public function ─────────────────────────────────────────────────────

def prepare_rv_for_orbit(
    rv_data: dict,
    *,
    mask: np.ndarray | None = None,
    time_scale: str = "mjd",
    time_object: Time | None = None,
    min_epochs: int = 3,
) -> dict:
    """
    Validate and prepare RV data for orbital fitting.

    Takes an epoch-RV dictionary (loader shape; see the module docstring) and
    returns a clean copy with only the arrays needed for orbit fitting,
    after applying masking and removing non-finite entries.

    Parameters
    ----------
    rv_data : dict
        Must contain at least the keys ``radial_velocity``,
        ``radial_velocity_err``, and ``obs_time_rv``.  All values are 1-D
        arrays of the same length.  RVs are expected in **km/s** and times
        in **days** (interpretation set by *time_scale*).
    mask : np.ndarray of bool, optional
        Boolean array (same length as the data arrays).  ``True`` means
        *keep* the epoch.  Applied before the finite-value filter.
        Use this for sigma-clipping, quality cuts, or any external
        selection.  If ``None`` (default), all epochs are kept.
    time_scale : str, default ``"mjd"``
        How to interpret ``rv_data["obs_time_rv"]``:

        - ``"mjd"``        — already Modified Julian Date.
        - ``"gaia_obmt"``  — Gaia OBMT (days since J2010.0 TCB).
        - ``"jd"``         — Julian Date.

        The RV loader and the demo simulator both emit **J2010-days**, so
        ``"gaia_obmt"`` is the correct choice for loader-shaped data; the
        default ``"mjd"`` is a pass-through and now RAISES on epochs that
        look like J2010-days, because a joint fit would otherwise place the
        two channels 55 197 d apart while sharing one time of periastron
        (see :func:`assert_rv_epochs_are_absolute_mjd`).

        Ignored when *time_object* is provided.
    time_object : astropy.time.Time, optional
        If provided, used directly instead of ``rv_data["obs_time_rv"]``
        and *time_scale*.  Must have the same length as the data arrays.
        Converted to MJD internally.
    min_epochs : int, default 3
        Minimum number of valid epochs after masking and finite filtering.
        Raises ``ValueError`` if fewer remain.

    Returns
    -------
    prepared : dict
        Dictionary with keys:

        - ``"epochs_mjd"``  — 1-D float array, observation times in MJD.
        - ``"rv"``          — 1-D float array, radial velocities in km/s.
        - ``"rv_err"``      — 1-D float array, RV uncertainties in km/s.
        - ``"n_epochs"``    — int, number of valid epochs.

    Raises
    ------
    KeyError
        If a required key is missing from *rv_data*.
    ValueError
        If array lengths are inconsistent, *mask* has the wrong length,
        or fewer than *min_epochs* valid epochs remain.

    Examples
    --------
    >>> rv_data = load_simulated_inputs(seed=0).rv_data  # or your loader's dict
    >>> prepared = prepare_rv_for_orbit(rv_data, time_scale="mjd")
    >>> prepared["n_epochs"]
    24
    """
    # ── 1. Check required keys ────────────────────────────────────────────
    for key in _REQUIRED_KEYS:
        if key not in rv_data:
            raise KeyError(
                f"Required key '{key}' not found in rv_data. "
                f"Expected keys: {_REQUIRED_KEYS}"
            )

    # ── 2. Extract and validate arrays ────────────────────────────────────
    rv = np.asarray(rv_data["radial_velocity"], dtype=float)
    rv_err = np.asarray(rv_data["radial_velocity_err"], dtype=float)
    times_raw = np.asarray(rv_data["obs_time_rv"], dtype=float)

    n_input = rv.shape[0]
    if rv_err.shape[0] != n_input or times_raw.shape[0] != n_input:
        raise ValueError(
            f"Array length mismatch: radial_velocity has {n_input} elements, "
            f"radial_velocity_err has {rv_err.shape[0]}, "
            f"obs_time_rv has {times_raw.shape[0]}."
        )

    # ── 3. Apply user-supplied mask ───────────────────────────────────────
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != n_input:
            raise ValueError(
                f"mask length ({mask.shape[0]}) does not match "
                f"data length ({n_input})."
            )
        rv = rv[mask]
        rv_err = rv_err[mask]
        times_raw = times_raw[mask]

    # ── 4. Remove non-finite values ───────────────────────────────────────
    # An epoch is kept only if rv and rv_err are finite.
    # When time_object is provided, we use it for times instead of
    # times_raw, so we do not filter on times_raw finiteness.
    finite = np.isfinite(rv) & np.isfinite(rv_err)
    if time_object is None:
        # Also require finite obs_time_rv when it is used for times.
        finite = finite & np.isfinite(times_raw)

    rv = rv[finite]
    rv_err = rv_err[finite]
    times_raw = times_raw[finite]

    # ── 5. Convert times to MJD ───────────────────────────────────────────
    if time_object is not None:
        # Use the astropy Time object directly.
        # Apply the same mask + finite filter to it.
        t = time_object
        if mask is not None:
            t = t[mask]
        t = t[finite]
        epochs_mjd = t.mjd.astype(float)
    else:
        epochs_mjd = _convert_time_to_mjd(times_raw, time_scale)

    # ── 6. Minimum-epoch guard ────────────────────────────────────────────
    n_valid = len(rv)
    if n_valid < min_epochs:
        raise ValueError(
            f"Only {n_valid} valid epoch(s) after filtering. "
            f"Minimum required: {min_epochs}."
        )

    # ── 7. Build output ──────────────────────────────────────────────────
    out = {
        "epochs_mjd": epochs_mjd,
        "rv": rv,
        "rv_err": rv_err,
        "n_epochs": n_valid,
    }

    # ── 8. Carry scan_angle through if present ────────────────────────────
    # Apply the same mask + finite filter so it stays aligned.
    if "scan_angle" in rv_data:
        sa = np.asarray(rv_data["scan_angle"], dtype=float)
        if mask is not None:
            sa = sa[mask]
        sa = sa[finite]
        out["scan_angle"] = sa

    return out
