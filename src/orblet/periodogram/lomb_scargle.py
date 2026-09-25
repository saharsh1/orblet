"""Lomb–Scargle periodogram of epoch radial velocities."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle


def compute_lomb_scargle_periodogram(
    radial_velocity,
    radial_velocity_err,
    obs_time_rv,
    *,
    return_lombscargle_obj=False,
    nterms=1,
    normalization="standard",
    center_data=True,
    fit_mean=True,
    frequency=None,              # Explicit frequency grid (array or Quantity), overrides rest if set
    min_frequency=None,
    max_frequency=None,
    spacing="auto",              # Grid spacing: "auto", "lin", or "log"
    n_freq=100000,               # # steps for lin/log grid
    samples_per_peak=10,         # for autopower grid
    nyquist_factor=5             # for autopower grid
):
    """
    Compute and return the Lomb-Scargle periodogram for radial velocity data.

    The arrays are used as given: selecting epochs (a survey validity flag,
    an outlier cut) is the caller's step, done before the call.  Every
    argument after the three arrays is keyword-only.

    Parameters
    ----------
    radial_velocity : array
        Measured radial velocities (Quantity or float).
    radial_velocity_err : array
        Uncertainties on the radial velocities.
    obs_time_rv : array
        Observation times of each RV point.
    return_lombscargle_obj : bool, default: False
        If True, also return the LombScargle object itself.
    nterms, normalization, center_data, fit_mean, frequency, min_frequency, max_frequency, spacing, n_freq, samples_per_peak, nyquist_factor
        Configuration for LombScargle and period search grid. See astropy LombScargle documentation.

    Returns
    -------
    frequency : ndarray
        Frequency grid evaluated (1/period).
    power : ndarray
        Lomb-Scargle periodogram power for each frequency.
    best_frequency : float
        Frequency of maximum periodogram power.
    best_period : float
        Period (1/best_frequency).
    ls : astropy.timeseries.LombScargle
        (only if return_lombscargle_obj=True) Fitted LombScargle object.

    Raises
    ------
    ValueError
        If any RV, uncertainty or time is non-finite (mask it first), or if
        fewer than 4 points are given.
    """

    # 1. Non-finite input is refused, not dropped: masking is the caller's.
    times = obs_time_rv
    rad_vels = np.asarray(radial_velocity)
    errors = np.asarray(radial_velocity_err)
    if not (
        np.all(np.isfinite(rad_vels))
        and np.all(np.isfinite(errors))
        and np.all(np.isfinite(np.asarray(times, dtype=float)))
    ):
        raise ValueError(
            "radial_velocity, radial_velocity_err and obs_time_rv must be "
            "finite; mask the non-finite epochs before calling."
        )

    # 2. Need enough data points for meaningful periodogram
    if len(rad_vels) < 4:
        raise ValueError("Need at least 4 valid RV points for a meaningful periodogram.")

    # 3. Set up the LombScargle model
    ls = LombScargle(
        times, rad_vels, errors,
        nterms=nterms,
        normalization=normalization,
        center_data=center_data,
        fit_mean=fit_mean,
    )

    # 4. Frequency grid: use explicit frequency argument if given, else construct grid
    if frequency is not None:
        # If times carry units, make sure frequency has compatible units
        if hasattr(times, "unit") and not hasattr(frequency, "unit"):
            frequency = np.asarray(frequency) * (1 / times.unit)
        power = ls.power(frequency)

    else:
        # Compute default frequency limits if not provided
        t_min, t_max = np.min(times), np.max(times)
        baseline = t_max - t_min
        min_frequency = 1.0 / baseline if min_frequency is None else min_frequency

        if max_frequency is None:
            # Set max freq based on median time spacing (or fall back to baseline)
            ts = np.sort(times)
            dts = np.diff(ts)
            try:
                dt_med = np.median(dts)
                max_frequency = (0.5 / dt_med) * (nyquist_factor if spacing == "auto" else 1.0)
            except Exception:
                max_frequency = 0.5 / baseline

        # 5. Choose frequency grid and compute periodogram power
        if spacing == "auto":
            frequency, power = ls.autopower(
                minimum_frequency=min_frequency,
                maximum_frequency=max_frequency,
                samples_per_peak=samples_per_peak,
                nyquist_factor=nyquist_factor,
            )
        elif spacing == "lin":
            frequency = np.linspace(min_frequency, max_frequency, int(n_freq))
            power = ls.power(frequency)
        elif spacing == "log":
            frequency = np.geomspace(min_frequency, max_frequency, int(n_freq))
            power = ls.power(frequency)
        else:
            raise ValueError("spacing must be one of: 'auto', 'lin', 'log'")

    # 6. Identify highest peak (best frequency, period)
    imax = int(np.nanargmax(power))
    best_frequency = frequency[imax]
    best_period = 1.0 / best_frequency

    # 7. Return result, possibly including the LombScargle object
    if return_lombscargle_obj:
        return frequency, power, best_frequency, best_period, ls
    return frequency, power, best_frequency, best_period
