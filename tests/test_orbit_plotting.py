"""
Plotting regression tests for ``plot_astrometric_orbit``.

Batch P (plotting regression fix) — verifies that the cosmetic
across-scan (AC) whisker is OFF by default, and that passing an
explicit ``whisker_length_mas`` re-enables it with the requested
half-length.

Two design decisions under test:

1. ``whisker_length_mas=None`` (default) → AC whiskers are NOT drawn.
   Only the along-scan (AL) tick remains, anchored at each epoch.

2. ``whisker_length_mas=<positive float>`` → AC whiskers are drawn
   with that explicit half-length, in addition to the AL ticks.

Both tests use a fully synthetic, hand-built input pair (no real-data
fixture, no Julia, no archive access).  The plotting code is exercised
on N=10 toy circular epochs with deterministic RNG seed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Headless backend; must precede ``pyplot``.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from orblet.plotting import plot_astrometric_orbit


def _make_synthetic_inputs(
    n_epochs: int = 10, n_samples: int = 50, seed: int = 12345
) -> tuple[dict, dict]:
    """
    Build a synthetic ``(astro_data, result)`` pair sufficient to drive
    ``plot_astrometric_orbit`` end-to-end.

    The ``result`` dict mimics the structure produced by
    ``fit_astrometry_orbit``: a ``chains`` sub-dict carrying Keplerian
    posterior samples (``P_days``, ``e``, ``omega_rad``, ``inc_rad``,
    ``Omega_rad``, ``tp_mjd``, ``a_rel_au``, ``plx_mas``, ``m2_msun``,
    ``M_total_msun``, ``log_density``)
    plus a ``simulate`` sub-dict with ``resid_al``.

    All values are deterministic given ``seed``; no real targets, no
    real catalog parameters.

    Units (matching plotting.py expectations):
      ``P_days``       — days
      ``e``            — dimensionless
      ``omega_rad``, ``inc_rad``, ``Omega_rad`` — radians
      ``tp_mjd``       — MJD (TCB days)
      ``a_rel_au``     — AU (companion semi-major axis)
      ``plx_mas``      — mas
      ``m2_msun``      — M_sun (companion mass)
      ``M_total_msun`` — M_sun (TOTAL system mass)
      ``obs_time`` — days from J2010.0 TCB
      ``scan_angle`` — radians
      ``centroid_pos``, ``centroid_pos_err`` — mas
    """
    rng = np.random.default_rng(seed)

    # ── Toy circular orbit with mild posterior dispersion ─────────────
    # Tight posterior (~1% scatter) keeps the rendered orbit near a
    # single mode so the AC/AL tick counting in the assertions is
    # unambiguous.
    P_days_med = 400.0                       # ~400 d circular orbit
    P_days = P_days_med * (1.0 + 0.01 * rng.standard_normal(n_samples))
    e_arr = np.full(n_samples, 1e-3) + 1e-4 * rng.standard_normal(n_samples)
    e_arr = np.clip(e_arr, 0.0, 0.99)
    omega_arr = (np.pi / 4.0) + 0.01 * rng.standard_normal(n_samples)
    inc_arr = (np.pi / 4.0) + 0.01 * rng.standard_normal(n_samples)
    Omega_arr = (np.pi / 6.0) + 0.01 * rng.standard_normal(n_samples)
    # Periastron near J2016.0 (MJD ≈ 57389.0 in TCB days).
    tp_arr = 57389.0 + 0.5 * rng.standard_normal(n_samples)
    a_arr = 1.0 + 0.01 * rng.standard_normal(n_samples)        # AU
    plx = 10.0 + 0.05 * rng.standard_normal(n_samples)        # mas
    mass_arr = 1.0 + 0.01 * rng.standard_normal(n_samples)      # M_sun
    M = 2.0 + 0.01 * rng.standard_normal(n_samples)           # M_sun (total)
    log_density = -0.5 * rng.standard_normal(n_samples) ** 2

    chains = {
        "P_days": P_days,
        "e": e_arr,
        "omega_rad": omega_arr,
        "inc_rad": inc_arr,
        "Omega_rad": Omega_arr,
        "tp_mjd": tp_arr,
        "a_rel_au": a_arr,
        "plx_mas": plx,
        "m2_msun": mass_arr,
        "M_total_msun": M,
        "log_density": log_density,
    }

    # ── Synthetic observation arrays ──────────────────────────────────
    # Span ~one period, deterministic scan angles (avoid duplicates
    # to keep AL tick lengths distinct from any AC half-length).
    t_obs = np.linspace(0.0, 400.0, n_epochs)        # days from J2010.0
    psi = np.linspace(0.1, 0.1 + np.pi, n_epochs)    # radians
    f_pi = 0.5 * np.cos(psi)                          # arbitrary
    centroid_pos = np.zeros(n_epochs)                 # mas
    centroid_pos_err = np.full(n_epochs, 0.1)         # mas

    astro_data = {
        "obs_time": t_obs,
        "scan_angle": psi,
        "centroid_pos": centroid_pos,
        "centroid_pos_err": centroid_pos_err,
        "parallax_factor_al": f_pi,
    }

    result = {
        "chains": chains,
        "simulate": {"resid_al": np.zeros(n_epochs)},
    }
    return astro_data, result


def _count_two_point_segments(ax) -> int:
    """
    Count Line2D artists that represent a single 2-point segment
    (i.e., a whisker or tick), excluding the multi-point model orbit
    polylines and the single-point barycenter marker.
    """
    return sum(1 for line in ax.lines if len(line.get_xdata()) == 2)


def _two_point_segment_lengths(ax) -> list[float]:
    """Return Euclidean lengths of every 2-point segment on ``ax``."""
    out = []
    for line in ax.lines:
        if len(line.get_xdata()) == 2:
            dx = float(np.diff(line.get_xdata())[0])
            dy = float(np.diff(line.get_ydata())[0])
            out.append(float(np.hypot(dx, dy)))
    return out


def test_plot_astrometric_orbit_no_ac_whiskers_by_default():
    """
    With ``whisker_length_mas`` left at its default (``None``), the
    AC cosmetic whisker must NOT be drawn.  Only the AL tick remains
    at each of the N=10 epochs → exactly 10 two-point segments.
    """
    astro_data, result = _make_synthetic_inputs()
    fig = plot_astrometric_orbit(
        astro_data, result,
        ra_deg=180.0,
        dec_deg=0.0,
        parallax_mas=10.0,
        pmra_masyr=0.0,
        pmdec_masyr=0.0,
    )
    try:
        ax = fig.axes[0]
        n_segments = _count_two_point_segments(ax)
        assert n_segments == 10, (
            f"expected 10 AL ticks (no AC whiskers), got {n_segments}"
        )
    finally:
        plt.close(fig)


def test_plot_astrometric_orbit_ac_whiskers_when_enabled():
    """
    Passing an explicit ``whisker_length_mas`` re-enables the AC
    cosmetic whisker at the requested half-length.  Each of the N=10
    epochs then contributes 1 AC + 1 AL = 2 two-point segments → 20
    total, with at least one segment whose length matches 2 × the
    requested half-length (full whisker = 2·5 = 10 mas).
    """
    astro_data, result = _make_synthetic_inputs()
    fig = plot_astrometric_orbit(
        astro_data, result,
        ra_deg=180.0,
        dec_deg=0.0,
        parallax_mas=10.0,
        pmra_masyr=0.0,
        pmdec_masyr=0.0,
        whisker_length_mas=5.0,
    )
    try:
        ax = fig.axes[0]
        n_segments = _count_two_point_segments(ax)
        assert n_segments == 20, (
            f"expected 20 segments (10 AL + 10 AC), got {n_segments}"
        )
        seg_lengths = _two_point_segment_lengths(ax)
        # At least one segment should match the explicit AC full length
        # (2 × 5.0 = 10.0 mas).  AL ticks are 2 × 0.1 = 0.2 mas, so the
        # AC segments are clearly distinguishable.
        assert any(abs(L - 10.0) < 1e-9 for L in seg_lengths), (
            f"no segment with length ~10.0 mas found; lengths = {seg_lengths}"
        )
    finally:
        plt.close(fig)
