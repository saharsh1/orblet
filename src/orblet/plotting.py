"""
Plotting functions for orbital fit results.

Three figures:
1. Phase-folded RV with best-fit curve.
2. O−C residuals vs time and vs phase.
3. Corner plot of selected orbital elements.

ω convention
------------
``omega_rad`` from the chains-dict is interpreted as the **primary's**
argument of periastron (binary-star convention).  The engines emit
ω in this frame, so plots that label ω can do so without extra
qualification.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from orblet.constants import (
    MJD_J2010_TCB,
    MJD_J2016_TCB,
    OMEGA_CONVENTION_PRIMARY,
)
from orblet.elements import (
    _keplerian_rv,
    compute_rv_model_curve,
    compute_residuals,
    extract_orbital_elements,
)
from orblet.constants import DAYS_PER_KEPLER_YEAR
from orblet.kepler import campbell_xy


def _sample_orbit_curves(
    result: dict,
    t: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Draw up to ``n_samples`` joint RV-orbit curves from the chain.

    Returns an ``(n_draw, len(t))`` array of model RVs (km/s) — one row
    per drawn chain sample.  Joint-row sampling preserves the
    ``(P, e, ω, tp, K, γ)`` correlations; this is the whole point of
    posterior sampling vs the median-each parameter combination, which
    can land between modes of a bimodal chain and correspond to no real
    sample.

    Returns ``None`` if the chain is missing any of the required keys
    (``P_days``, ``e``, ``omega_rad``, ``tp_mjd``, ``K_kms``) — the
    caller then falls back to the single median-each curve from
    :func:`compute_rv_model_curve`.

    The ω-convention sentinel is enforced here (mirrors the guard in
    :func:`compute_rv_model_curve`); a chain tagged
    companion-frame raises rather than silently mis-orienting the model.
    """
    chains = result.get("chains", {}) if isinstance(result, dict) else {}
    if not isinstance(chains, dict):
        return None

    omega_conv = chains.get("_meta", {}).get("omega_convention") if isinstance(chains, dict) else None
    if omega_conv != OMEGA_CONVENTION_PRIMARY:
        raise ValueError(
            f"plot_orbit_fit requires primary-frame ω; got "
            f"omega_convention={omega_conv!r}.  If the chain's ω is the "
            f"primary's, add ``_meta['omega_convention'] = "
            f"'primary_argument_of_periastron'`` to the chain dict; "
            f"otherwise convert it first (reference manual, §1.4)."
        )

    required = ("P_days", "e", "omega_rad", "tp_mjd", "K_kms")
    if any(k not in chains for k in required):
        return None

    P_days = np.asarray(chains["P_days"]).ravel()
    e_arr  = np.asarray(chains["e"]).ravel()
    om_arr = np.asarray(chains["omega_rad"]).ravel()
    tp_arr = np.asarray(chains["tp_mjd"]).ravel()
    K_arr  = np.asarray(chains["K_kms"]).ravel()
    if "gamma_kms" in chains:
        gamma_arr = np.asarray(chains["gamma_kms"]).ravel()
    else:
        gamma_arr = np.zeros_like(P_days)

    n_total = P_days.size
    if not all(arr.size == n_total
               for arr in (e_arr, om_arr, tp_arr, K_arr, gamma_arr)):
        return None  # length-mismatched chain — bail to median-each fallback

    # Drop non-finite rows: one bad value disqualifies the joint sample.
    finite_mask = (
        np.isfinite(P_days) & np.isfinite(e_arr) & np.isfinite(om_arr)
        & np.isfinite(tp_arr) & np.isfinite(K_arr) & np.isfinite(gamma_arr)
    )
    if not finite_mask.any():
        return None
    finite_idx = np.flatnonzero(finite_mask)
    n_draw = min(int(n_samples), int(finite_idx.size))
    pick = rng.choice(finite_idx, size=n_draw, replace=False)

    curves = np.empty((n_draw, int(t.size)), dtype=float)
    for k, i in enumerate(pick):
        curves[k] = _keplerian_rv(
            t,
            float(P_days[i]),
            float(e_arr[i]),
            float(om_arr[i]),
            float(K_arr[i]),
            float(gamma_arr[i]),
            float(tp_arr[i]),
        )
    return curves


def plot_orbit_fit(
    prepared_data: dict,
    result: dict,
    *,
    min_points_per_period: int = 25,
    figsize: tuple = (10, 8),
    height_ratios: tuple = (3, 3, 2),
    source_id: int | str | None = None,
    # ── Axis limits ───────────────────────────────────────────
    xlim_time: tuple[float, float] | None = None,
    # ── Data point styling ────────────────────────────────────
    data_color: str = "black",
    data_marker: str = "o",
    data_ms: float = 4,
    data_capsize: float = 0,
    data_elinewidth: float = 0.8,
    data_zorder: int = 3,
    # ── Model line styling ────────────────────────────────────
    model_color: str = "gray",
    model_lw: float = 0.8,
    model_ls: str = "-",
    model_zorder: int = 2,
    # ── Posterior-band styling ────────────────────────────────
    # The displayed model curve is the MEDIAN-OF-CURVES across
    # ``n_posterior_samples`` joint chain draws; the band is the
    # 16/84 % envelope across the same samples.  This is faithful even
    # when the chain is bimodal in (ω, tp) — the legacy single-curve
    # median-each parameter combination is silently mis-oriented in
    # that case (see ``project_median_each_plot_artifact`` in memory).
    band_color: str = "gray",
    band_alpha: float = 0.25,
    band_zorder: int = 1,
    n_posterior_samples: int = 200,
    posterior_seed: int = 0,
    # ── Systemic velocity line ────────────────────────────────
    gamma_color: str = "gray",
    gamma_ls: str = ":",
    gamma_lw: float = 0.8,
    # ── Annotation styling ────────────────────────────────────
    annotation_fontsize: float = 9,
    annotation_bbox: dict | None = None,
    # ── Title ─────────────────────────────────────────────────
    title_fontsize: float = 11,
    show_legend: bool = False,
    include_jitter: bool = True,
) -> plt.Figure:
    """
    Three-panel figure: RV vs time, phase-folded RV, and O−C residuals.

    Parameters
    ----------
    prepared_data : dict
        Output of :func:`~orblet.prepare.prepare_rv_for_orbit`.
    result : dict
        A fit result dict (``"chains"`` + ``"summary"``) as an RV sampler returns it.
    min_points_per_period : int, default 25
        Minimum number of model-curve samples per orbital period.
        The total number of model points is
        ``max(500, n_periods × min_points_per_period)``.
    figsize : tuple, default (10, 8)
        Figure size in inches.
    height_ratios : tuple, default (3, 3, 2)
        Relative heights of the three panels.
    source_id : int or str, optional
        Source identifier for the figure title.
    data_color, data_marker, data_ms, data_capsize, data_elinewidth,
    data_zorder
        Styling for the data error bars.
    model_color, model_lw, model_ls, model_zorder
        Styling for the model curve.
    gamma_color, gamma_ls, gamma_lw
        Styling for the systemic-velocity horizontal line.
    annotation_fontsize : float, default 9
        Font size for the parameter and residual annotations.
    annotation_bbox : dict, optional
        Bounding-box style for annotations.  Defaults to a
        semi-transparent white box.
    title_fontsize : float, default 11
        Font size for the source-ID title.
    show_legend : bool, default False
        Whether to show a legend in the top panel.
    include_jitter : bool, default True
        Whether to inflate the data error bars (panels 1-3) by the
        chain-median jitter in quadrature.  Has no effect on the model
        curve / band or on the reported RMS.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if annotation_bbox is None:
        annotation_bbox = dict(facecolor="white", alpha=0.8, edgecolor="none")

    t_obs = prepared_data["epochs_mjd"]
    rv_obs = prepared_data["rv"]
    rv_err = prepared_data["rv_err"]
    n_epochs = len(rv_obs)

    # ── Model curve on a fine grid ────────────────────────────────────────
    # Ensure at least min_points_per_period samples per period.
    s = result["summary"]
    period_days = s["P_days"]["median"]
    time_span = t_obs.max() - t_obs.min()
    n_periods = max(time_span / period_days, 1.0)
    n_model_points = max(500, int(np.ceil(n_periods * min_points_per_period)))

    t_pad = 0.05 * time_span
    t_grid = np.linspace(t_obs.min() - t_pad, t_obs.max() + t_pad, n_model_points)

    # Posterior-sample path: draw joint chain rows, evaluate the Keplerian
    # RV at each row, then take the MEDIAN curve at each evaluation point.
    # This is faithful to the chain even when (ω, tp) are bimodal — unlike
    # the legacy median-each parameter combination, which lands between
    # modes and produces a curve no chain sample actually visited.
    rng_grid = np.random.default_rng(int(posterior_seed))
    curves_grid = _sample_orbit_curves(
        result, t_grid,
        n_samples=int(n_posterior_samples), rng=rng_grid,
    )
    # γ and the tp used for the phase fold come from the chain medians
    # regardless of which path is taken (the fold is an x-axis transform,
    # not a fit-quality statement).
    gamma = float(s.get("gamma_kms", {}).get("median", 0.0))
    tp_for_phase = float(s["tp_mjd"]["median"])

    if curves_grid is None:
        # Fallback: legacy single-curve path (chain missing keys, length-
        # mismatched, or all non-finite — e.g. an old engine version).
        rv_model_grid = compute_rv_model_curve(result, t_grid)["rv_model"]
        rv_grid_lo = rv_grid_hi = None
        resid_out = compute_residuals(
            prepared_data, result, include_jitter=include_jitter,
        )
        oc = resid_out["residuals"]
        rms = float(resid_out["rms"])
        rv_err_plot = resid_out["rv_err_eff"] if include_jitter else rv_err
    else:
        rv_model_grid = np.median(curves_grid, axis=0)
        rv_grid_lo = np.percentile(curves_grid, 16, axis=0)
        rv_grid_hi = np.percentile(curves_grid, 84, axis=0)
        # Re-sample at the OBSERVATION epochs with the same seed so the
        # residuals are computed against the same joint draws that shaped
        # the displayed band.  The RMS is then against the median-curve at
        # the data epochs — consistent with what the eye fits in panel 2.
        rng_obs = np.random.default_rng(int(posterior_seed))
        curves_obs = _sample_orbit_curves(
            result, np.asarray(t_obs, dtype=float),
            n_samples=int(n_posterior_samples), rng=rng_obs,
        )
        rv_model_obs = np.median(curves_obs, axis=0)
        oc = np.asarray(rv_obs) - rv_model_obs
        rms = float(np.sqrt(np.mean(oc**2)))
        # Error bars: inflate by the chain-median jitter when requested.
        if include_jitter:
            jit_arr = np.asarray(
                result["chains"].get("rv_jitter_kms", np.zeros(1)),
            ).ravel()
            jit_arr = jit_arr[np.isfinite(jit_arr)]
            jit_med = float(np.median(jit_arr)) if jit_arr.size else 0.0
            rv_err_plot = np.sqrt(np.asarray(rv_err)**2 + jit_med**2)
        else:
            rv_err_plot = rv_err

    # ── Phase-fold ────────────────────────────────────────────────────────
    phase_obs = np.mod(t_obs - tp_for_phase, period_days) / period_days
    phase_grid = np.mod(t_grid - tp_for_phase, period_days) / period_days
    sort_idx = np.argsort(phase_grid)

    # ── Common errorbar kwargs ────────────────────────────────────────────
    err_kw = dict(
        fmt=data_marker, ms=data_ms, capsize=data_capsize,
        elinewidth=data_elinewidth, color=data_color, zorder=data_zorder,
    )
    model_kw = dict(
        color=model_color, lw=model_lw, ls=model_ls, zorder=model_zorder,
    )

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, height_ratios=list(height_ratios),
    )

    # --- Panel 1: RV vs time ---
    ax = axes[0]
    ax.errorbar(t_obs, rv_obs, yerr=rv_err_plot, **err_kw)
    if rv_grid_lo is not None:
        ax.fill_between(
            t_grid, rv_grid_lo, rv_grid_hi,
            color=band_color, alpha=band_alpha, lw=0, zorder=band_zorder,
        )
    ax.plot(t_grid, rv_model_grid, **model_kw)
    ax.axhline(gamma, ls=gamma_ls, color=gamma_color, lw=gamma_lw)
    ax.set_xlabel("MJD (days)")
    ax.set_ylabel("RV (km/s)")
    if show_legend:
        ax.legend(fontsize=annotation_fontsize)
    if source_id is not None:
        ax.set_title(f"Source {source_id}", fontsize=title_fontsize)
    if xlim_time is not None:
        ax.set_xlim(xlim_time)

    # --- Panel 2: phase-folded RV ---
    ax = axes[1]
    ax.errorbar(phase_obs, rv_obs, yerr=rv_err_plot, **err_kw)
    if rv_grid_lo is not None:
        ax.fill_between(
            phase_grid[sort_idx], rv_grid_lo[sort_idx], rv_grid_hi[sort_idx],
            color=band_color, alpha=band_alpha, lw=0, zorder=band_zorder,
        )
    ax.plot(phase_grid[sort_idx], rv_model_grid[sort_idx], **model_kw)
    ax.axhline(gamma, ls=gamma_ls, color=gamma_color, lw=gamma_lw)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("RV (km/s)")

    # Panel-2 info text: chain medians for P / e / K (the displayed band
    # already shows the spread, so the summary numbers stay as medians).
    info_text = (
        f"P = {period_days:.2f} d\n"
        f"e = {s['e']['median']:.3f}\n"
        f"K = {s.get('K_kms', {}).get('median', float('nan')):.2f} km/s"
    )
    ax.text(
        0.98, 0.97, info_text, transform=ax.transAxes,
        va="top", ha="right", fontsize=annotation_fontsize,
        bbox=annotation_bbox,
    )

    # --- Panel 3: O−C residuals ---
    ax = axes[2]
    ax.errorbar(t_obs, oc, yerr=rv_err_plot, **err_kw)
    ax.axhline(0, ls="-", color=gamma_color, lw=gamma_lw)
    ax.set_xlabel("MJD (days)")
    ax.set_ylabel("O−C (km/s)")

    rms_text = (
        f"N = {n_epochs}    "
        f"RMS = {rms:.2f} km/s"
    )
    ax.text(
        0.98, 0.95, rms_text, transform=ax.transAxes,
        va="top", ha="right", fontsize=annotation_fontsize,
        bbox=annotation_bbox,
    )

    fig.tight_layout()
    return fig


def plot_orbit_corner(
    result: dict,
    *,
    params: list[str] | None = None,
    labels: list[str] | None = None,
    figsize: tuple | None = None,
    label_fontsize: float = 12,
    tick_fontsize: float = 9,
    title_fontsize: float = 11,
    quantiles: list[float] | None = None,
    show_titles: bool = True,
    color: str = "gray",
    hist_kwargs: dict | None = None,
    **corner_kwargs,
) -> plt.Figure:
    """
    Corner plot of posterior orbital element samples using ``corner.py``.

    Parameters
    ----------
    result : dict
        A fit result dict (``"chains"`` + ``"summary"``) as an RV sampler returns it.
    params : list of str, optional
        Chain keys to include.  Defaults to the main orbital elements:
        ``["P_days", "e", "omega_rad", "m2_msun"]``.
        Period is displayed in days, as carried by the chain.
        Silently skips keys not present in the chain.
    labels : list of str, optional
        Custom axis labels (one per parameter).  If ``None``, uses
        built-in labels with units.
    figsize : tuple, optional
        Figure size.  Defaults to ``(n*2.5, n*2.5)``.
    label_fontsize : float, default 12
        Font size for axis labels.
    tick_fontsize : float, default 9
        Font size for tick labels.
    title_fontsize : float, default 11
        Font size for quantile titles on the diagonal.
    quantiles : list of float, optional
        Quantile lines to draw on 1-D histograms.
        Default: ``[0.16, 0.5, 0.84]``.
    show_titles : bool, default True
        Show median and 68% CI above each diagonal panel.
    color : str, default ``"gray"``
        Colour for histograms and contours.
    hist_kwargs : dict, optional
        Extra keyword arguments passed to the 1-D histogram.
    **corner_kwargs
        Any additional keyword arguments are forwarded to
        ``corner.corner()``  (e.g. ``bins``, ``smooth``,
        ``plot_contours``, ``levels``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import corner as corner_pkg

    chains = result["chains"]

    if params is None:
        # fm_ast: present on TI chains only (they report observables)
        # — the filter below drops it elsewhere.  It is the TI chain's
        # headline mass quantity, so it belongs in the default corner.
        params = ["P_days", "e", "omega_rad", "m2_msun", "fm_ast_msun"]
    # Keep only those present in the chain.
    params = [p for p in params if p in chains]

    n = len(params)
    if figsize is None:
        figsize = (n * 2.5, n * 2.5)
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]
    if hist_kwargs is None:
        hist_kwargs = {}

    # Default label map — covers both Kepler and Thiele-Innes parameterisations.
    _label_map = {
        # ``P_days`` is in days in the chain, so the axis label is days
        # and no conversion happens here (a chain in Keplerian years would
        # need multiplying by DAYS_PER_KEPLER_YEAR first; the fits export
        # days).
        "P_days": "P (days)",
        "e": "e",
        "omega_rad": "ω (rad)",
        "Omega_rad": "Ω (rad)",
        "inc_rad": "i (rad)",
        "m2_msun": "m₂ (M☉)",
        "a_rel_au": "a (AU)",
        "tp_mjd": "tp (MJD)",
        "tau": "τ",
        # Thiele-Innes amplitudes — PHOTOCENTRE orbit, mas (the sampled
        # A,B,F,G are photocentre amplitudes, not the relative orbit's).
        #
        "A_mas": "A (mas)",
        "B_mas": "B (mas)",
        "F_mas": "F (mas)",
        "G_mas": "G (mas)",
        "a_phot_mas": "α (mas)",
        # Astrometric mass function a_phot³/P² (β=0): m₂³/(m₁+m₂)².
        "fm_ast_msun": "f$_{M,\\mathrm{ast}}$ (M☉)",
        # Thiele-Innes amplitudes in Gaia DR3 NSS convention (photocenter, mas).
        "A_nss_mas": "A$_\\mathrm{NSS}$ (mas)",
        "B_nss_mas": "B$_\\mathrm{NSS}$ (mas)",
        "F_nss_mas": "F$_\\mathrm{NSS}$ (mas)",
        "G_nss_mas": "G$_\\mathrm{NSS}$ (mas)",
        "Omega_nss_rad": "Ω$_\\mathrm{NSS}$ (rad)",
        "omega_nss_rad": "ω$_\\mathrm{NSS}$ (rad)",
        "tp_nss_days": "t$_\\mathrm{p,NSS}$ (days, offset)",
        "M_total_msun": "M$_\\mathrm{tot}$ (M☉)",
        "plx_mas": "ϖ (mas)",
        "K_kms": "K (km/s)",
        "fm_spec_msun": "f(m) (M☉)",
        "gamma_kms": "γ (km/s)",
        "rv_jitter_kms": "jitter (km/s)",
    }

    # Build sample arrays; every key is plotted in its chain unit.
    samples = []
    auto_labels = []
    for p in params:
        arr = chains[p].copy()
        samples.append(arr)
        auto_labels.append(_label_map.get(p, p))

    samples = np.column_stack(samples)
    if labels is None:
        labels = auto_labels

    fig = corner_pkg.corner(
        samples,
        labels=labels,
        quantiles=quantiles,
        show_titles=show_titles,
        title_kwargs={"fontsize": title_fontsize},
        label_kwargs={"fontsize": label_fontsize},
        color=color,
        hist_kwargs=hist_kwargs,
        fig=plt.figure(figsize=figsize),
        **corner_kwargs,
    )

    # Apply tick font size.
    for ax in fig.get_axes():
        ax.tick_params(labelsize=tick_fontsize)

    return fig


# ── MCMC chain trace diagnostic ──────────────────────────────────────────────

def plot_traces(
    chain: np.ndarray,
    labels: list[str] | None = None,
    *,
    color: str = "k",
    alpha: float = 0.3,
    lw: float = 0.5,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Per-parameter MCMC trace plot — one stacked panel per dimension.

    A thin, styling-preserving convenience for the emcee trace loop that
    recurs across the fitting-example and sampler-diagnostics notebooks: it
    overplots every walker's path vs step for each sampled dimension, so the
    eye can judge burn-in and mixing.  It computes and transforms NOTHING —
    it visualises the raw chain exactly as passed, so no scientific
    convention (units, ω frame, parameter order) is hidden at the call site;
    label and interpret the dimensions in the notebook as before.

    Parameters
    ----------
    chain : np.ndarray
        The emcee chain, shape ``(n_steps, n_walkers, n_dim)`` (the output
        of ``sampler.get_chain()``).  A 2-D ``(n_steps, n_dim)`` array is
        accepted and treated as a single walker.
    labels : list of str, optional
        One y-axis label per dimension.  Defaults to ``["dim 0", ...]``;
        length must equal ``n_dim`` when supplied.
    color, alpha, lw : str, float, float
        Per-walker line styling.  The defaults (``"k"`` / ``0.3`` / ``0.5``)
        reproduce the notebooks' trace styling verbatim — preserve them
        unless you have a specific reason to change them.
    figsize : tuple, optional
        Figure size.  Defaults to ``(10, 1.6 * n_dim)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    arr = np.asarray(chain, dtype=float)
    if arr.ndim == 2:
        # (n_steps, n_dim) -> (n_steps, 1, n_dim): a single walker.
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(
            "chain must have shape (n_steps, n_walkers, n_dim) or "
            "(n_steps, n_dim)."
        )
    n_dim = arr.shape[2]

    if labels is None:
        labels = [f"dim {i}" for i in range(n_dim)]
    elif len(labels) != n_dim:
        raise ValueError(
            "labels length must match the number of chain dimensions "
            "(chain.shape[-1])."
        )

    if figsize is None:
        figsize = (10, 1.6 * n_dim)

    # squeeze=False keeps `axes` 2-D even for n_dim == 1, so the indexing
    # below is uniform.
    fig, axes = plt.subplots(
        n_dim, 1, figsize=figsize, sharex=True, squeeze=False,
    )
    axes = axes[:, 0]
    for i in range(n_dim):
        # All walkers overplotted for dimension i: matplotlib plots each
        # column of the (n_steps, n_walkers) slice as its own line.
        axes[i].plot(arr[:, :, i], color=color, alpha=alpha, lw=lw)
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    return fig


# ── Astrometric orbit on the sky ─────────────────────────────────────────────

def astrometric_display_points(
    astro_data: dict,
    *,
    per_ccd: bool = False,
) -> dict:
    """Along-scan points to DISPLAY for an astrometry plot (one per visit default).

    When ``astro_data`` carries the per-visit ``"visit"`` block (attached by
    a loader that groups per-CCD rows into transits) and ``per_ccd`` is
    False, returns ONE point per visit (transit); otherwise returns the raw per-CCD
    points.  Legacy bundles / the simulator / pickle caches have no ``"visit"`` key
    and so fall back to per-CCD automatically.

    The displayed error bar is the LARGER of the two visit errors
    (``max(err_ivw, err_scatter)``, with a NaN ``err_scatter`` — e.g. a
    single-survivor visit — falling back to ``err_ivw``).  This is a DISPLAY choice
    for honesty and carries NO scientific weight: the fits consume the raw per-CCD
    stream, never a visit error (the visit inverse-variance error is
    anti-conservative because the within-visit CCD errors are correlated).

    Parameters
    ----------
    astro_data : dict
        Loader-shaped astrometry bundle (may or may not carry ``"visit"``).
    per_ccd : bool, optional
        Force the raw per-CCD points even when a visit block is present.

    Returns
    -------
    dict
        ``obs_time``, ``scan_angle`` (rad), ``centroid_pos`` (mas),
        ``centroid_pos_err`` (mas), ``parallax_factor_al`` (dimensionless) —
        arrays of length ``n_visits`` (visit mode) or ``n_ccd`` (per-CCD mode).
        The parallax factor rides along so a caller can evaluate the along-scan
        MODEL at the display points (it is a per-transit scalar in DR4, passed
        through the visit reduction unchanged).
    """
    visit = astro_data.get("visit") if isinstance(astro_data, dict) else None
    if visit is not None and not per_ccd:
        err_ivw = np.asarray(visit["err_ivw"], dtype=float)
        err_scatter = np.asarray(visit["err_scatter"], dtype=float)
        # Larger of the two bars; NaN scatter (n=1 visit) → the IVW bar.
        display_err = np.where(
            np.isfinite(err_scatter),
            np.maximum(err_ivw, err_scatter),
            err_ivw,
        )
        return {
            "obs_time": np.asarray(visit["obs_time"], dtype=float),
            "scan_angle": np.asarray(visit["scan_angle"], dtype=float),
            "centroid_pos": np.asarray(visit["centroid_pos"], dtype=float),
            "centroid_pos_err": display_err,
            "parallax_factor_al": np.asarray(
                visit["parallax_factor_al"], dtype=float,
            ),
        }

    return {
        "obs_time": np.asarray(astro_data["obs_time"], dtype=float),
        "scan_angle": np.asarray(astro_data["scan_angle"], dtype=float),
        "centroid_pos": np.asarray(astro_data["centroid_pos"], dtype=float),
        "centroid_pos_err": np.asarray(astro_data["centroid_pos_err"], dtype=float),
        "parallax_factor_al": np.asarray(
            astro_data["parallax_factor_al"], dtype=float,
        ),
    }


def plot_astrometric_orbit(
    astro_data: dict,
    result: dict,
    *,
    ra_deg: float,
    dec_deg: float,
    parallax_mas: float,
    pmra_masyr: float,
    pmdec_masyr: float,
    t_ref: float | None = None,
    n_model_points: int = 300,
    n_draws: int = 50,
    whisker_length_mas: float | None = None,
    figsize: tuple = (7, 7),
    source_id: int | str | None = None,
    data_color: str = "black",
    model_color: str = "C0",
    draw_alpha: float = 0.08,
    whisker_lw: float = 0.8,
    model_lw: float = 1.5,
) -> plt.Figure:
    """
    Plot the astrometric orbit on the sky plane (Δα*, Δδ).

    Shows the Keplerian orbit of the photocenter after subtracting
    parallax and proper motion, with 1-D along-scan measurements
    displayed as "whisker" lines at each epoch.

    Each whisker is a short line segment along the scan direction,
    centered on the measured along-scan centroid position (after
    removing the 5-parameter model).  The whisker lies on the locus
    of sky positions consistent with the 1-D measurement.

    Parameters
    ----------
    astro_data : dict
        An epoch-astrometry dict in the loader shape
        :class:`~orblet.simulate.orbit.OrbitSimulator` documents.
    result : dict
        A fit result dict (``"chains"`` + ``"summary"``) as an
        astrometric sampler returns it.
    ra_deg, dec_deg : float
        Catalog position (degrees, ICRS).
    parallax_mas : float
        Catalog parallax (mas).
    pmra_masyr, pmdec_masyr : float
        Catalog proper motion (mas/yr).
    t_ref : float, optional
        Reference epoch in days from J2010.0 TCB.
        Default: 2191.5 (J2016.0).
    n_model_points : int, default 300
        Number of points for the model orbit curve.
    n_draws : int, default 50
        Number of posterior draws to overlay.
    whisker_length_mas : float or None, optional
        Fixed half-length of each across-scan whisker (mas).  This is
        a *cosmetic* length: the across-scan direction is unconstrained
        by an AL measurement.  If None (default), the AC whisker is
        NOT drawn — only the along-scan (AL) error tick, which is the
        physically meaningful 1-σ AL precision, is shown at each epoch.
        Pass a positive float to draw cosmetic AC whiskers with that
        explicit half-length.  The AL tick is drawn unconditionally
        from ``astro_data["centroid_pos_err"]`` when available.
    figsize : tuple, default (7, 7)
        Figure size.
    source_id : int or str or None, optional
        Shown in the title.
    data_color : str, default ``"black"``
        Colour of the whisker lines.
    model_color : str, default ``"C0"``
        Colour of the median model orbit.
    draw_alpha : float, default 0.08
        Transparency of individual posterior draws.
    whisker_lw : float, default 0.8
        Line width of whiskers.
    model_lw : float, default 1.5
        Line width of the median model orbit.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # This function builds its along-scan O−C inline below and imports no
    # data-loading code: plotting needs arrays, not a data source, which
    # is what lets orblet stand alone.
    #
    #
    from orblet.elements import ti_to_kepler

    chains = result["chains"]

    # ── Auto-detect Thiele-Innes chain and convert to Keplerian ──────────
    # The rest of this function draws the orbit in Keplerian elements
    # (P, e, ω, i, Ω, tp, a_phot).  If the fit was done in the T-I basis
    # the chain contains (A, B, F, G, e, τ, mass) instead; convert so
    # downstream code is basis-agnostic.  The T-I fit also exposes
    # (P_days, a_rel_au) as derived variables in the @variables block,
    # so only (inc_rad, omega_rad, Omega_rad) need to be computed here.
    if "A_mas" in chains and "inc_rad" not in chains:
        chains = ti_to_kepler(chains)

    # ── Reference epoch ──────────────────────────────────────────────────
    # J2016.0 expressed in days from J2010.0; derived from constants
    # (= MJD_J2016_TCB − MJD_J2010_TCB = 2191.5).
    _J2016_TCB = MJD_J2016_TCB - MJD_J2010_TCB
    if t_ref is None:
        t_ref = _J2016_TCB

    # ── Observation data ─────────────────────────────────────────────────
    t_obs = astro_data["obs_time"]        # days from J2010.0
    psi = astro_data["scan_angle"]        # radians
    centroid = astro_data["centroid_pos"]  # mas
    f_pi = astro_data["parallax_factor_al"]

    # ── Subtract the 5-parameter model ────────────────────────────────
    # For a binary, the catalog astrometric solution (position, PM,
    # parallax) is biased because Gaia fitted a single-star model to
    # data that contains orbital motion.  Using the catalog values to
    # compute residuals leaves large systematic offsets (~tens of mas)
    # that dwarf the true orbital signal (~few mas).
    #
    # Nuisance keys are read by their plain names; no chain produced by
    # orblet carries a mission prefix on them.
    #
    #

    # The Gaia archive centroid_pos_al already has the catalog
    # 5-parameter model subtracted.  What remains is approximately
    # the orbital signal plus noise plus a small correction for the
    # orbital bias in the catalog solution.
    #
    # The 5-parameter subtraction uses the engine's own ``simulate``
    # sub-dict (``result["simulate"]``, built at fit time by
    # ``orbit/parallax_factors.build_simulate_sub_dict`` from the MAP
    # sample), so it shares the fit's parallax factors and epoch
    # convention.  Raw chain values are not used as position / PM
    # offsets here; for the sky plot the centroid itself is the best
    # proxy for the along-scan orbital residual.
    # ── Along-scan O−C residuals ─────────────────────────────────────
    # The full O−C residual (data minus full model including orbit)
    # measures how far each observation deviates from the model along
    # the scan direction.  These are small (noise-like) for a good fit.
    sim = result.get("simulate")
    if sim is not None and "resid_al" in sim:
        resid_al = sim["resid_al"]
    else:
        resid_al = np.zeros(len(t_obs))

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # ── Model orbit from posterior chains ─────────────────────────────────
    # Compute the photocenter orbit (Δα*, Δδ) for each posterior draw.
    # The orbit is parameterised by Visual{KepOrbit} elements.
    # We compute it from Kepler's equation directly.
    P_days = chains.get("P_days")
    ecc = chains.get("e")
    omega = chains.get("omega_rad")      # argument of periastron (rad)
    inc = chains.get("inc_rad")        # inclination (rad)
    Omega = chains.get("Omega_rad")      # longitude of ascending node (rad)
    tp = chains.get("tp_mjd")        # time of periastron (MJD)
    a_au = chains.get("a_rel_au")       # semi-major axis (AU; Campbell only)
    plx_chain = chains.get("plx_mas")  # parallax (mas)
    mass_msun = chains.get("m2_msun")  # companion mass (M_sun)
    M_sys = chains.get("M_total_msun")  # system mass = totalmass (M_sun)
    # Photocentre amplitude straight from the TI inversion (mas).
    # Present on TI chains, which report observables and carry NO b_a /
    # b_mass / M — the alpha-first branch below is the TI path; the
    # mass-ratio round trip remains for Campbell chains, which do carry
    # the masses (algebraically the same quantity).
    alpha_mas = chains.get("a_phot_mas")

    if any(v is None for v in [P_days, ecc, omega, inc, Omega, tp]) or (
        a_au is None and alpha_mas is None
    ):
        raise ValueError(
            "Result chains missing required orbital elements for "
            "astrometric orbit plot."
        )

    # Photocenter semi-major axis in mas.
    #
    # In the companion-frame ω convention (common in planet codes) the
    # photocentre offset is the relative orbit scaled by −m₂/M_total: a
    # NEGATIVE amplitude.  orblet's chains carry ``omega_rad`` in the
    # PRIMARY frame, and with primary-frame ω the same photocentre orbit
    # has a POSITIVE amplitude, a_phot = +a·(m₂/M_total)·ϖ.  Mixing the
    # two (primary-frame ω with a negative amplitude, or the reverse)
    # would draw the orbit rotated by half a turn.  Both conventions
    # describe one orbit; see the reference manual, §1.4 and §3.2.
    #
    #
    # This matches
    # :meth:`orblet.simulate.orbit.OrbitSimulator.along_scan_orbit`'s
    # convention.
    if alpha_mas is not None:
        # TI chain: the measured photocentre amplitude, no mass round
        # trip needed (algebraically identical to the legacy
        # a_au·(m₂/M)·plx on chains that carry both — the masses
        # cancel).
        a_phot_mas = alpha_mas
    elif mass_msun is not None and M_sys is not None and plx_chain is not None:
        a_phot_mas = a_au * (mass_msun / M_sys) * plx_chain
    elif plx_chain is not None:
        a_phot_mas = a_au * plx_chain
    else:
        a_phot_mas = a_au * parallax_mas

    # Time grid for model orbit (one full period, centered on data).
    t_mid_mjd = float(np.median(t_obs)) + MJD_J2010_TCB  # MJD
    P_days_median = float(np.median(P_days))
    t_model_mjd = np.linspace(
        t_mid_mjd - 0.6 * P_days_median,
        t_mid_mjd + 0.6 * P_days_median,
        n_model_points,
    )

    # ── Find best (MAP) sample ────────────────────────────────────────────
    # For multimodal posteriors (e.g., the ω/Ω 180° degeneracy, or the
    # post-audit-batch-4 full-sphere (i, ω, Ω) ↔ (π−i, π−ω, π−Ω) mirror),
    # element-wise medians fall between modes and produce a nonsensical
    # orbit.  Use the single posterior sample with the highest log-posterior
    # density as the representative "best" orbit.
    #
    # The chain key is ``logpost`` (cross-engine schema parity contract),
    # with ``log_density`` accepted as an older spelling.  Without either,
    # the function falls back to the sample closest to the median period,
    # which is a quasi-random draw from the posterior: the plotted orbit
    # then need not match the data, and that mismatch is the sign the
    # chain carried no log-posterior.  Fits made with orblet always carry
    # ``logpost``.
    log_density = chains.get("logpost", chains.get("log_density"))
    if log_density is not None:
        best_idx = int(np.argmax(log_density))
    else:
        # Fallback: use the sample closest to the median period
        # (least sensitive to angular degeneracies).  Reached only if
        # neither ``logpost`` nor ``log_density`` is present — i.e. a
        # synthetic chain dict the caller built by hand.
        # Deterministic tie-break (trap #34): an even-length chain's
        # median is the mean of the two central draws, so two indices
        # can be EXACTLY equidistant and plain argmin becomes bit-level
        # unstable under any rescale.  Stable argsort pins the lowest.
        best_idx = int(np.argsort(
            np.abs(P_days - np.median(P_days)), kind="stable",
        )[0])

    # ── Draw posterior orbits ─────────────────────────────────────────────
    n_samples = len(P_days)
    draw_idx = np.random.choice(
        n_samples, size=min(n_draws, n_samples), replace=False,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ``campbell_xy`` takes the period in Keplerian YEARS (see
    # :func:`orblet.kepler.campbell_xy`); the chain key
    # ``P_days`` is in days, hence the division at every call below.
    for idx in draw_idx:
        dra, ddec = campbell_xy(
            t_model_mjd,
            P_days[idx] / DAYS_PER_KEPLER_YEAR,
            ecc[idx], omega[idx], inc[idx],
            Omega[idx], tp[idx], a_phot_mas[idx],
        )
        ax.plot(dra, ddec, color=model_color, alpha=draw_alpha, lw=0.6)

    # Best (MAP) orbit (thicker line).
    dra_med, ddec_med = campbell_xy(
        t_model_mjd,
        P_days[best_idx] / DAYS_PER_KEPLER_YEAR, ecc[best_idx],
        omega[best_idx], inc[best_idx],
        Omega[best_idx], tp[best_idx],
        a_phot_mas[best_idx],
    )
    ax.plot(dra_med, ddec_med, color=model_color, lw=model_lw, label="Best orbit")

    # ── Whisker anchor: orbit position projected onto the AL locus ──
    # An AL observation η_obs constrains the locus
    #     sin(ψ)·Δα* + cos(ψ)·Δδ = η_obs
    # which is a *line* in (Δα*, Δδ) — the across-scan direction is
    # entirely unconstrained by a single 1-D measurement.  We pick the
    # point on this line closest to the model orbit at the same epoch
    # as the anchor for both the AC whisker and the AL error tick:
    #     anchor = orbit_model + resid_al * (sin ψ, cos ψ)
    # where resid_al = η_obs − η_model is taken from the simulator output.
    t_obs_mjd = t_obs + MJD_J2010_TCB
    dra_at_obs, ddec_at_obs = campbell_xy(
        t_obs_mjd,
        P_days[best_idx] / DAYS_PER_KEPLER_YEAR, ecc[best_idx],
        omega[best_idx], inc[best_idx],
        Omega[best_idx], tp[best_idx],
        a_phot_mas[best_idx],
    )
    x_anchor = dra_at_obs + resid_al * sin_psi
    y_anchor = ddec_at_obs + resid_al * cos_psi

    # ── Across-scan whisker half-length (cosmetic) ───────────────────
    # AL does not constrain the across-scan direction, so this length
    # is purely visual.  Default (``None``) is OFF: the AC whisker is
    # only drawn when the caller passes an explicit positive
    # ``whisker_length_mas``.  The AL tick (drawn separately below) is
    # the only physically meaningful error bar at each epoch.
    if whisker_length_mas is not None:
        w_ac = np.full(len(t_obs), float(whisker_length_mas))
    else:
        w_ac = None

    # ── Along-scan tick half-length: actual AL measurement σ ────────
    # This is the real error bar: the 1-σ AL uncertainty drawn along
    # the scan direction.  Set to None if no per-epoch errors exist.
    centroid_err = astro_data.get("centroid_pos_err")
    if centroid_err is not None:
        w_al = np.asarray(centroid_err, dtype=float)
    else:
        w_al = None

    # ── Draw whiskers and AL error ticks ─────────────────────────────
    # Each epoch contributes:
    #   - a short along-scan tick   = ±1σ_AL precision (always drawn
    #     when ``centroid_pos_err`` is available)
    #   - an optional across-scan line  = AL locus (unconstrained
    #     direction); drawn only when ``whisker_length_mas`` is set.
    # Both are anchored at the same point.  No data "dot" is drawn,
    # because the dot's position depends on the AC projection, which
    # the data does not constrain.
    for i in range(len(t_obs)):
        if w_ac is not None:
            # Across-scan unit vector: (-cos ψ, sin ψ) in (Δα*, Δδ).
            ac_ra = -cos_psi[i]
            ac_dec = sin_psi[i]
            ax.plot(
                [x_anchor[i] - w_ac[i] * ac_ra, x_anchor[i] + w_ac[i] * ac_ra],
                [y_anchor[i] - w_ac[i] * ac_dec, y_anchor[i] + w_ac[i] * ac_dec],
                color=data_color, lw=whisker_lw, alpha=0.3,
                solid_capstyle="round",
            )

        # Along-scan unit vector: (sin ψ, cos ψ).  Drawn opaque so the
        # error tick remains visible against the cosmetic AC whisker.
        if w_al is not None:
            al_ra = sin_psi[i]
            al_dec = cos_psi[i]
            ax.plot(
                [x_anchor[i] - w_al[i] * al_ra, x_anchor[i] + w_al[i] * al_ra],
                [y_anchor[i] - w_al[i] * al_dec, y_anchor[i] + w_al[i] * al_dec],
                color=data_color, lw=whisker_lw, alpha=1.0,
                solid_capstyle="round",
            )

    # ── Barycenter marker ─────────────────────────────────────────────────
    ax.plot(0, 0, "+", color="gray", ms=10, mew=1.5, zorder=5)

    # ── Labels and styling ────────────────────────────────────────────────
    ax.set_xlabel("Δα* (mas)")
    ax.set_ylabel("Δδ (mas)")
    ax.set_aspect("equal")
    ax.invert_xaxis()  # RA increases to the left on the sky
    ax.legend(loc="upper right", fontsize=9)

    title = "Astrometric orbit"
    if source_id is not None:
        title += f" — {source_id}"
    ax.set_title(title)

    fig.tight_layout()
    return fig


# ── Sky-plane overlay (lifted from notebook cell 8dee6b61) ────────────────

def plot_astrometric_sky_overlay(
    astro_data: dict,
    chains: dict,
    *,
    figsize: tuple = (7, 7),
    source_id: int | str | None = None,
    per_ccd: bool = False,
):
    """
    Sky-plane overlay of astrometric and RV best-fit orbits with AL data.

    Lifted from ``notebooks/single_source_analysis.ipynb`` cell
    ``8dee6b61``.  Renders three layers in a single ``(Δα*, Δδ)``
    panel:

      1. Astrometric best fit — median of ``chains`` posterior, drawn
         as the photocenter orbit (parallax+PM subtracted by
         construction; the curve is the orbital signal only).
      2. RV-only edge-on (``sin i = 1``, ``i = π/2``) — the minimum
         sky-plane size compatible with the RV posterior.  Uses RV
         medians for ``(P, e, ω, tp, K)`` and inherits ``Ω`` from the
         astrometric chain (RV cannot constrain ``Ω``).
      3. AL data points — each observed centroid (assumed to already be
         parallax+PM-subtracted) is 1-D, so the marker is placed at the
         observed AL value while the across-scan position is locked to
         the astrometric model's prediction at that transit time.

    The figure shows the data it is given, and ``source_id`` if one is
    passed: whether it may be shared is decided by whoever owns the
    data.

    Photocenter convention: β = 0 (dark companion) ⇒
    ``a_phot = a · m₂ / M_total · plx`` with POSITIVE sign in mas.
    ``omega_rad`` is the primary's argument of periastron
    (binary-star textbook convention), so combining a
    primary-frame ω with the textbook (companion-frame) T-I formula
    requires a positive amplitude — matching ``plot_astrometric_orbit``
    and the simulator.

    Parameters
    ----------
    astro_data : dict
        Required keys: ``obs_time`` (J2010 days), ``scan_angle``
        (rad), ``centroid_pos`` (mas, parallax+PM subtracted),
        ``centroid_pos_err`` (mas), ``parallax_factor_al``.
    chains : dict
        Posterior chains.  Required keys: ``P_days`` (days), ``e``,
        ``omega_rad`` (rad), ``inc_rad`` (rad), ``Omega_rad`` (rad),
        ``tp_mjd`` (MJD), ``plx_mas`` (mas), ``K_kms`` (km/s), plus
        EITHER ``a_phot_mas``
        (photocentre amplitude, mas — TI chains, which carry no mass
        keys) OR the legacy triple
        ``a_rel_au`` (AU) + ``m2_msun`` (M☉) + ``M_total_msun`` (M☉,
        total) for Campbell / legacy chains.
    figsize : tuple, default (7, 7)
        Figure size.
    source_id : int or str or None, optional
        Shown in the title when given; ``None`` (the default) leaves it
        out.
    per_ccd : bool, optional
        By default the AL data are drawn VISIT-AVERAGED (one point per transit) when
        ``astro_data`` carries the per-visit ``"visit"`` block, falling back to raw
        per-CCD points otherwise.  ``per_ccd=True`` forces the raw per-CCD points.
        See :func:`astrometric_display_points`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes carrying all three plotted layers.
    """
    from orblet.kepler import campbell_xy

    # ── ω convention sentinel guard ───────────────────────────────────
    # The Thiele-Innes projection below combines ``omega_rad`` with the
    # photocenter amplitude.  Reject chains that lack the sentinel or
    # carry a non-primary convention; pairing companion-frame ω with the
    # post-realignment positive amplitude would phase-flip the rendered
    # orbit by 180°.  Mirrors the guards in
    # :func:`orblet.elements.compute_rv_model_curve` and
    # :func:`orblet.elements.compute_residuals`.
    from orblet.constants import OMEGA_CONVENTION_PRIMARY
    omega_conv = (
        chains.get("_meta", {}).get("omega_convention")
        if isinstance(chains, dict) else None
    )
    if omega_conv != OMEGA_CONVENTION_PRIMARY:
        raise ValueError(
            f"plot_astrometric_sky_overlay requires primary-frame ω; got "
            f"omega_convention={omega_conv!r}.  If the chain's ω is the "
            f"primary's, add ``_meta['omega_convention'] = "
            f"'primary_argument_of_periastron'`` to the chain dict; "
            f"otherwise convert it first (reference manual, §1.4)."
        )

    # ── MAP-sample selection ──────────────────────────────────────────
    # Per-parameter medians would yield a
    # NONSENSICAL orbit on multimodal posteriors: the median (ω, Ω, i)
    # rarely sits at the median (a, e, P) point in the chain, so the
    # drawn ellipse corresponds to no actual sample (plot_orbit_fit
    # avoids the same pathology for RV the same way).  Here we
    # use the MAP sample (a single coherent posterior draw with the
    # highest log-posterior) as the representative orbit; the legacy
    # ``log_density`` key is the fallback for hand-built chain dicts.
    log_density = chains.get("logpost", chains.get("log_density"))
    if log_density is not None:
        best_idx = int(np.argmax(np.asarray(log_density, dtype=float)))
    else:
        # Hand-built chain dict without either key — fall back to the
        # median-of-period heuristic (least sensitive to angular
        # degeneracies); not robust but better than nothing.
        P_days_arr = np.asarray(chains["P_days"], dtype=float)
        # Deterministic tie-break — see trap #34 and the twin site above.
        best_idx = int(np.argsort(
            np.abs(P_days_arr - np.median(P_days_arr)), kind="stable",
        )[0])

    def _at_map(key: str) -> float:
        """Read ``chains[key][best_idx]`` as a Python float."""
        return float(np.asarray(chains[key], dtype=float)[best_idx])

    # Astrometric MAP values (one coherent posterior sample).
    P_days_a = _at_map("P_days")
    e_a = _at_map("e")
    omega_a = _at_map("omega_rad")
    inc_a = _at_map("inc_rad")
    Omega_a = _at_map("Omega_rad")
    tp_a = _at_map("tp_mjd")
    plx_a = _at_map("plx_mas")

    # Positive photocenter amplitude: post-realignment cycle (commit
    # f249aa3) ``omega_rad`` is the primary's argument of periastron.  Pairing
    # primary-frame ω with the standard textbook (companion-frame) T-I
    # formula in :func:`campbell_xy` requires the POSITIVE amplitude
    # — matches ``plot_astrometric_orbit`` (line 555) and the simulator.
    # See the reference manual, §1.4, for the ω convention.
    #
    # TI chains carry the measured photocentre amplitude directly
    # (``a_phot_mas``) and — reporting observables first —
    # NO ``a_rel_au``/``m2_msun``/``M_total_msun``; Campbell / legacy chains
    # keep the mass-ratio round trip (algebraically the same quantity).
    if "a_phot_mas" in chains:
        a_phot_mas_a = _at_map("a_phot_mas")
    else:
        a_au_a = _at_map("a_rel_au")
        mass_a = _at_map("m2_msun")
        M_a = _at_map("M_total_msun")
        a_phot_mas_a = a_au_a * (mass_a / M_a) * plx_a

    # RV side: same MAP sample (a single coherent posterior draw).
    P_days_rv = _at_map("P_days")  # falls back to astrometric P
    if "K_kms" in chains:
        K_rv = _at_map("K_kms")
    else:
        K_rv = 0.0
    e_rv = e_a
    omega_rv = omega_a
    tp_rv = tp_a

    # RV-only a₁·sin(i) → AU, then mas.  At sin(i)=1 this IS a₁ (the
    # minimum sky-plane size compatible with the RV posterior).
    _AU_PER_KMS_DAY = 86400.0 / 1.495978707e8
    a1_sini_au_rv = (
        K_rv * P_days_rv * np.sqrt(1.0 - e_rv ** 2) / (2.0 * np.pi)
        * _AU_PER_KMS_DAY
    )
    # sin(i)=1 ⇒ multiply by 1; convert AU → mas via the same parallax
    # used for the astrometric photocenter.
    a_edge_on_mas = a1_sini_au_rv * plx_a

    # ── Smooth model curves ───────────────────────────────────────────
    # Per-curve time grids spanning exactly one period each, anchored at
    # periastron, so both ellipses close cleanly.
    t_grid_a = np.linspace(0.0, P_days_a, 400) + tp_a
    t_grid_rv = np.linspace(0.0, P_days_rv, 400) + tp_rv

    # ``campbell_xy`` takes ``period_yr`` in Keplerian YEARS while the
    # chain key ``P_days`` is in days — hence the division here and at
    # the two calls below.
    dRA_a, dDec_a = campbell_xy(
        t_mjd=t_grid_a, period_yr=P_days_a / DAYS_PER_KEPLER_YEAR, ecc=e_a,
        omega_rad=omega_a, inc_rad=inc_a, Omega_rad=Omega_a,
        tp_mjd=tp_a, a_mas=a_phot_mas_a,
    )
    # RV-only edge-on: i=π/2; Ω inherited from astrometric fit.
    dRA_rv, dDec_rv = campbell_xy(
        t_mjd=t_grid_rv, period_yr=P_days_rv / DAYS_PER_KEPLER_YEAR,
        ecc=e_rv,
        omega_rad=omega_rv, inc_rad=np.pi / 2.0, Omega_rad=Omega_a,
        tp_mjd=tp_rv, a_mas=a_edge_on_mas,
    )

    # ── Data points: AL measurements snapped to model AC direction ───
    # ``astro_data["centroid_pos"]`` is taken as the
    # parallax+PM-subtracted along-scan residual (i.e., the "data minus
    # 5-par" output from the engine's simulate dict, or equivalently the
    # output of ``subtract_5parameter_astrometry``).  Compute the
    # astrometric model at each observed transit, decompose into (AL,
    # AC), and place the data point at AL = w_resid_obs, AC = AC_model.
    # DISPLAY points: visit-averaged by default (one per transit), falling back to
    # per-CCD when the bundle has no visit block (legacy / simulator / pickle).  The
    # error bar is a display choice (larger of err_ivw / err_scatter) with no
    # scientific weight — the fits use the raw per-CCD stream.  ``per_ccd=True``
    # forces the raw points.
    _disp = astrometric_display_points(astro_data, per_ccd=per_ccd)
    t_obs = _disp["obs_time"]
    psi = _disp["scan_angle"]
    w_resid = _disp["centroid_pos"]
    sigma = _disp["centroid_pos_err"]

    # Convert obs_time (J2010 days) to MJD for campbell_xy.
    t_mjd = t_obs + MJD_J2010_TCB

    dRA_model_obs, dDec_model_obs = campbell_xy(
        t_mjd=t_mjd, period_yr=P_days_a / DAYS_PER_KEPLER_YEAR, ecc=e_a,
        omega_rad=omega_a, inc_rad=inc_a, Omega_rad=Omega_a,
        tp_mjd=tp_a, a_mas=a_phot_mas_a,
    )
    sin_psi_obs = np.sin(psi)
    cos_psi_obs = np.cos(psi)

    # AC unit vector is the scan unit vector rotated +90°: (cos ψ, -sin ψ).
    AC_model = dRA_model_obs * cos_psi_obs - dDec_model_obs * sin_psi_obs

    # Reconstruct (Δα*, Δδ) of the data point: AL · (sin ψ, cos ψ) +
    # AC · (cos ψ, -sin ψ).
    data_dRA = w_resid * sin_psi_obs + AC_model * cos_psi_obs
    data_dDec = w_resid * cos_psi_obs - AC_model * sin_psi_obs

    # 1σ along-scan error bars as line segments from the data point along
    # the scan direction.  NaN-separated arrays render as a single
    # ax.plot call (no LineCollection / Python loop needed).
    err_lo_x = data_dRA - sigma * sin_psi_obs
    err_hi_x = data_dRA + sigma * sin_psi_obs
    err_lo_y = data_dDec - sigma * cos_psi_obs
    err_hi_y = data_dDec + sigma * cos_psi_obs
    nan_pad = np.full(err_lo_x.shape, np.nan)
    errbar_x = np.column_stack([err_lo_x, err_hi_x, nan_pad]).ravel()
    errbar_y = np.column_stack([err_lo_y, err_hi_y, nan_pad]).ravel()

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # 1σ AL error bars (under the curves so they don't obscure the model).
    ax.plot(errbar_x, errbar_y, "-", color="0.6", lw=0.6, alpha=0.7, zorder=1)

    # Model curves.
    ax.plot(
        dRA_a, dDec_a, "-", lw=1.8, color="C3", zorder=3,
        label=fr"Astrometric best fit  ($i = {np.rad2deg(inc_a):.1f}^\circ$, "
              fr"$a_\mathrm{{phot}} = {a_phot_mas_a:.3f}$ mas)",
    )
    ax.plot(
        dRA_rv, dDec_rv, "--", lw=1.5, color="C0", zorder=3,
        label=fr"RV edge-on  ($\sin i = 1$, "
              fr"$a_1 = {a_edge_on_mas:.3f}$ mas)",
    )

    # Data points — drawn last so they sit on top of everything else.
    ax.scatter(
        data_dRA, data_dDec,
        s=14, marker="o", facecolor="white", edgecolor="black", lw=0.9,
        zorder=5,
        label=f"Gaia AL transits  (N={len(data_dRA)})",
    )
    ax.scatter(0.0, 0.0, marker="+", color="k", s=70, zorder=4,
               label="barycenter")

    ax.set_xlabel(r"$\Delta\alpha^{*}$ [mas]")
    ax.set_ylabel(r"$\Delta\delta$ [mas]")
    ax.set_aspect("equal")
    ax.invert_xaxis()  # standard astronomical convention
    title = "Sky-plane orbit overlay"
    if source_id is not None:
        title += f" — source {source_id}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return ax
