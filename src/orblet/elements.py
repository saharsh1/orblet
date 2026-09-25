"""
Post-processing of orbital fit results.

Extracts orbital elements, computes model RV curves and residuals
from the chain dictionary an RV sampler returns (``"chains"`` and
``"summary"`` keys; the schema each function needs is stated on it).

All public functions work with plain numpy arrays and the ``result``
dict returned by the fitter.

Units
-----
- Period        : returned in **days** (the chain's ``P_days`` is
                  already in days).
- RV, offset    : returned in **km/s** (bridge already converts back).
- Jitter        : returned in **km/s**.
- Masses        : companion in M_Jup, stellar in M_sun  (unchanged).
- Semi-major axis : AU  (unchanged).
- ω, tp         : radians, MJD  (unchanged).

ω convention
------------
``omega_rad`` is reported in the **primary** frame (binary-star convention,
``ω = primary's argument of periastron``), as emitted by the Python
engine.  This means :func:`to_nss_convention` produces ``omega_nss_rad``
aligned with the Halbwachs et al. 2023 (A&A 674, A9, §2.2) catalogue
convention.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time

from orblet.constants import DAYS_PER_KEPLER_YEAR

# Value-free (no epoch value, path or identifier in the text).
_NSS_NO_CATALOGUE_EPOCH_MSG = (
    "an NSS t_periastron needs the catalogue reference epoch, and this "
    "chain does not carry one: _meta has no 'epoch_ref_mjd'. NSS reports "
    "t_periastron relative to gaia_source.ref_epoch, so an absolute time "
    "cannot be converted without it. Astrometric and joint fits record "
    "that epoch automatically; an RV-ONLY chain carries 'epoch_ref' "
    "instead, which is the MEDIAN OBSERVATION EPOCH (a gauge for omega) "
    "and not a catalogue epoch — an NSS periastron time cannot be formed "
    "from it. Pass include_keplerian=False for the amplitude-only "
    "conversion, which needs no epoch."
)


# ── Orbital element extraction ────────────────────────────────────────────────

def extract_orbital_elements(result: dict) -> dict:
    """
    Summarise posterior orbital elements from fit result.

    For each parameter, reports the median and 68 % credible interval
    (16th–84th percentile).  Period is reported in days, the unit the
    chain's ``P_days`` already carries.

    Parameters
    ----------
    result : dict
        A fit result dict (``"chains"`` + ``"summary"``) as an RV sampler returns it.
        Must contain ``"chains"`` and ``"summary"`` keys.

    Returns
    -------
    elements : dict
        Keys are human-readable parameter names.  Each value is a dict
        with ``"median"``, ``"q16"``, ``"q84"``, ``"unit"``.

    Examples
    --------
    >>> elements = extract_orbital_elements(result)
    >>> elements["period_days"]["median"]
    123.4
    """
    chains = result["chains"]
    summary = result["summary"]

    def _stat(key):
        s = summary[key]
        return {"median": s["median"], "q16": s["q16"], "q84": s["q84"]}

    elements = {}

    # Period: the chain key is already in days.
    if "P_days" in chains:
        p_days = chains["P_days"]
        elements["period_days"] = {
            "median": float(np.median(p_days)),
            "q16": float(np.percentile(p_days, 16)),
            "q84": float(np.percentile(p_days, 84)),
            "unit": "days",
        }

    # Eccentricity (dimensionless).
    if "e" in chains:
        elements["eccentricity"] = {**_stat("e"), "unit": ""}

    # Argument of periastron (radians).
    if "omega_rad" in chains:
        elements["omega_rad"] = {**_stat("omega_rad"), "unit": "rad"}

    # Epoch of periastron (MJD).
    if "tp_mjd" in chains:
        elements["tp_mjd"] = {**_stat("tp_mjd"), "unit": "MJD"}

    # Companion mass (solar masses — converted from M_Jup in the bridge).
    if "m2_msun" in chains:
        elements["companion_mass_msun"] = {**_stat("m2_msun"), "unit": "M_sun"}

    # Total system mass (solar masses; Campbell/joint chains only).
    # Read from ``M_total_msun``, the key the fits emit, so the summary
    # always carries a total-mass entry.
    if "M_total_msun" in chains:
        elements["total_mass_msun"] = {
            **_stat("M_total_msun"), "unit": "M_sun",
        }

    # Semi-major axis (AU).
    if "a_rel_au" in chains:
        elements["semi_major_axis_au"] = {**_stat("a_rel_au"), "unit": "AU"}

    # RV semi-amplitude K (km/s) — derived per sample in the bridge.
    if "K_kms" in chains:
        elements["K_kms"] = {**_stat("K_kms"), "unit": "km/s"}

    # Spectroscopic mass function f(m) = K³ P (1-e²)^{3/2} / (2πG).
    # Model-independent observable: f(m) = m₂³ sin³i / (m₁ + m₂)².
    # Derived per sample in the bridge.
    if "fm_spec_msun" in chains:
        elements["mass_function_msun"] = {
            **_stat("fm_spec_msun"), "unit": "M_sun",
        }

    # RV offset (km/s — already converted by bridge).
    if "gamma_kms" in chains:
        elements["gamma_kms"] = {
            **_stat("gamma_kms"), "unit": "km/s",
        }

    # RV jitter (km/s — already converted by bridge).
    if "rv_jitter_kms" in chains:
        elements["rv_jitter_kms"] = {
            **_stat("rv_jitter_kms"), "unit": "km/s",
        }

    return elements


# ── Keplerian RV model ────────────────────────────────────────────────────────

def _solve_kepler(mean_anomaly: np.ndarray, ecc: float) -> np.ndarray:
    """
    Solve Kepler's equation ``M = E - e sin(E)`` via Newton-Raphson.

    Thin wrapper around :func:`orblet.kepler.solve_kepler`;
    kept for backward-compatible imports.
    """
    from orblet.kepler import solve_kepler
    return solve_kepler(mean_anomaly, ecc)


def _keplerian_rv(
    times: np.ndarray,
    period_days: float,
    ecc: float,
    omega: float,
    K: float,
    gamma: float,
    tp: float,
) -> np.ndarray:
    """
    Evaluate a Keplerian radial-velocity curve.

    Parameters
    ----------
    times : array
        Observation times (MJD, days).
    period_days : float
        Orbital period in days.
    ecc : float
        Eccentricity.
    omega : float
        Argument of periastron of the PRIMARY (radians; primary-frame
        convention — binary-star textbook convention; reference manual,
        §1.4).  Every fit in orblet emits ``omega_rad`` in this frame.

    K : float
        RV semi-amplitude (km/s).
    gamma : float
        Systemic velocity / RV offset (km/s).
    tp : float
        Epoch of periastron passage (MJD, days).

    Returns
    -------
    rv : array
        Model radial velocities (km/s).
    """
    # Mean anomaly.
    M = 2.0 * np.pi * (times - tp) / period_days

    # Eccentric anomaly.
    E = _solve_kepler(M, ecc)

    # True anomaly.
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )

    # Positive sign: ω is the primary's argument of periastron
    # (binary-star textbook convention; reference manual, §1.4).  Standard
    # textbook
    # primary-frame RV:  v_r = γ + K [cos(ν + ω) + e · cos(ω)].
    # The Python RV forward model in
    # :mod:`orblet.model` uses the same form.
    return gamma + K * (np.cos(nu + omega) + ecc * np.cos(omega))


def _stellar_K_from_elements(
    mass_mjup: float,
    period_yr: float,
    ecc: float,
    M_total_msun: float,
) -> float:
    """
    Compute the stellar RV semi-amplitude K_star (km/s) from the orbital
    elements and the total system mass.

    The chain below uses the total system mass (``M_total``), for
    chains that sample the total mass directly:


    1. ``a = cbrt(P² × M_total)``  — semi-major axis (AU) from
       standard Kepler III with total mass.
    2. ``J = 2π a / P_yr / sqrt(1 - e²)``  — companion velocity
       semi-amplitude in AU/yr.
    3. ``K_comp = J × AU_to_m / sec_per_yr``  — companion K in m/s.
    4. ``K_star = (M_comp / M_total) × K_comp``  — stellar K.

    This is the standard Keplerian formula:
        K_star = (2π / P)^{1/3} × M_comp / M_total^{2/3} / sqrt(1-e²)

    Parameters
    ----------
    mass_mjup : float
        Companion mass in Jupiter masses.
    period_yr : float
        Period in Keplerian years.
    ecc : float
        Eccentricity.
    M_total_msun : float
        Total system mass (M_star + M_comp) in solar masses.

    Returns
    -------
    K_star : float
        Stellar RV semi-amplitude in km/s (positive).
    """
    from orblet.priors import MSUN_IN_MJUP

    # Step 1: semi-major axis from Kepler III with total mass.
    # a³ = P² × M_total  →  a in AU, P in years, M in M_sun.
    a_au = (period_yr**2 * M_total_msun) ** (1.0 / 3.0)

    # Step 2: companion velocity semi-amplitude [AU/yr]
    J = (2.0 * np.pi * a_au) / period_yr / np.sqrt(1.0 - ecc**2)

    # Step 3: convert to m/s
    from orblet.constants import AU_M
    sec_per_yr = DAYS_PER_KEPLER_YEAR * 86400.0
    K_comp_ms = J * AU_M / sec_per_yr

    # Step 4: stellar K = M_comp / M_total × K_companion
    # M_total = system.M (total mass at system level).
    m_comp_msun = mass_mjup / MSUN_IN_MJUP
    K_star_ms = (m_comp_msun / M_total_msun) * K_comp_ms

    return K_star_ms / 1000.0  # m/s → km/s


# ── Model curve and residuals ─────────────────────────────────────────────────

def compute_rv_model_curve(
    result: dict,
    t_grid: np.ndarray,
) -> dict:
    """
    Evaluate the best-fit RV curve on a fine time grid.

    Uses median posterior values for all orbital parameters.

    Parameters
    ----------
    result : dict
        A fit result dict (``"chains"`` + ``"summary"``) as an RV sampler returns it.
    t_grid : np.ndarray
        Times (MJD, days) at which to evaluate the model.

    Returns
    -------
    model : dict
        ``"t"`` — the input time grid.
        ``"rv_model"`` — model RV curve (km/s).
        ``"params"`` — dict of the orbital parameters used.
    """
    # ── ω convention sentinel guard ───────────────────────────────────
    # ``_keplerian_rv`` requires primary-frame ω (binary-star textbook
    # convention).  Reject chains that lack the sentinel or carry a
    # non-primary convention; a silent π offset on ω would phase-flip
    # the RV model.  Mirrors the guard in
    # :func:`orblet.rv_chain.astrometric_priors_from_rv_chain`.
    from orblet.constants import OMEGA_CONVENTION_PRIMARY
    chains = result.get("chains", {}) if isinstance(result, dict) else {}
    omega_conv = (
        chains.get("_meta", {}).get("omega_convention")
        if isinstance(chains, dict) else None
    )
    if omega_conv != OMEGA_CONVENTION_PRIMARY:
        raise ValueError(
            f"compute_rv_model_curve requires primary-frame ω; got "
            f"omega_convention={omega_conv!r}.  If the chain's ω is the "
            f"primary's, add ``_meta['omega_convention'] = "
            f"'primary_argument_of_periastron'`` to the chain dict; "
            f"otherwise convert it first (reference manual, §1.4)."
        )

    s = result["summary"]

    period_days = s["P_days"]["median"]
    ecc = s["e"]["median"]
    omega = s["omega_rad"]["median"]
    tp = s["tp_mjd"]["median"]
    gamma = s.get("gamma_kms", {}).get("median", 0.0)

    # Use pre-computed K from the chains if available (derived in the
    # bridge).  Fall back to computing from mass/period/ecc/M_total,
    # noting that b_mass in the chains is in M_sun (converted by bridge),
    # and M is the system-level total mass (M_star + M_comp).
    if "K_kms" in s:
        K = s["K_kms"]["median"]
    else:
        from orblet.priors import MSUN_IN_MJUP
        mass_mjup = s["m2_msun"]["median"] * MSUN_IN_MJUP
        # ``_stellar_K_from_elements`` takes Keplerian YEARS; the chain
        # key is in days, so divide here.
        K = _stellar_K_from_elements(
            mass_mjup,
            s["P_days"]["median"] / DAYS_PER_KEPLER_YEAR,
            ecc,
            s["M_total_msun"]["median"],
        )

    rv_model = _keplerian_rv(t_grid, period_days, ecc, omega, K, gamma, tp)

    return {
        "t": t_grid,
        "rv_model": rv_model,
        "params": {
            "period_days": period_days,
            "ecc": ecc,
            "omega": omega,
            "K_kms": K,
            "gamma_kms": gamma,
            "tp_mjd": tp,
        },
    }


def compute_residuals(
    prepared_data: dict,
    result: dict,
    *,
    include_jitter: bool = True,
) -> dict:
    """
    Compute observed-minus-computed (O−C) residuals at the data epochs.

    Parameters
    ----------
    prepared_data : dict
        Output of :func:`~orblet.prepare.prepare_rv_for_orbit`.
    result : dict
        A fit result dict (``"chains"`` + ``"summary"``) as an RV sampler returns it.
    include_jitter : bool, default True
        If True, add the fitted jitter in quadrature to ``rv_err``
        when computing reduced χ².  Set to False to evaluate
        goodness-of-fit against measurement errors alone.

    Returns
    -------
    residuals : dict
        ``"epochs_mjd"`` — observation times.
        ``"rv_obs"``     — observed RV (km/s).
        ``"rv_model"``   — model RV at observed epochs (km/s).
        ``"rv_err"``     — RV uncertainties (km/s).
        ``"rv_err_eff"`` — effective errors used for χ² (km/s).
        ``"residuals"``  — O−C (km/s).
        ``"rms"``        — RMS of residuals (km/s).
        ``"chi2_red"``   — reduced χ² (residuals / effective errors).
    """
    # ── ω convention sentinel guard ───────────────────────────────────
    # ``_keplerian_rv`` requires primary-frame ω.  Surface the mismatch
    # here (in addition to inside ``compute_rv_model_curve``) so the
    # error message names this function — easier to debug from a
    # ``compute_residuals`` call site.
    from orblet.constants import OMEGA_CONVENTION_PRIMARY
    chains = result.get("chains", {}) if isinstance(result, dict) else {}
    omega_conv = (
        chains.get("_meta", {}).get("omega_convention")
        if isinstance(chains, dict) else None
    )
    if omega_conv != OMEGA_CONVENTION_PRIMARY:
        raise ValueError(
            f"compute_residuals requires primary-frame ω; got "
            f"omega_convention={omega_conv!r}.  If the chain's ω is the "
            f"primary's, add ``_meta['omega_convention'] = "
            f"'primary_argument_of_periastron'`` to the chain dict; "
            f"otherwise convert it first (reference manual, §1.4)."
        )

    t_obs = prepared_data["epochs_mjd"]
    rv_obs = prepared_data["rv"]
    rv_err = prepared_data["rv_err"]

    model_out = compute_rv_model_curve(result, t_obs)
    rv_model = model_out["rv_model"]

    oc = rv_obs - rv_model
    rms = float(np.sqrt(np.mean(oc**2)))

    # Effective error bars for chi-squared.
    if include_jitter:
        jitter_kms = result["summary"].get(
            "rv_jitter_kms", {}
        ).get("median", 0.0)
        rv_err_eff = np.sqrt(rv_err**2 + jitter_kms**2)
    else:
        rv_err_eff = rv_err

    # Reduced chi-squared.
    # Approximate degrees of freedom: default model has 8 free parameters
    # (P, e, ω, τ, mass, M, offset, jitter).  This count is approximate
    # and does not adjust for custom priors.
    n_data = len(rv_obs)
    n_params = 8
    dof = max(n_data - n_params, 1)
    chi2_red = float(np.sum((oc / rv_err_eff) ** 2) / dof)

    return {
        "epochs_mjd": t_obs,
        "rv_obs": rv_obs,
        "rv_model": rv_model,
        "rv_err": rv_err,
        "rv_err_eff": rv_err_eff,
        "residuals": oc,
        "rms": rms,
        "chi2_red": chi2_red,
    }


# ── Thiele-Innes → Keplerian element inversion ───────────────────────────────

def ti_to_kepler(
    chains: dict,
    *,
    plx_key: str = "plx_mas",
    M_key: str = "M_total_msun",
) -> dict:
    """
    Return a new chains dict augmented with Keplerian elements derived
    from a Thiele-Innes fit's posterior samples.

    Input must contain ``A_mas, B_mas, F_mas, G_mas`` (mas).  Output adds:
    ``a_phot_mas`` (mas, photocenter), ``a_phot_au`` (AU, photocenter),
    ``inc_rad`` (rad), ``omega_rad`` (rad), ``Omega_rad`` (rad) — and,
    ONLY when the input carries the mass keys (``m2_msun`` and the
    ``M_key`` total mass), the mass-scaled ``a_rel_au`` (AU,
    total-relative) plus a REWRITTEN ``P_days`` (days, Kepler III on
    that ``a_rel_au``).

    The mass-key guard is LOAD-BEARING: engine TI chains carry no mass
    latents, so on them this function emits the geometry keys only and
    can neither build the prior-contaminated ``a_rel_au`` nor overwrite
    the sampled ``P_days`` (see
    ``docs/model_and_likelihoods.md`` §3.8).  On legacy /
    Campbell-shaped inputs the derived keys are written
    unconditionally over any pre-existing values.  Other input keys
    (``e``, ``plx_mas``, observation-block columns, sampler
    diagnostics, ``_meta``) are preserved unchanged.

    Parameters
    ----------
    chains : dict of str → numpy array
        Posterior samples, as in ``fit_astrometry_orbit()["chains"]``.
        Must contain Thiele-Innes amplitudes ``A_mas, B_mas, F_mas, G_mas``.
        For ``a_phot_au`` computation, also expects the ``plx_key``
        parallax; for ``a_rel_au`` (total-relative) and ``P_days``,
        expects ``m2_msun`` and the ``M_key`` total mass (both M☉).
    plx_key, M_key : str
        Chain keys for parallax (mas) and total mass (M☉).

    Returns
    -------
    augmented : dict
        Shallow copy of ``chains`` with the derived Keplerian keys
        appended/overwritten.  Original arrays for keys not listed
        above are referenced, not copied, so this is cheap.

    Notes
    -----
    Inversion follows the standard Thiele-Innes identities (see e.g.
    Hilditch 2001, §5.12):

        u = (A² + B² + F² + G²) / 2
        v = A G − B F
        α = √(u + √((u+v)(u−v)))        # mas  (photocenter semi-major axis)
        a_phot = α / plx                 # AU   (photocenter)
        a = a_phot × M / m_comp          # AU   (total-relative)
        P = √(a³ / M_tot) × DAYS_PER_KEPLER_YEAR   # days (Kepler III
                                                   # gives years)

        ω + Ω = atan2(B − F, A + G)
        ω − Ω = atan2(−(B + F), A − G)
        ω = [(ω+Ω) + (ω−Ω)] / 2
        Ω = [(ω+Ω) − (ω−Ω)] / 2

        tan⁴(i/2) = [(A−G)² + (B+F)²] / [(A+G)² + (B−F)²]
        i = 2 · arctan((num / den)^(1/4))

    Inclination is returned in ``[0, π]``.  Ω is returned wrapped into
    ``[−π, π]`` (from arctan2); downstream code that compares across
    samples should handle the ±π/±2π ambiguity of (ω, Ω) explicitly
    (there are typically two degenerate modes 180° apart).

    Semi-major-axis convention
    --------------------------
    The sampled (A, B, F, G) amplitudes are PHOTOCENTER amplitudes
    (mas), so ``α = √(u + √((u+v)(u−v)))`` is the photocenter semi-major
    axis ``a_phot × plx``.  Two derived axes are exposed:

    - ``a_phot_au`` = ``α / plx`` (AU): direct from inversion.  The
      photocenter sky amplitude is ``a_phot_au × plx`` (mas).
    - ``a_rel_au`` = ``a_phot_au × M / m2`` (AU): total-relative axis,
      matching the Campbell-basis Kepler-III ``(P² × M)^{1/3}`` —
      emitted only when the input chain carries ``m2_msun`` and the
      ``M_key`` total mass (engine TI chains do not).

    The chains-dict ``_meta["a_convention"]`` sentinel records the
    convention (``A_CONVENTION_TOTAL_RELATIVE`` for the default emit
    path); the identity ``a_rel_au × m2 = a_phot_au × M`` holds to
    ``rtol=1e-12``.

    Raises
    ------
    KeyError
        If ``A_mas``, ``B_mas``, ``F_mas``, or ``G_mas`` are missing.
        This usually means the fit was run with ``basis="kepler"``,
        in which case no conversion is needed.

    Examples
    --------
    >>> result = fit_astrometry_orbit(..., basis="thiele_innes")
    >>> chains_kep = ti_to_kepler(result["chains"])
    >>> fig = plot_orbit_corner({"chains": chains_kep},
    ...                         params=["P_days", "e", "inc_rad", "m2_msun"])
    """
    required = ("A_mas", "B_mas", "F_mas", "G_mas")
    missing = [k for k in required if k not in chains]
    if missing:
        raise KeyError(
            f"Thiele-Innes amplitudes missing from chains: {missing}. "
            f"ti_to_kepler expects output of a basis='thiele_innes' fit."
        )

    # Shallow copy — we only add/overwrite the derived keys; we never
    # mutate the original arrays held by the caller.
    out = dict(chains)

    # Local aliases; name F_ti to avoid shadowing matplotlib Figure etc.
    A = out["A_mas"]
    B = out["B_mas"]
    F_ti = out["F_mas"]
    G = out["G_mas"]

    # ── (α, a_phot[, a, P]) from the T-I amplitudes ──────────────────
    # The sampled (A, B, F, G) are PHOTOCENTER amplitudes (mas), so the
    # inversion yields α = a_phot × plx directly.  ``a_phot_au`` is the
    # photocenter axis (AU).  The mass-scaled ``a_rel_au`` (and the
    # ``P_days`` rewrite) are emitted ONLY when the input carries the
    # ``m2_msun`` / ``M_key`` mass keys — that guard is LOAD-BEARING:
    # TI chains report observables and carry no mass keys, so their
    # sampled ``P_days`` can never be overwritten here.  Inputs that do
    # carry mass keys get ``a_rel_au`` and the Kepler-III period.  See
    # the docstring "Semi-major-axis convention"
    # section above.
    u = (A ** 2 + B ** 2 + F_ti ** 2 + G ** 2) / 2.0
    v = A * G - B * F_ti
    # Guard: (u+v)(u−v) = u² − v² must be ≥ 0 for physical samples.
    # Clip tiny negative values arising from float noise.
    disc = np.maximum((u + v) * (u - v), 0.0)
    alpha_mas = np.sqrt(u + np.sqrt(disc))
    out["a_phot_mas"] = alpha_mas

    if plx_key in out:
        a_phot_AU = alpha_mas / out[plx_key]
        out["a_phot_au"] = a_phot_AU
        if "m2_msun" in out and M_key in out:
            # ``a_rel_au`` (total-relative) = ``a_phot_au × M / m2``.
            # Identity ``a_rel_au × m2 = a_phot_au × M`` holds to
            # rtol=1e-12 (pinned by a regression test).
            a_AU = a_phot_AU * out[M_key] / out["m2_msun"]
            out["a_rel_au"] = a_AU
            # Kepler III with M in M☉ and a in AU gives the period in
            # Keplerian YEARS; ``P_days`` is exported in days.
            out["P_days"] = (
                np.sqrt(a_AU ** 3 / out[M_key]) * DAYS_PER_KEPLER_YEAR
            )

    # ── (ω + Ω) and (ω − Ω) ───────────────────────────────────────────
    omega_plus_Omega = np.arctan2(B - F_ti, A + G)
    omega_minus_Omega = np.arctan2(-(B + F_ti), A - G)
    omega = 0.5 * (omega_plus_Omega + omega_minus_Omega)
    Omega = 0.5 * (omega_plus_Omega - omega_minus_Omega)

    # ── Inclination from the ratio of (A±G, B∓F) vector magnitudes ───
    # Derivation:
    #     (A+G)² + (B−F)² = a_phot² (1 + cos i)²
    #     (A−G)² + (B+F)² = a_phot² (1 − cos i)²
    # Their ratio is tan⁴(i/2), so i = 2 · arctan((num/den)^(1/4)).
    num = (A - G) ** 2 + (B + F_ti) ** 2
    den = (A + G) ** 2 + (B - F_ti) ** 2
    tan_half_i = np.power(num / den, 0.25)
    inc = 2.0 * np.arctan(tan_half_i)

    out["omega_rad"] = omega
    out["Omega_rad"] = Omega
    out["inc_rad"] = inc

    return out


# ── T-I → Gaia DR3 NSS convention ────────────────────────────────────────────

# Provenance sentinels recorded in ``chains["_meta"]`` so a downstream
# reader can tell WHICH transform produced the ``*_nss`` amplitudes.
# Identity-compared by tests, hence module-level constants.
_NSS_SOURCE_PHOTOCENTRE_IDENTITY = "photocentre_identity"
_NSS_SOURCE_RELATIVE_SCALED = "relative_scaled"


def to_nss_convention(
    chains: dict,
    *,
    mass_key: str = "m2_msun",
    M_key: str = "M_total_msun",
    include_keplerian: bool = True,
    sign_flip: bool = True,
) -> dict:
    """
    Return a chains dict augmented with Gaia DR3 NSS-convention quantities.

    Gaia DR3 NSS (``nss_two_body_orbit``) reports **photocentre**
    Thiele-Innes amplitudes (a_phot × plx, mas).  Our engine samples
    photocentre amplitudes DIRECTLY, so on a current chain the
    amplitudes need NO conversion: they are passed through unchanged
    (``_meta["nss_amplitude_source"] ==
    ``photocentre_identity``).  Only the angle/time conventions differ,
    and those are handled below.

    Two branches, chosen by whether the chain carries the legacy mass
    keys:

    - **Photocentre (current engines).** Identity on (A, B, F, G).
    - **Legacy relative-orbit input.** If ``mass_key``/``M_key`` are
      present the historical scaling ``±M_comp/M_tot`` is applied, for
      chains that genuinely hold RELATIVE-orbit amplitudes.  Kept
      byte-identical for archived comparisons; ``_meta`` records
      ``nss_amplitude_source == "relative_scaled"`` and the
      ``nss_sign_flip`` actually used.  (Campbell-basis chains can never
      reach this function at all — they carry no ``A_mas..G_mas`` and the
      guard below raises first.)

    Sign position (do not re-open casually).  Three
    separate questions are easy to conflate here, and NONE of them
    leaves the identity branch ambiguous:

    1. the relative → photocentre ``−M_comp/M_tot`` factor — MOOT for
       current chains, since nothing needs converting;
    2. the global ``(A,B,F,G) → (−A,−B,−F,−G)`` branch — a DEGENERACY,
       not an unknown: it is the ``(ω+π, Ω+π)`` twin, i.e. the same
       physical orbit, and both we and NSS collapse it by folding Ω into
       ``[0, π)`` (below), so it cancels;
    3. the DR4 along-scan / scan-angle sign — SETTLED, locked by
       ``tests/test_bh3_prerelease_convention_lock.py`` against ESA's
       published Gaia BH3 orbit (Panuzzo+ 2024) at the 0.5σ level; a
       flip would in any case land in branch 2 and fold away.

    Output keys
    -----------
    ``A_nss_mas, B_nss_mas, F_nss_mas, G_nss_mas`` (mas, photocenter).
    If ``include_keplerian=True``:
        ``tp_nss_days`` — periastron **OFFSET IN DAYS from the catalogue
            reference epoch**, folded into ``[−P/2, +P/2]``.  NOT a date:
            NSS defines ``t_periastron`` as "epoch at periastron …
            relative to ``gaia_source.ref_epoch``" (see
            the Gaia DR4 datamodel).  An absolute
            Julian Date would be wrong on the zero point and unbounded
            in range.

            **Where the epoch comes from.** It is INHERITED from the chain
            — ``chains["_meta"]["epoch_ref_mjd"]``, the catalogue epoch the
            fit itself was referenced to (required by every astrometric and
            joint fit; ``require_epoch_ref_mjd`` raises on ``None``).  It
            is deliberately NOT a parameter of this function: re-supplying
            a value the chain already carries adds no check, only a second
            place for two epochs to disagree.

            **The pitfall.** An RV-only chain carries ``epoch_ref``, which
            is the MEDIAN OBSERVATION EPOCH — a gauge for ω, chosen for
            convenience, with no catalogue meaning.  Converting against it
            yields a number that looks like an NSS periastron time and is
            not one.  (The ACTIVE guard is the ``A_mas..G_mas``
            amplitude requirement, which rejects RV chains before any
            epoch check.)  Use ``include_keplerian=False``
            for the amplitude-only conversion, which needs no epoch.
        ``Omega_nss_rad`` — Ω folded into ``[0, π)`` to match NSS wrap.
        ``omega_nss_rad`` — ω shifted by π whenever Ω was, so the
            ``(ω, Ω)`` pair remains physically consistent.  Wrapped
            into ``[−π, π]``.
    Original chain keys are preserved.

    Why ω shifts with Ω
    -------------------
    The identity ``(ω, Ω) ↔ (ω+π, Ω+π)`` produces the *same*
    physical orbit — (A, B, F, G), e, i, τ, tp are all invariant under
    it.  Folding Ω into ``[0, π)`` effectively chooses one
    representative from each equivalence class; ω must shift in lockstep
    whenever Ω did, otherwise the resulting ``(b_ω, b_Ω_nss)`` pair
    refers to a *different* orbit than the sample produced.

    τ and tp are *not* shifted: the periastron time is invariant under
    ``(ω+π, Ω+π)``.  (τ *would* shift by 0.5 under the separate
    T-I sign-flip degeneracy ``(A,B,F,G,τ) ↔ (−A,−B,−F,−G, τ+0.5)``,
    but NSS doesn't break that one by convention — use a fold
    post-processor if you need a unimodal posterior.)

    Parameters
    ----------
    chains : dict of str → ndarray
        Posterior samples from ``fit_astrometry_orbit(basis='thiele_innes')``.
        Must contain ``A_mas, B_mas, F_mas, G_mas``.  ``mass_key``/``M_key`` are
        OPTIONAL and select the legacy branch when both are present.
        For ``include_keplerian=True`` must also contain ``tp_mjd`` and
        ``Omega_rad`` (run :func:`ti_to_kepler` first if the latter is absent).
    mass_key, M_key : str
        Chain keys for companion mass and total mass (both in M☉), used
        ONLY by the legacy relative-orbit branch.  Absent on every
        observables-first chain, which carries no mass columns.
    include_keplerian : bool, default True
        If True, also convert ``tp_mjd`` (MJD → JD) and ``Omega_rad`` (fold
        into [0, π)).  If False, only the (A, B, F, G) amplitudes are
        converted.
    sign_flip : bool, default True
        LEGACY BRANCH ONLY — ignored on the identity path.  Whether to
        apply the ``−1`` factor when scaling relative-orbit amplitudes
        to photocentre ones (the primary moves opposite to the companion
        in the barycentric frame).  The value used is recorded in
        ``_meta["nss_sign_flip"]``, since ``"relative_scaled"`` alone
        does not say which sign produced the numbers.

    Returns
    -------
    augmented : dict
        Shallow copy of ``chains`` with NSS keys appended.

    Notes
    -----
    **Light ratio (luminous secondaries).**  For a dark companion (BH,
    NS, unlit WD) the scaling factor is ``M_comp / M_tot``.  For a
    luminous secondary with light ratio β = L_comp / L_pri, the correct
    photocenter factor is ``(M_comp/M_tot − β/(1+β))``.  This function
    implements only the dark-companion case — extend if you need SB2
    or mixed-light solutions.

    **Verification.**  The identity branch is validated end-to-end by
    ``tests/test_bh3_prerelease_convention_lock.py``, which reproduces
    ESA's published Gaia BH3 orbit from public DR4 pre-release epoch
    astrometry within 0.5σ (including a_phot and both angles).  When
    comparing a NEW target against a published NSS ``Orbital`` solution,
    expect the amplitudes to match up to the global sign branch of
    point 2 above; fix one shared sign across all four before comparing,
    exactly as that test does.

    **Time origin.**  ``t_periastron`` is reported by **every** NSS table
    as days relative to ``gaia_source.ref_epoch``, wrapped into
    ``[−period/2, +period/2]`` (the Gaia DR4 datamodel's table
    descriptions).  This function emits exactly
    that, inheriting the epoch from ``_meta["epoch_ref_mjd"]``.
    No NSS table reports a JD, and a JD is not a more general form but a
    different quantity.

    Examples
    --------
    >>> result = fit_astrometry_orbit(..., basis='thiele_innes')
    >>> chains_kep = ti_to_kepler(result['chains'])
    >>> chains_nss = to_nss_convention(chains_kep)
    >>> plot_orbit_corner(
    ...     {'chains': chains_nss, 'summary': result['summary']},
    ...     params=['A_nss_mas', 'B_nss_mas', 'F_nss_mas', 'G_nss_mas',
    ...             'e', 'm2_msun'],
    ... )
    """
    required = ("A_mas", "B_mas", "F_mas", "G_mas")
    missing = [k for k in required if k not in chains]
    if missing:
        raise KeyError(
            f"Thiele-Innes amplitudes missing from chains: {missing}. "
            f"to_nss_convention expects output of a basis='thiele_innes' fit."
        )
    out = dict(chains)

    # Which branch?  The legacy scaling applies ONLY to chains that
    # actually carry relative-orbit amplitudes, and those are exactly the
    # ones that still ship the mass columns.  Observables-first chains
    # (TI chains) sample the photocentre orbit directly, so their
    # amplitudes are already in the NSS convention.
    is_legacy = mass_key in chains and M_key in chains

    if is_legacy:
        # Relative → photocentre, dark-companion case (β = 0):
        # a_phot / a_rel = M_comp / M_tot.  The primary moves opposite to
        # the companion in the barycentric frame, hence the sign flip.
        mass_ratio = chains[mass_key] / chains[M_key]
        factor = -mass_ratio if sign_flip else mass_ratio
        provenance = _NSS_SOURCE_RELATIVE_SCALED
    else:
        # Identity: no scaling and NO sign claim.  See the docstring's
        # three-point sign discussion — nothing here is ambiguous.
        factor = 1.0
        provenance = _NSS_SOURCE_PHOTOCENTRE_IDENTITY

    out["A_nss_mas"] = factor * chains["A_mas"]
    out["B_nss_mas"] = factor * chains["B_mas"]
    out["F_nss_mas"] = factor * chains["F_mas"]
    out["G_nss_mas"] = factor * chains["G_mas"]

    # Provenance goes in ``_meta`` (the established home for non-array
    # sentinels), COPIED not mutated: ``out = dict(chains)`` is shallow,
    # so writing into the existing dict would edit the caller's chain.
    meta = dict(chains.get("_meta") or {})
    meta["nss_amplitude_source"] = provenance
    if is_legacy:
        meta["nss_sign_flip"] = bool(sign_flip)
    out["_meta"] = meta

    if include_keplerian:
        if "tp_mjd" in chains:
            # NSS `t_periastron` is an OFFSET, not a date: "epoch at
            # periastron … relative to `gaia_source.ref_epoch`, in the
            # range [−period/2, +period/2]" (DR4 table descriptions, see
            # the Gaia DR4 datamodel).  So it is NOT an absolute Julian
            # Date such as `Time(b_tp).jd`, which would be wrong on the
            # zero point AND unbounded in range.
            #
            # WHERE THE EPOCH COMES FROM (and why the caller is not asked):
            # every astrometric/joint fit already REQUIRES `epoch_ref_mjd`
            # — the catalogue reference epoch — and records it in `_meta`.
            # Reading it there ties the export to the same epoch the fit
            # used.  Asking the caller to supply it again would add no
            # check, only a second place for two epochs to disagree
            # silently.  RV-only chains carry
            # `epoch_ref` (the MEDIAN observation epoch, a gauge for ω) and
            # no catalogue epoch, so they are refused rather than silently
            # given a number that looks like an NSS time and is not one.
            ref_epoch_mjd = (chains.get("_meta") or {}).get("epoch_ref_mjd")
            if ref_epoch_mjd is None:
                raise ValueError(_NSS_NO_CATALOGUE_EPOCH_MSG)
            tp_offset = (
                np.asarray(chains["tp_mjd"], dtype=float) - float(ref_epoch_mjd)
            )
            if "P_days" in chains:
                # Fold into [−P/2, +P/2). Periastron is only defined
                # modulo the period, so this is a relabelling, not a
                # change of orbit.
                P_days = np.asarray(chains["P_days"], dtype=float)
                tp_offset = (
                    np.mod(tp_offset + P_days / 2.0, P_days) - P_days / 2.0
                )
            out["tp_nss_days"] = tp_offset
        if "Omega_rad" in chains:
            # NSS restricts Ω to [0, π) to break the visual-orbit
            # (ω, Ω) ↔ (ω+π, Ω+π) degeneracy.  We implement this
            # by first wrapping to [0, 2π), then folding the second
            # half into [0, π) with a compensating π shift on ω.
            Omega = chains["Omega_rad"]
            Omega_2pi = np.mod(Omega, 2.0 * np.pi)           # → [0, 2π)
            needs_shift = Omega_2pi >= np.pi
            out["Omega_nss_rad"] = np.where(
                needs_shift, Omega_2pi - np.pi, Omega_2pi,
            )
            if "omega_rad" in chains:
                omega = chains["omega_rad"]
                omega_shifted = np.where(
                    needs_shift, omega - np.pi, omega,
                )
                # Wrap ω into a canonical [−π, π] range for display.
                out["omega_nss_rad"] = (
                    (omega_shifted + np.pi) % (2.0 * np.pi) - np.pi
                )

    return out


def mirror_inclination(seed: dict) -> dict:
    """The inclination-mirror image of a level-4 seed: ``i -> pi - i``.

    An EXPLORATORY RESTART, not an equal posterior image: with
    ``omega`` and ``Omega`` held fixed the flip leaves the Thiele-Innes
    constants invariant only at ``i = 90 deg`` — the exact astrometric
    degeneracy is ``(omega, Omega) -> (omega + pi, Omega + pi)``, which
    the RV channel and the gauge fold already remove.  Documented
    usage: ``warmstart_theta=[seed, mirror_inclination(seed)]`` with
    ``n_chains=2`` — the same hemisphere insurance level 3 applies at
    walker level.  Masses are unchanged (``sin i`` is equal for ``i``
    and ``pi - i``).
    """
    out = dict(seed)
    out["inc_rad"] = float(np.pi) - float(seed["inc_rad"])
    return out
