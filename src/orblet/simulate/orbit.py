"""
:class:`OrbitSimulator` — synthetic SB1 orbit + system parameters that
emit data in *exactly* the dict shapes a Gaia epoch-data loader returns.

Conventions (matched to the rest of the package)
------------------------------------------------
- Period in days (user-facing); the simulator works in days everywhere.
- Masses in solar masses.  Total mass M = m1 + m2 follows the
  ``Visual{KepOrbit}`` convention (system-level total).
- Photocenter under β = 0 (dark companion):  a_phot = a1 = a × m2/M.
- Times in days from J2010.0 TCB (matches
  the loader's ``obs_time`` field).
- ``tp`` is given in MJD; the J2010-day → MJD offset is +55197.0.
- Angles in radians (ω, Ω, i).
- All randomness routes through ``numpy.random.default_rng(seed)``;
  the global ``np.random`` state is never touched.

Forward-model fidelity
----------------------
- RV uses the standard SB1 line-of-sight expression
  ``v(t) = γ + K1 [cos(ν+ω) + e cos ω]`` with ν the true anomaly.
- Astrometry uses :func:`orblet.kepler.thiele_innes_al`
  (which delegates to :func:`orblet.kepler.campbell_xy`) —
  the *same* Thiele-Innes routine the postprocessing path uses, so a
  bug in the projection would surface in both forward and inverse
  models simultaneously rather than masking itself.
- Optional 5-parameter astrometric model (offset + PM + parallax) is
  added on top of the orbit, so the fitted ``astro_data`` looks
  exactly like a real Gaia source's centroid_pos_al.  The simulator
  doesn't subtract anything; that's the fitter's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from orblet.constants import (
    AU_M,
    DAYS_PER_KEPLER_YEAR,
    MJD_J2010_TCB,
    MJD_J2016_TCB,
)
from orblet.kepler import solve_kepler, thiele_innes_al


# Days from J2010.0 TCB → MJD (J2010.0 TCB = JD 2455197.5 → MJD 55197.0).
# Local alias kept so existing internal usages (``_J2010_TO_MJD``) don't
# need to be renamed; both names refer to the same float.
_J2010_TO_MJD = MJD_J2010_TCB
# AU per (km/s · day) = 86 400 / (AU in km).
_AU_PER_KMS_DAY = 86400.0 / (AU_M / 1000.0)


# ── The simulator ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrbitSimulator:
    """
    Frozen orbital + system truth.  Methods produce dict-shaped data
    matched to the loader return values, so the existing fitters can
    consume synthetic data with no modifications.

    Data-safety note
    ----------------
    ``OrbitSimulator`` is intended for **synthetic truths**.  The
    auto-generated ``__repr__`` redacts ``source_id`` (via
    ``repr=False``), but the astrometric fields (``ra_deg``, ``dec_deg``,
    ``parallax_mas``, ``pmra_masyr``, ``pmdec_masyr``) identify a real
    source when set to real values, and its ``repr`` shows them.  If
    synthetic data built from it are to be shared, give it invented sky
    parameters; seeded from a real fit, its output carries that fit's
    numbers.  Use this class for synthetic
    truths: construct it directly, or start from the one preset,
    :meth:`toy_orbit`, whose sky parameters are invented.

    Examples
    --------
    >>> from orblet.simulate.cadence import DemoParallaxConsistentCadence
    >>> sim   = OrbitSimulator.toy_orbit()
    >>> cad   = DemoParallaxConsistentCadence(ra_deg=sim.ra_deg, dec_deg=sim.dec_deg, seed=0)
    >>> rv    = sim.simulate_rv(cad, seed=0)
    >>> astro = sim.simulate_astrometry(cad, seed=0)
    """

    # Orbit (Visual{KepOrbit} parameterisation).
    P_days: float
    e: float
    omega_rad: float
    Omega_rad: float
    i_rad: float
    tp_mjd: float

    # Masses (M☉).  m1 is the spectroscopic primary, m2 the (dark) companion.
    m1_msun: float
    m2_msun: float

    # System geometry.
    ra_deg: float
    dec_deg: float
    parallax_mas: float
    pmra_masyr: float = 0.0
    pmdec_masyr: float = 0.0

    # Systemic velocity, km/s.
    gamma_kms: float = 0.0

    # Noise scales.  These override the cadence's per-transit defaults
    # — set both to zero for noise-free closure debugging.
    rv_sigma_kms: float = 1.0
    centroid_sigma_mas: float = 0.5
    rv_jitter_kms: float = 0.0
    astro_jitter_mas: float = 0.0

    # Photocenter convention.  β = 0 ⇒ dark companion ⇒ a_phot = a1.
    # β > 0 not yet supported by the downstream postprocess (TODO upstream).
    light_ratio_beta: float = 0.0

    # Reference epoch for the local-plane astrometric origin (deg).
    # When None, the simulator uses (ra_deg, dec_deg) as the origin
    # (no offset term in the centroid_pos signal).
    ra0_deg: float | None = None
    dec0_deg: float | None = None

    # Reference epoch (MJD) for proper-motion subtraction.  J2016.0 TCB
    # (MJD 57388.5, :data:`orblet.constants.MJD_J2016_TCB`).
    t_ref_mjd: float = MJD_J2016_TCB

    # Optional metadata, copied through to outputs.
    source_id: int = field(default=-1, repr=False)  # display-only redaction; asdict/vars/attribute/pickle unchanged

    # ── Validation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not (0.0 <= self.e < 1.0):
            raise ValueError(f"e must lie in [0,1); got {self.e}")
        if self.P_days <= 0.0:
            raise ValueError(f"P_days must be > 0; got {self.P_days}")
        if self.m1_msun <= 0.0 or self.m2_msun <= 0.0:
            raise ValueError("m1, m2 must be > 0.")
        if self.parallax_mas <= 0.0:
            raise ValueError(
                f"parallax must be > 0 (positive distance); got {self.parallax_mas}"
            )
        if self.light_ratio_beta != 0.0:
            raise NotImplementedError(
                "light_ratio_beta != 0 (luminous secondary) is not "
                "wired through to_nss_convention yet — the closure "
                "test would compare incompatible photocenters.  "
                "Leave β = 0 until the upstream code supports it."
            )

    # ── Derived quantities ────────────────────────────────────────────────

    @property
    def M_total_msun(self) -> float:
        return self.m1_msun + self.m2_msun

    @property
    def P_yr(self) -> float:
        return self.P_days / DAYS_PER_KEPLER_YEAR

    @property
    def a_au(self) -> float:
        """
        Total semi-major axis in AU.

        Kepler III in solar units:  a³ [AU] = P² [yr] × M_total [M☉].
        """
        return float((self.P_yr ** 2 * self.M_total_msun) ** (1.0 / 3.0))

    @property
    def a1_au(self) -> float:
        """Primary semi-major axis around the system barycenter (AU)."""
        return self.a_au * (self.m2_msun / self.M_total_msun)

    @property
    def a_phot_au(self) -> float:
        """Photocenter semi-major axis (AU). Equals a1 under β = 0."""
        # β > 0 generalisation:  a_phot = a × |m2/M_total - β/(1+β)|.
        # Disabled in __post_init__ for now.
        return self.a1_au

    @property
    def a_phot_mas(self) -> float:
        return self.a_phot_au * self.parallax_mas

    @property
    def K1_kms(self) -> float:
        """RV semi-amplitude of the primary, in km/s."""
        # K1 = (2π/P) · a1·sin(i) / sqrt(1 − e²)
        a1_sini_au = self.a1_au * np.sin(self.i_rad)
        return float(
            (2.0 * np.pi / self.P_days)
            * a1_sini_au
            * (1.0 / _AU_PER_KMS_DAY)
            / np.sqrt(1.0 - self.e ** 2)
        )

    # ── Forward model (pure functions of t) ───────────────────────────────

    def primary_radial_velocity(self, t_mjd: np.ndarray) -> np.ndarray:
        """
        Line-of-sight velocity of the primary at MJD epochs ``t``,
        in km/s, *without* observational noise.
        """
        # Mean anomaly relative to periastron.
        M_anom = (2.0 * np.pi * (t_mjd - self.tp_mjd) / self.P_days) % (
            2.0 * np.pi
        )
        E = solve_kepler(M_anom, self.e)
        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + self.e) * np.sin(E / 2.0),
            np.sqrt(1.0 - self.e) * np.cos(E / 2.0),
        )
        return (
            self.gamma_kms
            + self.K1_kms
            * (np.cos(nu + self.omega_rad) + self.e * np.cos(self.omega_rad))
        )

    def along_scan_orbit(
        self, t_j2010_days: np.ndarray, scan_angle_rad: np.ndarray,
    ) -> np.ndarray:
        """
        Photocenter along-scan signal at given Gaia transits, in mas.

        Pure orbital contribution; no parallax / proper-motion / offset.
        """
        t_mjd = t_j2010_days + _J2010_TO_MJD
        return thiele_innes_al(
            t_mjd=t_mjd,
            psi=scan_angle_rad,
            period_yr=self.P_yr,
            ecc=self.e,
            omega_rad=self.omega_rad,
            inc_rad=self.i_rad,
            Omega_rad=self.Omega_rad,
            tp_mjd=self.tp_mjd,
            a_mas=self.a_phot_mas,
        )

    # ── Public API (loader-shaped dicts) ──────────────────────────────────

    def simulate_rv(self, cadence: Any, seed: int | None = None) -> dict:
        """
        Synthetic epoch RV dict in the loader shape.

        Adds white Gaussian noise with σ = ``rv_sigma_kms`` (or the
        cadence's per-transit ``rv_err`` if ``rv_sigma_kms`` is set
        to ``None`` via a future API change), plus optional jitter.
        Validity flags are all True.
        """
        rng = np.random.default_rng(seed)
        t = np.asarray(cadence.rv_obs_time, dtype=float)
        t_mjd = t + _J2010_TO_MJD
        rv_clean = self.primary_radial_velocity(t_mjd)

        # Combine measurement σ with optional intrinsic jitter.
        sigma = np.full_like(t, float(self.rv_sigma_kms))
        if self.rv_jitter_kms > 0.0:
            sigma = np.sqrt(sigma ** 2 + self.rv_jitter_kms ** 2)

        rv = rv_clean + rng.normal(0.0, sigma)

        # Match the loader's astropy-Quantity convention so the
        # downstream prepare_rv_for_orbit doesn't have to be told
        # "this is synthetic".
        import astropy.units as u

        return {
            "radial_velocity":     rv * (u.km / u.s),
            "radial_velocity_err": sigma * (u.km / u.s),
            "obs_time_rv":         t * u.d,
            "rv_validity_flag":    np.ones(t.shape, dtype=bool),
            "rv_invalid_reason":   np.full(t.shape, "", dtype=object),
            "transit_id_spec":     np.asarray(cadence.rv_transit_id),
            "scan_angle":          np.asarray(cadence.rv_scan_angle, dtype=float),
        }

    def simulate_astrometry(
        self,
        cadence: Any,
        seed: int | None = None,
        *,
        include_5param_model: bool = True,
    ) -> dict:
        """
        Synthetic epoch-astrometry dict in the loader shape.

        Parameters
        ----------
        include_5param_model : bool, default True
            If True, the centroid_pos signal carries the full Gaia
            content: *position offset* (Δα0*, Δδ0 from ra0/dec0 vs
            ra/dec) + *proper motion* + *parallax × parallax_factor_al*
            + *orbital signal* + Gaussian noise.  The synthetic data
            then exercises the fitter's joint (5-param + orbit) model
            exactly as a real source would.  If False, only the orbit
            + noise are written — convenient when the test wants to
            isolate the orbital fit.
        """
        rng = np.random.default_rng(seed)
        t = np.asarray(cadence.astro_obs_time, dtype=float)         # J2010 days
        psi = np.asarray(cadence.astro_scan_angle, dtype=float)
        f_pi = np.asarray(cadence.astro_parallax_factor_al, dtype=float)

        signal = self.along_scan_orbit(t, psi)

        if include_5param_model:
            # Position offset relative to (ra0, dec0).  When ra0/dec0
            # are not provided by the cadence, take them = (ra, dec)
            # so the offset term is zero and the test source is
            # exactly aligned with the local-plane origin.
            ra0  = cadence.ra0_deg  if cadence.ra0_deg  is not None else self.ra_deg
            dec0 = cadence.dec0_deg if cadence.dec0_deg is not None else self.dec_deg
            cos_d = np.cos(np.deg2rad(self.dec_deg))
            d_ra_mas  = (self.ra_deg  - ra0)  * cos_d * 3_600_000.0
            d_dec_mas = (self.dec_deg - dec0) *         3_600_000.0
            signal = signal + d_ra_mas * np.sin(psi) + d_dec_mas * np.cos(psi)

            # Proper motion.  Δt in years from t_ref_mjd.
            dt_yr = (t + _J2010_TO_MJD - self.t_ref_mjd) / DAYS_PER_KEPLER_YEAR
            signal = signal + (
                self.pmra_masyr * np.sin(psi) + self.pmdec_masyr * np.cos(psi)
            ) * dt_yr

            # Parallax × scan-direction parallax factor.
            signal = signal + self.parallax_mas * f_pi

        sigma = np.full_like(t, float(self.centroid_sigma_mas))
        if self.astro_jitter_mas > 0.0:
            sigma = np.sqrt(sigma ** 2 + self.astro_jitter_mas ** 2)

        centroid = signal + rng.normal(0.0, sigma)

        return {
            "transit_id":           np.asarray(cadence.astro_transit_id),
            "obs_time":             t,
            "scan_angle":           psi,
            "centroid_pos":         centroid,
            "centroid_pos_err":     sigma,
            "parallax_factor_al":   f_pi,
            "ra0_deg":              cadence.ra0_deg if cadence.ra0_deg is not None else self.ra_deg,
            "dec0_deg":             cadence.dec0_deg if cadence.dec0_deg is not None else self.dec_deg,
            # The simulator's reference epoch, in days-from-J2010 (matching obs_time's
            # scale).  A real loader sets this from the catalogue; here it flows from
            # the sim's own t_ref_mjd so the batch epoch matches the simulated truth.
            "ref_epoch_days":       float(self.t_ref_mjd) - MJD_J2010_TCB,
            "columns":              [],
        }

    # ── Canonical truths (factory @classmethods) ─────────────────────────

    @classmethod
    def toy_orbit(cls, **overrides: Any) -> "OrbitSimulator":
        """
        The one preset: a readable truth both channels detect strongly.

        P = 185 d, e = 0.45, a 9.6 M☉ dark companion to a 0.9 M☉ primary,
        so K₁ ≈ 66.9 km/s and a_phot ≈ 2.67 mas — well above the noise in
        RV and astrometry alike.  That is deliberate: a newcomer choosing
        parameters freely can land face-on or below the noise and conclude
        the code is broken.  The sky position, parallax and proper motion
        are invented round numbers (RA 180°, Dec −20°, ϖ 2.1 mas) and the
        ``source_id`` is the synthetic sentinel ``-1``, so the output
        identifies nothing.

        Pass kwargs to override any field; for anything else construct
        :class:`OrbitSimulator` directly.  Pair it with
        :class:`~orblet.simulate.cadence.DemoParallaxConsistentCadence`
        (as :func:`~orblet.simulate.bundles.load_simulated_inputs` does)
        for data a fit can close on.
        """
        defaults = dict(
            P_days=185.0,
            e=0.45,
            omega_rad=np.deg2rad(60.0),
            Omega_rad=np.deg2rad(110.0),
            i_rad=np.deg2rad(127.0),
            tp_mjd=55300.0,
            m1_msun=0.9,
            m2_msun=9.6,
            ra_deg=180.0,
            dec_deg=-20.0,
            parallax_mas=2.1,
            pmra_masyr=2.0,
            pmdec_masyr=-3.0,
            gamma_kms=0.0,
            rv_sigma_kms=3.0,
            centroid_sigma_mas=0.15,
            source_id=-1,
        )
        defaults.update(overrides)
        return cls(**defaults)
