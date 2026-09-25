"""
The demo cadence :class:`DemoParallaxConsistentCadence`.

This cadence is the default skeleton for the demo bundle
(:func:`orblet.simulate.bundles.load_simulated_inputs`).  Its parallax
factor follows from the source's sky position, the epoch and the scan
angle, as real ones do, rather than being drawn independently of the scan
angles (a geometry no satellite can produce).  The same factor array both
injects the parallax signal (in :meth:`OrbitSimulator.simulate_astrometry`)
and removes it (in the fit), so it is suitable for forward/inverse closure
checks.

Parallax factors from the ephemeris
-----------------------------------
The along-scan parallax factor is
:func:`orblet.parallax.along_scan_parallax_factor` at the cadence's epochs,
scan angles and sky position — with its defaults: the observer at Sun-Earth
L2, where Gaia is, and astropy's built-in ephemeris (no download, no extra
package).  So the demo's parallax wobble has the real Earth's timing and
Gaia's amplitude, to ~0.2 % of Gaia's own factor.

ψ schedule
----------
Deterministic rotating sequence ``ψ_k = (ψ0 + k·Δψ) mod π`` with the
golden-ratio step ``Δψ = π·(√5−1)/2``.  The golden step gives a
wide-span, incommensurate set of scan directions (so the 5-parameter
along-scan design is well conditioned) that never closes on a short
cycle with the 1-yr parallax period.

Time sampling (anti-alias)
--------------------------
The RV and astrometric observation TIMES are placed on an even grid over
the 5-yr baseline and then jittered by a seeded uniform draw of up to
±0.45 of the grid spacing (see :data:`_TIME_JITTER_FRAC`).  An evenly
spaced schedule folds into only a few discrete orbital phases at periods
commensurate with the grid step, leaving phase gaps that admit the 2×
period alias; the seeded jitter breaks that commensurability so folding
at ANY period in the tens-to-hundreds-of-days range gives dense,
near-uniform phase coverage.  The sampling is ORBIT-AGNOSTIC (it never
uses the period) and FULLY DETERMINISTIC given ``(ra_deg, dec_deg,
seed)``; ``seed`` selects the jitter realisation (``seed=None`` uses a
fixed default seed so the bundle stays reproducible).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from orblet.constants import DAYS_PER_KEPLER_YEAR, MJD_J2010_TCB
from orblet.parallax import along_scan_parallax_factor


__all__ = ["DemoParallaxConsistentCadence"]


# ── Module constants (documented design choices) ─────────────────────

# Sampling.
_N_ASTRO_EPOCHS = 48                         # ≥ 40 astro transits
_N_RV_EPOCHS = 36                            # ≥ 12 RV epochs (anti-alias)
_BASELINE_DAYS = 5.0 * DAYS_PER_KEPLER_YEAR  # 5 Julian years (J2010-days)
_T_START_J2010_DAYS = 0.0

# Quasi-random time-sampling jitter (fraction of the even grid spacing).
# Both the RV and astro epochs are placed on an even grid and then jittered
# by ±_TIME_JITTER_FRAC × (grid spacing) using a seeded RNG.  An evenly
# spaced (linspace) schedule folds into a SMALL NUMBER of discrete phase
# groups at any period that is rationally commensurate with the grid step
# (e.g. the 16-epoch RV grid clustered into 4 groups at P = 185 d, leaving
# two phase bins empty and admitting the 2× period alias).  Jittering the
# times breaks that commensurability so that folding at ANY period in the
# tens-to-hundreds-of-days range gives dense, near-uniform phase coverage.
# The schedule is ORBIT-AGNOSTIC: it never uses the period.  A jitter of
# 0.45 keeps the times monotonically increasing (no reordering) while
# spreading each grid point across most of its cell.
_TIME_JITTER_FRAC = 0.45

# Per-transit noise template (overridden by OrbitSimulator's own σ).
_ASTRO_CENTROID_ERR_MAS = 0.15
_RV_ERR_KMS = 3.0

# ψ schedule: golden-ratio step (radians), and the starting angle.
_PSI_STEP_RAD = math.pi * (math.sqrt(5.0) - 1.0) / 2.0
_PSI0_RAD = 0.0

# Synthetic sentinel source_id (never a real Gaia source_id).
_SYNTHETIC_SOURCE_ID = -1


def _jittered_grid_times(n_epochs: int, *, seed: int) -> np.ndarray:
    """Quasi-random, monotonically increasing epoch times (J2010-days).

    Places ``n_epochs`` points on an even grid spanning the 5-yr baseline
    and adds a seeded uniform jitter of up to ±``_TIME_JITTER_FRAC`` of the
    grid spacing.  ORBIT-AGNOSTIC (the period is never used).  Endpoints are
    clipped to the baseline so all times stay in
    ``[_T_START_J2010_DAYS, _T_START_J2010_DAYS + _BASELINE_DAYS]``.

    Parameters
    ----------
    n_epochs : int
        Number of epochs.
    seed : int
        Seed for the local RNG (deterministic output per seed).

    Returns
    -------
    np.ndarray
        Sorted float64 epoch times in days from J2010.0 TCB, shape
        ``(n_epochs,)``.
    """
    grid = np.linspace(
        _T_START_J2010_DAYS,
        _T_START_J2010_DAYS + _BASELINE_DAYS,
        n_epochs,
    )
    spacing = _BASELINE_DAYS / (n_epochs - 1)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-_TIME_JITTER_FRAC, _TIME_JITTER_FRAC, size=n_epochs)
    t = grid + jitter * spacing
    # Clip to the baseline and re-sort: jitter < 0.5·spacing cannot reorder
    # interior points, but the clipped endpoints must stay ordered.
    t = np.clip(t, _T_START_J2010_DAYS, _T_START_J2010_DAYS + _BASELINE_DAYS)
    t.sort()
    return t.astype(np.float64)


@dataclass(frozen=True)
class DemoParallaxConsistentCadence:
    """
    Closure-grade demo cadence: epochs, scan angles and the parallax
    factors they imply for a sky position.

    Time arrays are in days from J2010.0 TCB.  Scan angles are in
    radians.  The parallax factor is
    :func:`orblet.parallax.along_scan_parallax_factor` (Gaia at L2,
    astropy's built-in ephemeris) for ``(ra_deg, dec_deg)``.
    """

    astro_obs_time: np.ndarray
    astro_scan_angle: np.ndarray
    astro_parallax_factor_al: np.ndarray
    astro_transit_id: np.ndarray
    astro_centroid_pos_err: np.ndarray
    ra0_deg: float | None
    dec0_deg: float | None
    rv_obs_time: np.ndarray
    rv_scan_angle: np.ndarray
    rv_transit_id: np.ndarray
    rv_err: np.ndarray
    source_id: int
    _ra_deg: float
    _dec_deg: float

    def __init__(self, *, ra_deg: float, dec_deg: float, seed: int | None = None) -> None:
        # ``seed`` selects the quasi-random time-jitter realisation (see the
        # module docstring).  A None seed maps to a fixed default so the
        # default bundle stays reproducible.  Astro and RV draw from
        # independent RNG streams (seed and seed+1) so their jitters are not
        # correlated.
        astro_t = _jittered_grid_times(_N_ASTRO_EPOCHS, seed=0 if seed is None else seed)
        rv_t = _jittered_grid_times(
            _N_RV_EPOCHS, seed=1 if seed is None else seed + 1,
        )

        # Deterministic rotating ψ schedule (golden-ratio step, mod π).
        astro_psi = np.mod(
            _PSI0_RAD + np.arange(_N_ASTRO_EPOCHS) * _PSI_STEP_RAD, math.pi,
        )
        rv_psi = np.mod(
            _PSI0_RAD + np.arange(_N_RV_EPOCHS) * _PSI_STEP_RAD, math.pi,
        )

        astro_f = along_scan_parallax_factor(
            astro_t + MJD_J2010_TCB, astro_psi, ra_deg, dec_deg,
        )

        object.__setattr__(self, "astro_obs_time", astro_t.astype(np.float64))
        object.__setattr__(self, "astro_scan_angle", astro_psi.astype(np.float64))
        object.__setattr__(
            self, "astro_parallax_factor_al", astro_f.astype(np.float64),
        )
        object.__setattr__(
            self, "astro_transit_id", np.arange(_N_ASTRO_EPOCHS, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "astro_centroid_pos_err",
            np.full(_N_ASTRO_EPOCHS, _ASTRO_CENTROID_ERR_MAS, dtype=np.float64),
        )
        object.__setattr__(self, "ra0_deg", None)
        object.__setattr__(self, "dec0_deg", None)

        object.__setattr__(self, "rv_obs_time", rv_t.astype(np.float64))
        object.__setattr__(self, "rv_scan_angle", rv_psi.astype(np.float64))
        object.__setattr__(
            self, "rv_transit_id", np.arange(_N_RV_EPOCHS, dtype=np.int64),
        )
        object.__setattr__(
            self, "rv_err", np.full(_N_RV_EPOCHS, _RV_ERR_KMS, dtype=np.float64),
        )
        object.__setattr__(self, "source_id", _SYNTHETIC_SOURCE_ID)
        object.__setattr__(self, "_ra_deg", float(ra_deg))
        object.__setattr__(self, "_dec_deg", float(dec_deg))

    def parallax_factor_al_at(
        self, t_j2010_days: np.ndarray, psi_rad: np.ndarray,
    ) -> np.ndarray:
        """The along-scan parallax factor at arbitrary epochs (J2010-days)
        and scan angles (radians) for this cadence's sky position, computed
        the same way as the cadence's own factors.
        """
        t = np.asarray(t_j2010_days, dtype=float) + MJD_J2010_TCB
        return along_scan_parallax_factor(t, psi_rad, self._ra_deg, self._dec_deg)
