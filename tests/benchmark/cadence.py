"""
Observation cadences for the synthetic-data benchmark framework.

Three implementations:

- :class:`SourceCadence`  — load a real source's cached epoch dicts and
  expose ``obs_time`` / ``scan_angle`` / ``parallax_factor_al`` /
  ``transit_id`` for both astrometry and RV.  This guarantees that the
  time sampling and scan-angle structure are *exactly* what Gaia
  produced for that source — periodogram aliasing, scan-angle
  coupling, and the per-source RV-vs-AL transit ratio are all
  preserved.
- :class:`BH3Cadence`     — :class:`SourceCadence` with the BH3
  source_id pre-filled.  This is the default test fixture.  Requires
  the BH3 caches to have been populated by
  ``scripts/prefetch_bh_candidates.py --sources 4318465066420528000``.
- :class:`ScanLawCadence` — **stub**, raises ``NotImplementedError``.
  Intended to drive cadence + scan angle + parallax factor purely
  from a sky position via Gaia's published scanning law (e.g. via the
  GOST tool).  Tracked as a future improvement; do not use until
  implemented.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


# Gaia BH3 (Panuzzo et al. 2024) — the canonical SB1 + astrometric
# black-hole companion benchmark.  Used as the default cadence skeleton.
BH3_SOURCE_ID = 4318465066420528000


# ── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class Cadence(Protocol):
    """
    Schedule of Gaia transits for a synthetic source.

    All time arrays are in **days from J2010.0 TCB** (matching
    the epoch-astrometry loader's ``obs_time`` convention).  Scan angles are in radians.
    """

    # Astrometric transits.
    astro_obs_time: np.ndarray
    astro_scan_angle: np.ndarray
    astro_parallax_factor_al: np.ndarray
    astro_transit_id: np.ndarray
    astro_centroid_pos_err: np.ndarray   # mas, default σ_AL per transit
    # ra0 / dec0 of the local-plane origin (deg).  None ⇒ simulator
    # treats centroid_pos as already in the source-centred frame
    # (i.e. no offset term).
    ra0_deg: float | None
    dec0_deg: float | None

    # RV transits.  Subset of astrometric transits in general; the
    # per-source ratio falls out of which Gaia visits triggered RVS.
    rv_obs_time: np.ndarray
    rv_scan_angle: np.ndarray
    rv_transit_id: np.ndarray
    rv_err: np.ndarray                   # km/s, default σ_RV per transit


# ── Real-source skeletons ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _LoadedCadence:
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


def _default_cache_root() -> Path:
    """Repo-root-relative ``data/raw/`` (matches the YAML's defaults)."""
    # tests/benchmark/cadence.py  →  repo root is two parents up.
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def SourceCadence(
    source_id: int,
    *,
    cache_root: str | Path | None = None,
    rv_default_err_kms: float = 1.0,
    astro_default_err_mas: float = 0.5,
) -> Cadence:
    """
    Build a :class:`Cadence` from a source's cached epoch dicts.

    Parameters
    ----------
    source_id : int
        Gaia DR4 source_id.  Must already be present in both the
        astrometry and RV caches under ``cache_root``.
    cache_root : str or Path, optional
        Directory containing ``astrometry_cache/`` and ``rv_cache/``.
        Defaults to ``<repo>/data/raw/``.
    rv_default_err_kms / astro_default_err_mas : float
        Used only as a per-transit noise template if the cached dict
        does not already carry a per-transit error array.  The
        :class:`OrbitSimulator` overrides these via its own
        ``rv_sigma_kms`` / ``centroid_sigma_mas`` parameters.

    Raises
    ------
    FileNotFoundError
        If the astrometry or RV pickle is missing.  The raised error
        names the prefetch invocation that would create it.
    """
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    astro_path = root / "astrometry_cache" / f"{int(source_id)}.pkl"
    rv_path    = root / "rv_cache"         / f"{int(source_id)}.pkl"
    for label, path in (("astrometry", astro_path), ("RV", rv_path)):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} cache missing for source {source_id}: {path}\n"
                "Populate it first with:\n"
                f"    python scripts/prefetch_bh_candidates.py "
                f"--config config/bh_candidates_prefetch.yaml "
                f"--sources {int(source_id)}"
            )

    with astro_path.open("rb") as f:
        astro = pickle.load(f)
    with rv_path.open("rb") as f:
        rv = pickle.load(f)

    # Astrometric arrays — fall back to the per-transit centroid_pos_err
    # if available; otherwise broadcast the default scalar.
    astro_obs_time = np.asarray(astro["obs_time"], dtype=float)
    astro_scan = np.asarray(astro["scan_angle"], dtype=float)
    astro_pf = np.asarray(astro.get("parallax_factor_al", np.zeros_like(astro_obs_time)), dtype=float)
    astro_tid = np.asarray(astro["transit_id"])
    cpos_err = astro.get("centroid_pos_err")
    if cpos_err is None or np.all(~np.isfinite(np.asarray(cpos_err, dtype=float))):
        astro_err = np.full_like(astro_obs_time, astro_default_err_mas)
    else:
        astro_err = np.asarray(cpos_err, dtype=float)
        astro_err = np.where(np.isfinite(astro_err) & (astro_err > 0),
                             astro_err, astro_default_err_mas)

    # RV arrays.  obs_time_rv may be a Quantity (astropy unit), unwrap.
    def _arr(v):
        return np.asarray(getattr(v, "value", v), dtype=float)

    rv_obs_time = _arr(rv["obs_time_rv"])
    rv_scan = _arr(rv.get("scan_angle", np.zeros_like(rv_obs_time)))
    rv_tid = np.asarray(rv["transit_id_spec"])
    rv_err_raw = rv.get("radial_velocity_err")
    if rv_err_raw is None:
        rv_err = np.full_like(rv_obs_time, rv_default_err_kms)
    else:
        rv_err = _arr(rv_err_raw)
        rv_err = np.where(np.isfinite(rv_err) & (rv_err > 0),
                          rv_err, rv_default_err_kms)

    return _LoadedCadence(
        astro_obs_time=astro_obs_time,
        astro_scan_angle=astro_scan,
        astro_parallax_factor_al=astro_pf,
        astro_transit_id=astro_tid,
        astro_centroid_pos_err=astro_err,
        ra0_deg=astro.get("ra0_deg"),
        dec0_deg=astro.get("dec0_deg"),
        rv_obs_time=rv_obs_time,
        rv_scan_angle=rv_scan,
        rv_transit_id=rv_tid,
        rv_err=rv_err,
        source_id=int(source_id),
    )


def BH3Cadence(*, cache_root: str | Path | None = None) -> Cadence:
    """Default test cadence: Gaia BH3 (source_id = 4318465066420528000)."""
    return SourceCadence(BH3_SOURCE_ID, cache_root=cache_root)


# ── Stub for future scanning-law-driven cadence ────────────────────────────


class ScanLawCadence:
    """
    **Stub** — synthesise a cadence from a sky position via Gaia's
    published scanning law.  Not yet implemented.

    Intended API::

        cad = ScanLawCadence(
            ra=..., dec=...,
            t_start=..., t_end=...,         # MJD or J2010.0-TCB days
            rvs_fraction=0.4,               # frac. of AL transits with RVS
            seed=0,
        )

    Will need either a port of Gaia's GOST scan-prediction utility or
    a direct read of the published Nominal Scanning Law.  Tracked here
    so anyone glancing at the framework sees what's missing.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "ScanLawCadence is not implemented.  TODO: port the Gaia "
            "scanning-law cadence + scan-angle predictor (GOST-equivalent) "
            "and remove this stub.  Until then, use SourceCadence(source_id) "
            "or the BH3Cadence default."
        )
