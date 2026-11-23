"""Helpers for the Gaia DR4 pre-release tutorials.

Tutorial 01 builds every function here step by step in its own cells.
Tutorials 02 and 03 import them from this file so they can start at the
fit.  The code is the same; the docstrings here are the reference.

This is pipeline code (archive table -> orblet input dict).  It lives
beside the notebooks on purpose: orblet itself does not know what a Gaia
transit is.

Data: ESA Gaia DR4 pre-release epoch astrometry (June 2026), 12 sources,
public, no login.  https://www.cosmos.esa.int/web/gaia/dr4-prerelease
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from astropy.table import Table

#: Public ESA download (anonymous FTP over HTTPS).
PRERELEASE_URL = (
    "https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/"
    "dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip"
)
#: The one VOTable inside the zip.
PRERELEASE_XML = "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"

#: DR4 reference epoch J2017.5 TCB, in days since J2010.0 TCB
#: (= orblet.constants.MJD_J2017_5_TCB - orblet.constants.MJD_J2010_TCB).
REF_EPOCH_DAYS_J2010 = 2739.375

#: Nanoseconds per day: obs_time_tcb is stored in ns since J2010.0 TCB.
NS_PER_DAY = 86400.0e9

#: Names of the ten CCD slots in each per-transit array.
CCD_NAMES = ["SM"] + [f"AF{i}" for i in range(1, 10)]

#: The 12 pre-release sources (ESA page).  Approximate G and parallax.
SOURCES = {
    4181040337841125632: dict(name="HD 183633", group="parallax", g_mag=9.0, plx_mas=1.0),
    3926186255616949504: dict(name=None, group="parallax", g_mag=19.0, plx_mas=1.0),
    435469040545191680: dict(name=None, group="parallax", g_mag=14.0, plx_mas=1.0),
    1663617687609809280: dict(name="LSPM J1324+6124", group="magnitude", g_mag=14.3, plx_mas=10.1),
    2309425390592896: dict(name=None, group="magnitude", g_mag=14.0, plx_mas=0.9),
    20694084440761600: dict(name=None, group="magnitude", g_mag=14.1, plx_mas=4.9),
    2237987199365376: dict(name=None, group="qso", g_mag=19.8, plx_mas=0.0),
    60730287810150016: dict(name=None, group="qso", g_mag=20.1, plx_mas=0.0),
    10973744521070720: dict(name=None, group="qso", g_mag=19.9, plx_mas=0.0),
    3937211745905473024: dict(name="HD 114762", group="orbit", g_mag=7.2, plx_mas=25.6),
    1457486023639239296: dict(name="Gaia-4", group="orbit", g_mag=11.9, plx_mas=13.6),
    4318465066420528000: dict(name="Gaia BH3", group="orbit", g_mag=11.2, plx_mas=2.0),
}


def source_id_of(name: str) -> int:
    """Return the DR4 source_id of a named pre-release source."""
    for sid, info in SOURCES.items():
        if info["name"] == name:
            return sid
    raise KeyError(name)


def download_prerelease(cache_dir: str | Path = "_data") -> Path:
    """Download and unpack the pre-release zip; return the path of the XML.

    The zip is fetched only when it is not in ``cache_dir`` yet.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / PRERELEASE_URL.rsplit("/", 1)[-1]
    if not zip_path.exists():
        urllib.request.urlretrieve(PRERELEASE_URL, zip_path)
    xml_path = cache_dir / PRERELEASE_XML
    if not xml_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
    return xml_path


def read_prerelease(xml_path: str | Path) -> Table:
    """Read the pre-release VOTable: one row per field-of-view transit."""
    return Table.read(str(xml_path), format="votable")


def _row_array(value, dtype=float) -> np.ndarray:
    """One per-CCD array column entry -> plain numpy array (masked -> NaN/False)."""
    arr = np.ma.asarray(value)
    if dtype is bool:
        return np.asarray(arr.filled(False), dtype=bool)
    return np.asarray(arr.astype(float).filled(np.nan), dtype=float)


def _row_scalar(value, default=np.nan) -> float:
    """One scalar column entry -> float (masked -> ``default``)."""
    if np.ma.is_masked(value) or value is None:
        return float(default)
    return float(value)


def explode_transits(
    table: Table,
    source_id: int,
    *,
    keep_only_agis: bool = True,
    add_excess_noise: bool = False,
) -> dict:
    """Turn the per-transit rows of one source into one datum per CCD.

    Steps, in this order:

    1. Select the rows of ``source_id``.
    2. Flatten the ten CCD slots (SM, AF1..AF9) of every row.
    3. Time: ``obs_time_tcb`` [ns since J2010.0 TCB] -> days since J2010.0,
       plus the barycentric (Roemer) correction ``obs_time_bary_corr``
       where it is given (ESA does the same; it is at most 6 minutes).
    4. Scan angle: degrees -> radians.
    5. Keep a CCD datum only when ``used_by_agis_al`` is True and the
       time, angle, abscissa and error are finite (``keep_only_agis``).
       This drops the SM slot (never used in AL) and the CCDs that the
       Gaia solution rejected.
    6. Optionally add ``agis_source_excess_noise`` to the error in
       quadrature, as ESA's ``gaiasupdate`` does (``add_excess_noise``).
    7. Colour fields (tutorial 4): ``colour_factor_al`` per CCD, and the
       per-transit ``nu_eff_used_in_astrometry`` and ``nu_eff_error``
       repeated on the CCDs of the transit.  They do NOT enter the
       selection of step 5, so the data of tutorials 1 to 3 do not change;
       a kept datum can have a non-finite colour factor.

    Returns the orblet astrometry dict: ``obs_time`` (days since J2010.0
    TCB), ``scan_angle`` (rad), ``centroid_pos`` (mas), ``centroid_pos_err``
    (mas), ``parallax_factor_al``, plus ``transit_id``, ``ccd_index``,
    ``ra0_deg``, ``dec0_deg``, ``ref_epoch_days``, ``excess_noise_mas`` and
    the counts ``n_transits``, ``n_ccd_total``; and the colour inputs
    ``colour_factor_al`` (mas um), ``nu_eff_used`` (1/um) and
    ``nu_eff_error`` (1/um), one value per kept datum.
    """
    rows = table[np.asarray(table["source_id"]) == source_id]
    if len(rows) == 0:
        raise KeyError(f"source_id {source_id} is not in the table")

    t_days, psi_rad, w_mas, sig_mas, pf, used = [], [], [], [], [], []
    transit_id, ccd_index = [], []
    colour_factor, nu_eff_used, nu_eff_error = [], [], []
    for row in rows:
        t_ns = _row_array(row["obs_time_tcb"])
        bary_ns = _row_scalar(row["obs_time_bary_corr"], default=0.0)
        n = t_ns.size
        t_days.append((t_ns + bary_ns) / NS_PER_DAY)
        psi_rad.append(np.deg2rad(_row_array(row["scan_pos_angle"])))
        w_mas.append(_row_array(row["centroid_pos_al"]))
        sig_mas.append(_row_array(row["centroid_pos_error_al"]))
        used.append(_row_array(row["used_by_agis_al"], dtype=bool))
        pf.append(np.full(n, _row_scalar(row["parallax_factor_al"])))
        transit_id.append(np.full(n, int(row["transit_id"]), dtype=np.int64))
        ccd_index.append(np.arange(n))
        colour_factor.append(_row_array(row["colour_factor_al"]))
        nu_eff_used.append(np.full(n, _row_scalar(row["nu_eff_used_in_astrometry"])))
        nu_eff_error.append(np.full(n, _row_scalar(row["nu_eff_error"])))

    t_days = np.concatenate(t_days)
    psi_rad = np.concatenate(psi_rad)
    w_mas = np.concatenate(w_mas)
    sig_mas = np.concatenate(sig_mas)
    pf = np.concatenate(pf)
    used = np.concatenate(used)
    transit_id = np.concatenate(transit_id)
    ccd_index = np.concatenate(ccd_index)
    colour_factor = np.concatenate(colour_factor)
    nu_eff_used = np.concatenate(nu_eff_used)
    nu_eff_error = np.concatenate(nu_eff_error)
    n_ccd_total = t_days.size

    finite = (np.isfinite(t_days) & np.isfinite(psi_rad) & np.isfinite(w_mas)
              & np.isfinite(sig_mas) & (sig_mas > 0) & np.isfinite(pf))
    keep = finite & used if keep_only_agis else finite

    excess = _row_scalar(rows["agis_source_excess_noise"][0], default=0.0)
    if add_excess_noise:
        sig_mas = np.sqrt(sig_mas ** 2 + excess ** 2)

    return {
        "obs_time": t_days[keep],
        "scan_angle": psi_rad[keep],
        "centroid_pos": w_mas[keep],
        "centroid_pos_err": sig_mas[keep],
        "parallax_factor_al": pf[keep],
        "transit_id": transit_id[keep],
        "ccd_index": ccd_index[keep],
        "colour_factor_al": colour_factor[keep],
        "nu_eff_used": nu_eff_used[keep],
        "nu_eff_error": nu_eff_error[keep],
        "ra0_deg": float(np.nanmedian(np.asarray(rows["ra0"], dtype=float))),
        "dec0_deg": float(np.nanmedian(np.asarray(rows["dec0"], dtype=float))),
        "ref_epoch_days": REF_EPOCH_DAYS_J2010,
        "excess_noise_mas": excess,
        "n_transits": len(rows),
        "n_ccd_total": int(n_ccd_total),
    }


# ----------------------------------------------------------------------
# Markdown display helpers (IPython is only needed when these are called).
# ----------------------------------------------------------------------

def matrix_markdown(X, columns, *, n_rows: int = 6, fmt: str = ".4f",
                    row_labels=None, caption: str | None = None):
    """Render the first ``n_rows`` rows of a design matrix as a markdown table."""
    from IPython.display import Markdown

    X = np.asarray(X, dtype=float)
    head = "| row | " + " | ".join(columns) + " |"
    sep = "|---|" + "|".join(["---:"] * len(columns)) + "|"
    lines = [head, sep]
    for i in range(min(n_rows, X.shape[0])):
        label = str(row_labels[i]) if row_labels is not None else str(i)
        cells = " | ".join(format(v, fmt) for v in X[i])
        lines.append(f"| {label} | {cells} |")
    if X.shape[0] > n_rows:
        lines.append("| ... | " + " | ".join(["..."] * len(columns)) + " |")
    text = "\n".join(lines)
    if caption:
        text = f"**{caption}** (shape {X.shape[0]} x {X.shape[1]})\n\n" + text
    return Markdown(text)


def params_markdown(names, values, sigmas=None, *, units=None, reference=None,
                    ref_label: str = "reference", fmt: str = ".4f",
                    caption: str | None = None):
    """Render fitted parameters (value, sigma, optional reference) as markdown."""
    from IPython.display import Markdown

    head = ["parameter"] + (["unit"] if units is not None else []) + ["value"]
    if sigmas is not None:
        head.append("sigma")
    if reference is not None:
        head += [ref_label, "diff / sigma" if sigmas is not None else "diff"]
    n_text = 2 if units is not None else 1
    lines = ["| " + " | ".join(head) + " |", "|---|" * n_text + "---:|" * (len(head) - n_text)]
    for i, name in enumerate(names):
        cells = [name] + ([units[i]] if units is not None else []) + [format(values[i], fmt)]
        if sigmas is not None:
            cells.append(format(sigmas[i], fmt))
        if reference is not None:
            ref = reference[i]
            if ref is None or (isinstance(ref, float) and np.isnan(ref)):
                cells += ["-", "-"]
            else:
                cells.append(format(ref, fmt))
                diff = values[i] - ref
                cells.append(format(diff / sigmas[i], ".2f") if sigmas is not None
                             else format(diff, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    if caption:
        text = f"**{caption}**\n\n" + text
    return Markdown(text)
