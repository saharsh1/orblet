"""The simulator's numbers, fingerprinted — so a surface change is proven not to move them.

A surface change must leave every number here byte-identical; a deliberate
change of number re-blesses here in its own commit and says what moved.

Each case is the SHA-256 of (shape, float64 bytes) of an array, or of the
sorted numeric fields of a dict, with the sum kept beside it for the failure
message. Units are stripped (astropy Quantities are compared by value).
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _simulator_baseline import SIMULATOR_BASELINE  # noqa: E402

from orblet.simulate.bundles import load_simulated_inputs  # noqa: E402
from orblet.constants import MJD_J2010_TCB  # noqa: E402
from orblet.parallax import along_scan_parallax_factor  # noqa: E402
from orblet.simulate.cadence import DemoParallaxConsistentCadence  # noqa: E402
from orblet.simulate.orbit import OrbitSimulator  # noqa: E402


def _digest(x) -> dict:
    a = np.ascontiguousarray(np.asarray(getattr(x, "value", x), dtype=np.float64))
    h = hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()
    return {"sha256": h, "sum": float(np.nansum(a))}


def _numeric_fields(d: dict) -> dict:
    """The numeric entries of a loader-shaped dict, by sorted key."""
    out = {}
    for k in sorted(d):
        v = d[k]
        if isinstance(v, (list, str)) or v is None:
            continue
        if np.asarray(getattr(v, "value", v)).dtype.kind not in "biuf":
            continue                      # e.g. the string reason codes
        out[k] = _digest(v)
    return out


def _preset_fields(sim: OrbitSimulator) -> dict:
    vals = [getattr(sim, f.name) for f in dataclasses.fields(sim)]
    return _digest([np.nan if v is None else float(v) for v in vals])


def _compute() -> dict[str, dict]:
    out: dict[str, dict] = {}

    # The demo preset (bh1_like before the thinning; the same numbers).
    demo = OrbitSimulator.toy_orbit()
    out["preset.demo"] = _preset_fields(demo)

    # The default bundle, two seeds: every array of both channels, the
    # catalogue row and the truth.
    for seed in (0, 1):
        b = load_simulated_inputs(seed=seed)
        for name in ("rv_data", "astro_data", "catalog_row"):
            for k, v in _numeric_fields(getattr(b, name)).items():
                out[f"bundle.seed{seed}.{name}.{k}"] = v
        out[f"bundle.seed{seed}.truth"] = _preset_fields(b.truth)

    # The demo cadence on its own, at a second sky position.
    cad = DemoParallaxConsistentCadence(ra_deg=45.0, dec_deg=60.0, seed=3)
    for k in ("astro_obs_time", "astro_scan_angle", "astro_parallax_factor_al",
              "rv_obs_time", "rv_scan_angle"):
        out[f"cadence.ra45_dec60_seed3.{k}"] = _digest(getattr(cad, k))

    # The along-scan parallax factor on fixed inputs: the defaults (Gaia at
    # L2, astropy's built-in ephemeris) and the geocentre.
    t = np.linspace(0.0, 5.0 * 365.25, 200) + MJD_J2010_TCB
    psi = np.linspace(0.0, np.pi, 200)
    for ra, dec in ((180.0, -20.0), (294.8, 14.9)):
        out[f"along_scan_parallax_factor.ra{ra:g}_dec{dec:g}"] = _digest(
            along_scan_parallax_factor(t, psi, ra, dec))
    out["along_scan_parallax_factor.ra180_dec-20.geocentre"] = _digest(
        along_scan_parallax_factor(t, psi, 180.0, -20.0, observer="geocentre"))
    return out


@pytest.fixture(scope="module")
def computed() -> dict[str, dict]:
    return _compute()


@pytest.mark.parametrize("case", sorted(SIMULATOR_BASELINE))
def test_simulator_is_bit_identical_to_the_baseline(case, computed) -> None:
    got = computed[case]
    expected = SIMULATOR_BASELINE[case]
    assert got["sha256"] == expected["sha256"], (
        f"{case}: bytes changed (sum {expected['sum']!r} -> {got['sum']!r}). "
        "A surface change moves no number; a deliberate one re-blesses in its "
        "own commit."
    )


def test_every_computed_case_is_baselined(computed) -> None:
    assert sorted(computed) == sorted(SIMULATOR_BASELINE)
