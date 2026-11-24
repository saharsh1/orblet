"""The first example of the README runs, as it is printed.

The README shows three calls: the Thiele-Innes design matrix, the linear
solve, and the conversion to Campbell elements. This test takes the first
``python`` block of ``README.md``, gives it the arrays it names from the
simulated demo orbit, and executes it. A change of a signature or of a
dictionary key that breaks the printed example fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from orblet import resolve_epochs_mjd
from orblet.simulate.bundles import load_simulated_inputs

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_python_block(markdown: str) -> str:
    match = re.search(r"^```python\n(.*?)^```", markdown, flags=re.MULTILINE | re.DOTALL)
    assert match, "README.md has no ```python block"
    return match.group(1)


def test_readme_example_runs_and_recovers_the_orbit() -> None:
    code = _first_python_block((_REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    for name in ("ti_design_matrix", "linear_solve_ti", "ti_to_kepler"):
        assert name in code, f"the README example no longer calls {name}"

    # The names the README leaves to the reader, from the demo orbit at its true shape.
    bundle = load_simulated_inputs(seed=0)
    astro = bundle.astro_data
    truth = bundle.truth
    epoch_ref_mjd = float(truth.t_ref_mjd)
    namespace = {
        "t_mjd": resolve_epochs_mjd(astro),
        "psi_rad": np.asarray(astro["scan_angle"], dtype=float),
        "parallax_factor_al": np.asarray(astro["parallax_factor_al"], dtype=float),
        "along_scan_mas": np.asarray(astro["centroid_pos"], dtype=float),
        "sigma_mas": np.asarray(astro["centroid_pos_err"], dtype=float),
        "P_days": truth.P_days,
        "e": truth.e,
        "tau": ((truth.tp_mjd - epoch_ref_mjd) / truth.P_days) % 1.0,
        "epoch_ref_mjd": epoch_ref_mjd,
    }

    exec(compile(code, "README.md", "exec"), namespace)

    fit = namespace["fit"]
    elements = namespace["elements"]
    assert namespace["X"].shape == (namespace["t_mjd"].size, 9)
    assert fit.beta.shape == (9,)
    # Loose bounds: the example must recover the injected orbit, not a number.
    assert abs(fit.beta[8] - truth.parallax_mas) < 0.1                     # mas
    assert abs(elements["a_phot_mas"] - truth.a_phot_mas) < 0.1            # mas
    assert abs(elements["inc_rad"] - truth.i_rad) < 0.05                   # rad
    assert abs(elements["omega_rad"] - truth.omega_rad) < 0.1              # rad
    assert abs(elements["Omega_rad"] - truth.Omega_rad) < 0.1              # rad
