"""orblet — a standalone orbit fitter: arrays in, orbital elements out.

No mission, no data files, no network. orblet is a CONSUMER of data and never
a producer: you hand it epochs, measurements and uncertainties, and it hands
back a model, a likelihood, a solution. Where those arrays came from, and
whether you are allowed to have them, is not orblet's business.

Work the open way
-----------------
orblet ships ATOMS, not a black box. A Kepler solver, forward models, design
matrices, the two linear cores, proposed likelihoods, default priors, a
frequency search, the element conversions, the plots. You assemble the
optimiser or sampler you actually want and plug your own target function into
it; the default recipes are written out in the manuals and the example
notebooks rather than hidden inside a wrapper.

That is a deliberate trade. A one-line `fit()` is quicker to call and harder
to trust: when it disagrees with your expectation you cannot see which step
disagreed. Here every step is a function you can call, print and plot on its
own.

The shape of the thing
----------------------
    model       the forward models: elements and epochs -> observables
    design      the design matrices, one column per linear parameter
    solve       the linear cores — fix the geometry, solve the rest exactly
    search      the linearised Thiele-Innes frequency scan
    periodogram Lomb-Scargle, the PDC and partial PDC, scan-angle coupling
    likelihood  proposed likelihoods (Gaussian with jitter), not mandates
    priors      prior classes, defaults, and the spec parser
    prepare     epoch and time-scale handling
    sampling    MCMC atoms: chains, stacking, R-hat, ESS, dead chains
    elements    conversions between parameter conventions
    residuals   the 5-parameter residual routine
    parallax    observer position -> on-sky parallax factors
    interpret   what a fitted orbit implies: masses, limits, flux ratios
    plotting    the orbit, the corner, the sky plane
    simulate    synthetic data with known truth, for testing and teaching

Why the split matters: the geometry of a Keplerian orbit is non-linear, but
once it is fixed the amplitudes enter LINEARLY and have a closed-form best
fit. `solve` is that closed form. A sampler over a handful of non-linear
parameters, with `solve` as its inner step, beats one over all of them.

Import cost
-----------
Nothing heavy is imported when you `import orblet`. Every name below is
resolved on first attribute access (PEP 562), so scipy, emcee and matplotlib
arrive only if you touch something that needs them.

Where to read more
------------------
    docs/model_and_likelihoods.md   the reference: every model, likelihood
                                    and prior, with its assumptions (§7)
    docs/open_api.md                every public name, where it lives, units
    notebooks/quickstart/           the atoms, one topic per notebook
    notebooks/fitting_examples/     the fits assembled from them
"""

from __future__ import annotations

import importlib

# public name -> (module, attribute). One mechanism, one table, grouped by
# what the name is FOR rather than by which file it happens to live in.
#: The package version. The ONLY place it is written: the build reads it from
#: here (``[tool.setuptools.dynamic]`` in orblet's pyproject), and the packaging
#: test pins that the installed metadata agrees. A plain literal on purpose —
#: setuptools parses it from the source without importing the package.
__version__ = "0.2.1"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # ── Design matrices: one column per linear parameter ───────────────
    "rv_design_matrix": ("orblet.design.columns", "rv_design_matrix"),
    "ti_design_matrix": ("orblet.design.columns", "ti_design_matrix"),
    "acceleration_columns": ("orblet.design.columns", "acceleration_columns"),
    "astrometric_5param_design_matrix": (
        "orblet.solve.astrometry",
        "astrometric_5param_design_matrix",
    ),

    # ── Linear cores: geometry fixed, amplitudes solved exactly ────────
    "linear_solve_rv": ("orblet.solve.rv", "linear_solve_rv"),
    "linear_solve_ti": ("orblet.solve.astrometry", "linear_solve_ti"),
    "best_linear_params_rv": ("orblet.solve.rv", "best_linear_params_rv"),
    "best_linear_params_ti": (
        "orblet.solve.astrometry",
        "best_linear_params_ti",
    ),
    "fit_astrometric_5param": (
        "orblet.solve.astrometry",
        "fit_astrometric_5param",
    ),
    "recover_K": ("orblet.solve.rv", "recover_K"),
    "recover_omega": ("orblet.solve.rv", "recover_omega"),
    "semi_amplitude_kms": ("orblet.model", "semi_amplitude_kms"),

    # ── Searching for the period ───────────────────────────────────────
    "scan_ti_frequency": ("orblet.search", "scan_ti_frequency"),

    # ── Seeds: turning one solution into a starting point for the next ─
    "compose_rv_seed": ("orblet.solve.rv", "compose_rv_seed"),
    "compose_ti_seed": ("orblet.search", "compose_ti_seed"),
    "ti_amplitude_chains": ("orblet.solve.astrometry", "ti_amplitude_chains"),
    "mirror_inclination": ("orblet.elements", "mirror_inclination"),

    # ── Epochs and time scales ─────────────────────────────────────────
    "prepare_rv_for_orbit": ("orblet.prepare", "prepare_rv_for_orbit"),
    "resolve_epochs_mjd": ("orblet.prepare", "resolve_epochs_mjd"),

    # ── Where the observer is, as on-sky parallax factors ──────────────
    "per_direction_parallax_factors": (
        "orblet.parallax",
        "per_direction_parallax_factors",
    ),
    "along_scan_parallax_factor": (
        "orblet.parallax",
        "along_scan_parallax_factor",
    ),
    "OBSERVER_GEOCENTRE": ("orblet.parallax", "OBSERVER_GEOCENTRE"),
    "OBSERVER_L2": ("orblet.parallax", "OBSERVER_L2"),
    "L2_OFFSET_AU": ("orblet.parallax", "L2_OFFSET_AU"),

    # ── Element conversions and derived curves ─────────────────────────
    "extract_orbital_elements": (
        "orblet.elements",
        "extract_orbital_elements",
    ),
    "ti_to_kepler": ("orblet.elements", "ti_to_kepler"),
    "to_nss_convention": ("orblet.elements", "to_nss_convention"),
    "compute_rv_model_curve": ("orblet.elements", "compute_rv_model_curve"),
    "compute_residuals": ("orblet.elements", "compute_residuals"),

    # ── Carrying one channel's chain into the other's prior ────────────
    "a1_sini_from_rv_chain": ("orblet.rv_chain", "a1_sini_from_rv_chain"),
    "astrometric_priors_from_rv_chain": (
        "orblet.rv_chain",
        "astrometric_priors_from_rv_chain",
    ),
    "predicted_k_kms_from_astrometric_chain": (
        "orblet.rv_chain",
        "predicted_k_kms_from_astrometric_chain",
    ),

    # ── Plots (matplotlib arrives here, not before) ────────────────────
    "plot_orbit_fit": ("orblet.plotting", "plot_orbit_fit"),
    "plot_orbit_corner": ("orblet.plotting", "plot_orbit_corner"),
    "plot_astrometric_orbit": ("orblet.plotting", "plot_astrometric_orbit"),
    "plot_astrometric_sky_overlay": (
        "orblet.plotting",
        "plot_astrometric_sky_overlay",
    ),
}


def __getattr__(name: str):
    """Resolve a public name on first access (PEP 562).

    Keeps ``import orblet`` cheap: nothing below the front door is executed
    until something is actually asked for.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module_name, attr = target
    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list[str]:
    return sorted(__all__)


# Kept as a literal, same names as the table above.
__all__ = [
    "rv_design_matrix",
    "ti_design_matrix",
    "acceleration_columns",
    "astrometric_5param_design_matrix",
    "linear_solve_rv",
    "linear_solve_ti",
    "best_linear_params_rv",
    "best_linear_params_ti",
    "fit_astrometric_5param",
    "recover_K",
    "recover_omega",
    "semi_amplitude_kms",
    "scan_ti_frequency",
    "compose_rv_seed",
    "compose_ti_seed",
    "ti_amplitude_chains",
    "mirror_inclination",
    "prepare_rv_for_orbit",
    "resolve_epochs_mjd",
    "per_direction_parallax_factors",
    "along_scan_parallax_factor",
    "OBSERVER_GEOCENTRE",
    "OBSERVER_L2",
    "L2_OFFSET_AU",
    "extract_orbital_elements",
    "ti_to_kepler",
    "to_nss_convention",
    "compute_rv_model_curve",
    "compute_residuals",
    "a1_sini_from_rv_chain",
    "astrometric_priors_from_rv_chain",
    "predicted_k_kms_from_astrometric_chain",
    "plot_orbit_fit",
    "plot_orbit_corner",
    "plot_astrometric_orbit",
    "plot_astrometric_sky_overlay",
]
