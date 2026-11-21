# Changelog

orblet follows semantic versioning. While the version is 0.x, a change to
the public surface (a name added, moved or removed, a convention changed) is
a MINOR bump; fixes and additions behind the same surface are PATCH bumps.

## Unreleased

**The astrometric mass-ratio function (AMRF) leaves orblet — names
removed.** `orblet.interpret.amrf` is gone, with every name in it: `amrf`,
`amrf_curve`, `amrf_ms_locus`, `amrf_unresolved_pair_locus`,
`ms_amrf_boundary`, `triple_amrf_boundary`, `classify_amrf`,
`validate_beta_relation`, `ms_flux_ratio`, `BetaRelation`, the label
constants and `AMRF_LABELS`, `MS_FLUX_RATIO_EXPONENT` and `PAIR_MODEL`.
The reason is the placement test: the triage needs a mass–luminosity
relation, a metallicity, an age and a pair model, which are a stellar
model and not an orbit. It lives in the separate `amrf-triage` package.
orblet keeps what an orbit gives on its own: the mass functions, the
companion-mass solve, the astrometric upper limit and the photocentre
flux-ratio relations of `orblet.interpret.flux_ratio`.

- `AMRF_BETA_BAND` (the band of the flux ratio, `"G"`) is now defined in
  `orblet.interpret.flux_ratio`, the one module that still uses it.
  Import it from there.
- No computed number moves: nothing that remains used the removed code.
- The quickstart notebook `04_interpretation_mass.ipynb` loses its AMRF
  part.

**The Lomb–Scargle power is exact on every frequency grid — a numeric
change.** `compute_lomb_scargle_periodogram` now sets the astropy method
itself: `"cython"` for one term, `"chi2"` for more. Before, astropy chose,
and on a regular grid (`spacing="auto"`, `spacing="lin"`, a regular
explicit `frequency`) it took an FFT-based approximation whose error
depends on the astropy version. What moved: the power on those grids, by
up to 6.6e-3 on the baseline series of 40 epochs (peak power 0.98); five
baseline cases are re-blessed (`ls_auto`, `ls_lin`, `ls_flagged`, `ls_nan`,
`ls_nterms2`, the `.power` entry of each). What did not move: the
frequency grids, the best frequency and best period of every case, and
the power on a logarithmic grid, which was exact already. The power is
now the same number, bit for bit, under astropy 7.2 and 8.0. Cost: about
1 s for 800 epochs and 100 000 frequencies.

**astropy 8 and Python 3.13 are supported** (`astropy>=7,<9`,
`requires-python = ">=3.11,<3.14"`). The suite passes on Python 3.12 with
astropy 7.2 and on Python 3.13 with astropy 7.2 and 8.0 (conda-forge
builds), every baseline bit for bit.

Three tutorial notebooks on the public Gaia DR4
pre-release epoch astrometry, in `notebooks/tutorials/` (single star,
astrometric orbit of Gaia-4 b, joint astro-spectroscopic orbit of Gaia
BH3), with the loader `notebooks/tutorials/dr4_prerelease.py`. The
notebooks download the data; nothing real is committed.

## 0.2.2 — 2026-09-29

A patch: no public name added, moved or removed.

**Faster PDC periodogram.** `compute_pdc_periodogram` is 3.6–3.9× faster
in the ordinary mode and 4.2–4.4× in the semi-partial mode (5,000 trial
periods at 597 and 824 epochs; 824 epochs: 34.5 s → 8.8 s and
37.2 s → 8.5 s). Peak memory is never higher. The work that does not
depend on the trial period is done once, the per-period matrices reuse two
buffers, and the phase is reduced as `P · frac(Δt · (1/P))` instead of
`np.mod(Δt, P)`.

**Numbers moved.** The phase reduction rounds differently: PDC scores move
by about 2e-15 (1.3e-13 at the 4-epoch minimum), and the best period is
unchanged in every tested case. The periodogram baseline is re-blessed for
the PDC scores and their false-alarm probabilities; the distance kernels,
the scan-angle coupling, Lomb–Scargle and the peak widths are unchanged.

**Fixed.**

- `EccOmegaDiskPrior` rejects the unit circle (`h² + k² = 1`, i.e.
  `e = 1`), as its docstring stated; before, only the likelihood rejected
  it. `UniformCircularPrior` (the ω and τ disks) still admits the circle,
  where a phase is valid.
- `solve_companion_mass` returns NaN for `sin_i > 1`, as it does for every
  other unphysical input; `sin_i = 1` is unchanged.
- The four quickstart notebooks are valid nbformat 4 again (18 code cells
  had lost their `execution_count`), so GitHub renders them.

**Added.**

- `CITATION.cff`, so GitHub offers "Cite this repository".
- Tests: the constants against `astropy.constants` (with the two deliberate
  differences, `GM_SUN_SI` and `MSUN_IN_MJUP`, pinned); notebook structure
  and the citation version; the PDC sweep against a frozen copy of the
  pre-optimisation sweep; `benchmarks/pdc_sweep.py` to time it.
- The joint-fit notebook's all-parameter rung samples a jitter per channel.

## 0.2.1 — 2026-09-25

First release. Thirty-six public names behind one front door, `orblet`:

- **Forward models** — the RV curve, the along-scan measurement equation,
  Thiele-Innes and Campbell parameterisations of the photocentre orbit, the
  Kepler solver.
- **Likelihoods** — RV, along-scan, and the shared-inclination joint form,
  each a diagonal Gaussian with a jitter term.
- **Design matrices and linear solves** — the three-column RV and
  nine-column Thiele-Innes designs, the reduced joint designs, the
  generalised-least-squares solves with proper marginal evidence and a
  flat-prior ranking score, and the single-star five-parameter solve.
- **Period search** — the linearised Thiele-Innes frequency scan with alias
  and geometry flags, and the `orblet.periodogram` sub-package: Lomb–Scargle,
  the phase distance correlation (PDC) periodogram and its partial form
  controlling for the scan angle, distance kernels for scalars, along-scan
  segments, scan angles and spectra, the scan-angle coupling D_cpl of a
  source, and the pointwise χ² false-alarm probability of a PDC score.
- **Parallax factors** — per-direction and along-scan factors from a
  solar-system ephemeris, for an observer at Gaia's L2 point by default
  (or the geocentre), using astropy's built-in ephemeris by default (no
  download); a JPL kernel is one extra (`[ephemeris]`) and one
  `fetch_ephemeris` call away.
- **Elements and seeds** — Thiele-Innes to Campbell, the Gaia NSS
  convention, the inclination mirror, and the two seed composers that carry
  a linearised solution into an all-parameter sampler.
- **Interpretation** — companion mass from the spectroscopic and
  astrometric mass functions with an external primary mass, the astrometric
  upper limit, and the flux-ratio branches of a joint fit.
- **Sampler atoms** — reproducible multi-chain emcee runs, split R-hat and
  effective sample size, per-chain summaries and quantiles.
- **Simulation** — an orbit simulator with one preset, `toy_orbit`, and a
  Gaia-like cadence whose parallax factors come from the ephemeris, so
  injecting and removing the parallax signal use the same array;
  `load_simulated_inputs(simulator=, cadence=, seed=)` builds a complete
  synthetic data set with its truth attached.

Four runtime dependencies: numpy, scipy, astropy, matplotlib. emcee, corner,
pandas and jplephem are optional extras, each imported inside the one
function that needs it.

Eight notebooks: four quickstarts (forward models and simulation, design
matrices and linear solves, period search, interpretation and mass) and four
fitting examples (RV linearised, RV all parameters, astrometric linearised,
joint with its three rungs of shared parameters).

Two manuals under `docs/`: the open-surface contract, and the reference
manual stating every model, likelihood and prior in symbols with the
`file:line` that computes it.
