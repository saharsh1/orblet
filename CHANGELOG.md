# Changelog

orblet follows semantic versioning. While the version is 0.x, a change to
the public surface (a name added, moved or removed, a convention changed) is
a MINOR bump; fixes and additions behind the same surface are PATCH bumps.

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
