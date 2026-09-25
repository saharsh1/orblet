# orblet <img src="https://raw.githubusercontent.com/saharsh1/orblet/main/docs/assets/orblet.png" alt="orblet" height="44" align="right">

Atoms for Keplerian orbit analysis. Forward models, likelihoods, design
matrices and linear solves, period search, element conversion, plotting —
each a function that takes arrays and returns a result, with its units,
frames and assumptions stated in its docstring.

orblet is **not a solver**. It does not know what a Gaia epoch is, does not
read files, and does not contact the network. You build the pipeline; orblet
supplies the pieces.

```python
from orblet import ti_design_matrix, linear_solve_ti, ti_to_kepler

X = ti_design_matrix(t_mjd, psi_rad, parallax_factor_al, P_days, e, tau, epoch_ref_mjd)
fit = linear_solve_ti(along_scan_mas, sigma_mas, X)
elements = ti_to_kepler({"A": fit.beta[0], "B": fit.beta[1], "F": fit.beta[2], "G": fit.beta[3]})
```

## Install

```
pip install "orblet[all]"
```

To follow the repository instead of a release, install from GitHub:

```
pip install "orblet[all] @ git+https://github.com/saharsh1/orblet.git"
```

or from a local clone, which is the right choice while you are editing it:

```
git clone https://github.com/saharsh1/orblet.git
pip install -e "./orblet[all]"
```

Drop `[all]` for the four core dependencies only — numpy, scipy, astropy,
matplotlib — which is all a fresh install needs to run the atoms and the
quickstarts. The extras (`emcee`, `corner`, `pandas`, `jplephem`; also
available one at a time as `[sampling]`, `[plots]`, `[tables]`,
`[ephemeris]`) are each imported inside the one function that uses them, so
a missing extra fails that call with a message naming it and leaves
everything else working.

Parallax factors use astropy's built-in solar-system ephemeris by default,
with no download. To use a JPL kernel instead:

```
pip install "orblet[ephemeris]"
python -c "from orblet.parallax import fetch_ephemeris; fetch_ephemeris('de432s')"
```

then pass `ephemeris="de432s"`. The fetch is a one-time ~10 MB download into
astropy's cache.

## What is here

| | |
|---|---|
| `orblet.model` | forward models: RV curve, along-scan astrometry, Thiele-Innes and Campbell photocentre orbits |
| `orblet.likelihood` | Gaussian log-likelihoods with jitter, per channel |
| `orblet.kepler` | the Kepler-equation solver |
| `orblet.design` / `orblet.solve` | design-matrix builders and the generalised-least-squares linear solves for RV and astrometry |
| `orblet.search` / `orblet.periodogram` | the Thiele-Innes frequency scan; Lomb-Scargle and phase-distance-correlation periodograms |
| `orblet.elements` | Thiele-Innes → Campbell, NSS convention, element extraction from chains |
| `orblet.priors` | prior classes and the log-prior composer |
| `orblet.sampling` / `orblet.chain_stats` | emcee helpers; quantiles and circular summaries of chains |
| `orblet.interpret` | companion mass and the astrometric mass-ratio function |
| `orblet.simulate` | a synthetic-orbit simulator and a parallax-consistent cadence, for tests and tutorials |
| `orblet.parallax` | parallax factors from a solar-system ephemeris (Gaia at L2 by default) |
| `orblet.plotting` | orbit, residual, sky-overlay and corner plots |

The front door — `from orblet import <name>` — exposes 35 names and imports
nothing heavy: `import orblet` pulls in no scipy, no matplotlib, no astropy.
Each name resolves on first use.

## Conventions

Every public function states them in its docstring. The ones that bite:

- radial velocity: positive is receding; `omega` is the **primary's** argument of periastron
- period in **days** at the public surface (Keplerian years only inside the Kepler solver)
- `tau` is the periastron phase in `[0, 1)`; `tp = tau * P + epoch_ref_mjd`
- astrometric amplitudes `A, B, F, G` are **photocentre** amplitudes in mas, positive-amplitude convention; the parallax term enters as `parallax_mas * parallax_factor_al`, additive
- Gaia's `pmra` is already `mu_alpha*` — never apply `cos(dec)` again
- **a seed is initialisation, never a prior**: a starting point for a sampler carries no evidence

## Four laws, each a test

1. A public name **is** the object it claims to be — no silent wrappers.
2. The numerical core does not move: byte-identity baselines pin the forward
   models, the likelihoods and the design columns.
3. Conventions hold: RV sign, primary-frame ω, τ → tp, time scales.
4. orblet imports nothing outside itself, the four dependencies and the
   standard library.

## Simulated data

One simulator preset ships, `OrbitSimulator.toy_orbit()`: an illustrative
orbit that both channels detect strongly, on an invented sky position. The
quickstarts load it through `load_simulated_inputs(seed=...)`, on a cadence
whose parallax factor is coupled to that position, so injecting and removing
the parallax signal use the same array and closure is exact. For any other
truth, construct `OrbitSimulator(...)` directly and pass it as
`load_simulated_inputs(simulator=..., cadence=..., seed=...)`.

## Licence

MIT.
