# Roadmap

What is planned, what it would touch, and what has to be decided first.
Items are not in priority order except where the text says so. Anything
that changes a computed number or the public surface lands as a minor
release with a changelog entry (see `CHANGELOG.md` for the rule).

## 1. `astropy.units` and `astropy.constants` at the interfaces — as an option

Arrays stay plain floats inside; the change is at the boundary. Public
functions accept `Quantity` inputs and convert to the documented unit on
entry (a float stays a float in the documented unit), and results can be
requested as `Quantity`. Constants: `G_SI`, `AU_M`, `MSUN_KG` compared
against `astropy.constants` in a test, with a decision to make first —
`GM_SUN_SI` is deliberately `4π²·AU³/yr²` so that `fm_spec` and `fm_ast`
share one mass unit, and that differs from astropy's `GM_sun` at the
1e-4 level. Keep ours as the authority and test the difference, or switch
and accept that the two mass functions no longer close exactly.

## 2. A JAX sibling, `orblet_jax` — alongside, not instead

A separate package with the same public names, taking and returning JAX
arrays: the forward models, likelihoods, design matrices and linear solves
are pure functions and port directly; the Kepler solver needs a fixed
iteration count instead of a tolerance loop; the frequency scan
parallelises over the grid. Conventions and tests are shared by
construction: the byte-identity baselines of this package become the
oracle the sibling must match to float tolerance. The sampler atoms stay
here (emcee is numpy).

## 3. The modified mass function and `q_min`

Port the estimators from github.com/saharsh1/BinaryMassFunction into
`orblet.interpret`: the modified mass function, the minimum mass ratio and
whatever else that repository derives from `(K, P, e)` and the primary
mass, with their assumptions stated the way `companion_mass` states its
sentinels. Read the source first; the placement test applies (an atom
answers a question on its own).

## 4. References — what to cite

A `docs/references.md` and a `CITATION.cff`, plus a "References" line in
the docstring of every function that implements a published method, so a
user knows what to cite for what. Candidates, each to be verified before
it is written down: the phase distance correlation periodogram; Lomb and
Scargle; the Thiele-Innes constants and their inversion; the along-scan
parallax-factor convention (Lindegren et al. 2012); Gelman & Rubin 1992
and Vehtari et al. 2021 for R-hat; emcee; the Gaia NSS conventions the
`to_nss_convention` step targets.

**What needs a reference** (every entry to be verified):

| Function | What needs citing |
|---|---|
| `compute_pdc_periodogram` | PDC (Zucker 2018); the partial / semi-partial mode (Binnenfeld et al. 2022) |
| U-centring, `distance_correlation` | Székely & Rizzo 2014; Székely, Rizzo & Bakirov 2007 |
| `scalar_distance_matrix` | the absolute-difference distance for PDC |
| `astrometric_segment_distance_matrix` | the segment dissimilarity and `L` (Binnenfeld et al. 2023) |
| `scan_angle_distance_matrix` | the circular scan-angle distance |
| `scan_angle_coupling` | D_cpl |
| `spectral_distance_matrix` | the chord distance between spectra |
| `pdc_false_alarm_probability` | Shen, Panda & Vogelstein 2022 |
| uncertainty-aware distance (item 5) | "PDC with errors" |
| `compute_lomb_scargle_periodogram` | Lomb; Scargle; astropy's implementation |

An unpublished manuscript is not cited here until it is on arXiv.

## 5. The periodograms — from new capability to stable

The periodogram sub-package is young; its baseline is expected to be
re-blessed once more when it is declared stable.

- **Lock the numbers**, after a look at real data (BH3, or public Gaia
  epoch astrometry) with the default segment length `L = 0.01 mas`. `L` is
  the across-scan length of the segment that stands in for each 1-D
  measurement — a scale, not a detector quantity — and what matters is the
  residual amplitude σ against it: for σ ≪ `L` the dissimilarity reduces to
  a function of the scan-angle difference alone and D_cpl → 1 whatever the
  data; for σ ≳ `L` results no longer depend on `L`. On synthetic Gaia-scale
  data (0.1 mas orbit + 0.1 mas noise) the partial-PDC periodogram at
  `L` = 1 mas correlates 0.87 with the converged one, at 0.1 mas 0.9975, at
  0.01 mas and below 1.0000.
- **Uncertainty-aware scalar distance** ("PDC with errors"): opt-in `err=` on
  `scalar_distance_matrix`, following `calc_pdc_distance_matrix` in the
  reference implementation (github.com/SPARTA-dev/PDC). No `err` ⇒ numbers
  unchanged.
- **Faster PDC sweep**: with the observation matrix U-centred,
  `⟨Ã, B̃⟩ = Σ_{i≠j} Ãᵢⱼ Bᵢⱼ / (n(n−3))`, and `‖B̃‖` follows from row sums, so
  the phase matrix need not be centred per trial period. Changes rounding:
  own commit, own re-bless.

## 6. Correlated noise within a transit, and merging CCDs into transits — PARKED

Parked until there is time to work through the mathematics in depth. The
problem: on per-CCD Gaia data the ~9 CCD crossings of one transit share
attitude and geometry, so a diagonal covariance makes formal errors and
χ²/dof optimistic; BH3's measured per-transit correlated term is
0.031 ± 0.007 mas. Material for that discussion, none of it decided:

- **Claim to demonstrate:** with `C_T = D + s_c²·11ᵀ`,
  `D = diag(σ_k² + s²)`, the inverse-variance transit mean is the GLS
  estimate, and `ln L` splits exactly into a merged term
  `ln N(w̄_T | μ_T, σ_T² + s_c²)` and a within-transit term that depends on
  `s` alone. If it holds, the existing `loglike_along_scan` and
  `linear_solve_ti` on merged data, with `s_c` in the jitter slot, already
  are the block model; the library additions reduce to `merge_transits`,
  a within-transit log-likelihood atom, and a per-CCD simulator mode.
- **Caveat:** in DR4, `scan_angle` and `obs_time` are per CCD. Merging them
  with the same weights as the abscissa cancels the first-order model
  error; what remains is second order in the within-transit spread.
- **Back-of-envelope at BH3 levels** (σ = 0.085 mas/CCD, s_c = 0.031, 9
  CCDs): true transit σ 0.042 mas; independent-CCD model 0.028 (0.030 with
  a fitted jitter). Formal errors ≈ 1.4× optimistic; a white jitter
  recovers little of it.
- **Study outline:** on BH3, the 9×9 per-CCD residual covariance
  (equicorrelated or not decides the design), the within-transit ψ/t
  spread, the stability of s_c, heavy tails; then synthetic
  injection–recovery comparing per-CCD + jitter, merged, and a dense block
  reference (coverage, χ²/dof, scan null, cost). The scramble null in
  `03_period_search` must permute whole transits on per-CCD data.
- **Open decisions:** grouping key (`transit_id` only?); within-transit
  rejection inside `merge_transits` or upstream; dense block likelihood
  public or test-only.

## 7. Examples and documentation

- An astrometric all-parameter example (the fourteen-parameter twin of
  `fit_rv_orbit_all_parameters.ipynb`); §3.9 of the reference manual
  points at the RV one until it exists.
- Jitter sampled at rung 3 of `fit_joint_orbit.ipynb` (both channels).
- The three open items of the reference manual (§8).

## 8. Smaller

- `chain_stats` without pandas at all: `chain_summary_table` is the one
  function that needs it; a plain-dict table would drop the `[tables]`
  extra.
