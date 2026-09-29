# Roadmap

What is planned, what it would touch, and what has to be decided first.
Items are not in priority order except where the text says so. Anything
that changes a computed number or the public surface lands as a minor
release with a changelog entry (see `CHANGELOG.md` for the rule).

## 1. `astropy.units` and `astropy.constants` at the interfaces — as an option

Arrays stay plain floats inside; the change is at the boundary. Public
functions accept `Quantity` inputs and convert to the documented unit on
entry (a float stays a float in the documented unit), and results can be
requested as `Quantity`.

The constants are compared with astropy in `tests/test_constants.py`:
`C_KMS`, `AU_M`, `G_SI`, `MSUN_KG`, `DAYS_PER_KEPLER_YEAR` and the MJD
epoch anchors are bit-identical. Two differ, and the test pins by how much:

- `GM_SUN_SI` is deliberately `4π²·AU³/yr²`, so that `fm_spec` and
  `fm_ast` share one mass unit; it sits 3.8e-5 above astropy's `GM_sun`.
  Switching to astropy's value would break the exact closure of the two
  mass functions.
- `MSUN_IN_MJUP = 1047.35` is the IAU 2009 Sun/Jupiter ratio; astropy's
  `M_sun / M_jup` (IAU 2015 nominal) is 1047.5655, 2.1e-4 higher. No result
  depends on it: its one use (`elements.py`) multiplies a solar mass by it
  and divides by it again. Kept as is.

Steps, in order:

1. One entry helper, `Quantity` → float in the documented unit, a float
   passed through untouched. It must recognise a `Quantity` without
   importing astropy at module scope (duck-typed on `.unit`), or the
   lazy-import test fails. Rolled out function by function across the
   public surface, each with a test that a float and the equivalent
   `Quantity` give identical bytes.
2. `Quantity` outputs on request — a public-surface change: minor release,
   consumer pin checked.

Rough size: a few hours for the helper, one to two days for the roll-out.

## 2. A JAX sibling, `orblet_jax` — alongside, not instead

A separate package with the same public names, taking and returning JAX
arrays: the forward models, likelihoods, design matrices and linear solves
are pure functions and port directly; the Kepler solver needs a fixed
iteration count instead of a tolerance loop; the frequency scan
parallelises over the grid. Conventions and tests are shared by
construction: the byte-identity baselines of this package become the
oracle the sibling must match to float tolerance. The sampler atoms stay
here (emcee is numpy).

**Decide first: what it is for.** Gradients (NUTS/HMC), throughput (`vmap`
over stars or grids), or a GPU. Apple-GPU JAX has no float64, so on Apple
Silicon it is CPU only; a GPU gain needs a CUDA machine. The answer sets
the order below.

- **Ports mechanically:** the forward models, design matrices, residuals,
  likelihoods and element conversions. In-place writes (`elements.py`,
  `design/columns.py`) become `.at[].set()`; `scipy.linalg.cho_*` and the
  `scipy.stats` priors have `jax.scipy` equivalents.
- **Needs real work:**
  - *Kepler solver:* the tolerance loop becomes a fixed iteration count
    (the bisection repair is already `where`-based). For gradients, a
    `custom_jvp` from the Kepler equation itself —
    `∂E/∂M = 1/(1 − e cos E)`, `∂E/∂e = sin E/(1 − e cos E)` — not
    differentiation through the Newton steps.
  - *Validation:* the value-dependent `raise`s (about forty, in `solve/`,
    `priors.py`, `model.py`) cannot live inside `jit`. Each function
    splits into a validating numpy wrapper and a pure core. JAX's Cholesky
    returns NaN instead of raising, so the singular-design error becomes a
    NaN check outside the core.
  - *Likelihood guards* (`e` out of range → −inf, `isfinite`) become
    `jnp.where`.
  - *float64:* `jax_enable_x64` on, or agreement stops near 1e-7.
- **Stays numpy:** `parallax` (astropy ephemeris; compute once, pass
  arrays), `prepare`, `plotting`, `chain_stats`, `sampling`, `simulate`,
  most of `interpret`.
- **The PDC sweep:** the cost is memory passes over N×N arrays (item 5),
  which XLA can fuse — possibly the largest CPU speed-up on offer, at
  float tolerance instead of byte identity. A self-contained experiment.
- **Tests:** orblet as the oracle at `rtol ≈ 1e-12`; gradients against
  finite differences.
- **Packaging decision:** own repository or a subdirectory; `orblet_jax`
  depends on `orblet` (constants, conventions, validators shared, not
  copied), JAX an ordinary dependency of the sibling only.

Stages, each a go/no-go for the next: a spike (Kepler + `rv_curve` +
`loglike` + one gradient, against orblet; about a day), the RV channel end
to end (several days), the full core with astrometry and the search (two
to three weeks).

## 3. The modified mass function and `q_min`

Port the estimators from github.com/saharsh1/BinaryMassFunction into
`orblet.interpret`: the modified mass function, the minimum mass ratio and
whatever else that repository derives from `(K, P, e)` and the primary
mass, with their assumptions stated the way `companion_mass` states its
sentinels. Read the source first; the placement test applies (an atom
answers a question on its own).

Not to be confused with `interpret/amrf.py` (the astrometric mass-ratio
function), which already exists. Steps: read the repository and list each
quantity with its inputs and assumptions; apply the placement test to
each; port with tests on synthetic cases whose answer is known in closed
form. New names: minor release. Rough size: a day after the reading.

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

The work is the verification (authors, year, journal, the exact method
each function follows), not the writing. No number moves. Rough size: a
day.

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
  unchanged. A new keyword: minor release. Rough size: a day.
- **Faster PDC sweep — done (patch release 0.2.2).** One period at a time,
  as before; batching over periods was measured and is not faster (one
  period is passes over N×N arrays, the loop overhead is microseconds).
  - *Byte-identical part:* `Δt` built once; for the semi-partial mode
    `E = A − (⟨A,Z⟩/⟨Z,Z⟩)Z` and `⟨E,E⟩` formed once; the phase matrix,
    its U-centring and the inner-product terms written into two reused
    N×N buffers. 1.5× (ordinary), 1.75× (semi-partial), peak memory never
    higher.
  - *Rounding-changing part:* the phase reduced as `P · frac(Δt · (1/P))`
    instead of `np.mod(Δt, P)`, which was two-thirds of a period. Scores
    move by about 2e-15 (1.3e-13 at the 4-epoch minimum); the baseline was
    re-blessed for it. Against the sweep before this work, 5,000 periods:
    3.6–3.9× (ordinary), 4.2–4.4× (semi-partial); at 824 epochs
    34 s → 8.8 s and 37 s → 8.5 s.
  - *Guards:* `tests/_pdc_reference.py` is the pre-optimisation sweep,
    frozen; `tests/test_pdc_lean_sweep.py` compares against it on 28
    cases (same NaN positions and exceptions, scores within 1e-12, same
    best period); `benchmarks/pdc_sweep.py` times both, interleaved.
  - *Found on the way:* the partial PDC is ill-conditioned when the
    nuisance explains the observations almost fully (coupling → 1): `E`
    is then rounding residue and every score is noise over noise.
- **Faster PDC sweep, further — not planned.** Two more rounding-changing
  options were measured on top: the algebraic shortcut (with `A`
  U-centred, `⟨Ã, B̃⟩ = ⟨Ã, B⟩`, and `‖B̃‖²` follows from the row sums of
  `B`, so the phase matrix need not be centred; about 1.2×), and BLAS dot
  products for the two inner products (about 1.05×). The shortcut needs a
  written proof before it is considered. A JAX sweep (item 2) is the
  other route.
- **Both curves in one sweep — PARKED.** The phase matrix depends only on
  the times and the trial period, so a caller wanting the ordinary and
  the partial curve today builds it twice. A shared sweep adds ~0.4 ms
  per period for the second score instead of ~5 ms, roughly halving the
  cost of the pair. Parked until it is clear the pair is wanted; if it
  is, the choice is a private combined path, a new entry point returning
  both result dicts (minor release), or `partial_mode="both"` (avoid:
  `"scores"` would change meaning). The regression test is that it
  equals the two separate calls.
- A further "additional partial" statistic is out of scope until its
  mathematical definition and intended scientific meaning are supplied.

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
  points at the RV one until it exists. Rough size: a day.
- The open item of the reference manual (§8): a fixed `sin i ≠ 1`
  labelled `MASS_CONVENTION_TRUE`. A third sentinel for an assumed
  inclination is a convention-surface change — minor release, consumer pin
  checked.

## 8. Smaller

- `chain_stats` without pandas at all: `chain_summary_table` is the one
  function that needs it; a plain-dict table would drop the `[tables]`
  extra. It changes a return type: minor release, consumer checked first
  for DataFrame use. Rough size: a few hours.
