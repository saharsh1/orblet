# Fitted models, likelihoods and priors — the reference

**Status.** Every formula below was read out of the implementation, not from
a docstring; where implementation and docstring disagree, this page follows
the **code** and the disagreement is listed in §8. Citations are guarded by a
test that checks each `file:line` still points at the named function.

**What this is.** The single place that states, in symbols, what orblet's
atoms compute: the forward models, the likelihoods, the design matrices and
the marginal evidences they make possible, the priors and their Jacobians,
the sampler atoms, and the step from a fit to a mass. Every equation carries
the `file:line` that computes it, relative to `src/orblet/`.

**What this is not.** It is not a fitter's manual. orblet ships atoms; the
notebooks under `notebooks/` compose them into fits, and the three
`fitting_examples` are the reference recipes. Where a recipe and this page
disagree, the code is the fact and the disagreement is a bug in one of the
three.

**Notation.** Unicode maths inline; display equations in fenced blocks so
they survive any renderer. Symbols: `P` period, `e` eccentricity, `ω`
argument of periastron (primary frame), `Ω` node, `i` inclination, `τ`
periastron phase, `ν` true anomaly, `E` eccentric anomaly, `ψ` scan angle,
`ϖ` parallax, `K` RV semi-amplitude, `γ` systemic velocity, `a_phot`
photocentre semi-major axis, `s` jitter.

---

## 1. Shared foundations

### 1.1 Constants

All in `constants.py`.

| symbol | value | line |
| --- | --- | --- |
| `DAYS_PER_KEPLER_YEAR` | 365.25 (Julian year) | `constants.py:73` |
| `G_SI` | 6.674 30 × 10⁻¹¹ (CODATA 2018) | `constants.py:63` |
| `AU_M` | 1.495 978 707 × 10¹¹ (IAU 2012, exact) | `constants.py:59` |
| `MSUN_KG` | `GM_SUN_SI / G_SI` (≈1.988 41 × 10³⁰; DERIVED, never an independent literal) | `constants.py:83` |
| `GM_SUN_SI` | 4π²·AU³/yr² — the unit-system GM, so `fm_spec` and `fm_ast` come out in ONE mass unit | `constants.py:104` |
| `MJD_J2010_TCB` | 55197.0 | `constants.py:198` |
| `MJD_J2016_TCB` | 57388.5 (DR3 epoch) | `constants.py:204` |
| `MJD_J2017_5_TCB` | 57936.375 (**DR4** epoch) | `constants.py:216` |
| `SOLAR_SYSTEM_EPHEMERIS_PIN` | `"builtin"` (astropy's own; within ~5 km of DE432s) | `constants.py:272` |

**Units, by layer — the most common source of confusion.** The period lives
in three different units depending on where you look:

| layer | unit | anchor |
| --- | --- | --- |
| forward models (`period_yr`) | Keplerian **years** | `rv_curve`, `model.py:197`; `kepler_xy_orbit`, `model.py:389` |
| **RV** design matrix (`period_yr`) | Keplerian **years** | `rv_design_matrix`, `design/columns.py:158` |
| `best_linear_params_rv` | **days** | `solve/rv.py:493` |
| **TI** design matrix, `best_linear_params_ti`, `scan_ti_frequency` | `f_per_day` = cycles per **DAY** | `solve/astrometry.py:753`, `search.py:425` |
| the default period prior | **days** | `default_companion_priors`, `priors.py:537` |
| the seed dicts (`P_yr`) | Keplerian **years** | `compose_ti_seed`, `search.py:219`; `compose_rv_seed`, `solve/rv.py:647` |

⚠️ **A silent factor-365.25 trap lives in this table.** The RV design matrix
takes **years** while `best_linear_params_rv` takes **days**, and the seed
dicts carry years while the default prior is in days — the wrong one raises
nothing. The prior families themselves (§5.1) are unit-agnostic: a prior is
in whatever unit you evaluate it in.

Everything else: velocities km/s, angles radians, positions and amplitudes
mas, proper motions mas/yr, parallax mas, masses M☉, times MJD (TCB).

### 1.2 Time, phase and the reference epoch

One map converts the sampled phase to a periastron time, used identically
everywhere (`tp_from_disk_angle`, `model.py:367`; the τ fraction from a
unit-disk latent is `_tau_from_disk_angle`, `model.py:354`):

```
P_days = P_yr × 365.25
t_p    = τ · P_days + t_ref          τ ∈ [0, 1)
```

`t_ref` (`epoch_ref_mjd`) carries **two different meanings** by channel, and
this asymmetry is deliberate — do not "harmonise" it:

- **Astrometric and joint models:** `t_ref` is the catalogue reference
  epoch and is the zero-point of proper motion *and* parallax. It is
  **REQUIRED** — `require_epoch_ref_mjd` (`constants.py:219`) raises on
  `None`. It tracks the data release: DR3 J2016.0, **DR4 J2017.5 =
  MJD 57936.375**. A wrong epoch here leaves a real scan-modulated
  residual.
- **RV-only models:** `t_ref` is *purely* the ω gauge — shifting it rotates
  ω while leaving `K` and χ² invariant. The linear core still requires it
  explicitly (`epoch_ref_mjd`, `solve/rv.py:38`); a convenient choice is
  the median observation epoch, and the choice must then travel with every
  result it produced.

**The ASTROMETRIC observation times are auto-detected, the epoch is not.**
`resolve_epochs_mjd` (`prepare.py:47`, public as `orblet.resolve_epochs_mjd`)
accepts `obs_time` as either MJD or J2010-days and decides by a median
threshold: `median(obs_time) < 30000 → obs_time + 55197.0`. So the time
origin is inferred from the data while `epoch_ref_mjd` must be supplied —
an asymmetry worth knowing, since both live on the same axis and a wrong
origin produces the same scan-modulated residual.

**The RV observation times are NOT auto-detected — YOU choose the scale, at
`prepare_rv_for_orbit`** (`prepare.py:212`). This is the single most
dangerous parameter in the RV path, so it gets its own rule:

> Gaia-style RV tables carry their epochs in **days-from-J2010** — the same
> clock as the astrometric `obs_time`. Prepare them with
> **`time_scale="gaia_obmt"`**. The default, `time_scale="mjd"`, is a
> pass-through: it converts nothing.

`prepare_rv_for_orbit` returns `epochs_mjd`, and a joint model takes that
array **verbatim** while resolving its astrometric partner to MJD. Pick the
wrong scale and the two channels end up 55 197 d apart *while sharing one
time of periastron* — the RV phase is wrong by `55197/P` of a cycle and
nothing looks unusual, because each channel is internally consistent.

This cannot happen silently: `assert_rv_epochs_are_absolute_mjd`
(`prepare.py:118`) raises whenever epochs on the **`"mjd"`** branch sit in
the J2010-day regime (the same 30 000 d threshold as above). Call it on any
RV epochs you hand to a joint likelihood. The check is a *regime* test, not
a distance between channels — so an archival RV set that legitimately
predates the Gaia epochs by years is not flagged, while the J2010/MJD
mix-up always is.

**Where a reference epoch comes from.** Several day-counts live on the same
axis and only some are catalogue epochs. Read this before writing anything
that subtracts one:

| Epoch | MJD | What it is | Valid as a catalogue epoch? |
|---|---|---|---|
| `MJD_J2010_TCB` | 55 197.0 | OBMT origin; the zero of Gaia's `obs_time` day-counts | **no** |
| `MJD_J2016_TCB` | 57 388.5 | DR3 `gaia_source.ref_epoch` (J2016.0) | yes, for DR3 |
| `MJD_J2017_5_TCB` | 57 936.375 | **DR4** `gaia_source.ref_epoch` (J2017.5) | yes, for DR4 |
| the target's own | its catalogue `ref_epoch` | the epoch a real target's data carries — what a real fit uses | yes |
| RV-only `t_ref` | e.g. the median observation epoch | a **gauge** for ω: shifting it rotates ω and leaves `K` and χ² invariant | **no** |

Carry the epoch with the result it produced; never re-supply it downstream.
`to_nss_convention` (`elements.py:656`) works this way: it reads the epoch
recorded beside the chain to convert a periastron time into the Gaia NSS
`t_periastron` offset. Re-supplying a value the result already carries adds
no check, only a second place for two epochs to disagree.

**The pitfall this creates.** An RV-only result's epoch is a gauge choice,
meaningful only for ω. Subtracting it would produce a number shaped exactly
like an NSS periastron time and meaning nothing. `to_nss_convention`
therefore **raises** on an RV-only result rather than converting it; pass
`include_keplerian=False` for the amplitude-only conversion, which is
epoch-independent.

**What the guard does NOT settle.** It is a coarse regime test, so it closes
the 55 197 d error and nothing finer. That an RV column's zero-point is
*exactly* J2010.0 TCB is the caller's assumption to verify against their
archive documentation; the at-Gaia vs barycentric Rømer term (±370 s) is
also unapplied. A residual offset of that size would pass the guard
untouched. Note that an **RV-only** fit is referenced to its own epoch, so a
constant offset cancels out of `K`, `P`, `e` and ω; the damage is
joint-fit-only, which is exactly why a wrong scale can look fine right up
until the two channels are combined.

### 1.3 The Kepler solve

`solve_kepler`, `kepler.py:38`. Solves `M = E − e sin E` by Newton from
`E₀ = M`, `tol = 1e-10`, `maxiter = 50` (`kepler.py:42`).

Newton from `E₀ = M` **diverges** for a scattered subset of anomalies near
periastron at `e ≳ 0.96`. Non-converged entries (those with residual >
1e-9) are repaired by 64 bisections on the wrapped mean anomaly
(`kepler.py:100-114`); converged entries are untouched by the repair.

True anomaly, half-angle form (`_true_anomaly`, `model.py:151`):

```
ν = 2 · atan2( √(1+e) · sin(E/2) ,  √(1−e) · cos(E/2) )
```

This atom is shared by the RV forward model, the RV design matrix and the
reduced joint designs — one source of truth.

### 1.4 Orbit geometry

Orbital-plane coordinates, `kepler_xy_orbit`, `model.py:389`:

```
r/a = (1 − e²) / (1 + e cos ν)
X   = (r/a) cos ν          Y = (r/a) sin ν
```

Analytically these are the standard elliptical rectangular coordinates
`X = cos E − e`, `Y = √(1−e²) sin E`; the code computes them in the
`(r/a, ν)` form.

**Conventions, as coded:**

- **ω is the PRIMARY's** argument of periastron (textbook binary-star), used
  as-is with no ω → ω+π translation (`model.py:517`). Sentinel
  `OMEGA_CONVENTION_PRIMARY` (`constants.py:163`); the companion's ω is
  this plus π.
- **Ω** is the longitude of the ascending node, counter-clockwise from
  north, as written into the Thiele-Innes constants by `campbell_xy`
  (`model.py:474`).
- **Inclination spans the full sphere**, `i = arccos(1 − 2u) ∈ [0, π]` with
  `u ∈ [0,1]`, so `cos i` is uniform on `[−1, +1]` — the isotropic prior
  (`CosUniformInclinationPrior`, `priors.py:472`). Sentinel `I_DOMAIN_FULL`
  (`constants.py:151`).
- **`M` is TOTAL mass** (`m₁ + m₂`), never stellar mass, wherever a
  forward model takes one; the default prior dict keeps the short key
  `"M"` (`default_system_priors`, `priors.py:590`).

---

## 2. The spectroscopic (RV) channel

One forward model, used two ways:

| | what is sampled | amplitudes | jitter | worked example |
| --- | --- | --- | --- | --- |
| **linearised** | the shape `(P, e, τ)` — 3 | `(γ, K cos ω, K sin ω)` solved in closed form, evidence marginalised over them | **FIXED** | `fit_rv_orbit.ipynb` |
| **all parameters** | `(P, e, τ, ω, K, γ, s)` — 7 | sampled | sampled | `fit_rv_orbit_all_parameters.ipynb` |

### 2.1 Forward model

`rv_curve`, `model.py:197`:

```
v(t) = γ + K [ cos(ν + ω) + e cos ω ]
```

**Sign convention — pinned: positive RV means RECEDING.** The canonical
SB1 form with `K > 0` adopts the Gaia RVS convention
(`radial_velocity` positive = receding), asserted by
`tests/test_rv_sign_convention.py`. External RVs must enter in the same
convention.

From masses, when `K` is not sampled directly (`semi_amplitude_kms`,
`model.py:114`, with **sin i ≡ 1**; `rv_model`, `model.py:259`, is the
mass-parameterised wrapper around `rv_curve`):

```
a_tot = (P_yr² · M_tot)^{1/3}   [AU]      # Kepler III, TOTAL mass
K₁    = (2π a_tot) / (P √(1−e²)) · (m₂ / M_tot)
```

No `sin i` factor appears anywhere, so `mass_msun` in this path *is*
`m₂ sin i`. `rv_model_with_inclination` (`model.py:927`) is the variant a
joint model uses when the inclination is shared with the astrometry (§4.4).

### 2.2 The linear form — why the linearised fit exists

Expanding `cos(ν+ω)`, the model is **linear in three amplitudes** given the
shape `(P, e, τ)`:

```
v = γ·1  +  (K cos ω)·(cos ν + e)  +  (K sin ω)·(−sin ν)
```

so the design matrix is (`rv_design_matrix`, `design/columns.py:158`,
public as `orblet.rv_design_matrix`):

```
X = [ 1 ,  cos ν + e ,  −sin ν ]        β = (γ, C, S) = (γ, K cos ω, K sin ω)
```

Recovery (`solve/rv.py:377`, `:399`):

```
K = √(C² + S²)              ω = atan2(S, C) ∈ (−π, π]
```

**At e ≈ 0, ω is pure gauge** (degenerate with `T₀`), and `C`, `S`
*individually* are gauge-dependent — only `K` is physical. Do not interpret
`K cos ω` / `K sin ω` individually — decode only through `recover_K` /
`recover_omega` (`solve/rv.py:378-406`).

### 2.3 Likelihood

`loglike`, `likelihood.py:93` (the sum at `likelihood.py:155-161`):

```
σ_eff,i² = σ_i² + s²
ln L     = −½ Σᵢ [ (vᵢ − v_model,i)² / σ_eff,i²  +  ln(2π σ_eff,i²) ]
```

Jitter enters **in quadrature as a linear km/s quantity**; the
all-parameter example samples it directly under a uniform prior (sampling
`ln s` instead is a choice that owes its Jacobian, §5.3). The
normalisation term is retained — it must be, since `s` is free.
Returns `−inf` for `e ∉ [0,1)` or `s < 0`.

### 2.4 Marginal evidence

`linear_solve_rv`, `solve/rv.py:175`. With `Σ_d = diag(σ_eff²)`,
prior `β ~ N(μ_β, Σ_β)`, and `M = XᵀΣ_d⁻¹X + Σ_β⁻¹`:

```
β̂ = M⁻¹ (XᵀΣ_d⁻¹d + Σ_β⁻¹μ_β)          cov = M⁻¹
r = d − X μ_β        b = XᵀΣ_d⁻¹ r
log Z = −½ [ (rᵀΣ_d⁻¹r − bᵀM⁻¹b) + (Σ ln σᵢ² + ln|Σ_β| + ln|M|) + n ln 2π ]
```

`solve/rv.py:393-416`. This is `log N(d | Xμ_β, XΣ_βXᵀ + Σ_d)`
with **all constants retained**, computed via the matrix-determinant lemma
on the reduced 3×3 system, so it is comparable across models and additive
with the astrometric evidence (§4).

A second, distinct quantity is also returned: `logL_marginal`
(`solve/rv.py:366`), a **ranking score** for comparing shapes on
a grid:

```
logL_marginal = −½ [ χ²(β̂_d) + ln|M_d| + n ln 2π + Σ ln σᵢ² ],   M_d = XᵀΣ_d⁻¹X
```

It keeps the `ln|M_d|` Occam term but drops the flat prior's constant and the
k-dependent `(k/2) ln 2π` (shape-independent within one model). **It means
the same thing on every path:** `M_d` and `β̂_d` in this formula are always
the data-only normal matrix and the least-squares amplitudes, whether or
not a `β` prior was supplied; the prior affects only `beta` / `cov` (the
regularised posterior) and `log_evidence` (the proper prior-weighted
integral, every constant kept). It is `None` only when a prior was supplied
and the data-only matrix is singular. Rules: sum the same score across
independent channels at one shape; never add a `logL_marginal` to a
`log_evidence`; never compare `logL_marginal` across models with different
numbers of amplitudes. A sampler over the shape (rung 1, §4.1) targets
`log_evidence`, not this score; the score is for ranking a grid.

**Amplitude prior.** `beta_prior` is any object exposing `mean` and `cov`
(`solve/rv.py:232`); `None` is the flat path. The natural choice, and the
one `02_design_matrices_and_linear_solves.ipynb` builds, is
`μ_β = (γ_mean, 0, 0)`, diagonal with `(σ_γ², σ_cs², σ_cs²)`. Zero-mean
isotropic on `(C,S)` ⇒ **Rayleigh prior on K** (mode at `σ_cs`, density → 0
as K → 0) **× uniform ω**.

### 2.5 Mass function

`fm_from_K`, `model.py:63` — the single source:

```
f(m) = K³ P (1 − e²)^{3/2} / (2πG)          [M☉]
```

with `K` in km/s and `P` in Keplerian years internally, returning M☉.

Computed from a **measured** `K`, this is a pure observable: it carries no
inclination assumption and no mass prior. The `sin i` assumption enters
only when you *invert* it for a companion mass (§6.1) — where `sin i = 1`
yields a minimum mass. (Distinct from §2.1, where `sin i ≡ 1` is a genuine
model assumption because `K` is being *derived* from masses.)

### 2.6 What the RV channel does not measure

Inclination. `K` and `fm_spec` are the observables, and neither is a mass:
with RV alone `sin i = 1` gives a *minimum* companion mass, and the
interpretation layer labels it so (`MASS_CONVENTION_PROJECTED`,
`constants.py:115`).

---

## 3. The astrometric channel

**Two parameterisations of one orbit.** The Thiele-Innes amplitudes
`(A, B, F, G)` are linear in the along-scan data at a fixed shape
`(P, e, τ)`: they are the basis of the design matrix, the frequency scan and
the linearised fit (`fit_astrometric_orbit.ipynb`). The Campbell elements
`(ω, Ω, i)` with one amplitude `a_phot` are what the joint ladder samples at
rungs 2 and 3 (§4). Both are exact descriptions of the same photocentre
orbit; neither measures a mass. Astrometry alone measures
`fm_ast = a_phot³/P_yr²` (§3.8), and everything from §3.2 onwards describes
the Thiele-Innes machinery unless stated otherwise.

### 3.1 The along-scan measurement equation

`along_scan_model`, `model.py:580`. Gaia measures one number
per CCD crossing: the projection of the source position onto the scan
direction.

```
w(t) = Δα*·sin ψ + Δδ·cos ψ                        (orbit)
     + Δα*₀·sin ψ + Δδ₀·cos ψ                      (position offset)
     + (μ_α*·sin ψ + μ_δ·cos ψ)·Δt                 (proper motion)
     + ϖ · f_AL                                    (parallax)

Δt = (t − t_ref) / 365.25   [Keplerian years]
```

Three conventions are load-bearing here:

- **ψ assignment is `sin ψ ↔ α*`, `cos ψ ↔ δ`** (`model.py:607`).
- **Parallax is purely additive**: `ϖ_mas × parallax_factor_al`, with **no
  extra trigonometry and no ψ projection** — the factor already contains
  the geometry (`model.py:659`).
- **`pmra` is μ_α*, already including cos δ.** Never apply cos δ again
  (`model.py:610`).

The single-star part of this equation — the last three lines — is its own
closed-form solve, `fit_astrometric_5param` (`solve/astrometry.py:926`)
over the design `astrometric_5param_design_matrix`
(`solve/astrometry.py:792`, columns 4–8 of §3.3). Its residuals are what a
period search should run on (`03_period_search.ipynb`): position, proper
motion and parallax are a smooth secular signal that phases coherently at
long periods and drives any periodogram to the edge of its range.

### 3.2 Thiele-Innes

`campbell_xy`, `model.py:474`, primary frame, positive
photocentre amplitude:

```
a_rel  = (P_yr² · M_tot)^{1/3}   [AU]
a_phot = (m₂ / M_tot) · a_rel · ϖ    [mas]        # β = 0, dark companion

A = a_phot ( cos Ω cos ω − sin Ω sin ω cos i )
B = a_phot ( sin Ω cos ω + cos Ω sin ω cos i )
F = a_phot (−cos Ω sin ω − sin Ω cos ω cos i )
G = a_phot (−sin Ω sin ω + cos Ω cos ω cos i )
```

The mass-free form of the same four constants — one amplitude times the
unit direction — is `_ti_constants_unit` (`model.py:420`), which the joint
rungs use (§4.1); `thiele_innes_xy` (`model.py:544`) applies either set to
the orbital-plane `(X, Y)`.

**Note the pairing** (`model.py:528`) — flagged as a footgun in the
builder itself:

```
Δα* = B·X + G·Y            Δδ = A·X + F·Y
```

so, combined with §3.1, `w = (B X + G Y) sin ψ + (A X + F Y) cos ψ`.

### 3.3 The design matrix

`ti_design_matrix`, `design/columns.py:372`, public as
`orblet.ti_design_matrix`. **Nine columns**, in this fixed order
(`_BETA_COLUMNS`, `solve/astrometry.py:109`):

```
[ A , B , F , G , Δα*₀ , Δδ₀ , μ_α* , μ_δ , ϖ ]

cols 0–3 :  X·cos ψ ,  X·sin ψ ,  Y·cos ψ ,  Y·sin ψ
cols 4–8 :  sin ψ , cos ψ , sin ψ·Δt , cos ψ·Δt , f_AL
```

The four orbital amplitudes are linear given `(P, e, τ)`; that is what
makes the linearised fit and the frequency scan possible.

### 3.4 Likelihood

`loglike_along_scan`, `likelihood.py:182` — same shape as RV:

```
σ_eff,i² = σ_i² + s²
ln L     = −½ Σᵢ [ (wᵢ − w_model,i)² / σ_eff,i²  +  ln(2π σ_eff,i²) ]
```

**The covariance is diagonal, and on Gaia data each row is one CCD, not
one transit.** Gaia observes a source ~10× per transit (SM + AF1–AF9), and
the usual preparation keeps every CCD as an independent measurement. Those
~10 points share attitude and geometry, so treating them as independent
makes formal errors and χ²/dof **optimistic**. Point estimates are
validated (BH3 to 0.02–0.03σ); significance and false-positive tests on
such a stream are **not calibrated**. The jitter `s` is the only term
absorbing that correlation, and `Σ_d = diag(σᵢ² + s²)` is an approximation
to a block-correlated covariance. See §7.

### 3.5 Marginal evidence and amplitude priors

`linear_solve_ti`, `solve/astrometry.py:315` — algebraically identical
to §2.4 with nine columns (the routine is width-generic, despite docstrings
that say "nine"). Both the proper `log_evidence` and the `logL_marginal`
ranking score are returned, under the same three-number contract as §2.4
(the ranking score from the data alone on every path; the prior only in
`beta`/`cov` and `log_evidence`).

**Amplitude prior.** As for RV, any object with `mean` and `cov` over the
nine columns can be passed as `beta_prior` (`solve/astrometry.py:320`), or
`None` for the flat path, which is what the linearised example runs. On a
short baseline the parallax column is poorly conditioned against the
orbit, and the usual cure is a Gaussian parallax prior centred on the
catalogue value, with wide Gaussians on the offsets and proper motion and
a zero-mean Gaussian on `(A,B,F,G)` at a width you choose. Know that the
catalogue parallax is not fully external to the data — see §7, item 4.

### 3.6 The frequency scan

`scan_ti_frequency`, `search.py:425`. Scans a 3-D grid in
(frequency, e, τ), uniform in frequency with `Δf = 1/(oversample·T)`,
default oversample 4. At each node it builds the design matrix and solves
with a **flat** prior (non-flat priors are refused, for cross-frequency
comparability). Candidates are ranked by `logL_marginal`; local maxima in
frequency are returned.

Per peak (`_alpha_mas_from_beta`, `search.py:326`):

```
u = (A² + B² + F² + G²)/2        v = AG − BF
α = √( u + √((u+v)(u−v)) )       a_phot = α / ϖ
```

**Peaks are flagged, not pruned** (`_prune_peak_flags`,
`search.py:356-393`) — each carries `flagged` / `flag_reasons` for exactly
two reasons: the **1-yr alias** window (`1/365.25 ± 2%` and its 6-month
harmonic, intrinsically degenerate with the parallax column) and the
**geometry screen** `a_phot > 0` (a vanishing amplitude or a non-positive
fitted parallax; NOT an `i → 0` effect: α = a_phot·ϖ at every inclination).
A peak carries observables only — shape node, score, the nine amplitudes
with covariance, `a_phot_au`, and the fitted parallax `plx_mas ±
plx_sigma_mas` (the σ is conditional on the node's fixed shape and carries
no jitter — a lower bound). A small-but-positive fitted parallax is
deliberately NOT flagged: it is printed beside the period so the reader
sees it, and the seeded fit is where a poor starting point gets corrected.
No automation acts on it.

⚠️ **No false-alarm probability is computed anywhere in this path.** The
scan ranks candidates; it does not tell you whether the best one is
significant. The alias flag is the only alias handling in the astrometric
periodicity path. `03_period_search.ipynb` shows the honest substitute: a
scramble null over the same epochs and scan law.

### 3.7 The sign gauge

The gauge is on the ANGLES: `(ω, Ω) → (ω+π, Ω+π)` at **fixed**
`(A,B,F,G)` is the same orbit — the amplitudes are exactly invariant under
that shift, and it and the identity are the ONLY invariances among the
angle and inclination transforms at fixed amplitudes, at every
eccentricity, on both the amplitude and along-scan surfaces.

Negating the amplitudes is **not** a gauge: `(A,B,F,G) → (−A,−B,−F,−G)` is
a point reflection of the photocentre orbit, a different orbit with a
different along-scan track (measured at several mas on a 2.5 mas orbit),
and the paired "half-period degeneracy `(−A,−B,−F,−G, τ+0.5)`" is likewise
not an invariance at `e > 0`. Transforms that flip the amplitude sign
together with an angle are a different family: `(Ω+π, a_phot → −a_phot)`
is an exact invariance — the Klein-4 `(+,−)` row of §4.3, and the
irreducible source of the β branch ambiguity in §6.5.

When amplitudes are decoded into elements, `ti_to_kepler`
(`elements.py:468`) returns one representative of the pair, from the two
`atan2` combinations of `(A, B, F, G)`; `to_nss_convention`
(`elements.py:656`) is where the choice **Ω ∈ [0, π)** is imposed, with
the compensating π shift on ω, to match the Gaia NSS wrap. The amplitudes
are unchanged by either choice, so a seed composed from decoded elements
and one composed from the amplitudes are the same orbit.

### 3.8 Mass is not in the Thiele-Innes model

The TI forward model consumes `A, B, F, G`; `m₂` and `M` never enter it
(verified bit-exactly: Δ log L = 0 across 0.01 → 33 M☉ for any mass
attached to a TI fit). Sampling masses beside the amplitudes therefore
returns their priors unchanged, however wide or narrow — an 11 M☉ black
hole under a solar-mass total-mass prior comes back at 1 M☉ regardless of
the photocentre. The data-measured mass quantity is

```
fm_ast = a_phot³ / P_yr²  = m₂³/(m₁+m₂)²     under β = 0     [M☉]
```

(`mass_function_au`, `interpret/astrometric_upper_limit.py:123`; §6.3),
with no inclination assumption and `m₂ > fm_ast` for any `m₁ > 0`. A
companion mass comes from the interpretation layer with an EXTERNAL
primary mass (§6.4), never from the fit.

`ti_to_kepler` builds the mass-scaled `a_rel_au` and a Kepler-III period
**only** when handed mass keys beside the amplitudes (`elements.py:481`).
Do not hand it any: a "semi-major axis" so built is the prior, not a
measurement.

**The data-bearing quantities of a TI fit**: `a_phot_mas` (angular,
straight from A,B,F,G), `a_phot_au` (that ÷ parallax), the sampled `P`,
and `fm_ast`. Nothing else in it is mass-like.

### 3.8a What each parameterisation fits — sampled vs derived

Same data, same likelihood surface, two parameterisations. The data only
ever constrain the mas-level photocentre amplitudes — the combination
`fm_ast = m₂³/(m₁+m₂)²` (times parallax). The two differ in *where* the
interpretation happens:

**Thiele-Innes — measurement outside-in:**

    sampled:  P, e, τ,  A, B, F, G [mas],  ϖ, offsets, PM, jitter
              └─ shape ─┘ └─ four sky-amplitude coefficients ─┘

    derived (after the fit):  a_phot (mas, geometric combination of A,B,F,G);
                              a_phot/ϖ (AU);  fm_ast = (a_phot/ϖ)³ / P_yr²;
                              the angles ω, i, Ω decoded from A,B,F,G.
    mass:                     nowhere in the model — the interpretation
                              layer only (§3.8, §6).

The amplitudes ARE the fitted quantities; everything physical is decoded
afterwards.

**Campbell — model inside-out:**

    sampled:  P, e, τ,  ω, i, Ω,  a_phot,  ϖ, offsets, PM, jitter     (rung 3, §4.4)
        or:   P, e, τ,  ω, i, Ω,  m₂, M_total,  ϖ, …                  (campbell_xy, §3.2)

    computed INSIDE the forward model, every likelihood call:
        with masses:  a_rel (AU) = (P_yr² · M_total)^⅓        ← Kepler III
                      a_phot (AU) = a_rel · m₂/M_total         ← dark companion, β = 0
        A, B, F, G  = a_phot(mas) × angle factors  → compared to data

Campbell samples ELEMENTS; the amplitudes are derived on the way to the
data. Sampling `a_phot` directly (the joint rungs) keeps the interpretation
outside the model as TI does; sampling masses puts it inside, and then
`m₂` and `M_total` are constrained by the data ONLY through the `fm_ast`
combination — the orthogonal mass direction is filled by the priors. Both
parameterisations measure the same thing; choose by where you want the
assumption to live, and say which you chose.

### 3.9 Recommended practice — search, seed, sample

A sampler over all fourteen astrometric parameters is a **refiner, not a
searcher**. Started cold (walkers drawn from the priors) it cannot find the
right basin on hard cases — measured, not argued — even though the same
likelihood has one clear peak at the truth. The linearised path finds that
peak cheaply. The recommended practice is therefore three explicit steps,
all made by you:

```python
peaks  = scan_ti_frequency(t_mjd, psi, pf, d_obs, sigma, ..., top_k=6)   # 1. find: 3-D grid + closed-form amplitudes
seed   = compose_ti_seed(peaks[0], epoch_ref_mjd=EPOCH_REF)             # 2. translate the peak into a plain dict
run    = run_emcee_chains(log_post, ..., draw_init_fn=from_seed(seed))  # 3. refine: your posterior, started there
```

**Times.** `scan_ti_frequency` takes MJD, while `obs_time` may hold either
MJD or days-from-J2010. Convert with `resolve_epochs_mjd` (§1.2), never by
hand, so the seed and the likelihood agree on the clock.

**What the seed is** (`compose_ti_seed`, `search.py:219`). A plain dict of
named, unit-suffixed physical quantities (`P_yr`, `e`, `A_mas` …
`plx_mas`, `pmra_masyr`, `tp_mjd`, `epoch_ref_mjd`; `astro_jitter_mas`
optional). Print it, edit any entry, build it by hand without a scan, or
swap in `peaks[1]`. The scan's `peak.beta` is the canonical amplitude
source; a fit's decoded elements are equally usable (§3.7).

**What the seed does — and only that.** It decides where the walkers
start. It never enters the likelihood, a prior, a bound or a fixed value;
the posterior is byte-identical with and without it, and the walkers are
free to leave it if the data pull elsewhere. **A seed is initialisation,
never a prior.**

**The choices, all visible.** `run_emcee_chains` (`sampling.py:110`)
starts every walker of a chain from `draw_init_fn(rng)`, or — with
`centres=` — spreads them around one centre per chain by a width `sigma`
you set. Put rival peaks from `top_k` on different chains and the
per-chain R-hat (`rhat_per_param`, `sampling.py:64`) reports their
disagreement as signal; put both inclination hemispheres on different
chains (`mirror_inclination`, `elements.py:919`) for the same reason.
Fix the spread width **before** looking at any result. There is
deliberately **no automatic mode**: nothing runs the scan for you, so the
basin you seed is always your decision, on the record.

**Honesty notes.** (i) A seeded fit is **mode-conditional** — it describes
the solution near where the chains started, and `R̂ ≈ 1` certifies
within-basin agreement only; the per-chain list is the probe for rivals.
(ii) The seed's `epoch_ref_mjd` must equal the likelihood's: the seed's τ
is defined at its own epoch, and a mismatch silently shifts the phase.
(iii) A seed outside the prior support starts the walkers at `−inf`; check
it against the prior before sampling rather than after.

**The RV twin.** A sampler over all RV parameters has the same finding
problem, measured on two synthetic cases: on an eccentric, sparsely
sampled orbit every cold start collapses to "no orbit" (`K → 0`); on a
near-circular one a start can land at `e ≈ 0`, where ω and τ are not
separately determined (only their sum is) and the walkers wander that
degenerate direction instead of climbing. Both are finding-step failures,
and the same hand-off removes them:

```python
ql     = <rung-1 sampling of (P, e, τ) with linear_solve_rv inside log_post>   # 1. find  (fit_rv_orbit.ipynb)
seed   = compose_rv_seed(ql, prepared, epoch_ref_mjd=EPOCH_REF)               # 2. translate: medians of (P, e, τ) + one closed-form (γ, K, ω)
run    = run_emcee_chains(log_post_all, ..., draw_init_fn=from_seed(seed))    # 3. refine, jitter freed  (fit_rv_orbit_all_parameters.ipynb)
```

`compose_rv_seed` (`solve/rv.py:647`) returns `P_yr` (years — the
sampler's own key; the linear core and `best_linear_params_rv` take days),
`e`, `omega_rad` (primary frame), `tau`, `K_kms`, `offset_kms`,
`epoch_ref_mjd`; `jitter_kms` optional (no quick-look source → each chain
draws its start from the jitter prior). It RAISES when the quick-look's
recorded epoch differs from the one you pass. On a near-circular orbit the
honest posterior is broad in ω and τ individually — that is the physics,
not a sampler fault.

---

## 4. The joint channel

One orbit, seen two ways. The two channels share the Keplerian shape
`(P, e, τ)` and, when the model says so, the orientation `(ω, Ω, i)`; each
channel keeps its own amplitudes. How much of that is *shared in the model*
versus *checked afterwards* is the only thing that changes between the three
rungs below, and it is the whole design. `notebooks/fitting_examples/
fit_joint_orbit.ipynb` walks all three on one synthetic orbit.

| rung | sampled | solved in closed form | couples |
| --- | --- | --- | --- |
| 1 — shared shape | `(P, e, τ)` — 3 | RV: `(γ, K cos ω, K sin ω)`; astro: `(A, B, F, G)` + 5 nuisances | shape only |
| 2 — shared angles | `(P, e, τ, ω, Ω, cos i)` — 6 | RV: `(γ, K)`; astro: `a_phot` + 5 nuisances | shape + orientation |
| 3 — everything | all 14 (+ jitters) | nothing | everything |

**The shared epoch is load-bearing.** Both channels use the same reference
epoch and the same periastron phase, `t_p = τ P + t_ref`, in the RV model
and in the along-scan model. That is what makes `τ` one parameter rather
than two that happen to share a name.

### 4.1 Rungs 1 and 2: the evidence factorises

Because the two amplitude blocks are **disjoint** (`K` is spectroscopic,
`a_phot` is astrometric), the joint marginal evidence at a fixed set of
sampled parameters is exactly the sum of the two channels' evidences —
no approximation, no inner coupling step, the prior counted **once**:

```
log target(θ) = log p(θ) + log Z_rv(θ) + log Z_ast(θ)
```

Each `log Z` is the §2.4 / §3.5 closed form — `linear_solve_rv`
(`solve/rv.py:175`) and `linear_solve_ti` (`solve/astrometry.py:315`), which
return it as `logL_marginal` — evaluated on whichever design matrix the rung
leaves linear.

At **rung 1** the designs are the full ones of §2 and §3: `rv_design_matrix`
(`design/columns.py:158`), three columns, and `ti_design_matrix`
(`design/columns.py:372`), nine.

At **rung 2** the orientation is sampled, so what is left linear shrinks.
`reduced_rv_design` (`design/columns.py:216`) has two columns, because `ω`
is no longer the RV solve's to choose; `reduced_astro_design`
(`design/columns.py:246`) has six, because with `(ω, Ω, i)` fixed the four
Thiele-Innes constants are all one scale times a known direction:

```
RV    β = (γ, K)                     X = [ 1 ,  cos(ν+ω) + e cos ω ]
ASTRO β = (a_phot, Δα*₀, Δδ₀, μ_α*, μ_δ, ϖ)
      orbit column = (B_u X + G_u Y) sin ψ + (A_u X + F_u Y) cos ψ
```

with `A_u…G_u` the **unit** Thiele-Innes constants, `_ti_constants_unit`
(`model.py:420`): the geometry fixes the *direction* of the TI ray and one
amplitude scales it.

### 4.2 What rung 2 shares — and what it does not

| | RV design | astro design |
| --- | --- | --- |
| `P`, `e`, `τ`, **`ω`** | yes | yes |
| `Ω`, `cos i` | **no** | yes |
| amplitude | `K` | `a_phot` |

⚠️ **Inclination does not enter the RV channel at rungs 1 or 2, and there
is no `a_phot sin i = a₁` constraint in the target.** The gain of rung 2
over rung 1 is that the four free TI constants collapse to a single ray
fixed by the shared geometry, and that `ω` is one number — *not* that the
amplitude/inclination tie is enforced. Inclination is therefore constrained
only by the astrometric epoch *shape*, which is deliberately weak: the
safe, falsifiable choice for compact-object claims.

The tie is instead surfaced **afterwards** as a deficit, all axes in AU,
`P` in Keplerian years:

```
a_spec = K · P · √(1−e²) / 2π            a₁ = a_spec / sin i
a_phot = a_phot_mas / ϖ_mas              D  = 1 − a_phot / a₁
```

`D → 0` for a dark companion. Formed per posterior draw, so the channels'
shared-shape correlation cancels; the notebook's co-validation section does
exactly this.

The two mass functions that go with it (`model.py:66`):

```
fm_spec = K³ P (1−e²)^{3/2} / 2πG   =  (m₂ sin i)³ / M²
fm_ast  = a_phot³ / P_yr²           →  m₂³ / M²      (only if β = 0)
```

so under a dark companion `fm_ast / fm_spec = 1 / sin³i`. That ratio is the
quantity a compact-object argument leans on — and it is an *independent
diagnostic*, not a fitted constraint. Reading it as evidence when β > 0 (a
luminous companion suppresses `a_phot`) inverts the conclusion.

### 4.3 The gauge and the mirror

Two discrete symmetries, and they are different in kind.

**The sign gauge is the same orbit.** `(a_phot, ω, Ω)` and
`(−a_phot, ω + π, Ω + π)` predict the same along-scan track, and likewise
`(K, ω)` and `(−K, ω + π)` the same RV curve. At rung 2 the amplitudes are
solved with a sign, so the target is exactly invariant under the four sign
images of `(ω, Ω)` — a Klein-4 group:

```
(+K, +a) → identity
(−K, −a) → ω += π            , negate K and a_phot
(+K, −a) → Ω += π            , negate a_phot
(−K, +a) → ω += π , Ω += π   , negate K
```

Fold on the recovered amplitude signs when summarising — `to_nss_convention`
(`elements.py:656`) folds `Ω` into `[0, π)` with the compensating shift of
`ω`, and the notebook does the same by hand — or, at rung 3, impose
`a_phot > 0` and `K > 0` as priors and the gauge is gone before sampling
starts. In the weak-signal limit (`K̂ ≈ 0` or `â_phot ≈ 0`) the
representative is ambiguous and the corresponding angle is genuinely
unconstrained.

**The mirror is a different orbit.** `(i, ω, Ω)` and `(π − i, …)` give the
same along-scan prediction but are two geometries; `mirror_inclination`
(`elements.py:919`) writes down the partner. A sampler started in one basin
stays there. Seed one chain per hemisphere, or read the per-chain R-hat on
`cos i`, and let the RVs break the tie where they can.

### 4.4 Rung 3: everything sampled

Nothing is marginalised. The forward models are the **same two reduced
designs** of rung 2 multiplied by sampled amplitudes — `X_ast · β_ast` and
`X_rv · β_rv` — and the likelihood is the plain Gaussian of the residuals in
each channel, §2.3 and §3.4 side by side. What changes between the rungs is
only the division of labour between algebra and sampler; the physics is the
same object throughout.

Rung 3 is where three things become possible that the closed-form rungs
cannot give: priors on the amplitudes (positivity, an external parallax),
their correlations with the geometry measured rather than assumed Gaussian
at a fixed shape, and non-Gaussian noise terms. It is also where a cold
start wanders — fourteen dimensions is exactly the regime for it — so it
starts from rung 2's posterior.

**The fully tied model.** `loglike_joint` (`likelihood.py:239`) goes one
step further than the rung-3 notebook: it shares the inclination *into the
RV channel*, scaling `K` by `sin i`, and takes the primary's orbit as the
photocentre orbit — `a_phot = a₁`, a dark companion hard-wired. On a
luminous pair the recovered companion mass is then biased low, and the
result cannot by itself establish that the companion is dark. Use it when
that assumption is the one you want to make, and say so.

### 4.5 Jitter

Rungs 1 and 2 hold **one fixed jitter per channel**, never sampled: the
posterior is *conditional* on those values, and a reduced-χ² far from one
is the sign that a fixed jitter is wrong. Rung 3 can sample both — in log
space, as `fit_rv_orbit_all_parameters.ipynb` shows for one channel — and
that is one of the reasons to climb to it.

### 4.6 The ladder, and the seed

Each rung starts from the one below it: rung 1 finds the shape and the
period-agreement of the two channels says whether they may share one at all
(a tension of several σ means a triple, a blend or systematics, and a joint
fit run past that point yields an uninterpretable `D`); rung 2 refines the
geometry from rung 1's median shape, with `ω` from the RV solve and
`(Ω, i)` from the Thiele-Innes inversion; rung 3 starts from rung 2's median
geometry with the amplitudes solved there. The composers for the single
channels are atoms too: `compose_rv_seed` (`solve/rv.py:647`) and
`compose_ti_seed` (`search.py:219`).

**A seed is initialisation, never a prior — also across rungs on the same
data.** It decides where the walkers start and carries no evidence; the
posterior at every rung is the posterior of that rung's model alone.

Any convenience fitter wrapped around this ladder is a packaging of the
same model; where a wrapper and the open recipe disagree, the open recipe
is the reference.

---

## 5. Priors, transformations and the sampler atoms

orblet never applies a prior for you. The posterior a sampler sees is the
function you compose — a likelihood from §2–§4 plus the priors below — and
nothing in the package overrides, widens or reparameterises what you wrote.
Read the cell that composes `log_post` in each example: that is the whole
prior.

### 5.1 Families

`priors.py`. A prior is any object with `logpdf(*args) -> float`
(`PriorDistribution`, `priors.py:145`) or a plain callable
`theta -> float` (`LogPriorCallable`, `priors.py:132`). All families return
a float and `−inf` outside support for finite input. **NaN input is
undefined and fails open**: the families test `if x < lo or x > hi`, which
is `False` for NaN, so they return a finite density. Never rely on a prior
to reject NaN; guard the likelihood.

| family | log-pdf | line |
| --- | --- | --- |
| `Uniform(lo,hi)` | `−log(hi−lo)`, inclusive both ends | `:195` |
| `Normal(μ,σ)` | `norm.logpdf` | `:217` |
| `truncated_Normal(μ,σ,lo,hi)` | `truncnorm.logpdf`, inclusive | `:251` |
| `LogUniform(lo,hi)` | `−log x − log(log hi − log lo)` | `:286` |
| `UniformInFrequencyPeriod(lo,hi)` | `−2 log x − log(1/lo − 1/hi)` | `:345` |
| `UniformCircular` | `−log π` on the unit disk | `:404` |
| `EccOmegaDisk` | `−log π` on the unit disk | `:456` |
| `CosUniformInclination` | `0` on `u ∈ [0,1]`; `i = arccos(1−2u)` | `:515` |

`UniformInFrequencyPeriod` is ∝ 1/P² and documented as **not**
uninformative — do not reach for it as a default. The families are
unit-agnostic: evaluate a period prior in the unit you sample in (§1.1).

### 5.2 Defaults, and the two footguns

Two helpers return prior specifications in the tuple dialect
`("LogUniform", lo, hi)` that `parse_prior_spec` (`priors.py:656`) turns
into the objects above:

- `default_companion_priors(period_days)` (`priors.py:537`):
  `P ~ LogUniform(P₀/3, 3P₀)` **days**, `e ~ Uniform(0, 0.99)`,
  `ω ~ UniformCircular`, `τ ~ Uniform(0, 1)`,
  `mass ~ LogUniform(1e-4, 5)` M☉.
- `default_system_priors()` (`priors.py:590`):
  `M ~ truncated_Normal(1.0, 0.2, lo = 0.1)` M☉ — TOTAL mass.

Both are defaults for a solar-type primary with an ordinary companion, and
each carries a footgun for compact-object work:

1. **`M` defaults to `truncated_Normal(1.0, 0.2, 0.1)` M☉** — silent for
   any non-solar-total system. Always override.
2. **`mass` is bounded above at 5 M☉** — excludes every black hole.
   Override; a posterior piled against the bound is the sign.

The examples use the **√e–ω disk** for `(e, ω)` (§5.3), which means
`e ~ Uniform(0, 1)` with no ceiling: an eccentricity ceiling tighter than 1
needs a different parameterisation, not a tighter prior on the disk. The
boundary `e = 1` itself is admitted by the disk and rejected only by the
likelihood (§8, item 2).

### 5.3 Transformations and Jacobians

A prior is stated on a physical quantity; a sampler moves in whatever
coordinate you give it. Every change of variable owes its Jacobian, and
these are the ones the examples use:

| sampled coordinate | physical quantity | log-Jacobian to add |
| --- | --- | --- |
| `p = ln P` | `P = eᵖ` | `+ p` |
| `f = 1/P` | `P = 1/f`, `f ≤ 0 → −inf` | `− 2 ln f` |
| `(h, k)` on the unit disk, `EccOmegaDiskPrior` (`priors.py:416`) | `e = h² + k²`, `ω = atan2(k, h)` | none — uniform on the disk IS `e ~ U(0,1) × ω ~ U(0, 2π)` |
| `(x, y)` on the unit disk, `UniformCircularPrior` (`priors.py:391`) | `ω` or `τ` from the angle; the radius is unidentifiable | none |
| `u ∈ [0, 1]`, `CosUniformInclinationPrior` (`priors.py:472`) | `i = arccos(1 − 2u)`, full sphere | none — `cos i` uniform IS isotropic |
| `ln s` (jitter) | `s = e^{ln s}` | `+ ln s` |

`UniformInFrequencyPeriodPrior` (`priors.py:297`) already carries the
`1/P²` density: pair it with a sampler that moves in `P`, not in `f`, or
the factor is counted twice.

### 5.4 Starting the walkers

`run_emcee_chains` (`sampling.py:110`) runs `n_chains` INDEPENDENT emcee
ensembles on any `log_post(vec) -> float` and returns a `ChainRun`
(`sampling.py:94`). Two ways to start a chain:

- `draw_init_fn(rng) -> vec`: every walker is one call. The natural
  choice is a draw from the priors (a cold start) or a small perturbation
  of a seed (§3.9).
- `centres=` (one vector per chain) with `init_walkers_fn` and `sigma`:
  the walkers are spread around each chain's centre by a width you set —
  rival peaks or both inclination hemispheres on different chains.

A finite `log_post` on every starting walker is preferred, not required:
emcee needs SPREAD starts, and a spreader that insists on finite ones
collapses the cloud on a tight prior. A chain that never reaches a finite
log-posterior is found AFTER the run, from its `per_chain_logposts`, and
handed to `stack_and_diagnose` as `dead_chains` — reported, not hidden.

Randomness, in this order — the reproducibility contract:
`default_rng(seed)` → `n_chains` integers (the chain seeds); then per chain
`default_rng(chain_seed)` for the starting cloud and `RandomState(chain_seed)`
for emcee's moves. The master generator is used for nothing else, so it
does not matter what the caller drew before the call. A rerun with the same
`seed` reproduces every sample.

### 5.5 Convergence and summaries

Chains are independent of each other; walkers inside one chain are not
(the stretch move couples them). Convergence statistics therefore compare
CHAINS:

| atom | what it computes | line |
| --- | --- | --- |
| `rhat_per_param` | classic Gelman–Rubin R̂ per parameter over `(n_chains, n_draws, n_params)`; no rank normalisation, no folding — feed it split halves | `sampling.py:64` |
| `stack_and_diagnose` | drops the warm-up (`int(round(discard_fraction · iterations))`; halves round to even), stacks the chains, computes R̂ and ESS over ALL chains and, when a chain is dead, the `*_alive` twins over the living ones | `sampling.py:249` |
| `Diagnostics` | the result: flat samples with row-aligned log-posteriors and log-likelihoods, the kept block per chain, `rhat`, `ess`, `rhat_alive`, `ess_alive` | `sampling.py:225` |
| `summarize_chains` | `{median, std, q16, q84}` per key, pooled over the live chains, plus the same statistics per chain (dead chains included and flagged, never dropped) | `sampling.py:486` |
| `chain_quantiles` | quantiles of one array at the levels you ask | `chain_stats.py:46` |
| `chain_credible_interval` | a central interval | `chain_stats.py:77` |
| `chain_circular_summary` | the circular mean and spread of an angle — use it for ω, Ω, τ | `chain_stats.py:109` |
| `chain_summary_table` | the pooled summary as a table; needs `pandas` (`orblet[tables]`), everything above does not | `chain_stats.py:153` |

The pooled R̂ deliberately keeps dead chains — their blow-up is the signal
— and the `*_alive` twins are the numbers the pooled summary actually
describes. Reading order on a suspicious fit: which chains died → `rhat`
against `rhat_alive` (did death hide a disagreement) → the per-chain
summaries (WHERE the chains sit) → the ESS (are the error bars trustworthy
at all). Short chains read a HIGH ESS: treat one below emcee's
`n_kept ≥ 50·τ̂` rule as optimistic.

---

## 6. From a fit to a mass

### 6.1 The companion-mass layer

`solve_companion_mass`, `interpret/companion_mass.py:223`. Inverts

```
f(m) = (m₂ sin i)³ / (m₁ + m₂)²
```

for `m₂` — **not** by a cubic formula but by a vectorised bisection on the
provably monotone `h(m₂)`: a doubling bracket search (`:288`) then a
**fixed 100 bisections with no tolerance test**. Invalid inputs
(`fm ≤ 0`, `sin i ≤ 0`, `m₁ ≤ 0`) return `NaN`; the function never raises.

`sin i` handling (`_resolve_sin_i`, `:373`):

- **`sin_i = 1.0` (default)** → *minimum* companion mass, sentinel
  `MASS_CONVENTION_PROJECTED`.
- a `CosUniformInclinationPrior` → isotropic draw `i = arccos(1−2u)`,
  sentinel `MASS_CONVENTION_TRUE`.
- ⚠️ a supplied float is **not range-checked** despite an error constant
  promising it: `sin_i = 0` silently yields all-NaN, `sin_i = 5` is
  accepted (§8, open item 1).

`companion_mass_from_rv_posterior` (`:437`) applies the solver per
posterior draw of an RV fit with an external primary-mass prior;
`04_interpretation_mass.ipynb` is the worked example.

### 6.2 The astrometric upper limit

`interpret/astrometric_upper_limit.py`, assuming a dark companion (β = 0,
so `a_phot = a₁`):

```
a₁ sin i = K P √(1−e²) / 2π          (`a1sini_au`, `:61`, in AU)
a_phot   = a_phot_mas / ϖ_mas        (`a_phot_au`, `:92`)
C        = a³ / P_yr²                (`mass_function_au`, `:123`)
```

The "limit" (`astrometric_upper_limit_au`, `:151`) is the 95th percentile
of the finite draws. It is explicitly an **RV-shape-conditioned credible
bound, not a detection upper limit** — `a_phot_mas` is a folded
non-negative magnitude carrying a positive noise floor, so the bound can
only be pushed upward. `consistency_verdict` (`:235`) is a heuristic, not a
calibrated test; `companion_mass_bracket` (`:183`) turns the bound into a
scalar `[m₂_min, m₂_max]` through §6.1.

### 6.3 Is Kepler III enforced?

| parameterisation | enforced? |
| --- | --- |
| RV forward model from masses | **yes** — `a = (P² M_tot)^{1/3}` (`model.py:135`) |
| Campbell with masses | **yes** — same relation (`model.py:515`); but enforcement ≠ measurement: astrometry constrains only `fm_ast`, so mass marginals are prior-conditioned along the degenerate direction |
| Campbell with `a_phot` (joint rungs 2–3) | **no** — `a_phot` is a free amplitude; the tie to the RV amplitude is read afterwards as the deficit `D` (§4.2) |
| **Thiele-Innes** | **no** — no mass enters; the mass statement is `fm_ast` (§3.8) |

### 6.4 The astrometric companion mass

`companion_mass_from_astrometric_posterior`
(`interpret/companion_mass.py:594`) turns a TI fit's `fm_ast` draws into a
conditional companion mass with an EXTERNAL primary-mass prior — the
astrometric twin of §6.1:

- **No `sin i` enters.** The TI amplitudes measured the inclination, so
  `fm_ast = m₂³/(m₁+m₂)²` is projection-free by construction; passing
  `sin_i = 1` into the shared solver is EXACT (it selects the
  projection-free form), not an edge-on assumption. Sentinel
  `MASS_CONVENTION_ASTROMETRIC_EXTERNAL_M1` (`constants.py:134`).
- **β = 0 conditional.** With companion light,
  `fm_ast = |m₂/M − β/(1+β)|³ · M`; the `beta` keyword is RESERVED and
  raises `NotImplementedError`, so a β = 0 answer can never be returned
  under a caller-stated β.
- **Fail-closed input guard.** A result carrying spectroscopic keys
  (`K_kms`, `gamma_kms`, …) is REJECTED — a joint fit constrains the mass
  split without an external m₁, and feeding one in would silently
  override measured information. Only recognisable TI results are
  accepted.
- The result is CONDITIONAL on the m₁ prior and on β = 0; NaN draws
  (screened parallax or amplitude) propagate with nan-aware summaries.

### 6.5 The flux ratio, from a joint fit

`beta_from_joint_posterior` (`interpret/flux_ratio.py:392`). A joint fit
measures the photocentre amplitude `a_phot` and, through `K` and the shared
inclination, the primary's `a₁`; their folded ratio `r` per draw is
inverted into BOTH flux-ratio branches — the sign gauge of §4.3 is exactly
why the branch cannot be chosen here. The mass ratio `q` it needs comes
from `fm_spec` (§6.1, with the posterior's MEASURED per-draw `sin i`,
`MeasuredSinI`, `:224`, and an external primary mass): `fm_spec` is β-free,
because β scales the photocentre amplitude but not the ellipse shape or
the inclination. The astrometric adapter (§6.4) is FORBIDDEN as a `q`
source — it inverts `fm_ast` under β = 0, so using it would assume β = 0
in order to infer β. An astrometry-only result is refused. Nothing is
fitted and no sampler is touched.

---

## 7. Assumptions and limitations

**Physical assumptions baked into the models**

1. **The companion is dark (β = 0)** wherever a photocentre is converted
   to a mass. `a_phot = (m₂/M_tot)·a_rel·ϖ` assumes the companion
   contributes no light. A luminous companion makes `a_phot < a₁` and
   biases every derived mass. No forward model carries a flux-ratio
   parameter; the joint rungs avoid the assumption inside the fit
   (`a_phot` is a free amplitude), but β = 0 returns in the interpretation
   of `fm_ast` and `D`, and §6.5 is the only place β is inferred.
2. **`sin i = 1` in the RV path** unless the caller supplies `sin i` (a
   number, or a joint fit's per-draw values) ⇒ minimum masses.
3. **Single Keplerian orbit.** No third body, no acceleration term, no
   perspective acceleration.
4. **The nine-column solve wants an informative parallax prior** — normally
   the catalogue value. For a genuine astrometric binary that catalogue
   parallax was itself perturbed by the orbit now being fitted, so the
   prior is not fully external. It propagates directly:
   `a_phot[AU] = a_phot_mas / ϖ`, hence into `fm_ast`, `D` and every
   derived mass. `ϖ` is solved or sampled rather than fixed, but is
   prior-dominated on a short baseline. Treat a mass whose error budget is
   parallax-dominated as *conditional on the catalogue solution*, and check
   it against an independent distance where one exists.

**Statistical limitations**

5. **Per-CCD independence** (§3.4) on Gaia data. Formal errors and χ²/dof
   are optimistic; significance tests on such a stream are uncalibrated.
6. **Gaussian errors with a single global jitter, and no outlier model.**
   Whatever filtering happened before the arrays reached orblet conditions
   every χ²/dof statement; inside the likelihood there is no heavy tail,
   so one corrupted point enters at full 1/σ² weight.
7. **Fixed jitter at the linearised rungs** ⇒ posteriors are conditional
   on it; a mis-set value biases amplitudes, hence `K`, `a_phot` and every
   derived mass. `reduced_chi2_band` (`search.py:393`) is the only guard;
   rung 3 is where jitter is sampled (§4.5).
8. **The TI parameterisation carries no mass information beyond `fm_ast`**
   (§3.8): a TI fit alone cannot give `m₂` without an external `m₁`.
9. **The inclination mirror `i ↔ π−i` is a distinct orbit and nothing
   detects it for you.** Seed one chain per hemisphere
   (`mirror_inclination`, §4.3) and read the per-chain R̂; where the RVs
   cannot break the tie, report both.
10. **Nothing here is a detection statistic.** The scan ranks, the
    evidences compare shapes at one model, and neither computes a
    false-alarm probability (§3.6); the scramble null is the honest
    substitute.

**Numerical**

11. Kepler–Newton divergence at `e ≳ 0.96` is repaired by bisection
    (§1.3).
12. The mass bisection runs a **fixed** 100 iterations with no convergence
    test.
13. Parallax is **not** clipped to positivity by the linear solves;
    `plx_mas` may be ≤ 0, and only a caller's `a_phot_au` / `D` gating on
    `ϖ > 0` protects what is derived from it (the scan's geometry screen
    does this for peaks, §3.6).

---

## 8. Open items

Documentation and behaviour gaps found by reading the implementation against
its docstrings, not yet closed. Everything that was found and has since been
fixed is out of this list; the code and the tests are the record.

1. **`solve_companion_mass` does not validate `sin_i`** despite an error
   constant promising it, and its docstring says "~60 iterations" where the
   loop runs 100.
2. **The `e` boundary**: the disk priors reject `h²+k² > 1`, so `e = 1`
   exactly is admitted by the prior and caught only by the likelihood.
3. **`companion_mass_from_rv_posterior` labels any fixed `sin i ≠ 1` as
   `MASS_CONVENTION_TRUE`** (`companion_mass.py:542`) — so a mass computed
   from an *assumed* inclination carries a sentinel that reads as
   "measured"; only `sin i = 1` gets `PROJECTED`. The `sin_i` parameter
   docstring says so and names the result's `sin_i_mode` as the
   discriminator. The sentinel semantics themselves are unchanged: a third
   sentinel is a convention-surface change, filed as a follow-up.

---

## 9. Index — where each piece lives

| quantity | computed in | shown in |
| --- | --- | --- |
| RV curve | `rv_curve`, `model.py:197` | `01_forward_models_and_simulation`, `fit_rv_orbit*` |
| RV design matrix | `rv_design_matrix`, `design/columns.py:158` | `02_design_matrices_and_linear_solves` |
| RV likelihood | `loglike`, `likelihood.py:93` | `fit_rv_orbit_all_parameters` |
| RV marginal evidence | `linear_solve_rv`, `solve/rv.py:175` | `02_design_matrices_and_linear_solves`, `fit_rv_orbit` |
| mass function | `fm_from_K`, `model.py:63` | `04_interpretation_mass` |
| along-scan model | `along_scan_model`, `model.py:580` | `01_forward_models_and_simulation`, `fit_astrometric_orbit` |
| Thiele-Innes constants | `campbell_xy`, `model.py:474` (`_ti_constants_unit`, `:420`) | `fit_joint_orbit` |
| TI design matrix | `ti_design_matrix`, `design/columns.py:372` | `02_design_matrices_and_linear_solves` |
| single-star solve | `fit_astrometric_5param`, `solve/astrometry.py:926` | `02_design_matrices_and_linear_solves`, `03_period_search` |
| astro likelihood | `loglike_along_scan`, `likelihood.py:182` | `fit_joint_orbit` |
| TI marginal evidence | `linear_solve_ti`, `solve/astrometry.py:315` | `02_design_matrices_and_linear_solves`, `fit_astrometric_orbit` |
| reduced joint designs | `reduced_rv_design`, `design/columns.py:216`; `reduced_astro_design`, `:246` | `fit_joint_orbit` |
| frequency scan | `scan_ti_frequency`, `search.py:425` | `03_period_search`, `fit_astrometric_orbit` |
| seeds | `compose_ti_seed`, `search.py:219`; `compose_rv_seed`, `solve/rv.py:647` | `fit_astrometric_orbit`, `fit_rv_orbit` |
| chains | `run_emcee_chains`, `sampling.py:110`; `rhat_per_param`, `:64` | `fit_rv_orbit_all_parameters`, `fit_joint_orbit` |
| companion mass | `solve_companion_mass`, `interpret/companion_mass.py:223` | `04_interpretation_mass` |
| astrometric upper limit | `astrometric_upper_limit_au`, `interpret/astrometric_upper_limit.py:151` | `04_interpretation_mass` |
| prior families | `priors.py` | all |

---

*Maintenance: this page states behaviour, so it goes stale when the code
changes. Any change that touches a forward model, a likelihood, a prior
default or a public contract must update the affected section in the same
commit. The §8 list should shrink, not grow.*
