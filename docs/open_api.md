# The open surface — what you can build an analysis from

**Contract.** A name is on the open surface when it imports from the path
shown, its docstring states units and assumptions, and a test covers it.
The rule behind it: a user should be able to build
their own fitting routine from these pieces; the end-to-end solvers are
conveniences, kept and sub-priority.

**Reading order.** The quickstarts under `notebooks/quickstart/` show one
capability per cell on synthetic data; the open trio under
`notebooks/fitting_examples/` builds a whole fit from these pieces; the
solvers come last. Conventions that every row relies on: epochs in TCB
MJD (`resolve_epochs_mjd` converts the loader's J2010-days), angles in
radians, scan angle ψ counter-clockwise from north, Keplerian year =
`DAYS_PER_KEPLER_YEAR` days, the RV sign and ω conventions of
`docs/model_and_likelihoods.md` (§1 and §2).

**Not on the surface, on purpose:** the resolved-pair forward model and
the design-builder protocol (no user yet), the MAP-sample `simulate`
builders (solver layer), the bulk presearch driver, the Holl+2023 noise
law (imaging stack, paper pending), and `ms_flux_ratio` (a shipped
placeholder; supply your own flux-ratio relation to the AMRF functions).

## Forward models — `notebooks/quickstart/01_forward_models_and_simulation.ipynb`

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `rv_model` | `orblet.model` | primary radial velocity at epochs | km/s; P in Keplerian years; masses M☉ | `mass_msun` is the PROJECTED companion mass m₂ sin i; single instrument; γ added |
| `along_scan_model` | `orblet.model` | project (Δα*, Δδ) plus the 5-parameter model onto the scan | mas; ψ rad; PM mas/yr | consumes Gaia's bundled `parallax_factor_al`; PM linear about `epoch_ref_mjd` |
| `thiele_innes_xy` | `orblet.model` | photocentre (Δα*, Δδ) from A, B, F, G | mas | amplitudes are photocentre amplitudes, already × parallax |
| `campbell_xy` | `orblet.model` | photocentre (Δα*, Δδ) from Campbell elements and masses | mas; masses M☉; plx mas | **β = 0**: the primary's orbit a₁ is taken as the photocentre |
| `kepler_xy_orbit` | `orblet.model` | in-plane (x, y) track, normalised by a | dimensionless | Kepler solve inside |
| `tp_from_disk_angle` | `orblet.model` | periastron time from a unit-disk phase latent | MJD; P days | only the angle of (x, y) is identifiable |
| `along_scan_from_theta` | `orblet.model` | the engine's own along-scan model from a flat θ dict | mas | θ keys per basis; P in DAYS in θ |
| `rv_loglike` | `orblet.likelihood` | Gaussian log-likelihood with jitter, RV channel | nats; jitter km/s | σ_eff² = σ² + jitter²; epochs independent |
| `loglike_along_scan` | `orblet.likelihood` | Gaussian log-likelihood with jitter, along-scan channel | nats; jitter mas | per-epoch independence (no per-CCD correlation) |

## Simulation — same quickstart

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `per_direction_parallax_factors` | `orblet` | (F_α*, F_δ) from a solar-system ephemeris | dimensionless; epochs TCB MJD | observer = Gaia at L2 by default (0.18 % vs Gaia's own factor, measured on BH3; `observer="geocentre"` is 1 % low); ephemeris = astropy's built-in (no download); `ephemeris="de432s"` needs the `[ephemeris]` extra and `orblet.parallax.fetch_ephemeris` once |
| `along_scan_parallax_factor` | `orblet` | the along-scan projection F_α* sin ψ + F_δ cos ψ | dimensionless; epochs TCB MJD, ψ rad | same defaults; the demo cadence is built on it |
| `OBSERVER_GEOCENTRE`, `OBSERVER_L2` | `orblet` | the two observer options | — | — |
| `L2_OFFSET_AU` | `orblet` | Sun–Earth L2 distance beyond the geocentre | AU (0.010003) | Hill estimate; Gaia's Lissajous orbit not modelled |

## Design matrices and linear solves — `02_design_matrices_and_linear_solves.ipynb`

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `rv_design_matrix` | `orblet` | columns [1, cos ν + e, −sin ν] at a fixed (P, e, τ) | P Keplerian years | linear in (γ, K cos ω, K sin ω) |
| `linear_solve_rv` | `orblet` | generalised least squares for the three RV amplitudes | km/s | `logL_marginal` is the flat-prior marginal likelihood from the data alone, on every path (None if the data-only matrix is singular); `log_evidence` is the prior-weighted integral, only with a prior; sum the same kind across channels, never mix them |
| `best_linear_params_rv` | `orblet` | the RV solve at a shape, with optional covariance-correct draws | P days | jitter fixed |
| `recover_K`, `recover_omega` | `orblet` | K = √(C² + S²), ω = atan2(S, C) | km/s, rad | primary-frame ω |
| `semi_amplitude_kms` | `orblet` | K for given masses and period, edge-on | km/s | multiply by sin i |
| `ti_design_matrix` | `orblet` | nine columns [A, B, F, G, Δα*, Δδ, μα*, μδ, ϖ] at a fixed shape | mas | uses the bundled parallax factor |
| `linear_solve_ti` | `orblet` | generalised least squares for the nine amplitudes | mas | same three-number contract as the RV solve |
| `best_linear_params_ti` | `orblet` | the TI solve at a shape, with optional draws | P days | jitter fixed |
| `ti_amplitude_chains` | `orblet` | name the β columns as a chains dict for `ti_to_kepler` | mas | — |
| `astrometric_5param_design_matrix`, `fit_astrometric_5param` | `orblet` | the standard no-orbit Gaia model and its closed-form fit | mas, mas/yr | no orbit, no jitter, no priors |
| `acceleration_columns` | `orblet` | the two sky-plane acceleration columns (½ t² convention) | mas/yr² | same `epoch_ref_mjd` as the PM columns (load-bearing) |

## Period search — `03_period_search.ipynb`

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `scan_ti_frequency` | `orblet` | 3-D (f, e, τ) linear-TI scan, ranked peaks (observables only: shape node, score, amplitudes, `a_phot_au`, `plx_mas ± plx_sigma_mas`; no mass) | cycles/day, mas | marginal over amplitudes, PROFILED over (e, τ) — eccentric nodes favoured; flags only the 1-yr line, its 6-month harmonic and a non-positive parallax; a small positive parallax is shown, not flagged; no jitter; no false-alarm probability; it is the starting point of the seeded refiner, not a fit |
| `compute_lomb_scargle_periodogram` | `orblet.periodogram` | Lomb–Scargle on epoch RVs | 1/day | sinusoid model |
| `compute_pdc_periodogram` | `orblet.periodogram` | phase-distance-correlation periodogram from a distance matrix | days | needs n ≳ 20 |
| `scalar_distance_matrix`, `astrometric_segment_distance_matrix`, `scan_angle_distance_matrix` | `orblet.periodogram` | distance kernels for scalar, 1-D scan and scan-angle data | — | the scan-angle kernel treats ψ mod π; the segment one is a dissimilarity whose `L_mas` (default 0.01) must sit well below the residuals |
| `scan_angle_coupling` | `orblet.periodogram` | scan-angle coupling D_cpl of one source, no period scan | — | input is residuals after the 5-parameter model; U-centred, can be negative |
| `pdc_false_alarm_probability` | `orblet.periodogram` | χ² false-alarm probability of a PDC score | — | pointwise (per trial period), no look-elsewhere correction |
| `peak_fwhm_days` | `orblet.periodogram` | width of the tallest peak | days | — |

## Interpretation — `04_interpretation_mass.ipynb`

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `ti_to_kepler` | `orblet` | A, B, F, G → a_phot, i, ω, Ω | mas, AU, rad | i ∈ [0, π]; the i ↔ π − i mirror is not resolved |
| `to_nss_convention` | `orblet` | Gaia DR3 NSS-convention quantities from a chains dict | as NSS | see the docstring for the sign convention |
| `a1_sini_from_rv_chain`, `astrometric_priors_from_rv_chain`, `predicted_k_kms_from_astrometric_chain` | `orblet` | RV ↔ astrometry bridges | AU, km/s | truncated-normal widening; circular wrap for angles |
| `solve_companion_mass` | `orblet.interpret.companion_mass` | m₂ from the mass function, m₁ and sin i | M☉ | cubic solve; m₁ ASSUMED |
| `companion_mass_from_rv_posterior` | `orblet.interpret.companion_mass` | m₂ draws from `fm_spec_msun` draws | M☉ | two modes: `sin_i = 1` → minimum mass; your own `sin_i` (number or per-draw array) → mass, only as good as that `sin_i`; never drawn here |
| `companion_mass_from_astrometric_posterior` | `orblet.interpret.companion_mass` | m₂ draws from `fm_ast_msun` draws | M☉ | **β = 0** |
| `a1sini_au`, `a_phot_au`, `mass_function_au`, `astrometric_upper_limit_au`, `companion_mass_bracket`, `consistency_verdict` | `orblet.interpret.astrometric_upper_limit` | the observables-first mass ladder and the non-detection bound | AU, days, M☉ | β = 0 ⇒ a_phot = a₁ |
| `MeasuredSinI` | `orblet.interpret.flux_ratio` | row-aligned measured sin i carrier for the mass layer | — | draws must match the mass-function draws one to one |
| `signed_photocentre_axis_ratio`, `beta_branches_from_axis_ratio` | `orblet.interpret.flux_ratio` | a_phot / a₁ for (q, β) and its two-branch inverse | — | mirror branch degeneracy |
| `amrf`, `amrf_curve`, `ms_amrf_boundary`, `triple_amrf_boundary`, `classify_amrf`, `validate_beta_relation` | `orblet.interpret.amrf` | the astrometric mass-ratio function and its triage | — | pass your own `beta_relation(q, m1)`; the shipped MS relation is a placeholder |
| `chain_quantiles`, `chain_credible_interval`, `chain_circular_summary`, `chain_summary_table` | `orblet.chain_stats` | posterior summaries | as the chain | flat post-burn-in chains; circular summary via the resultant vector |
| `UniformInFrequencyPeriodPrior`, `EccOmegaDiskPrior`, `CosUniformInclinationPrior`, `LogUniformPrior`, `TruncatedNormalPrior`, `NormalPrior` | `orblet.priors` | the prior family the engines use, callable by you | physics units | the frequency prior favours short periods (∝ 1/P²) |
| `default_companion_priors`, `default_system_priors` | `orblet.priors` | the engines' defaults | M☉, days | companion ceiling 5 M☉ (excludes black holes); total mass 1 M☉ — override |

## Quality control and preparation

| name | import from | what it does | units | assumptions |
|---|---|---|---|---|
| `prepare_rv_for_orbit` | `orblet` | validate, mask and convert epoch RVs | km/s, MJD | `time_scale="gaia_obmt"` for loader-shaped data; the default RAISES on J2010-days |
| `resolve_epochs_mjd` | `orblet` | astrometry `obs_time` → TCB MJD by the engines' own rule | MJD | auto-detects J2010-days vs MJD |
| `assert_rv_epochs_are_absolute_mjd` | `orblet.prepare` | guard against the two-convention trap | MJD | — |

## Seeds and post-processing — `notebooks/fitting_examples/`

| name | import from | what it does | note |
|---|---|---|---|
| `compose_rv_seed`, `compose_ti_seed`, `mirror_inclination` | `orblet` | the seed bridge between levels | a seed is initialisation, never a prior |
| `extract_orbital_elements`, `compute_rv_model_curve`, `compute_residuals` | `orblet` | post-processing of a sampler result | primary-frame ω required |

## Sampler atoms

The stages of a fit that know nothing about the orbit. **You write the
posterior** (your prior plus your likelihood — one addition); these run it
as independent, reproducible emcee chains and turn the chains — theirs, or
those of ANY emcee run of your own — into a warm-up discard,
convergence diagnostics and summary tables. WHERE a chain starts is yours
too: give `run_emcee_chains` one centre per chain, or none. The three
nonlinear samplers above are built from these same functions, and are
proven byte-identical to their pre-atoms selves by
`tests/test_nonlinear_noop_baselines.py`. **Atoms take their helpers as
arguments**: pass the project's statistics (second table) or your own.
Layout everywhere: samples `(iterations, walkers, parameters)` per chain —
what `emcee`'s `get_chain()` returns.

| name | import from | what it does | note |
|---|---|---|---|
| `run_emcee_chains` | `orblet.sampling` | run `n_chains` independent emcee ensembles on any `log_post(vec) -> float`; returns a `ChainRun` (samples, log-posteriors, acceptance, the chain seeds) | the ONLY atom that draws random numbers, in a fixed order: `default_rng(seed)` → one seed per chain → per chain `default_rng(chain_seed)` for the starting cloud and `RandomState(chain_seed)` for emcee's moves; same `seed` ⇒ same samples. Chains are independent, walkers inside a chain are not — so convergence compares CHAINS |
| `stack_and_diagnose` | `orblet.sampling` | flatten the chains in one order, drop the warm-up, rebuild the per-sample `loglike`, split the post-warm-up run into halves for R-hat, ESS, and the same over live chains | warm-up = `int(round(discard_fraction · iterations))` (halves round to even); on a run too short to split, R-hat falls back to the FULL un-burned chains; refuses arrays whose shape does not match the declared sizes |
| `convergence_columns` | `orblet.sampling` | write `r_hat_<key>` / `ess_<key>` for each exported quantity | a quantity built from several sampled parameters gets the LARGEST R-hat and SMALLEST ESS among them; modifies `chains` in place |
| `alive_sample_mask`, `summarize_chains` | `orblet.sampling` | median / std / 16–84 % per quantity over live chains, plus a per-chain table | dead chains stay in `chains` and in the per-chain table, never in the pooled summary; `nan_aware_keys` governs the pooled table only |
| `rhat_per_param` | `orblet.sampling` | classic Gelman–Rubin R-hat, `(chains, draws, parameters)` → one value per parameter | basic estimator: no rank normalisation, no folding (Vehtari et al. 2021); certifies agreement between the chains you STARTED, not the absence of a rival basin |

| helper (pass it, or call it) | import from | what it does | note |
|---|---|---|---|

## Plotting — `notebooks/fitting_examples/`

| name | import from | what it does |
|---|---|---|
| `plot_orbit_fit`, `plot_orbit_corner`, `plot_astrometric_orbit`, `plot_astrometric_sky_overlay` | `orblet` | phase-fold, corner, sky-plane and along-scan overlays (resolved lazily; importing the package does not start pyplot) |
