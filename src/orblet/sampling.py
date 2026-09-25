"""Shared sampler atoms: the stages of a fit that know nothing about the orbit.

A nonlinear orbit fit runs the same pipeline whatever the channel:
validate → resolve priors → build the posterior → RUN THE CHAINS → stack
and diagnose → export → summarise.  The stages in this module are the ones
that are the same code for every channel.  They take plain arrays and
callables, never an engine, so they can be used on their own: hand them
the chains of ANY emcee run and you get the project's discard / split-R-hat
/ ESS bookkeeping and its summary tables.

Design rule — helpers are ARGUMENTS.  An atom never looks a helper up by
name.  The caller passes the R-hat function, the ESS function and so on.
Two reasons: a user can substitute their own statistic without touching
library code; and the samplers pass the objects they resolve in THEIR OWN
module namespace, so a test that replaces a helper on a sampler module
still reaches the code that runs.

Randomness lives in ONE atom: :func:`run_emcee_chains`, which owns the
per-chain seeds, the starting clouds and emcee's random state (its
docstring states the order of the draws — the reproducibility contract).
Every other atom is deterministic.  No atom raises a warning OF ITS OWN:
each project warning is raised by the sampler that called the atom, so
moving a stage here cannot change which line it blames (a warning that
emcee itself raises during the run does surface from this module).
Choosing WHERE a chain starts
(a seed, a MAP search) is not an atom — it differs per channel and stays
in each sampler; the runner only takes the resulting ``centres``.

Conventions
-----------
``per_chain_samples[c]`` has shape ``(iterations, n_walkers, n_dim)`` — the
emcee layout (iteration-major) — and ``per_chain_logposts[c]`` has shape
``(iterations, n_walkers)``.  Both are flattened with the SAME C order, so
row ``i`` of ``flat_samples`` pairs with entry ``i`` of ``flat_logposts``
and ``flat_loglikes``.  ``discard_fraction`` is the warm-up fraction of the
ITERATIONS dropped from the exported samples and from R-hat.  Latent
vectors are dimensionless sampler coordinates; units enter only when a
channel decodes them.

Lazy imports
------------
Only :mod:`numpy` and the standard library at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "Diagnostics",
    "rhat_per_param",
    "ChainRun",
    "run_emcee_chains",
    "stack_and_diagnose",
    "convergence_columns",
    "alive_sample_mask",
    "summarize_chains",
]


def rhat_per_param(samples_chains: np.ndarray) -> np.ndarray:
    """Classic Gelman-Rubin R-hat, one value per parameter.

    ``samples_chains`` has shape ``(n_chains, n_draws, n_params)``; the
    chains must have equal length.  With between-chain variance
    ``B = n_draws · var(chain means)`` and mean within-chain variance
    ``W``, the pooled estimate is ``(n_draws − 1)/n_draws · W + B/n_draws``
    and R-hat is its ratio to ``W``, square-rooted (Gelman & Rubin 1992).
    Dimensionless; ≈ 1 when the chains agree.

    This is the BASIC estimator: no rank normalisation and no folding
    (Vehtari et al. 2021).  Feed it split halves — as
    :func:`stack_and_diagnose` does — to make it sensitive to a chain that
    is still drifting.  Returns NaN for every parameter when fewer than
    two chains are given, and NaN for a parameter whose within-chain
    variance is not positive.
    """
    n_chains, n_draws, n_params = samples_chains.shape
    if n_chains < 2:
        return np.full(n_params, np.nan)
    chain_mean = samples_chains.mean(axis=1)               # (n_chains, n_params)
    B = n_draws * np.var(chain_mean, axis=0, ddof=1)
    W = np.mean(np.var(samples_chains, axis=1, ddof=1), axis=0)
    var_hat = (n_draws - 1) / n_draws * W + B / n_draws
    safe_W = np.where(W > 0, W, np.nan)
    return np.sqrt(var_hat / safe_W)


# ``eq=False`` for the same reason as ``Diagnostics``: array fields.
@dataclass(frozen=True, eq=False)
class ChainRun:
    """What :func:`run_emcee_chains` returns: one entry per chain.

    ``per_chain_samples[c]`` is emcee's ``get_chain()`` for chain ``c`` —
    shape ``(iterations, n_walkers, n_dim)``; ``per_chain_logposts[c]`` its
    ``get_log_prob()`` — ``(iterations, n_walkers)``;
    ``per_chain_acceptance[c]`` the mean walker acceptance fraction;
    ``chain_seeds`` the integer seed each chain ran under.
    """

    per_chain_samples: list[np.ndarray]
    per_chain_logposts: list[np.ndarray]
    per_chain_acceptance: list[float]
    chain_seeds: np.ndarray


def run_emcee_chains(
    log_post: Callable[[np.ndarray], float],
    *,
    n_chains: int,
    n_walkers: int,
    n_dim: int,
    iterations: int,
    seed: int,
    draw_init_fn: Callable[[np.random.Generator], np.ndarray],
    centres: Sequence[np.ndarray] | None = None,
    sigma: np.ndarray | None = None,
    init_walkers_fn: Callable[..., np.ndarray] | None = None,
    engine_name: str = "sampler",
) -> ChainRun:
    """Run ``n_chains`` INDEPENDENT, reproducible emcee ensembles on any posterior.

    ``log_post`` is yours: any ``fn(vec) -> float`` (return ``-inf`` outside
    the support).  This atom supplies what is tedious and easy to get
    wrong: one independent seed per chain drawn from a master generator,
    a starting cloud per chain, and emcee's own random state pinned per
    chain so a rerun with the same ``seed`` reproduces every sample.

    Two ways to start a chain:

    - ``centres`` given (one ``(n_dim,)`` vector per chain): the walkers
      are spread around that chain's centre by ``init_walkers_fn(rng,
      map_vec=, n_walkers=, log_post_fn=, sigma=, engine_name=,
      prior_draw_fn=)``, which must return ``(n_walkers, n_dim)``.  A
      finite ``log_post`` on every row is PREFERRED, NOT REQUIRED: emcee
      needs SPREAD starts, not finite ones.  The project's spreader places
      a spread fallback draw even when it finds no finite one (on tight
      priors none may exist to be found), and a chain that never reaches
      a finite log-posterior is caught AFTER the run — see
      ``dead_chain_indices``.  Do not "fix" a spreader by demanding finite
      starts; that collapses the walkers.  ``sigma`` is the per-dimension
      width of the cloud, the SAME for every chain.
    - ``centres`` is ``None``: every walker is one call of
      ``draw_init_fn(rng)``.

    Randomness, in this order — it is the reproducibility contract:
    ``default_rng(seed)`` → ``n_chains`` integers (the chain seeds); then
    per chain ``default_rng(chain_seed)`` for the starting cloud, and
    ``RandomState(chain_seed)`` handed to emcee for the moves.  The master
    generator is used for nothing else, so it does not matter when, before
    this call, the caller drew its own random numbers.

    Chains are independent of each other; walkers inside one chain are
    not (the stretch move couples them).  Convergence statistics should
    therefore compare CHAINS — see :func:`stack_and_diagnose`.

    emcee is imported inside this function, so importing the module stays
    cheap.  Raises whatever ``init_walkers_fn`` or emcee raise; emits no
    warning of its own.
    """
    import emcee

    if centres is not None:
        if init_walkers_fn is None or sigma is None:
            raise ValueError(
                "run_emcee_chains: `centres=` needs `init_walkers_fn=` and "
                "`sigma=` (a spreader takes a centre and a width and returns "
                "one starting point per walker); "
                "pass centres=None to start every walker from draw_init_fn."
            )
        if len(centres) != int(n_chains):
            raise ValueError(
                f"run_emcee_chains: got {len(centres)} centres for "
                f"n_chains={int(n_chains)}; one centre per chain is required."
            )

    master_rng = np.random.default_rng(seed)
    chain_seeds = master_rng.integers(0, 2**31 - 1, size=int(n_chains))

    per_chain_samples = []
    per_chain_logposts = []
    per_chain_acceptance = []
    for c_idx, c_seed in enumerate(chain_seeds):
        rng = np.random.default_rng(int(c_seed))
        if centres is not None:
            p0 = init_walkers_fn(
                rng,
                map_vec=centres[c_idx],
                n_walkers=int(n_walkers),
                log_post_fn=log_post,
                sigma=sigma,
                engine_name=engine_name,
                prior_draw_fn=draw_init_fn,
            )
        else:
            p0 = np.array([
                draw_init_fn(rng)
                for _ in range(int(n_walkers))
            ])
        rng_state = np.random.RandomState(int(c_seed)).get_state()
        sampler = emcee.EnsembleSampler(
            int(n_walkers), n_dim, log_post,
        )
        sampler.random_state = rng_state
        sampler.run_mcmc(p0, int(iterations), progress=False)
        per_chain_samples.append(sampler.get_chain())
        per_chain_logposts.append(sampler.get_log_prob())
        per_chain_acceptance.append(float(np.mean(sampler.acceptance_fraction)))

    return ChainRun(
        per_chain_samples=per_chain_samples,
        per_chain_logposts=per_chain_logposts,
        per_chain_acceptance=per_chain_acceptance,
        chain_seeds=chain_seeds,
    )


# ``eq=False``: the fields are numpy arrays, for which a generated ``__eq__``
# raises ("truth value of an array is ambiguous") and a generated
# ``__hash__`` fails.  Identity comparison is the honest default.
@dataclass(frozen=True, eq=False)
class Diagnostics:
    """What :func:`stack_and_diagnose` hands to the export stage.

    ``flat_samples`` ``(n_chains · n_kept · n_walkers, n_dim)``, with
    ``flat_logposts`` and ``flat_loglikes`` row-aligned to it;
    ``kept_per_chain`` the post-warm-up ``(n_kept, n_walkers, n_dim)``
    block of each chain; ``rhat`` / ``ess`` per latent dimension over ALL
    chains, and ``rhat_alive`` / ``ess_alive`` over the chains that are not
    dead (the same arrays, not copies, when no chain is dead).
    """

    n_discard: int
    n_kept: int
    burn: int
    kept_per_chain: list
    flat_samples: np.ndarray
    flat_logposts: np.ndarray
    flat_loglikes: np.ndarray
    rhat: np.ndarray
    ess: np.ndarray
    rhat_alive: np.ndarray
    ess_alive: np.ndarray


def stack_and_diagnose(
    per_chain_samples: Sequence[np.ndarray],
    per_chain_logposts: Sequence[np.ndarray],
    *,
    log_lik: Callable[[np.ndarray], float],
    dead_chains: Sequence[int],
    n_walkers: int,
    n_dim: int,
    iterations: int,
    discard_fraction: float,
    rhat_fn: Callable[[np.ndarray], np.ndarray],
    ess_fn: Callable[[np.ndarray], np.ndarray],
    alive_fn: Callable[..., tuple],
) -> Diagnostics:
    """Stack the chains, drop the warm-up, and compute split-R-hat and ESS.

    Parameters
    ----------
    per_chain_samples, per_chain_logposts
        One array per chain, emcee layout (see the module docstring).
    log_lik
        The likelihood callable the sampler used, ``fn(vec) -> float``.
        It is evaluated once per exported sample to build the
        ``loglike`` column.  Pass the SAME object the sampler used, so
        ``logpost - logprior == loglike`` holds for the exported samples.
    dead_chains
        Indices of chains that never reached a finite log-posterior.  The
        pooled ``rhat`` / ``ess`` deliberately keep them (their blow-up is
        the signal); the ``*_alive`` twins leave them out.
    n_walkers, n_dim, iterations
        Sizes of each emcee run.
    discard_fraction
        Warm-up fraction in ``[0, 1)``.  The number of iterations dropped
        is ``int(round(discard_fraction * iterations))`` — Python rounds
        halves to the EVEN integer, so 0.25 of 10 drops 2 and 0.25 of 14
        drops 4.
    rhat_fn
        ``fn(chains) -> (n_dim,)`` for ``chains`` of shape
        ``(m, n_rows, n_dim)``.
    ess_fn
        ``fn(walker_major) -> (n_dim,)`` for an array of shape
        ``(n_chains · n_walkers, n_kept, n_dim)``.
    alive_fn
        The live-chains twin, called as
        ``alive_fn(per_chain_samples, kept_per_chain, dead_chains,
        n_walkers=, burn=, rhat_fn=, ess_fn=)`` and returning
        ``(rhat_alive, ess_alive)``.  Called only when a chain is dead.

    Raises
    ------
    ValueError
        If an array does not have the declared shape — ``iterations`` and
        ``n_walkers`` enter the flattening only through their PRODUCT, so
        a swapped pair (or a pre-thinning ``iterations``) would otherwise
        pass silently and burn / split the wrong axis; or if
        ``discard_fraction`` leaves no iteration after the warm-up.

    Notes
    -----
    What this atom does for R-hat is build the INPUT: each chain's
    post-warm-up run is cut into two equal halves along the iteration
    axis, giving ``2 · n_chains`` sequences — the split-chain construction
    of Gelman & Rubin (1992; see Gelman et al., *Bayesian Data Analysis*,
    3rd ed.).  The statistic itself is whatever ``rhat_fn`` computes.  The
    project's own ``rhat_fn`` is the classic, non-rank-normalised ratio;
    the rank-normalised, folded estimator of Vehtari et al. (2021) is an
    improvement a caller may pass in its place.

    The exported samples, ESS and (normally) R-hat all describe the SAME
    post-warm-up material.  The one exception: when fewer than two
    iterations per half remain, R-hat falls back to the full, un-split,
    UN-BURNED chains — so on a very short run R-hat and the exported
    samples describe different material.  Walkers inside one emcee run are
    coupled by the stretch move, so flattening over walkers is a pragmatic
    choice, not a claim of independence; the independence that R-hat
    relies on is between chains.
    """
    # Shape guard.  Costs no arithmetic and changes no result; see Raises.
    expected = (int(iterations), int(n_walkers), int(n_dim))
    for samples_c, logposts_c in zip(per_chain_samples, per_chain_logposts):
        if tuple(np.shape(samples_c)) != expected:
            raise ValueError(
                "per_chain_samples entries must have shape "
                "(iterations, n_walkers, n_dim)"
            )
        if tuple(np.shape(logposts_c)) != expected[:2]:
            raise ValueError(
                "per_chain_logposts entries must have shape "
                "(iterations, n_walkers)"
            )

    full_chain_input = np.stack(
        [s.reshape(int(n_walkers) * int(iterations), n_dim)
         for s in per_chain_samples],
        axis=0,
    )  # (n_chains, iter*walkers, n_dim)

    n_discard = int(round(float(discard_fraction) * int(iterations)))
    n_kept = int(iterations) - n_discard
    if n_kept <= 0:
        raise ValueError(
            "discard_fraction leaves no samples after warm-up; reduce "
            "discard_fraction or increase iterations."
        )
    # Per-chain (iter, walker, dim) → (kept, walker, dim) → flat.
    kept_per_chain = [s[n_discard:] for s in per_chain_samples]
    kept_logposts_per_chain = [
        arr[n_discard:] for arr in per_chain_logposts
    ]
    flat_samples = np.concatenate(
        [k.reshape(int(n_walkers) * n_kept, n_dim) for k in kept_per_chain],
        axis=0,
    )
    flat_logposts = np.concatenate(
        [arr.reshape(-1) for arr in kept_logposts_per_chain], axis=0,
    )
    flat_loglikes = np.array(
        [log_lik(flat_samples[i])
         for i in range(flat_samples.shape[0])],
        dtype=float,
    )

    # The same formula as ``n_discard``, computed again on purpose: the
    # samplers this was lifted from did so, and the two are kept separate
    # so the exported samples and R-hat each state their own burn.
    burn = int(round(float(discard_fraction) * int(iterations)))
    half_len = (int(iterations) - burn) // 2
    if half_len >= 2:
        split_chains = []
        for s in per_chain_samples:
            kept = s[burn: burn + 2 * half_len]            # (2*half_len, walkers, dim)
            first_half = kept[:half_len]                    # (half_len, walkers, dim)
            second_half = kept[half_len:]                   # (half_len, walkers, dim)
            split_chains.append(
                first_half.reshape(half_len * int(n_walkers), n_dim)
            )
            split_chains.append(
                second_half.reshape(half_len * int(n_walkers), n_dim)
            )
        rhat_input = np.stack(split_chains, axis=0)         # (2*n_chains, half_len*walkers, dim)
    else:
        rhat_input = full_chain_input
    rhat = rhat_fn(rhat_input)                              # (n_dim,)
    # ESS on walker-major rows: (iter, walker, dim) → (walker, iter, dim).
    ess = ess_fn(                                           # (n_dim,)
        np.concatenate(
            [np.moveaxis(k, 1, 0) for k in kept_per_chain], axis=0,
        )
    )

    if dead_chains:
        rhat_alive, ess_alive = alive_fn(
            per_chain_samples, kept_per_chain, dead_chains,
            n_walkers=int(n_walkers), burn=burn,
            rhat_fn=rhat_fn,
            ess_fn=ess_fn,
        )
    else:
        rhat_alive, ess_alive = rhat, ess

    return Diagnostics(
        n_discard=n_discard, n_kept=n_kept, burn=burn,
        kept_per_chain=kept_per_chain,
        flat_samples=flat_samples, flat_logposts=flat_logposts,
        flat_loglikes=flat_loglikes,
        rhat=rhat, ess=ess, rhat_alive=rhat_alive, ess_alive=ess_alive,
    )


def convergence_columns(
    chains: dict,
    physics_to_latent: Mapping[str, int | tuple],
    *,
    rhat: np.ndarray,
    ess: np.ndarray,
    rhat_alive: np.ndarray,
    ess_alive: np.ndarray,
    n_chains: int,
    n_walkers: int,
    n_kept: int,
    ess_unreliable_fn: Callable[..., list],
) -> tuple[dict, dict, list]:
    """Write ``r_hat_<key>`` / ``ess_<key>`` into ``chains``, per physics key.

    ``physics_to_latent`` maps an exported chain key to the latent
    dimension it is derived from, or to a TUPLE of dimensions when several
    latents feed one quantity (an angle sampled as a unit-disk pair).  For
    a tuple the conservative reduction is reported: the LARGEST R-hat and
    the SMALLEST ESS of its members.

    ``chains`` is modified IN PLACE, in the iteration order of
    ``physics_to_latent`` (``r_hat_<key>`` then ``ess_<key>``), so the key
    order of the exported dict is the caller's to choose.

    Returns ``(r_hat_alive_meta, ess_alive_meta, ess_unreliable_keys)``:
    the live-chains twins keyed by physics key, and the keys whose ESS
    estimate is not trustworthy for this many rows and kept iterations
    (``ess_unreliable_fn(ess_by_key, n_rows=, n_kept=)``).
    """
    r_hat_alive_meta: dict = {}
    ess_alive_meta: dict = {}
    for key, idx in physics_to_latent.items():
        if isinstance(idx, tuple):
            r_vals = [rhat[i] for i in idx]
            e_vals = [ess[i] for i in idx]
            chains[f"r_hat_{key}"] = float(np.nanmax(r_vals))
            chains[f"ess_{key}"] = float(np.nanmin(e_vals))
            r_hat_alive_meta[key] = float(
                np.nanmax([rhat_alive[i] for i in idx])
            )
            ess_alive_meta[key] = float(
                np.nanmin([ess_alive[i] for i in idx])
            )
        else:
            chains[f"r_hat_{key}"] = float(rhat[idx])
            chains[f"ess_{key}"] = float(ess[idx])
            r_hat_alive_meta[key] = float(rhat_alive[idx])
            ess_alive_meta[key] = float(ess_alive[idx])
    unreliable = ess_unreliable_fn(
        {key: chains[f"ess_{key}"] for key in physics_to_latent},
        n_rows=int(n_chains) * int(n_walkers), n_kept=int(n_kept),
    )
    return r_hat_alive_meta, ess_alive_meta, unreliable


def alive_sample_mask(chains: Mapping, dead_chains: Iterable[int]) -> np.ndarray:
    """Boolean mask over the exported samples: ``False`` on dead chains.

    Uses the per-sample ``chains["chain_id"]`` label.  With no dead chain
    the mask is all ``True``.
    """
    alive = np.ones(chains["chain_id"].shape[0], dtype=bool)
    for c in dead_chains:
        alive &= chains["chain_id"] != c
    return alive


def summarize_chains(
    chains: Mapping,
    alive: np.ndarray,
    *,
    n_chains: int,
    bookkeeping_keys: Iterable[str],
    per_chain_summary_fn: Callable[[Mapping, int], dict],
    nan_aware_keys: Iterable[str] = (),
) -> tuple[dict, dict]:
    """Build ``summary`` (pooled, live chains only) and ``summary_per_chain``.

    Every 1-D array in ``chains`` that is not a bookkeeping column gets a
    ``{"median", "std", "q16", "q84"}`` block.  Arrays with one row per
    sample are restricted to the live chains first (``alive``); arrays of
    any other length are summarised whole.  Keys in ``nan_aware_keys`` use
    the NaN-ignoring estimators — for a derived quantity that is NaN by
    design on part of the posterior.

    ``summary_per_chain`` is ``per_chain_summary_fn(chains, n_chains)``,
    evaluated with invalid / overflow floating-point warnings silenced (a
    dead chain's columns are legitimately non-finite).  Dead chains are
    included there and flagged, never dropped.  ``nan_aware_keys`` governs
    the POOLED table only: ``per_chain_summary_fn`` owns its own NaN
    policy, so give it the same keys if the two tables must agree.  The
    floating-point silencing likewise covers the per-chain call only, not
    the pooled loop.

    Returns ``(summary, summary_per_chain)``; the key order of ``summary``
    follows ``chains``.
    """
    bookkeeping = frozenset(bookkeeping_keys)
    nan_aware = frozenset(nan_aware_keys)
    summary = {}
    for k, v in chains.items():
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim != 1:
            continue
        if k in bookkeeping:
            continue
        if v.shape[0] == alive.shape[0]:
            v = v[alive]
        if k in nan_aware:
            summary[k] = {
                "median": float(np.nanmedian(v)),
                "std": float(np.nanstd(v)),
                "q16": float(np.nanpercentile(v, 16)),
                "q84": float(np.nanpercentile(v, 84)),
            }
            continue
        summary[k] = {
            "median": float(np.median(v)),
            "std": float(np.std(v)),
            "q16": float(np.percentile(v, 16)),
            "q84": float(np.percentile(v, 84)),
        }
    with np.errstate(invalid="ignore", over="ignore"):
        summary_per_chain = per_chain_summary_fn(chains, int(n_chains))
    return summary, summary_per_chain
