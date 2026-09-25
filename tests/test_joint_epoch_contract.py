"""Regression guard for C4e bug #1 — the joint-engine epoch-reference contract.

The synthetic joint fixture (``_synthesize_joint`` in ``conftest.py``) must
generate its orbit at the SAME reference epoch the engine assumes by default
(``MJD_J2016_TCB``, the Gaia DR4 catalog epoch), so a consumer that calls
``fit_joint_orbit`` WITHOUT passing ``epoch_ref_mjd`` cannot silently mismatch
the proper-motion lever-arm origin.

Before the fix the fixture generated at MJD 60000; with the BH3-like fixture's
large proper motion that injected a ~747 mas systematic into the astrometric
model evaluated AT the true parameters (vs ~0.38 mas, the noise floor, at the
matched epoch). These checks are fast, deterministic, and MCMC-free: the data
is built by the fixture's forward emulator and scored by the engine's OWN
forward model at the analytic truth, so they pin the epoch-contract invariant,
not sampler behaviour.

Scope: bug #1 (epoch contract) ONLY. Bug #2 (sampler basin / inclination
convergence) is out of scope and keeps ``test_closure_joint_bh3_like`` xfail.
"""

from __future__ import annotations

import math

import numpy as np

from orblet.constants import MJD_J2010_TCB, MJD_J2016_TCB
from orblet.model import _along_scan_for_theta_campbell
from orblet.likelihood import loglike_along_scan

# BH3-like fixture block (LARGE proper motion) — identical to the closure test
# and the C4e diagnostic harness, so the guard exercises the defect-triggering
# case (the ~747 mas mismatch lived here). Parameters are synthetic round
# numbers approximating the published Gaia BH3 benchmark; no source identifier.
_BH3_KW = dict(
    seed=3,
    P_days=4253.1,
    e=0.7291,
    omega_rad=np.deg2rad(157.08),
    Omega_rad=np.deg2rad(195.4),
    inc_rad=float(np.pi - np.deg2rad(120.94)),
    m_comp_msun=33.0,
    M_total_msun=33.76,
    plx_mas=1.785,
    gamma_kms=-333.2,
    rv_sigma_kms=2.0,
    centroid_sigma_mas=0.40,
    pmra_masyr=-28.30,
    pmdec_masyr=-155.18,
    ra_deg=294.82716,
    dec_deg=14.93047,
    n_rv_epochs=48,
    n_astro_epochs=80,
    t_span_rv_periods=1.5,
    t_span_astro_periods=1.5,
)


def _astro_model_at_truth(joint, truth, *, epoch_ref_mjd):
    """Astro along-scan RMS + log-likelihood of the engine's OWN forward model
    at the TRUE parameters, scored against the fixture data, for a given model
    reference epoch. No MCMC.

    Unpacks the astro dict exactly as ``fit_joint_orbit`` does (the
    ``MJD_J2010_TCB`` guard maps a relative ``obs_time`` back to absolute MJD).
    Returns ``(rms_mas, ll_astro)``.
    """
    astro = joint["astro"]
    obs_time_raw = np.asarray(astro["obs_time"], dtype=float)
    astro_t_mjd = (
        obs_time_raw + MJD_J2010_TCB
        if np.nanmedian(obs_time_raw) < 30000.0
        else obs_time_raw
    )
    centroid_pos = np.asarray(astro["centroid_pos"], dtype=float)
    centroid_pos_err = np.asarray(astro["centroid_pos_err"], dtype=float)
    psi = np.asarray(astro["scan_angle"], dtype=float)
    plx_factor = np.asarray(astro["parallax_factor_al"], dtype=float)

    # Build theta straight from the analytic truth (tp_mjd is absolute and
    # epoch-independent; only the PM lever-arm term depends on epoch_ref_mjd).
    # NB: the fixture builds data with ``_kepler.campbell_xy`` (primary-frame ω,
    # +photocentre amplitude) while the engine forward uses ω+π and a negative
    # amplitude; the two negations cancel exactly (the documented campbell_xy
    # contract), so the model matches the data AT TRUTH. Do not "simplify" one
    # side's ω/amplitude sign without the other or this equivalence breaks.
    astro_theta = {
        "P_yr": truth["P_yr"], "e": truth["e"],
        "omega_rad": truth["omega_rad"], "inc_rad": truth["inc_rad"],
        "Omega_rad": truth["Omega_rad"], "m2_msun": truth["m_comp_msun"],
        "M_total_msun": truth["M_total_msun"], "plx_mas": truth["plx_mas"],
        "ra_offset_mas": 0.0, "dec_offset_mas": 0.0,
        "pmra_masyr": truth["pmra_masyr"], "pmdec_masyr": truth["pmdec_masyr"],
        "astro_jitter_mas": 0.0, "tp_mjd": truth["tp_mjd"],
    }
    model_along = _along_scan_for_theta_campbell(
        astro_theta, t_mjd=astro_t_mjd, psi=psi,
        parallax_factor_al=plx_factor, epoch_ref_mjd=epoch_ref_mjd,
    )
    resid = np.asarray(model_along, dtype=float) - centroid_pos
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ll = float(loglike_along_scan(
        model_along_scan=model_along, centroid_pos=centroid_pos,
        centroid_pos_err=centroid_pos_err, jitter_mas=0.0,
    ))
    return rms, ll


def test_joint_default_epoch_matches_engine_default(synth_joint_dataset):
    """The factory generates at the engine's None-resolved default epoch, so a
    no-epoch consumer cannot silently mismatch the PM lever-arm origin.

    Pinned against the imported constant (not a copied literal) so the test
    stays valid if ``MJD_J2016_TCB`` is ever re-defined.
    """
    _, truth_bh3 = synth_joint_dataset(**_BH3_KW)
    _, truth_zero = synth_joint_dataset(seed=0)
    assert truth_bh3["epoch_ref_mjd"] == MJD_J2016_TCB
    assert truth_zero["epoch_ref_mjd"] == MJD_J2016_TCB


def test_joint_model_at_truth_at_default_epoch_is_noise_floor(synth_joint_dataset):
    """PRIMARY GUARD. With the fixture and the engine default sharing the epoch,
    the model at truth reproduces the (large-PM) astrometry to the noise floor.

    Pre-fix this RMS was ~747 mas; the 0.5 mas bar sits ~3 orders of magnitude
    below that, so the guard is decisive and not seed-fragile. ``MJD_J2016_TCB``
    is exactly the epoch ``fit_joint_orbit`` uses when ``epoch_ref_mjd=None``.
    """
    joint, truth = synth_joint_dataset(**_BH3_KW)
    rms, ll = _astro_model_at_truth(joint, truth, epoch_ref_mjd=MJD_J2016_TCB)

    n_astro = _BH3_KW["n_astro_epochs"]
    sigma = _BH3_KW["centroid_sigma_mas"]
    floor_astro = -0.5 * (n_astro + n_astro * math.log(2.0 * math.pi * sigma ** 2))
    assert rms < 0.5, f"astro model-at-truth RMS {rms:.3f} mas (epoch mismatch?)"
    assert ll > floor_astro - 30.0, (
        f"ll_astro {ll:.1f} far below noise floor {floor_astro:.1f} "
        "=> model at truth does not reproduce the data (epoch mismatch?)"
    )


def test_joint_zero_pm_epoch_shift_is_harmless(synth_joint_dataset):
    """CONTROL. With PM = 0 the epoch origin shift cannot move the modeled
    astrometric signal (only the PM term is epoch-dependent), so the model-at-
    truth RMS stays at the noise floor at BOTH the default epoch and the old
    60000. Proves the defect is the proper-motion lever-arm (bug ∝ PM × Δepoch),
    explaining why the zero-PM consumers were never broken by the latent
    mismatch.
    """
    joint, truth = synth_joint_dataset(seed=0)  # factory default => PM = 0
    rms_default, _ = _astro_model_at_truth(joint, truth, epoch_ref_mjd=MJD_J2016_TCB)
    rms_shifted, _ = _astro_model_at_truth(joint, truth, epoch_ref_mjd=60000.0)
    assert rms_default < 0.5, f"zero-PM RMS at default epoch {rms_default:.3f} mas"
    assert rms_shifted < 0.5, f"zero-PM RMS at shifted epoch {rms_shifted:.3f} mas"
