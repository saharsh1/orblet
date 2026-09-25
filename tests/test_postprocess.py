"""
Tests for orblet.elements — orbital element extraction,
model curves, and residuals.

All tests use synthetic data and mock fit results.
"""

from __future__ import annotations

import numpy as np
import pytest

from orblet.constants import OMEGA_CONVENTION_PRIMARY
from orblet import elements as pp
from orblet.elements import (
    extract_orbital_elements,
    compute_rv_model_curve,
    compute_residuals,
    to_nss_convention,
    _solve_kepler,
    _keplerian_rv,
    _stellar_K_from_elements,
)
from orblet.priors import DAYS_PER_KEPLER_YEAR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_result(
    period_yr: float = 0.5,
    ecc: float = 0.3,
    omega: float = 1.0,
    tp: float = 58100.0,
    mass_msun: float = 0.5,
    M_star: float = 1.0,
    offset_kms: float = -50.0,
    jitter_kms: float = 0.5,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """Create a mock result dict mimicking fit_rv_orbit output.

    Mirrors the bridge output: m2_msun in M_sun, K_kms and fm_spec_msun
    derived.  ``P_days`` is in DAYS (the ``period_yr`` argument is in
    Keplerian years and is converted here).
    """

    rng = np.random.default_rng(seed)

    def _samples(center, width):
        return rng.normal(center, width, n_samples)

    chains = {
        "P_days": _samples(period_yr, 0.01) * DAYS_PER_KEPLER_YEAR,
        "e": np.clip(_samples(ecc, 0.02), 0, 0.99),
        "omega_rad": _samples(omega, 0.05),
        "tp_mjd": _samples(tp, 1.0),
        "m2_msun": np.clip(_samples(mass_msun, 0.05), 0.01, None),
        "M_total_msun": np.clip(_samples(M_star, 0.05), 0.1, None),
        "a_rel_au": _samples(0.8, 0.01),
        "gamma_kms": _samples(offset_kms, 0.5),
        "rv_jitter_kms": np.abs(_samples(jitter_kms, 0.1)),
    }

    # Derive K_kms and fm_spec_msun per sample (same as the bridge does).
    # M_total_msun is the system-level total mass (M_star + M_comp).
    # ``_stellar_K_from_elements`` takes the period in Keplerian YEARS,
    # so the days-valued chain key is converted back here.
    from orblet.priors import MSUN_IN_MJUP
    K_arr = np.array([
        _stellar_K_from_elements(m * MSUN_IN_MJUP, p, e, M)
        for m, p, e, M in zip(
            chains["m2_msun"], chains["P_days"] / DAYS_PER_KEPLER_YEAR,
            chains["e"], chains["M_total_msun"],
        )
    ])
    chains["K_kms"] = K_arr

    from orblet.constants import G_SI, MSUN_KG
    K_ms = K_arr * 1000.0
    P_sec = chains["P_days"] * 86400.0
    fm_kg = K_ms**3 * P_sec * (1.0 - chains["e"]**2)**1.5 / (2.0 * np.pi * G_SI)
    chains["fm_spec_msun"] = fm_kg / MSUN_KG

    summary = {}
    for key, arr in chains.items():
        summary[key] = {
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "q16": float(np.percentile(arr, 16)),
            "q84": float(np.percentile(arr, 84)),
        }

    # Mirror the ``_meta`` block both engines emit at chain export.  The
    # primary-frame ω sentinel is required by ``compute_rv_model_curve``
    # and ``compute_residuals``.
    chains["_meta"] = {"omega_convention": OMEGA_CONVENTION_PRIMARY}

    return {"chains": chains, "summary": summary}


def _make_mock_prepared(n: int = 20, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "epochs_mjd": np.linspace(58000, 58500, n),
        "rv": rng.normal(-50, 20, n),
        "rv_err": np.full(n, 2.0),
        "n_epochs": n,
    }


# ── Kepler solver ─────────────────────────────────────────────────────────────

class TestSolveKepler:
    def test_circular(self):
        """For e=0, E should equal M."""
        M = np.linspace(0, 2 * np.pi, 100)
        E = _solve_kepler(M, 0.0)
        np.testing.assert_allclose(E, M, atol=1e-12)

    def test_known_solution(self):
        """For e=0.5, M=π → E=π (by symmetry)."""
        M = np.array([np.pi])
        E = _solve_kepler(M, 0.5)
        np.testing.assert_allclose(E, [np.pi], atol=1e-10)

    def test_consistency(self):
        """E - e*sin(E) should give back M."""
        M = np.array([0.5, 1.0, 2.0, 4.0, 5.5])
        e = 0.6
        E = _solve_kepler(M, e)
        M_recovered = E - e * np.sin(E)
        np.testing.assert_allclose(M_recovered, M, atol=1e-12)


# ── Keplerian RV ──────────────────────────────────────────────────────────────

class TestKeplerianRV:
    def test_circular_sinusoid(self):
        """Circular orbit → pure sinusoid with amplitude K.

        The stellar RV formula uses a negative sign (star moves
        opposite to companion), so max = gamma + K still holds
        because the negation flips the phase, not the amplitude.
        """
        t = np.linspace(0, 100, 1000)
        K = 10.0
        gamma = -30.0
        rv = _keplerian_rv(t, period_days=50.0, ecc=0.0, omega=0.0,
                           K=K, gamma=gamma, tp=0.0)

        # Amplitude should still be K (sign flip just shifts phase).
        assert rv.max() == pytest.approx(gamma + K, abs=0.1)
        assert rv.min() == pytest.approx(gamma - K, abs=0.1)

    def test_period(self):
        """RV should repeat after one period."""
        P = 30.0
        t = np.array([10.0, 10.0 + P])
        rv = _keplerian_rv(t, period_days=P, ecc=0.3, omega=0.5,
                           K=20.0, gamma=0.0, tp=0.0)
        np.testing.assert_allclose(rv[0], rv[1], atol=1e-10)


# ── K from orbital elements ──────────────────────────────────────────────────

class TestStellarKFromElements:
    def test_positive(self):
        K = _stellar_K_from_elements(
            mass_mjup=1000.0, period_yr=1.0, ecc=0.0, M_total_msun=1.0
        )
        assert K > 0

    def test_higher_mass_higher_K(self):
        K1 = _stellar_K_from_elements(
            mass_mjup=100.0, period_yr=1.0, ecc=0.0, M_total_msun=1.0
        )
        K2 = _stellar_K_from_elements(
            mass_mjup=1000.0, period_yr=1.0, ecc=0.0, M_total_msun=1.0
        )
        assert K2 > K1


# ── extract_orbital_elements ──────────────────────────────────────────────────

class TestExtractOrbitalElements:
    def test_keys(self):
        result = _make_mock_result()
        elem = extract_orbital_elements(result)
        assert "period_days" in elem
        assert "eccentricity" in elem
        assert "companion_mass_msun" in elem

    def test_period_in_days(self):
        result = _make_mock_result(period_yr=1.0)
        elem = extract_orbital_elements(result)
        # 1 year ≈ 365.25 days.
        assert elem["period_days"]["median"] == pytest.approx(365.25, rel=0.05)

    def test_units_present(self):
        result = _make_mock_result()
        elem = extract_orbital_elements(result)
        assert elem["period_days"]["unit"] == "days"
        assert elem["eccentricity"]["unit"] == ""
        assert elem["gamma_kms"]["unit"] == "km/s"


# ── compute_rv_model_curve ────────────────────────────────────────────────────

class TestComputeRVModelCurve:
    def test_returns_expected_keys(self):
        result = _make_mock_result()
        t_grid = np.linspace(58000, 58500, 100)
        out = compute_rv_model_curve(result, t_grid)
        assert set(out.keys()) == {"t", "rv_model", "params"}
        assert len(out["rv_model"]) == 100

    def test_model_is_finite(self):
        result = _make_mock_result()
        t_grid = np.linspace(58000, 58500, 100)
        out = compute_rv_model_curve(result, t_grid)
        assert np.all(np.isfinite(out["rv_model"]))


# ── compute_residuals ─────────────────────────────────────────────────────────

class TestComputeResiduals:
    def test_returns_expected_keys(self):
        result = _make_mock_result()
        prepared = _make_mock_prepared()
        out = compute_residuals(prepared, result)
        expected = {"epochs_mjd", "rv_obs", "rv_model", "rv_err",
                    "rv_err_eff", "residuals", "rms", "chi2_red"}
        assert set(out.keys()) == expected

    def test_residuals_shape(self):
        result = _make_mock_result()
        prepared = _make_mock_prepared(n=15)
        out = compute_residuals(prepared, result)
        assert len(out["residuals"]) == 15

    def test_rms_positive(self):
        result = _make_mock_result()
        prepared = _make_mock_prepared()
        out = compute_residuals(prepared, result)
        assert out["rms"] > 0
        assert out["chi2_red"] > 0


# ── ω-convention regression tests ────────────────────────────────────────────

class TestKeplerianRVOmegaConvention:
    """Pin the primary-frame ω convention in :func:`_keplerian_rv`.

    Catches a regression where the formula was inadvertently using the
    pre-realignment companion-frame form ``γ - K [cos(ν+ω) + e cos(ω)]``
    while the chains carry primary-frame ω.  The sign would phase-flip
    the model RV by 180° (reference manual,
    §1.4).
    """

    def test_keplerian_rv_uses_primary_frame_omega(self):
        # At periastron (t = tp) the true anomaly ν = 0, so the formula
        # reduces to v_r(tp) = γ + K · [cos(ω) + e · cos(ω)]
        #                   = γ + K · (1 + e) · cos(ω).
        # For e=0.3, ω=1.0 rad, K=10 km/s, γ=0:
        #     v_r(tp) = 0 + 10 · 1.3 · cos(1.0) ≈ 13 · 0.5403 ≈ 7.025.
        # The OLD (companion-frame) formula would yield −7.025; this
        # test pins the correct sign.
        e = 0.3
        omega = 1.0
        K = 10.0
        gamma = 0.0
        P = 365.0
        tp = 0.0
        rv = _keplerian_rv(
            np.array([tp]),
            period_days=P, ecc=e, omega=omega, K=K, gamma=gamma, tp=tp,
        )
        expected = gamma + K * (1.0 + e) * np.cos(omega)
        np.testing.assert_allclose(rv, [expected], atol=1e-6)
        # Belt-and-suspenders numeric anchor — independent of the
        # symbolic expansion above.  Exact value: 10·1.3·cos(1) ≈ 7.02393.
        # The OLD (companion-frame) formula would yield ≈ −7.02393.
        np.testing.assert_allclose(rv, [7.02393], atol=1e-4)


class TestComputeRVModelCurveSentinel:
    """The sentinel guard rejects chains lacking the ω-convention tag."""

    def test_compute_rv_model_curve_requires_omega_convention_sentinel(self):
        result = _make_mock_result()
        # Strip the sentinel — the function must refuse to compute.
        del result["chains"]["_meta"]
        with pytest.raises(ValueError, match="omega_convention"):
            compute_rv_model_curve(result, np.linspace(58000, 58500, 10))

    def test_compute_rv_model_curve_rejects_wrong_convention(self):
        result = _make_mock_result()
        # Replace the sentinel with an unknown value.
        result["chains"]["_meta"] = {"omega_convention": "companion_frame"}
        with pytest.raises(ValueError, match="omega_convention"):
            compute_rv_model_curve(result, np.linspace(58000, 58500, 10))

    def test_compute_residuals_requires_omega_convention_sentinel(self):
        result = _make_mock_result()
        prepared = _make_mock_prepared()
        del result["chains"]["_meta"]
        with pytest.raises(ValueError, match="omega_convention"):
            compute_residuals(prepared, result)


# ── to_nss_convention (TI Stage-3a step 6) ────────────────────────────


def _ti_chain(n: int = 5, *, meta: dict | None = None) -> dict:
    """Mass-free TI chain, as the observables-first engine emits it.

    Carries ``P_days`` and ``_meta["epoch_ref_mjd"]`` by default because a
    real engine chain always does: the catalogue epoch is a REQUIRED input
    to every astrometric fit, and the NSS periastron conversion inherits
    it from there.  A fixture without them would be unfaithful to the
    chains it stands in for.
    """
    rng = np.random.default_rng(0)
    chains = {
        "P_days": np.full(n, 400.0),   # days
        "A_mas": rng.normal(2.0, 0.1, n),
        "B_mas": rng.normal(-1.5, 0.1, n),
        "F_mas": rng.normal(0.8, 0.1, n),
        "G_mas": rng.normal(-2.2, 0.1, n),
        "Omega_rad": np.linspace(0.1, 6.0, n),
        "omega_rad": np.full(n, 0.2),
        "tp_mjd": np.full(n, 60000.0),
    }
    chains["_meta"] = (
        {"epoch_ref_mjd": 57388.5} if meta is None
        else {"epoch_ref_mjd": 57388.5, **meta}
    )
    return chains


def _legacy_chain(n: int = 5) -> dict:
    """Legacy chain carrying the mass keys the old scaling needed."""
    chains = _ti_chain(n)
    chains["m2_msun"] = np.full(n, 0.5)
    chains["M_total_msun"] = np.full(n, 2.0)
    return chains


class TestToNssConvention:
    """The NSS-comparison converter after the observables-first fix.

    Context: our engine samples PHOTOCENTRE Thiele-Innes amplitudes
    directly, which is exactly what the Gaia NSS catalogue reports — so
    the amplitudes need no conversion at all.  Treating them as
    RELATIVE-orbit amplitudes and scaling them by the mass ratio would be
    wrong for these chains and would crash on them, since they carry no
    mass columns.
    """

    def test_photocentre_branch_no_keyerror_on_a_mass_free_chain(self):
        """This is the bug: red before the fix, green after."""
        out = to_nss_convention(_ti_chain())
        for key in (
            "A_nss_mas", "B_nss_mas", "F_nss_mas", "G_nss_mas",
            "omega_nss_rad", "Omega_nss_rad", "tp_nss_days",
        ):
            assert key in out

    def test_photocentre_branch_is_the_identity_on_amplitudes(self):
        """No scaling, no sign flip — the sampled amplitudes ARE the
        photocentre amplitudes NSS publishes.

        The residual global ``(A,B,F,G) → −(A,B,F,G)`` freedom is a
        DEGENERACY (the ``(ω+π, Ω+π)`` twin), which both we and NSS
        collapse by folding Ω into ``[0, π)`` — so no sign claim is
        needed here.  The DR4 scan-angle sign is separately locked by
        ``tests/test_bh3_prerelease_convention_lock.py``.
        """
        chains = _ti_chain()
        out = to_nss_convention(chains)
        for src, dst in (
            ("A_mas", "A_nss_mas"), ("B_mas", "B_nss_mas"),
            ("F_mas", "F_nss_mas"), ("G_mas", "G_nss_mas"),
        ):
            np.testing.assert_array_equal(out[dst], chains[src])
        assert (
            out["_meta"]["nss_amplitude_source"]
            is pp._NSS_SOURCE_PHOTOCENTRE_IDENTITY
        )

    @pytest.mark.parametrize("sign_flip", [True, False])
    def test_legacy_branch_values_byte_identical(self, sign_flip):
        """The mass-scaled path is the only one with archived callers;
        its numbers must not move by a bit."""
        chains = _legacy_chain()
        out = to_nss_convention(chains, sign_flip=sign_flip)
        factor = -0.25 if sign_flip else 0.25
        for src, dst in (
            ("A_mas", "A_nss_mas"), ("B_mas", "B_nss_mas"),
            ("F_mas", "F_nss_mas"), ("G_mas", "G_nss_mas"),
        ):
            np.testing.assert_allclose(
                out[dst], factor * chains[src], rtol=1e-15, atol=0.0
            )
        assert (
            out["_meta"]["nss_amplitude_source"]
            is pp._NSS_SOURCE_RELATIVE_SCALED
        )
        # The sign actually applied is recorded: "relative_scaled" alone
        # does not say WHICH sign produced the numbers.
        assert out["_meta"]["nss_sign_flip"] is sign_flip

    def test_meta_is_copied_not_mutated(self):
        """``out = dict(chains)`` is shallow, so writing into
        ``out['_meta']`` would silently mutate the CALLER's chain."""
        caller_meta = {"basis": "thiele_innes", "seed": 7}
        snapshot = dict(caller_meta)
        chains = _ti_chain(meta=caller_meta)

        out = to_nss_convention(chains)

        assert caller_meta == snapshot
        assert out["_meta"] is not caller_meta
        # Pre-existing entries survive alongside the new provenance.
        assert out["_meta"]["basis"] == "thiele_innes"
        assert out["_meta"]["seed"] == 7

    def test_works_with_no_meta_key_at_all(self):
        out = to_nss_convention(_ti_chain())
        assert (
            out["_meta"]["nss_amplitude_source"]
            is pp._NSS_SOURCE_PHOTOCENTRE_IDENTITY
        )

    def test_omega_Omega_fold_and_tp_conversion_unchanged(self):
        """The (ω, Ω) lockstep is untouched by the branch split —
        including the two boundary cases Ω = 0 and Ω = π.

        The tp assertion at the end checks the NSS form: ``t_periastron``
        is an OFFSET from the catalogue reference epoch, folded into
        ``[−P/2, +P/2]`` — not an absolute Julian Date.
        """
        n = 5
        chains = {
            "P_days": np.full(n, 400.0),
            "A_mas": np.ones(n), "B_mas": np.ones(n),
            "F_mas": np.ones(n), "G_mas": np.ones(n),
            "Omega_rad": np.array([0.0, 0.1, np.pi, 3.5, 6.0]),
            "omega_rad": np.full(n, 0.2),
            "tp_mjd": np.full(n, 60000.0),
            "_meta": {"epoch_ref_mjd": 57388.5},
        }
        legacy = {**chains, "m2_msun": np.full(n, 0.5), "M_total_msun": np.full(n, 2.0)}

        for source in (chains, legacy):
            out = to_nss_convention(source)
            Om = out["Omega_nss_rad"]
            assert np.all((Om >= 0.0) & (Om < np.pi))
            # Ω = π folds to exactly 0 (the code tests >= π).
            assert Om[2] == pytest.approx(0.0, abs=1e-12)
            # Ω = 0 is untouched, so ω is untouched too.
            assert Om[0] == pytest.approx(0.0, abs=1e-12)
            assert out["omega_nss_rad"][0] == pytest.approx(0.2, abs=1e-12)
            # ω shifts by −π exactly where Ω did.
            assert out["omega_nss_rad"][2] == pytest.approx(0.2 - np.pi, abs=1e-12)
            assert np.all(
                (out["omega_nss_rad"] >= -np.pi) & (out["omega_nss_rad"] <= np.pi)
            )
            # Offset from the catalogue epoch, folded into [−P/2, +P/2]:
            # 60000 − 57388.5 = 2611.5 d, and 2611.5 mod 400 d = 211.5,
            # which folds to 211.5 − 400 = −188.5.
            np.testing.assert_allclose(
                out["tp_nss_days"], -188.5, atol=1e-9
            )

    def test_missing_amplitudes_still_raise(self):
        with pytest.raises(KeyError):
            to_nss_convention({"Omega_rad": np.zeros(3)})

    def test_docstring_no_longer_calls_our_amplitudes_relative_orbit(self):
        """The mis-documentation was half the defect."""
        doc = to_nss_convention.__doc__
        assert "photocent" in doc
        # The old claim: our sampled amplitudes are relative-orbit ones.
        assert "represent the\n    **relative-orbit**" not in doc
        # The legacy branch is for legacy RELATIVE-orbit input — and
        # Campbell chains can never reach here (they carry no A_mas..G_mas),
        # so naming them would invite a real error.
        assert "legacy" in doc.lower()
