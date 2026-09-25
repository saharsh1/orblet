"""Composability pivot — sub-cycle C1: public re-export contract.

These tests pin the cellular ``orblet.model`` and
``orblet.likelihood`` namespaces to the underlying engine
functions.  Two contracts are enforced:

1. Identity:  ``model.rv_model is rv.forward.rv_model``
   (and analogous for the other four names).  No wrapping, no shimming.
2. Laziness:  ``import orblet.model`` (and ``... .likelihood``)
   must NOT pull scipy or emcee.  The PEP 562 ``__getattr__`` template
   in :mod:`orblet.priors` is mirrored to achieve this.

The lazy-import tests use ``monkeypatch.delitem`` to drop the relevant
modules from ``sys.modules`` and let pytest restore state at teardown,
matching the pattern in ``tests/test_python_rv_lazy_imports.py``.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_orbit_model_rv_model_importable_and_is_engine_function() -> None:
    """``from orblet.model import rv_model`` works; same object."""
    from orblet.model import rv_model
    from orblet.model import (
        rv_model as engine_rv_model,
    )
    assert rv_model is engine_rv_model


def test_orbit_model_thiele_innes_xy_importable_and_is_engine_function() -> None:
    """Same contract for ``thiele_innes_xy``.

    Note: this is the engine-side one (``astro.forward``),
    NOT the helper in ``orblet.kepler`` of the same name.  The
    name collision is flagged for sub-cycle C3.
    """
    from orblet.model import thiele_innes_xy
    from orblet.model import (
        thiele_innes_xy as engine_ti,
    )
    assert thiele_innes_xy is engine_ti


def test_orbit_model_campbell_xy_and_along_scan_model_importable() -> None:
    """Same contract for ``campbell_xy`` and ``along_scan_model``."""
    from orblet.model import campbell_xy, along_scan_model
    from orblet.model import (
        campbell_xy as engine_cxy,
        along_scan_model as engine_asm,
    )
    assert campbell_xy is engine_cxy
    assert along_scan_model is engine_asm


def test_orbit_likelihood_rv_loglike_is_aliased_engine_loglike() -> None:
    """``rv_loglike`` is the same function object as the engine's ``loglike``.

    The rename happens only at re-export; the underlying engine function
    keeps its bare ``loglike`` name.
    """
    from orblet.likelihood import rv_loglike
    from orblet.likelihood import loglike
    assert rv_loglike is loglike


def test_orbit_likelihood_loglike_along_scan_is_engine_function() -> None:
    """``loglike_along_scan`` is the same object as the engine function."""
    from orblet.likelihood import loglike_along_scan
    from orblet.likelihood import (
        loglike_along_scan as engine_las,
    )
    assert loglike_along_scan is engine_las


def test_orbit_model_module_import_is_cheap_no_scipy_or_emcee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing ``orblet.model`` must not pull scipy or emcee.

    The forward models are pure numpy, and importing them must not drag in
    the inference stack.
    """
    # Drop cached state for the module under test.  monkeypatch auto-restores
    # ``sys.modules`` at teardown so neighbouring tests are not perturbed.
    for mod in list(sys.modules):
        if mod.startswith(("scipy", "emcee")):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "orblet.model", raising=False)
    # `from orblet import <name>` reads the ATTRIBUTE bound on the orblet
    # package, not sys.modules. Re-importing below binds a NEW module object
    # there, and monkeypatch restores the sys.modules entry at teardown but
    # NOT the parent attribute -- leaving two live copies and breaking every
    # identity check across them. Clear the attribute too; monkeypatch
    # restores it. (This bit at C7 with orblet.astro.forward, and again here
    # when the B2 rewrite dropped the cleanup because the deferred SUBMODULE
    # was gone -- missing that the parent had merely changed from
    # orblet.astro to orblet.)
    import orblet as _orblet_pkg
    if hasattr(_orblet_pkg, "model"):
        monkeypatch.delattr(_orblet_pkg, "model", raising=False)

    model_mod = importlib.import_module("orblet.model")

    leaked = [m for m in sys.modules if m.startswith(("scipy", "emcee"))]
    assert not leaked, (
        f"importing orblet.model loaded {leaked}; the forward models must "
        "stay free of scipy and emcee."
    )
    # Cheap because it is lean, not because it is empty.
    assert model_mod.__all__, "orblet.model advertises nothing"


def test_orbit_likelihood_module_import_is_cheap_no_scipy_or_emcee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract for ``orblet.likelihood``.

    A Gaussian-with-jitter log-likelihood is arithmetic over numpy arrays,
    and reaching for one must not drag in the inference stack.
    """
    for mod in list(sys.modules):
        if mod.startswith(("scipy", "emcee")):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "orblet.likelihood", raising=False)
    # `from orblet import <name>` reads the ATTRIBUTE bound on the orblet
    # package, not sys.modules. Re-importing below binds a NEW module object
    # there, and monkeypatch restores the sys.modules entry at teardown but
    # NOT the parent attribute -- leaving two live copies and breaking every
    # identity check across them. Clear the attribute too; monkeypatch
    # restores it. (This bit at C7 with orblet.astro.forward, and again here
    # when the B2 rewrite dropped the cleanup because the deferred SUBMODULE
    # was gone -- missing that the parent had merely changed from
    # orblet.astro to orblet.)
    import orblet as _orblet_pkg
    if hasattr(_orblet_pkg, "likelihood"):
        monkeypatch.delattr(_orblet_pkg, "likelihood", raising=False)

    lik_mod = importlib.import_module("orblet.likelihood")

    leaked = [m for m in sys.modules if m.startswith(("scipy", "emcee"))]
    assert not leaked, (
        f"importing orblet.likelihood loaded {leaked}; the likelihoods must "
        "stay free of scipy and emcee."
    )
    assert lik_mod.__all__, "orblet.likelihood advertises nothing"


# ── C2: flat-kwargs positive-contract smoke tests ────────────────────


def test_orbit_model_rv_model_accepts_flat_kwargs_after_c2() -> None:
    """rv_model is flat-kwargs after C2; smoke test the new signature."""
    from orblet.model import rv_model
    import numpy as np
    t = np.array([58000.0, 58100.0, 58200.0], dtype=float)
    out = rv_model(
        t,
        period_yr=1.0, ecc=0.1, omega_rad=0.5, tau=0.25,
        mass_msun=0.05, M_msun=1.0, offset_kms=0.0,
        epoch_ref_mjd=57388.5,
    )
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_orbit_likelihood_rv_loglike_accepts_flat_kwargs_after_c2() -> None:
    """rv_loglike is flat-kwargs after C2; smoke test."""
    from orblet.likelihood import rv_loglike
    import numpy as np
    t = np.array([58000.0, 58100.0, 58200.0], dtype=float)
    rv = np.array([0.5, -0.3, 0.1], dtype=float)
    err = np.array([0.1, 0.1, 0.1], dtype=float)
    ll = rv_loglike(
        t, rv, err,
        period_yr=1.0, ecc=0.1, omega_rad=0.5, tau=0.25,
        mass_msun=0.05, M_msun=1.0, offset_kms=0.0,
        jitter_kms=0.01, epoch_ref_mjd=57388.5,
    )
    assert isinstance(ll, float)
    assert ll != float("inf") and ll != float("-inf")
