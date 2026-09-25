"""A complete synthetic dataset, ready to hand to a fitter.

:func:`load_simulated_inputs` builds one in memory: radial velocities, epoch
astrometry, a catalogue row, and the truth the data were generated from. It
reads no files and reaches no network — everything here is manufactured from
an orbit and a cadence.

The truth is the point. A fitter can only be shown to work by being given
data whose answer is already known, and that is what the ``truth`` field
carries: the :class:`~orblet.simulate.orbit.OrbitSimulator` whose parameters
produced the arrays. Recover those and the machinery works; do not, and it
does not.

The returned dicts match the SHAPE that real survey loaders return — same
keys, same units, same array conventions — so code written against a
simulated bundle runs unchanged against real data. That is what makes these
bundles usable for teaching, for the examples, and for contract tests.

Intended audience
-----------------
Loader-contract tests, prepare-step plumbing, and notebook examples that do
not assert posterior properties. See :func:`load_simulated_inputs` for the
full positive/negative-use list.
"""

from __future__ import annotations

from dataclasses import dataclass

from orblet.simulate.cadence import DemoParallaxConsistentCadence
from orblet.simulate.orbit import OrbitSimulator


# Sentinel: never references a real Gaia source.  Used as the
# ``source_id`` value in the catalog row and in the cadence's metadata
# so accidental cross-contamination between synthetic-demo data and
# real Gaia caches is loudly visible (a real Gaia source_id is a
# 19-digit positive integer; -1 cannot collide).
SYNTHETIC_SOURCE_ID = -1

@dataclass(frozen=True)
class SimulatedDemoBundle:
    """
    Container for synthetic-demo inputs returned by
    :func:`load_simulated_inputs`.

    Attributes
    ----------
    rv_data : dict
        Loader-shaped epoch RV dict (the shape :class:`OrbitSimulator` documents).
    astro_data : dict
        Loader-shaped epoch-astrometry dict (same source).
    catalog_row : dict
        Catalog-row dict (minimum keys: ``source_id``, ``ra``, ``dec``,
        ``parallax``, ``pmra``, ``pmdec``, ``m1``).  ``source_id`` is
        the synthetic sentinel (:data:`SYNTHETIC_SOURCE_ID`).
        Keys match the typical ``resolve_source_catalog`` return
        schema, so the bundle is drop-in compatible across simulated
        and real modes.

        Units (the dict keys are unsuffixed for
        cross-mode compatibility, but values carry the same units as
        the simulator attributes they come from):
        ``ra`` [deg], ``dec`` [deg], ``parallax`` [mas],
        ``pmra`` [mas/yr], ``pmdec`` [mas/yr], ``m1`` [M☉].
    truth : OrbitSimulator
        The :class:`OrbitSimulator` instance used to generate the
        synthetic data.  Stored directly (not as a dict) so tests can
        access truth via attribute.
    seed : int or None
        The seed forwarded to the simulator and the cadence; ``None``
        means system-entropy randomness.
    """

    rv_data: dict
    astro_data: dict
    catalog_row: dict
    truth: OrbitSimulator
    seed: int | None
    source_id: int | None


def load_simulated_inputs(
    *,
    simulator: OrbitSimulator | None = None,
    cadence=None,
    seed: int | None = None,
) -> SimulatedDemoBundle:
    """
    Build a :class:`SimulatedDemoBundle` from a truth and a cadence.

    With no arguments it builds the demo bundle: the one preset,
    :meth:`OrbitSimulator.toy_orbit`, observed on the analytic, closure-grade
    :class:`~orblet.simulate.cadence.DemoParallaxConsistentCadence` placed at
    the preset's own sky position.  That cadence couples its parallax factor
    to the sky position and time, and the SAME array both injects and
    removes the parallax signal, so the bundle is suitable for
    forward/inverse closure checks.  Nothing is read from disk.

    Pass ``simulator`` for another truth (the default cadence then follows
    its sky position), and ``cadence`` for another sampling: any object with
    the cadence attributes :meth:`OrbitSimulator.simulate_rv` and
    :meth:`OrbitSimulator.simulate_astrometry` read.  A cadence that draws
    scan angles and parallax factors independently describes a geometry no
    satellite can produce; closure assertions against it prove nothing.

    Parameters
    ----------
    simulator : OrbitSimulator, optional
        The truth.  Default :meth:`OrbitSimulator.toy_orbit`.
    cadence : object, optional
        When the observations happen.  Default
        ``DemoParallaxConsistentCadence(ra_deg=simulator.ra_deg,
        dec_deg=simulator.dec_deg, seed=seed)``.
    seed : int or None, default None
        Forwarded to the default cadence (its time jitter) and to the
        simulator's Gaussian-noise draws.  ``None`` ⇒ system-entropy noise.

    Returns
    -------
    SimulatedDemoBundle
        Frozen bundle with fields ``rv_data``, ``astro_data``,
        ``catalog_row``, ``truth``, ``seed``, ``source_id``.

    Examples
    --------
    >>> bundle = load_simulated_inputs(seed=0)
    >>> sim = OrbitSimulator.toy_orbit(m2_msun=3.0)
    >>> bundle = load_simulated_inputs(simulator=sim, seed=0)
    """
    if simulator is None:
        simulator = OrbitSimulator.toy_orbit()
    if cadence is None:
        # RA/Dec coupled to the truth so the analytic parallax factor is
        # self-consistent with the injected sky position.
        cadence = DemoParallaxConsistentCadence(
            ra_deg=simulator.ra_deg, dec_deg=simulator.dec_deg, seed=seed,
        )

    rv_data = simulator.simulate_rv(cadence, seed=seed)
    astro_data = simulator.simulate_astrometry(cadence, seed=seed)

    # Catalog row mirrors the schema of a survey catalogue lookup
    # (5-parameter astrometry plus the synthetic sentinel source_id).  All
    # numeric values come from the simulator, so the catalogue row describes
    # the same star whose orbit produced rv_data + astro_data.
    catalog_row = {
        "source_id": SYNTHETIC_SOURCE_ID,
        "ra":        simulator.ra_deg,
        "dec":       simulator.dec_deg,
        "parallax":  simulator.parallax_mas,
        "pmra":      simulator.pmra_masyr,
        "pmdec":     simulator.pmdec_masyr,
        "m1":        simulator.m1_msun,
    }

    return SimulatedDemoBundle(
        rv_data=rv_data,
        astro_data=astro_data,
        catalog_row=catalog_row,
        truth=simulator,
        seed=seed,
        source_id=SYNTHETIC_SOURCE_ID,
    )
