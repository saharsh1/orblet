"""
Thin re-export shim for backward-compatible test-side imports.

The :class:`OrbitSimulator` class was lifted to
:mod:`orblet.simulate.orbit` (Batch 7 sub-cycle 7a).  This module
preserves the existing ``from benchmark.simulator import …`` and
``from benchmark import OrbitSimulator`` import paths used throughout
``tests/`` by re-exporting the lifted symbol.

Identity contract: ``benchmark.simulator.OrbitSimulator`` is *the same
Python object* as ``orblet.simulate.orbit.OrbitSimulator`` — not a
sibling copy.  Bit-identical-output tests (test 7.8) rely on this
identity to guard against future divergence.
"""

from orblet.simulate.orbit import OrbitSimulator  # noqa: F401

# ``thiele_innes_al`` is also re-exported so that
# engine contract tests can do
# ``from benchmark.simulator import thiele_innes_al`` without an
# ImportError-driven skip path.  Both names refer to the canonical
# implementation in :mod:`orblet.kepler`.
from orblet.kepler import thiele_innes_al  # noqa: F401

__all__ = ["OrbitSimulator", "thiele_innes_al"]
