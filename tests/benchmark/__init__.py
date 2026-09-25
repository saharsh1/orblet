"""
Synthetic-data benchmark framework for the orbit fitters.

Public API
----------
- :class:`OrbitSimulator` — frozen orbital + system truth, with
  ``simulate_rv`` / ``simulate_astrometry`` methods that emit dicts
  shaped exactly like the real loaders' returns.
- :class:`Cadence` (Protocol) and concrete :class:`BH3Cadence`,
  :class:`SourceCadence`, :class:`ScanLawCadence` (stub).
- :func:`closure_test` — verify a posterior chain encloses the truth
  inside its (1 − α) credible interval for every named parameter.

Phase 1 deliberately stops at the closure tier.  Bias and coverage
helpers are intentionally absent until a use-case appears.
"""

from .simulator import OrbitSimulator  # noqa: F401
from .cadence import (                 # noqa: F401
    Cadence,
    BH3Cadence,
    SourceCadence,
    ScanLawCadence,
    BH3_SOURCE_ID,
)
from .closure import closure_test       # noqa: F401
from .coverage import coverage_test, ParamCoverage  # noqa: F401
