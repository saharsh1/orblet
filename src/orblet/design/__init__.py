"""Design matrices: the column functions and the builders that compose them.

The package has no leading underscore because :mod:`.columns` DEFINES
three names that ``docs/open_api.md`` publishes — ``rv_design_matrix``,
``ti_design_matrix`` and ``acceleration_columns`` — and a private-by-name
package should not host public API.  The definition site says what the
names are; their public path is orblet's front door.


This package is not itself part of the public surface: the public names reach
users through orblet's front door (``orblet.rv_design_matrix`` and friends),
and the BUILDERS below remain an internal composition seam.  The package
name is about the definition site telling the truth, not about widening
the public surface: ``orblet`` (the front door) is the only public path.


Layout, strictly one-way (top depends on bottom)::

    _constants → _kepler → rv.forward / astro.forward
                             → design.columns  (the arithmetic)
                               → design.builder (composes columns into blocks)
                                 → the channel solvers

:mod:`.columns` holds the arithmetic; :mod:`.builder` holds the pluggable
block contract. See :mod:`.builder` for that contract; the split keeps the
arithmetic importable by every channel without an import cycle.
"""

from orblet.design.builder import (
    AccelerationBlockBuilder,
    DesignBuilder,
    JointReducedAstroBuilder,
    JointReducedRvBuilder,
    RelativeDriftBuilder,
    RelativeTiBuilder,
    RvOrbitBuilder,
    SingleStarBlockBuilder,
    TiOrbitBuilder,
    concat_columns,
)

__all__ = [
    "AccelerationBlockBuilder",
    "DesignBuilder",
    "JointReducedAstroBuilder",
    "JointReducedRvBuilder",
    "RelativeDriftBuilder",
    "RelativeTiBuilder",
    "RvOrbitBuilder",
    "SingleStarBlockBuilder",
    "TiOrbitBuilder",
    "concat_columns",
]
