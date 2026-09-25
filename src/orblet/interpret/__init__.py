"""Turning a fitted orbit into a statement about the system.

Everything here sits AFTER the fit. The solvers produce orbital elements and
their uncertainties; these modules take that output and ask what it implies —
a companion mass, a flux ratio, an upper limit on an undetected companion, a
place on the astrometric mass-ratio function.

The separation is deliberate and it is the one in §9 of the project's
conventions: a measured quantity and an inferred one are not the same kind of
thing, and the inference carries assumptions the measurement does not. Keeping
the inference in its own subpackage makes it obvious which is which at the
import line.

The public names reach users through orblet's front door; this subpackage
is where they are defined, grouped by the question each one answers.
"""

from __future__ import annotations

__all__: list[str] = []
