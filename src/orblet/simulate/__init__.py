"""Making data that a fitter can be tested against.

orblet consumes data; this subpackage manufactures it. Give it an orbit and a
cadence and it returns epochs, radial velocities and along-scan abscissae with
known truth attached — which is the only way to ask whether a fitter recovers
what is actually there.

Everything here is synthetic by construction. Nothing reads a file, queries a
service, or knows what a mission is. That is what makes the examples runnable
by anyone, and what keeps the pedagogical notebooks free of the data-access
problem entirely.

    orbit.py     the orbit itself: elements in, observables out
    cadence.py   when the observations happen, and at what scan angle
"""

from __future__ import annotations

__all__: list[str] = []
