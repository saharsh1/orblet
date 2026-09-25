"""The linear cores: given the non-linear elements, solve the rest exactly.

A Keplerian orbit splits cleanly in two. Fix the geometry — period,
eccentricity, time of periastron, and for astrometry the inclination and node
— and what remains enters the model LINEARLY: the radial-velocity amplitudes,
the Thiele-Innes constants, the systemic velocity, the five astrometric
parameters. Linear means there is a closed-form best fit, so that half of the
problem never needs a sampler at all.

That is what these modules do, and it is why a sampler over a handful of
non-linear parameters can outrun one over all of them: the expensive half is
solved, not explored.

    rv.py           the radial-velocity linear core
    astrometry.py   the Thiele-Innes linear core

Both return the solution AND the marginal likelihood with the linear
parameters integrated out, which is what makes them usable as the inner step
of whatever outer optimiser or sampler you assemble.
"""

from __future__ import annotations

__all__: list[str] = []
