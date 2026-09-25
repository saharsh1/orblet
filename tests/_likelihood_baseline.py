"""Byte-identity baseline of the three log-likelihoods.

Compared exactly, not with a tolerance. If a value here changes, re-bless it
in its own commit and say what moved.
"""

from __future__ import annotations

LIKELIHOOD_BASELINE: dict[str, float] = {
    'loglike_along_scan': -28.268717853337947,
    'loglike_joint': -1461.4617123599187,
    'rv_loglike': -782.8789536221502,
}
