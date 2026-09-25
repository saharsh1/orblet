"""Input coercion shared by the periodogram modules."""

from __future__ import annotations

import numpy as np


def _as_1d_float(x, name: str) -> np.ndarray:
    """Coerce to 1-D float64 array with a clear error message."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr
