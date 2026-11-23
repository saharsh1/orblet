"""Does this numpy round matrix products like the build the baselines were made with?

The baseline tests compare SHA-256 digests of outputs, so they hold bit for
bit.  A few outputs go through a BLAS matrix product (matrix times vector, or
matrix times matrix).  Two BLAS libraries can round such a product
differently in the last bit: the baselines were made with an OpenBLAS numpy
(conda-forge), and a numpy linked to Apple Accelerate (the PyPI wheel on
macOS arm64) gives results that differ by about 1e-16.  Those results are
correct; only their last bit differs, and a digest cannot tolerate that.

``blas_rounds_like_the_reference()`` runs two fixed matrix products and
compares their bytes with the reference build.  When they differ, the
baseline tests SKIP the cases listed here, and only those, with the reason.
Every other case stays strict on every build.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: Digest of the two probe products on the reference build (OpenBLAS numpy).
REFERENCE_PROBE_SHA256 = "0830396f5f1b74fd2cbb33565e71f8defd2c9b4613020f490f73d352da174cf5"

#: Cases of ``test_periodogram_baseline`` whose arithmetic is inside astropy's
#: Lomb-Scargle and goes through BLAS.
PERIODOGRAM_CASES_THROUGH_BLAS = frozenset({
    "ls_auto.power",
    "ls_lin.power",
    "ls_log.power",
    "ls_nan.power",
    "ls_nterms2.power",
})

#: Cases of ``test_simulator_baseline`` that use the parallax factor, which
#: projects the observer position with a matrix-times-vector product.
SIMULATOR_CASES_THROUGH_BLAS = frozenset({
    "along_scan_parallax_factor.ra180_dec-20",
    "along_scan_parallax_factor.ra180_dec-20.geocentre",
    "along_scan_parallax_factor.ra294.8_dec14.9",
    "bundle.seed0.astro_data.centroid_pos",
    "bundle.seed0.astro_data.parallax_factor_al",
    "bundle.seed1.astro_data.centroid_pos",
    "bundle.seed1.astro_data.parallax_factor_al",
    "cadence.ra45_dec60_seed3.astro_parallax_factor_al",
})

SKIP_REASON = (
    "this numpy rounds BLAS matrix products differently from the build the "
    "baselines were made with (OpenBLAS); the result differs in the last bit "
    "only, which a digest cannot tolerate (see tests/_blas_reference.py)"
)


def _probe_sha256() -> str:
    """Digest of two fixed matrix products: matrix times vector, and a Gram matrix."""
    rng_probe = np.random.default_rng(5)          # seed 5: the fixed probe inputs
    matrix = rng_probe.standard_normal((50, 3))
    vector = np.array([-0.3, 0.9, 0.1])
    matrix_times_vector = matrix @ vector         # BLAS gemv
    gram = matrix.T @ matrix                      # BLAS gemm / syrk
    return hashlib.sha256(matrix_times_vector.tobytes() + gram.tobytes()).hexdigest()


def blas_rounds_like_the_reference() -> bool:
    """True when the two probe products have the bytes of the reference build."""
    return _probe_sha256() == REFERENCE_PROBE_SHA256
