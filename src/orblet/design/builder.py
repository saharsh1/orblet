"""Pluggable design-matrix builders (columns-only contract).

The linear orbit engines fit a model that is linear in a set of amplitudes at
a FIXED non-linear orbit shape ``(frequency/period, e, τ, …)``.  The design
matrix ``X`` holds one column per amplitude — the partial derivative of the
along-scan / RV model w.r.t. that amplitude.  This module makes column
construction a pluggable
INGREDIENT so future observables (relative-orbit, acceleration, two-source)
plug in as builders instead of forking a whole engine.

Scope of the contract (deliberately NARROW)
-------------------------------------------
A builder owns ONLY the design columns and their metadata:

- ``build(...)``            → the ``(n, n_col)`` float64 design matrix at a
                             fixed non-linear shape;
- ``column_names``         → the fixed column order (load-bearing: β-prior
                             slots and chain export index by it);
- ``column_units``         → per-column physical units — the unit of the
                             AMPLITUDE each column multiplies (the columns
                             themselves are the model partials);
- ``n_col``                → number of columns.

It does NOT own β-prior widths, latent decode / chain export, or gauge folds
— those stay engine-side, unchanged.  The DEFAULT builders here DELEGATE to
the column functions in :mod:`.columns`, so
every number is byte-for-byte identical to the engine path.

Composability
-------------
A future two-source / relative model is a horizontal concatenation of blocks
(single-star ⊕ orbit ⊕ relative).  :func:`concat_columns` provides the
column-stack primitive with the C-contiguity / dtype guarantees the GLS core
relies on; block builders compose through it without re-implementing geometry.

No echoed values
----------------
No value, index, or shape is logged, printed, or interpolated into any
message.  Every error string is a module-level constant.  The ``forward``
atoms are imported at module scope from :mod:`.columns`, which sits BELOW
the channel solvers -- so there is no cycle to avoid.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# The column arithmetic lives in this package, so these are MODULE-SCOPE
# imports.  Were the columns defined in the channel modules instead, these
# would have to be function-local imports reaching back UP into the
# modules that call these builders — a mutual dependency in which laziness
# would be the only thing keeping the package importable.
from orblet.design.columns import (
    acceleration_columns,
    astrometric_columns,
    reduced_astro_design,
    reduced_rv_design,
    rv_design_matrix,
    ti_design_matrix,
)

# Value-free error constants (no data value / shape ever interpolated).
_NEGATIVE_NCOL_MSG = "a design builder must declare a positive column count."
_NAMES_UNITS_LENGTH_MSG = (
    "column_names and column_units must have length n_col."
)
_CONCAT_EMPTY_MSG = "concat_columns requires at least one column block."
_CONCAT_ROWS_MSG = "all column blocks must share the same number of rows."


@runtime_checkable
class DesignBuilder(Protocol):
    """Columns-only design-matrix builder contract.

    Implementations expose the fixed column metadata as attributes and build
    the design matrix at a caller-supplied non-linear shape via keyword
    arguments (each concrete builder documents its own shape kwargs).  The
    returned array is ``(n, n_col)`` float64 and C-contiguous.
    """

    column_names: tuple[str, ...]
    column_units: tuple[str, ...]

    @property
    def n_col(self) -> int:  # pragma: no cover - trivial
        ...

    def build(self, *args, **kwargs) -> np.ndarray:  # pragma: no cover - protocol
        ...


def concat_columns(blocks: list[np.ndarray]) -> np.ndarray:
    """Horizontally stack design-column blocks into one ``(n, Σ n_col)`` matrix.

    The GLS core forms ``M = Xᵀ Σ⁻¹ X``; a contiguity change in ``X`` can alter
    the BLAS reduction order and drift the result at the ULP level.  This
    primitive therefore returns a C-contiguous float64 array explicitly, so
    composed builders match a hand-built ``np.empty((n, k)); X[:, a:b] = …``
    layout byte-for-byte.

    Parameters
    ----------
    blocks : list of np.ndarray
        Each ``(n, k_i)`` float64.  All must share the same ``n``.

    Returns
    -------
    np.ndarray
        ``(n, Σ k_i)`` float64, C-contiguous.
    """
    if not blocks:
        raise ValueError(_CONCAT_EMPTY_MSG)
    arrs = [np.asarray(b, dtype=float) for b in blocks]
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError(_CONCAT_ROWS_MSG)
    out = np.empty((n, sum(a.shape[1] for a in arrs)), dtype=float)
    col = 0
    for a in arrs:
        out[:, col:col + a.shape[1]] = a
        col += a.shape[1]
    return out


class SingleStarBlockBuilder:
    """The 5-column single-star along-scan block ``[Δα*, Δδ, μα*, μδ, ϖ]``.

    Delegates verbatim to
    :func:`.columns.astrometric_columns`
    — the block reused (byte-identical) by ``ti_design_matrix`` cols 4–8, the
    public ``astrometric_5param_design_matrix``, and the joint reduced-astro
    design.  Orbit-shape-INDEPENDENT: ``build`` takes only geometry.

    Column order / units are the engine's fixed convention: position offsets
    ``Δα*, Δδ`` in mas, proper motions ``μα*, μδ`` in mas/yr (``μα* = pmra``
    already includes ``cos δ``), parallax ``ϖ`` dimensionless-factor column.
    """

    column_names: tuple[str, ...] = (
        "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
    )
    # Amplitude units (the parallax AMPLITUDE ϖ is in mas; its column is the
    # dimensionless parallax factor — units name the amplitude, not the column).
    column_units: tuple[str, ...] = (
        "mas", "mas", "mas/yr", "mas/yr", "mas",
    )

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        psi: np.ndarray,
        parallax_factor_al: np.ndarray,
        *,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 5)`` single-star block (delegates, byte-identical).

        Parameters mirror ``astrometric_columns``: ``t_mjd`` / ``epoch_ref_mjd``
        share one time scale (MJD-TCB), ``psi`` in radians, ``parallax_factor_al``
        the dimensionless along-scan parallax factor.
        """
        return astrometric_columns(
            t_mjd, psi, parallax_factor_al, epoch_ref_mjd=epoch_ref_mjd,
        )


class TiOrbitBuilder:
    """The 9-column astrometric Thiele-Innes design
    ``[A, B, F, G, Δα*, Δδ, μα*, μδ, ϖ]``.

    Delegates verbatim to
    :func:`.columns.ti_design_matrix`
    (whose columns 4–8 are the single-star block — see
    :class:`SingleStarBlockBuilder`).  Orbit-shape-DEPENDENT: ``build`` takes
    the fixed non-linear shape ``(f_per_day, ecc, tau)`` plus geometry.
    Amplitude units: the four photocentre Thiele-Innes amplitudes A–G in mas,
    then the single-star amplitudes as in the 5-column block.
    """

    column_names: tuple[str, ...] = (
        "A", "B", "F", "G",
        "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
    )
    column_units: tuple[str, ...] = (
        "mas", "mas", "mas", "mas",
        "mas", "mas", "mas/yr", "mas/yr", "mas",
    )

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        psi: np.ndarray,
        parallax_factor_al: np.ndarray,
        *,
        f_per_day: float,
        ecc: float,
        tau: float,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 9)`` TI design (delegates, byte-identical).

        ``f_per_day`` is the orbital frequency in cycles/day
        (``P_days = 1/f_per_day``); ``tau`` the periastron-phase fraction in
        ``[0, 1)``; ``t_mjd`` / ``epoch_ref_mjd`` share one time scale.
        """
        return ti_design_matrix(
            t_mjd, psi, parallax_factor_al,
            f_per_day=f_per_day, ecc=ecc, tau=tau,
            epoch_ref_mjd=epoch_ref_mjd,
        )


class JointReducedRvBuilder:
    """The 2-column reduced joint RV design ``[γ, K]``.

    Delegates verbatim to
    ``.columns.reduced_rv_design`` — the K column is
    ``cos(ν + ω) + e·cos ω`` at fixed ``(P_yr, e, ω, τ)``.  Amplitudes γ, K
    in km/s.
    """

    column_names: tuple[str, ...] = ("gamma", "K")
    column_units: tuple[str, ...] = ("km/s", "km/s")

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        *,
        P_yr: float,
        e: float,
        omega: float,
        tau: float,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 2)`` reduced RV design (delegates, byte-identical)."""
        return reduced_rv_design(
            t_mjd, P_yr=P_yr, e=e, omega=omega, tau=tau,
            epoch_ref_mjd=epoch_ref_mjd,
        )


class JointReducedAstroBuilder:
    """The 6-column reduced joint astrometric design
    ``[a_phot, Δα*, Δδ, μα*, μδ, ϖ]``.

    Delegates verbatim to
    ``.columns.reduced_astro_design`` — column 0 is the
    unit-amplitude photocentre orbit column (so ``a_phot × col0`` reproduces
    the full along-scan orbit), columns 1–5 are the single-star block
    VERBATIM (:class:`SingleStarBlockBuilder`).  Amplitude units: a_phot in
    mas, then the single-star amplitudes.
    """

    column_names: tuple[str, ...] = (
        "a_phot", "ra_offset", "dec_offset", "pmra", "pmdec", "plx",
    )
    column_units: tuple[str, ...] = (
        "mas", "mas", "mas", "mas/yr", "mas/yr", "mas",
    )

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        psi: np.ndarray,
        parallax_factor_al: np.ndarray,
        *,
        P_yr: float,
        e: float,
        omega: float,
        Omega: float,
        cos_i: float,
        tau: float,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 6)`` reduced astro design (delegates, byte-identical)."""
        return reduced_astro_design(
            t_mjd, psi, parallax_factor_al,
            P_yr=P_yr, e=e, omega=omega, Omega=Omega, cos_i=cos_i, tau=tau,
            epoch_ref_mjd=epoch_ref_mjd,
        )


class RelativeDriftBuilder:
    """The 4-column RELATIVE linear-drift block ``[Δα*₀, Δδ₀, Δμα*, Δμδ]``.

    Track-1 rung 1: the relative separation of a resolved pair
    (Δ ≡ component 2 − component 1) modeled as a static offset + linear
    drift.  Delegates to :func:`.columns.astrometric_columns`
    and slices off the parallax column — the relative coordinate has NO
    parallax column because for a BOUND, common-distance pair the parallax
    (and center-of-mass proper motion) cancels; that cancellation is a MODEL
    ASSUMPTION (a chance-projection pair carries a differential parallax —
    a model built on this assumption cannot represent that).

    Sky-axis convention (same projection as ``along_scan_model``):
    sinψ ↔ Δα* (RA*), cosψ ↔ Δδ (Dec); ``d_pmra = Δμα*`` includes cos δ.
    """

    column_names: tuple[str, ...] = (
        "d_ra_offset", "d_dec_offset", "d_pmra", "d_pmdec",
    )
    column_units: tuple[str, ...] = ("mas", "mas", "mas/yr", "mas/yr")

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        psi: np.ndarray,
        *,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 4)`` relative drift block (delegated slice)."""
        # Dummy zero parallax factor: the delegate requires the argument but
        # the sliced columns 0-3 never touch it (the parallax column is
        # dropped BY DESIGN — the relative coordinate has none).
        t = np.asarray(t_mjd, dtype=float)
        zeros = np.zeros(t.shape[0], dtype=float)
        block = astrometric_columns(t, psi, zeros, epoch_ref_mjd=epoch_ref_mjd)
        # Column slice of a C-contiguous array is not C-contiguous; restore it
        # (values unchanged) to honor the catalog's contiguity guarantee.
        return np.ascontiguousarray(block[:, 0:4])


class AccelerationBlockBuilder:
    """The 2-column sky-plane acceleration block ``[a_ra*, a_dec]``.

    Delegates to :func:`.columns.acceleration_columns`
    (½-in-column convention: the amplitude IS the physical acceleration in
    mas/yr²).  MUST be built with the SAME ``epoch_ref_mjd`` as the drift /
    proper-motion block: under a common origin shift the acceleration is
    invariant but the PM absorbs +a·Δ and the offset μ·Δ + a·Δ²/2.
    Shared primitive for Track-1 rung 2 and the future t² single-star engine.
    """

    column_names: tuple[str, ...] = ("d_acc_ra", "d_acc_dec")
    column_units: tuple[str, ...] = ("mas/yr**2", "mas/yr**2")

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        psi: np.ndarray,
        *,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 2)`` acceleration block (delegates)."""
        return acceleration_columns(t_mjd, psi, epoch_ref_mjd=epoch_ref_mjd)


class RelativeTiBuilder:
    """The 4-column RELATIVE Thiele-Innes orbit block
    ``[A_rel, B_rel, F_rel, G_rel]`` at fixed ``(f_per_day, e, τ)``.

    Track-1 rung 3's column set.  Delegates to
    :func:`.columns.ti_design_matrix` columns 0–3 — the
    column FORM is identical to the photocentre TI orbit, but the amplitudes
    scale the RELATIVE orbit ``a_rel`` (component separation), NOT the
    photocentre ``a_phot``.  Footgun note: **A, F pair with cosψ (Dec); B, G
    pair with sinψ (RA*)** — ``d_ra = B·X + G·Y`` projects with sinψ and
    ``d_dec = A·X + F·Y`` with cosψ.

    Sign gauge: swapping the reference component (Δ → −Δ) negates all four
    amplitudes — at the angle level, (A,B,F,G) → −(A,B,F,G) ≡ Ω → Ω+π.  The
    reference-component choice is fixed by the upstream reduction and must
    feed ALL rungs identically; the along-scan fit cannot break the swap
    degeneracy.  (The rung-3 sampler must NOT reuse the photocentre Klein-4
    gauge fold — the relative gauge differs.)
    """

    column_names: tuple[str, ...] = ("A_rel", "B_rel", "F_rel", "G_rel")
    column_units: tuple[str, ...] = ("mas", "mas", "mas", "mas")

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        psi: np.ndarray,
        *,
        f_per_day: float,
        ecc: float,
        tau: float,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 4)`` relative TI block (delegated slice)."""
        # Dummy zero parallax factor: only the sliced orbit columns 0-3 are
        # used; the single-star block (cols 4-8) is discarded.  (A future
        # proven-no-op could extract a parallax-free _ti_orbit_columns.)
        t = np.asarray(t_mjd, dtype=float)
        zeros = np.zeros(t.shape[0], dtype=float)
        X = ti_design_matrix(
            t, psi, zeros, f_per_day=f_per_day, ecc=ecc, tau=tau,
            epoch_ref_mjd=epoch_ref_mjd,
        )
        return np.ascontiguousarray(X[:, 0:4])


class RvOrbitBuilder:
    """The 3-column SB1 RV design ``[γ, C, S] = [γ, K cosω, K sinω]``.

    Delegates verbatim to
    :func:`.columns.rv_design_matrix`.
    Orbit-shape-DEPENDENT: ``build`` takes the fixed non-linear shape
    ``(period_yr, ecc, tau)`` and the reference epoch.  Columns are the RV
    model partials ``∂v/∂γ = 1``, ``∂v/∂C = cos ν + e``, ``∂v/∂S = −sin ν``;
    ``C, S`` in km/s, γ in km/s (the offset column is dimensionless-1 but the
    amplitude it multiplies is km/s).
    """

    column_names: tuple[str, ...] = ("gamma", "C", "S")
    column_units: tuple[str, ...] = ("km/s", "km/s", "km/s")

    @property
    def n_col(self) -> int:
        return len(self.column_names)

    def build(
        self,
        t_mjd: np.ndarray,
        *,
        period_yr: float,
        ecc: float,
        tau: float,
        epoch_ref_mjd: float,
    ) -> np.ndarray:
        """Return the ``(n, 3)`` RV design (delegates, byte-identical).

        ``period_yr`` in Keplerian years, ``ecc`` in ``[0, 1)``, ``tau`` the
        periastron-phase fraction in ``[0, 1)``; ``t_mjd`` / ``epoch_ref_mjd``
        share one time scale.
        """
        return rv_design_matrix(
            t_mjd, period_yr=period_yr, ecc=ecc, tau=tau,
            epoch_ref_mjd=epoch_ref_mjd,
        )
