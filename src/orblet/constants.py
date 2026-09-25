"""Single source of truth for physical and calendar constants used across the package.

Every number a calculation depends on is defined here exactly once, so a
value and the claim made about it cannot drift apart. ``MSUN_KG``, for
instance, is DERIVED from the IAU 2015 nominal ``GM_sun`` rather than pinned
as a literal, because a literal and a comment saying where it came from had
already disagreed by 2.6e-4 once.

Migrating these to ``astropy.constants`` is on the roadmap (item 1);
it is a behaviour-visible change (the values move by ulps) and wants its own
baseline.

All values are plain Python floats and strings, used directly by this
package's numpy code; none is interpolated into another language or file
format, so ordinary float64 representation is all that matters. Where a
value is DERIVED from another (``MSUN_KG`` from the nominal GM), it is
computed here rather than typed, so the two cannot drift apart. The
astropy migration above would move values by ulps only.

Time-anchor constants:

- ``MJD_J2010_TCB`` (55197.0): MJD of J2010-01-01T00:00:00 TCB.  Used by Gaia
  obs_time arithmetic.
- ``MJD_J2016_TCB`` (57388.5): MJD of J2016.0 — the **DR3** reference
  epoch (``gaia_source.ref_epoch``).
- ``MJD_J2017_5_TCB`` (57936.375): MJD of J2017.5 — the **DR4** reference
  epoch.  Defined below; see ``docs/model_and_likelihoods.md``
  §1.2 for which of these day-counts is a catalogue epoch and which is
  not (the J2010 anchor and the RV median epoch are NOT).

These TCB-MJD anchors are bit-identical to
``astropy.Time("J2010.0", scale='tcb').mjd`` and
``astropy.Time("J2016.0", scale='tcb').mjd`` respectively.  astropy treats
``"J2010.0"`` / ``"J2016.0"`` as scale-agnostic calendar epoch labels: the
MJD of the same instant is bit-identical in any scale, so no rate
correction is applied.  The bit-exact agreement is enforced by
``tests/test_constants.py``.

The project's epoch astrometry (Gaia DR3/DR4 ``obs_time_tcb`` column) is
loaded as TCB-days-from-J2010, so these anchors are used as TCB throughout
(no scale conversion ever needed).

Which epoch is which matters.  The catalogue reference epoch is J2016.0
for DR3 and J2017.5 for DR4; J2010 is only the zero of the obs_time day
count and is never a reference epoch.  Astrometric and joint fits must be
given their reference epoch explicitly (``require_epoch_ref_mjd``):
picking one from memory, or from a comment, shifts every exported
periastron time and misapplies every catalogue-derived prior, and nothing
downstream raises.
"""

import math

# Speed of light in km/s — exact (defined value).
C_KMS = 299792.458

# Astronomical unit in metres (IAU 2012 exact).
# Python-only.
AU_M = 1.495978707e11

# Newtonian gravitational constant in SI units (CODATA 2018).
# Python-only.
G_SI = 6.67430e-11

# Solar mass in Jupiter masses (IAU 2015 nominal).
# A nominal value, typed exactly as published so it stays comparable
# with other codes that use the same nominal.
MSUN_IN_MJUP = 1047.35

# Days per Keplerian year: the Julian year.  Periods enter the forward
# models in these years, and proper motions per year of this length;
# the same constant sets both, so the two can never disagree.
DAYS_PER_KEPLER_YEAR = 365.25

# Solar mass in kg, DERIVED from the IAU 2015 nominal GM_sun.
# Python-only.
#
# Derived, not typed, so it is consistent with the nominal GM by
# construction.  A typed 1.98892e30 (a common tabulated value) would give
# G_SI × M = 1.327465e20 m³/s², 2.57e-4 ABOVE the nominal GM_sun, and every
# quantity computed through G × M would inherit that offset.  The measured
# quantity is GM, not M — see the next block.
MSUN_KG = 1.3271244e20 / G_SI

# ── The mass unit, and why it is a GM and not a G × M ────────────────────
#
# GM_sun is measured to ~1e-10; G alone is known only to ~2e-5, so any
# formula needing the solar mass parameter must use a GM DIRECTLY.
# Re-deriving it as ``G_SI * MSUN_KG`` reintroduces exactly the offset
# the derivation above avoids.
#
# ``GM_SUN_SI`` is the GM the project's UNIT SYSTEM implies: masses in
# M_sun, lengths in AU, times in Keplerian years, in which Kepler's third
# law is exactly ``a³ = P² M``.  That convention DEFINES the mass unit as
# 4π²AU³/yr² — and it is the unit the astrometric mass function
# ``fm_ast = a_phot³ / P²`` is already expressed in.  The spectroscopic
# ``fm_from_K`` MUST use the same unit, or the two mass functions — and so
# the deficit D, which compares them — carry a 2.19e-4 systematic.  That
# was exactly finding M-1.
#
# It sits 3.8e-5 from the IAU nominal GM_sun.  That residual is a property
# of defining a mass unit through the AU and the Julian year, not an
# error; the same test pins it so it cannot silently widen.
GM_SUN_SI = 4.0 * math.pi ** 2 * AU_M ** 3 / (
    DAYS_PER_KEPLER_YEAR * 86400.0
) ** 2

# Mass-convention sentinels for the cross-engine chains-dict ``_meta``
# entry.  RV-only fits implicitly fix sin(i)=1, so the sampled
# companion mass is the projected (minimum) mass m_2 × sin(i_true).
# Joint RV + astrometric fits measure the inclination, recover the true
# m_2 and emit the ``true_m2`` sentinel instead.  See the reference
# manual, §2.6 (what the RV channel does not measure) and §6.1.
#
MASS_CONVENTION_PROJECTED = "projected_m2_sini"
MASS_CONVENTION_TRUE = "true_m2"

# Third sentinel, for companion masses formed AFTER a Thiele-Innes
# astrometric fit from the exported observable ``fm_ast`` (Stage 3a of
# the observables-first track).  Three facts distinguish it, and all
# three must hold wherever it appears:
#   1. the inclination is MEASURED — it is already folded into the
#      photocentre semi-major axis recovered from the (A, B, F, G)
#      amplitudes, so no sin(i) is assumed and none is applied;
#   2. the primary mass is EXTERNAL — supplied by the caller, not
#      constrained by the astrometry, so the companion mass inherits
#      that input's uncertainty;
#   3. the photocentre light ratio is assumed ZERO (β = 0, dark
#      companion).  A luminous secondary biases ``fm_ast`` in EITHER
#      direction; the bias vanishes only at β/(1+β) = 2·m2/M.
# ``true_m2`` is NOT a substitute: it carries no β claim and no
# external-m1 claim, so folding this case into it would silently drop
# both caveats.
MASS_CONVENTION_ASTROMETRIC_EXTERNAL_M1 = "astrometric_external_m1"

# Inclination domain sentinels for the cross-engine ``_meta`` block.
#
# The inclination latent is ``i = arccos(1 − 2u)`` with ``u ∈ [0, 1]``,
# so ``cos i`` is uniform on ``[−1, +1]`` and the sampled domain is the
# full ``i ∈ [0, π]``: every fit emits ``I_DOMAIN_FULL``.  A half-sphere
# mapping (``u = 1 − cos i``, ``i ∈ [0, π/2]``) would silently choose one
# side of the i ↔ π − i mirror (reference manual, §4.3), which is why
# none is used.
#
# ``I_DOMAIN_UPPER_HEMI`` names that half-sphere domain; it is kept only
# so an archived chain that carries it still decodes, and nothing emits it.
#
# Sentinels are plain string literals (no Enum / dataclass) so the values
# round-trip bit-exactly through JSON / pickle.
I_DOMAIN_UPPER_HEMI = "[0, pi/2]"
I_DOMAIN_FULL = "[0, pi]"

# ω convention (chain-key sentinel for cross-engine and post-processing alignment).
# Fits emit chains-dict ``_meta["omega_convention"]`` ==
# ``OMEGA_CONVENTION_PRIMARY``, signalling that all ``omega_rad`` / ``cos_omega`` /
# ``sin_omega`` chain entries are in the primary's frame (binary-star textbook
# RV form, ``v_r = γ + K [cos(ν + ω) + e cos(ω)]``).  Many planet codes use
# the companion's argument of periastron instead; the two are related by
# ``ω_primary = ω_companion + π``, so convert at the boundary when chains
# are exchanged with such a code.  The sign conventions this rests on are
# in the reference manual, §1.4 and §2.1.
#
OMEGA_CONVENTION_PRIMARY = "primary_argument_of_periastron"

# The exported period column ``P_days`` is in DAYS (the sampler's
# latent stays in Keplerian years; the export is ONE
# exact multiply by DAYS_PER_KEPLER_YEAR).  Emitted as
# ``_meta["P_unit"]`` where a period-unit sentinel exists.
P_UNIT_DAYS = "days"

# Semi-major-axis convention sentinels.  ``a_rel_au`` carries one of
# two semantics across the cross-engine chains-dict, signalled by
# ``_meta["a_convention"]``:
#
# - ``A_CONVENTION_TOTAL_RELATIVE``: ``a_rel_au`` is the total-relative
#   semi-major axis (AU), i.e. ``a_rel = (P² × M_total)^{1/3}`` via
#   Kepler III with total mass.  This is what CAMPBELL-basis chains
#   carry directly.  Downstream consumers compute the photocenter sky
#   amplitude as ``a_phot_mas = b_a × (b_mass / M) × plx``.
#
# - ``A_CONVENTION_PHOTOCENTER``: the semi-axis column is the photocenter semi-major
#   axis (AU).  Reserved for non-default emit paths (e.g. legacy chain
#   dicts, or future engines that expose the
#   photocenter natively).
#
# THIELE-INNES chains carry NEITHER (they report observables first):
# they have no ``a_rel_au`` and no ``a_convention`` sentinel —
# a TI ``a_rel_au`` would be ``a_phot_au × M / m2``, the measured
# photocentre axis scaled by a PRIOR-ONLY mass ratio, so it is not
# exported.  TI chains expose ``a_phot_au`` (AU,
# photocenter) and ``a_phot_mas`` (mas, photocenter, = a_phot_au × plx)
# directly.
A_CONVENTION_TOTAL_RELATIVE = "total_relative_a_au"
A_CONVENTION_PHOTOCENTER = "photocenter_a_au"

# MJD of J2010-01-01T00:00:00 TCB.  Anchors Gaia OBMT and obs_time
# arithmetic (Gaia's "obs_time in days" is days since this epoch).
MJD_J2010_TCB = 55197.0

# MJD of J2016.0 — the Gaia **DR3** reference epoch.  Bit-identical to
# ``astropy.Time("J2016.0", scale='tcb').mjd`` — astropy treats the
# ``"J2016.0"`` label as scale-agnostic, so no rate correction is applied.
# The bit-exact agreement is enforced by ``tests/test_constants.py``.
MJD_J2016_TCB = 57388.5

# MJD of J2017.5 — the Gaia **DR4** reference epoch (``gaia_source.ref_epoch``).
# DR4 moved the reference epoch 1.5 yr later than DR3; the DR4 documentation
# (release 1.2) quotes J2017.5 for the 2-, 5- and 6-parameter solutions alike.
#
# The superseded TODO here proposed ``57935.0`` — that value is WRONG by 1.375 d.
# ``astropy.Time('2017.5', format='jyear').mjd`` gives 57936.375, and
# 57936.375 - 55197.0 = 2739.375 d = 7.5 x 365.25 exactly.  Using DR3's epoch on
# DR4 data leaves a spurious scan-modulated residual of |pm| x 1.5 yr (7.5 mas for
# a 5 mas/yr star, against ~0.1 mas errors), so this value is pinned by
# ``tests/test_epoch_astrometry_percc.py``.
MJD_J2017_5_TCB = 57936.375


def require_epoch_ref_mjd(epoch_ref_mjd, *, channel: str) -> float:
    """Return ``epoch_ref_mjd`` as a float, or raise if it is ``None``.

    Astrometric and joint fits must be TOLD their reference epoch — they may not
    assume one.  The reference epoch is the CATALOGUE epoch (DR3 = J2016.0 = MJD
    57388.5; DR4 = J2017.5 = MJD 57936.375), and it sets BOTH the periastron
    zero-point AND the position/proper-motion reference.  A wrong epoch silently
    shifts the exported ``tp`` and misapplies any catalogue-derived prior, so the
    solver refuses to guess.  With a loader-shaped bundle it is
    ``ref_epoch_days + MJD_J2010_TCB``.

    The RV channel does NOT use this: there the reference epoch is a pure ω gauge,
    correctly derived from the data as the median observation epoch.

    Parameters
    ----------
    epoch_ref_mjd
        The value to check (an MJD), or ``None``.
    channel
        ``"astrometric"`` or ``"joint"`` — named in the error for the caller.
    """
    if epoch_ref_mjd is None:
        raise ValueError(
            f"epoch_ref_mjd is required for a {channel} fit and was not supplied.  It "
            f"is the catalogue reference epoch (DR3 J2016.0 = MJD 57388.5; DR4 J2017.5 "
            f"= MJD 57936.375) and sets both the periastron zero-point and the "
            f"position/PM reference.  Refusing to assume it — a wrong epoch silently "
            f"shifts the exported tp.  The batch reads it from the data bundle's "
            f"'ref_epoch_days' (+ MJD_J2010_TCB); pass epoch_ref_mjd explicitly for a "
            f"standalone call."
        )
    return float(epoch_ref_mjd)


# Sample-convention sentinel for the engine-side ``simulate`` sub-dict.
# Records the rule the helper uses to pick the representative MAP row:
# the argmax of ``chains["logpost"]`` AFTER discard_fraction and AFTER
# walker-flatten (so the index is into the flat post-discard sample
# matrix).  Plain string literal so the value round-trips through JSON
# / pickle bit-exactly.
SIMULATE_SAMPLE_CONVENTION_MAP = "logpost_argmax_post_discard_post_flatten"

# Default ``astropy.coordinates.solar_system_ephemeris`` for the
# parallax-factor helpers in :mod:`orblet.parallax`.  ``"builtin"`` is
# astropy's own analytic ephemeris (ERFA ``epv00``): shipped with astropy,
# no download, no extra package.  It places the Earth within ~5 km of
# JPL DE432s over 2010-2020 — about 3e-8 AU, far below anything a
# parallax factor is used for (Gaia's L2 offset alone is 1.5e6 km).
# A JPL kernel is one argument away (``ephemeris="de432s"``) after a
# one-time ``orblet.parallax.fetch_ephemeris("de432s")``, which needs
# the ``[ephemeris]`` extra (jplephem) and a ~10 MB download.  Gaia's
# bundled ``parallax_factor_al`` has its own ephemeris vintage either
# way; for real data, use the bundled column.
SOLAR_SYSTEM_EPHEMERIS_PIN = "builtin"

# Channel sentinel for ``simulate["_meta"]["channel"]``.  The Python
# astrometric engine emits ``"astrometric"`` for both basis branches;
# the joint engine also emits this value because the simulate sub-dict
# carries the ASTROMETRIC channel only (the RV channel has no simulate
# sub-dict yet — see follow-up batch).
SIMULATE_CHANNEL_ASTROMETRIC = "astrometric"


#: The names user code may rely on. UNCHANGED by the B2 merge: a merge
#: changes where a name lives, not whether it is public. `GM_SUN_SI`,
#: `MJD_J2017_5_TCB`, `P_UNIT_DAYS` and `require_epoch_ref_mjd` are defined
#: above and deliberately NOT listed, exactly as before -- promoting them is
#: an additive surface decision of its own, not a side effect of merging.
__all__ = [
    "A_CONVENTION_PHOTOCENTER",
    "A_CONVENTION_TOTAL_RELATIVE",
    "AU_M",
    "C_KMS",
    "DAYS_PER_KEPLER_YEAR",
    "G_SI",
    "I_DOMAIN_FULL",
    "I_DOMAIN_UPPER_HEMI",
    "MASS_CONVENTION_ASTROMETRIC_EXTERNAL_M1",
    "MASS_CONVENTION_PROJECTED",
    "MASS_CONVENTION_TRUE",
    "MJD_J2010_TCB",
    "MJD_J2016_TCB",
    "MSUN_IN_MJUP",
    "MSUN_KG",
    "OMEGA_CONVENTION_PRIMARY",
    "SIMULATE_CHANNEL_ASTROMETRIC",
    "SIMULATE_SAMPLE_CONVENTION_MAP",
    "SOLAR_SYSTEM_EPHEMERIS_PIN",
]
