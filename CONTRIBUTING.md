# Contributing to orblet

## The one test for what belongs here

**Is it an atom, or is it pipeline?**

An atom answers a question on its own: arrays in, a result out, no knowledge
of where the arrays came from. A pipeline step only makes sense inside
someone's workflow — it needs a chain schema, a file layout, a mission's
epoch table, a catalogue. Atoms belong here. Pipeline does not, however
useful it is.

The test is sharper than "no Gaia, no files, no network", and it settles
cases that rule cannot: a function computing per-direction parallax factors
from a position is an atom; a function that assembles a diagnostic block from
a fitted chain is pipeline, even though both are "just maths".

## Conventions are not negotiable in silence

Every public function states its units, reference frames and sign
conventions in its docstring, and a change to any of them is a change to
the scientific meaning of every result downstream. If you must change one:
say so in the docstring, the commit message and the changelog, and update
the convention tests in the same commit. A convention that changes without
its test changing is a bug, whichever one of them is right.

The load-bearing ones are listed in the README. Add to that list rather than
to your memory.

## Measured, not inferred

Keep the line between what a function *measures* and what it *infers*
visible in its output. A companion-mass estimate is inferred under stated
assumptions about the primary and the inclination; a semi-amplitude is
measured. Name them differently, document the assumptions, and never let an
inferred quantity travel under a measured one's name.

**A seed is initialisation, never a prior.** A starting point handed to a
sampler carries no evidence and must not be reported as if it did.

## Change tiering

Default to the lightest change that is safe. Plotting, docstrings, comments
and mechanical fixes need a smoke test and one review. Anything that can move
a computed number — orbit mathematics, priors, likelihoods, conventions,
units, the public surface — needs its byte-identity baseline captured on the
unchanged code first, in its own commit, and compared exactly afterwards.

## Tests

- Synthetic data only. Nothing in this repository reads a real measurement.
- Baselines compare with `np.array_equal`, never `allclose`; a baseline is a
  record of what the code *did*, and a tolerance hides the drift it exists to
  catch.
- The import-closure test is the law that keeps orblet standalone. If it goes
  red, the fix is to move an import inside a function or to drop it — not to
  widen the allowed list.

## Where development happens

Here: this repository is the source. Structural changes — new modules,
moved names, changed conventions — land with their tests. A change to the
public surface (a name added, moved or removed, a convention changed) is a
minor-version bump, recorded in `CHANGELOG.md`; a change that moves a
computed number re-blesses its baseline in its own commit and says what
moved.
