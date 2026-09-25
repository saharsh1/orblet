"""Checks that code citations inside the reference manual still point
somewhere real.

``docs/model_and_likelihoods.md`` carries a `file:line` next to almost
every formula. Manuals are prose, not code — nothing stops a citation from
rotting after a rename or a line shifting. This module parses every
``path.py:NNN`` (or ``path.py:NNN-MMM``) citation out of the manual(s) in
``_MANUALS`` and checks two things:

1. the citation resolves to a real file (trying, in order, the literal
   repo-relative path, then the package-relative shorthand under
   ``src/orblet/``, then a unique-basename search) and the cited line is
   within that file's length;
2. the nearest backticked, plausible-python-name identifier appearing
   BEFORE the citation (same line, or up to two lines above) shows up
   somewhere within +/-5 lines of the cited line in the cited file. This
   is a loose anchor check — a substring match on the identifier's last
   dotted component — meant to catch a citation that drifted onto the
   wrong function, not to verify the citation is semantically perfect.

A manual line containing the literal marker ``doc-anchor-exempt`` is
skipped entirely (for a citation that is deliberately historical or
otherwise not checkable).

A failure here means a citation rotted: re-point it to the moved line, or —
for a deliberately historical citation — mark the manual line
``doc-anchor-exempt``. Do not dismiss a red as expected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# One entry today; add a second manual here (one line) when it exists.
_MANUALS = [
    _REPO_ROOT / "docs/model_and_likelihoods.md",
]

_CITATION = re.compile(r"[\w./]+\.py:\d+(?:-\d+)?")

# A backtick-delimited plausible python name: the backticks themselves
# anchor the match, so a citation like `` `solve/rv.py:175` `` (which
# contains "/" and ":") never matches this pattern and is never mistaken
# for its own anchor identifier.
_IDENTIFIER = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]+)`")

_EXEMPT_MARKER = "doc-anchor-exempt"

# Resolution order for a citation's path, tried in turn until a real file
# is found. Manual citations are written repo-relative or package-relative
# (``design/columns.py:216`` for the file under ``src/orblet/design/``).
_RESOLUTION_PREFIXES = (
    "",
    "src/orblet/",
)

#: Package roots searched by the bare-filename fallback below.
_SOURCE_ROOTS = ("src/orblet",)

_ANCHOR_WINDOW = 5
_MAX_REPORTED_OFFENDERS = 30
_MIN_CITATIONS = 40   # the manual carries well over a hundred; guard vacuity


def _resolve(cited_path: str) -> Path | None:
    """Return the first existing file among the resolution candidates.

    Falls back to a UNIQUE-basename search across the source roots for
    the manual's bare-filename shorthand (``companion_mass.py:542`` for the
    file under ``interpret/``): a unique match resolves; zero or multiple
    matches stay unresolved (ambiguity must be spelled out in the manual,
    not guessed here).
    """
    for prefix in _RESOLUTION_PREFIXES:
        candidate = _REPO_ROOT / (prefix + cited_path)
        if candidate.is_file():
            return candidate
    if "/" not in cited_path:
        matches = [
            p
            for root in _SOURCE_ROOTS
            for p in (_REPO_ROOT / root).rglob(cited_path)
            if p.is_file() and "__pycache__" not in p.parts
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _parse_citation(citation_text: str) -> tuple[str, int, int]:
    """Split ``path.py:NNN`` or ``path.py:NNN-MMM`` into (path, start, end)."""
    path_part, _, line_part = citation_text.rpartition(":")
    if "-" in line_part:
        start_text, _, end_text = line_part.partition("-")
        return path_part, int(start_text), int(end_text)
    line_no = int(line_part)
    return path_part, line_no, line_no


def _nearest_identifier(
    manual_lines: list[str], line_idx: int, col: int
) -> str | None:
    """Nearest backticked plausible-python-name before ``(line_idx, col)``,
    searching the current line up to ``col`` then up to two lines above.

    In a markdown TABLE row the search is scoped to the citation's own
    CELL (text after the last ``|`` before ``col``) — without this, the
    index table's search leaks into the previous cell and flags the
    neighbouring notebook name as a missing anchor.
    """
    same_line_prefix = manual_lines[line_idx][:col]
    if "|" in same_line_prefix:
        same_line_prefix = same_line_prefix.rsplit("|", 1)[1]
        # Cell-scoped: do NOT fall through to previous lines for table
        # rows — the rows above are unrelated entries.
        matches = list(_IDENTIFIER.finditer(same_line_prefix))
        return matches[-1].group(1) if matches else None
    same_line_matches = list(_IDENTIFIER.finditer(same_line_prefix))
    if same_line_matches:
        return same_line_matches[-1].group(1)
    for lines_back in (1, 2):
        idx = line_idx - lines_back
        if idx < 0:
            break
        matches = list(_IDENTIFIER.finditer(manual_lines[idx]))
        if matches:
            return matches[-1].group(1)
    return None


def _collect_offenders(manual_path: Path) -> tuple[list[str], int, int]:
    """Returns (offenders, citations seen, anchor checks skipped)."""
    manual_lines = manual_path.read_text(encoding="utf-8").splitlines()
    offenders: list[str] = []
    seen = 0
    anchor_skipped = 0

    for i, manual_line in enumerate(manual_lines):
        if _EXEMPT_MARKER in manual_line:
            continue
        for match in _CITATION.finditer(manual_line):
            seen += 1
            citation_text = match.group(0)
            cited_path, start_line, end_line = _parse_citation(citation_text)

            resolved = _resolve(cited_path)
            if resolved is None:
                offenders.append(
                    f"{manual_path.name}:{i + 1}: `{citation_text}` — "
                    "path does not resolve under any known source root"
                )
                continue

            file_lines = resolved.read_text(encoding="utf-8").splitlines()
            file_length = len(file_lines)
            if start_line > file_length or end_line > file_length:
                offenders.append(
                    f"{manual_path.name}:{i + 1}: `{citation_text}` — "
                    f"{resolved.relative_to(_REPO_ROOT)} has only "
                    f"{file_length} lines"
                )
                continue

            identifier = _nearest_identifier(manual_lines, i, match.start())
            if identifier is None:
                anchor_skipped += 1
                continue

            last_component = identifier.rsplit(".", 1)[-1]
            window_lo = max(1, start_line - _ANCHOR_WINDOW)
            window_hi = min(file_length, end_line + _ANCHOR_WINDOW)
            window_text = "\n".join(file_lines[window_lo - 1 : window_hi])
            if last_component not in window_text:
                offenders.append(
                    f"{manual_path.name}:{i + 1}: `{citation_text}` — "
                    f"anchor `{identifier}` (`{last_component}`) not found "
                    f"within lines {window_lo}-{window_hi} of "
                    f"{resolved.relative_to(_REPO_ROOT)}"
                )

    return offenders, seen, anchor_skipped


@pytest.mark.parametrize("manual_path", _MANUALS, ids=lambda p: p.name)
def test_manual_code_citations_resolve_and_anchor(manual_path: Path) -> None:
    offenders, seen, anchor_skipped = _collect_offenders(manual_path)
    assert seen >= _MIN_CITATIONS, (
        f"only {seen} citations parsed from {manual_path.name}; the regex or "
        "the manual changed shape and this check would pass vacuously"
    )

    message_lines = list(offenders[:_MAX_REPORTED_OFFENDERS])
    if len(offenders) > _MAX_REPORTED_OFFENDERS:
        message_lines.append(
            f"... and {len(offenders) - _MAX_REPORTED_OFFENDERS} more"
        )
    message_lines.append(
        f"[{anchor_skipped} citation(s) had no nearby backticked "
        "identifier — anchor check skipped for them]"
    )
    assert not offenders, "\n".join(message_lines)
