"""Files the repository shows to the outside: the notebooks and the
citation metadata.

GitHub renders a notebook only if it is valid nbformat 4, and refuses it
whole ("Invalid Notebook") over one missing key. The structural rules below
are the ones a stripped notebook breaks; they are checked with plain JSON so
the test needs no nbformat install. The citation metadata must name the
version that is actually released.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import orblet

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every published notebook; ``notebooks/_local/`` is gitignored scratch.
_NOTEBOOKS = sorted(
    p for p in (_REPO_ROOT / "notebooks").rglob("*.ipynb")
    if "_local" not in p.parts and ".ipynb_checkpoints" not in p.parts
)


def test_notebooks_found() -> None:
    assert len(_NOTEBOOKS) >= 8


@pytest.mark.parametrize(
    "path", _NOTEBOOKS, ids=lambda p: str(p.relative_to(_REPO_ROOT / "notebooks"))
)
def test_notebook_is_valid_nbformat4(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    for i, cell in enumerate(nb["cells"]):
        where = f"{path.name} cell {i}"
        assert {"cell_type", "metadata", "source"} <= cell.keys(), where
        if nb["nbformat_minor"] >= 5:
            assert "id" in cell, where
        if cell["cell_type"] == "code":
            # Required even when empty: a stripped cell carries
            # "execution_count": null and "outputs": [].
            assert "execution_count" in cell, where
            assert "outputs" in cell, where
        else:
            assert "execution_count" not in cell, where
            assert "outputs" not in cell, where


def test_citation_version_matches_package() -> None:
    text = (_REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    assert match, "CITATION.cff has no version line"
    assert match.group(1) == orblet.__version__
