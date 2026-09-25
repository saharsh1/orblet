"""orblet's own packaging contract.

Three claims a fresh install must satisfy, checked in fresh interpreters so a
warm ``sys.modules`` in the test session cannot mask a leak:

* the distribution is installed and reports a version;
* ``import orblet`` costs nothing heavy — no scipy, matplotlib, astropy, and
  no optional extra;
* the 36-name front door resolves, name by name, to the object it promises.

Synthetic only; no data files are read.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys

import pytest

FRONT_DOOR_SIZE = 36

#: Must NOT be loaded by ``import orblet``: the heavy end of its own
#: dependencies, which the lazy front door exists to keep out of import time.
MUST_NOT_LOAD = (
    "scipy", "matplotlib", "astropy",
    "emcee", "corner", "pandas", "jplephem",
)


def _fresh(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_distribution_is_installed_with_a_version():
    version = importlib.metadata.version("orblet")
    assert version, "orblet is importable but not installed as a distribution"
    parts = version.split(".")
    assert len(parts) >= 2 and all(p.isdigit() for p in parts[:2]), version


def test_version_attribute_matches_installed_metadata():
    """One version, written once: ``orblet.__version__`` is what the build shipped."""
    orblet = importlib.import_module("orblet")
    assert orblet.__version__ == importlib.metadata.version("orblet")


def test_import_orblet_loads_nothing_heavy():
    """The front door is lazy; importing it must not pay for what it can reach."""
    code = (
        "import sys, json; import orblet; "
        f"print(json.dumps(sorted(m for m in {MUST_NOT_LOAD!r} "
        "if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules))))"
    )
    result = _fresh(code)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert not loaded, f"import orblet loaded: {loaded}"


def test_front_door_size_and_each_name_resolves():
    orblet = importlib.import_module("orblet")
    names = list(orblet.__all__)
    assert len(names) == FRONT_DOOR_SIZE, (
        f"front door has {len(names)} names, expected {FRONT_DOOR_SIZE}; "
        "a change here is a public-API change and must be deliberate"
    )
    assert len(set(names)) == len(names), "duplicate names in __all__"
    for name in names:
        obj = getattr(orblet, name)
        assert obj is not None, name


@pytest.mark.parametrize("name", sorted(importlib.import_module("orblet").__all__))
def test_front_door_name_is_the_defining_object(name: str):
    """No silent wrappers: the public name IS the object its home module defines.

    Read from the front door's own map, ``orblet._LAZY_EXPORTS``, which names
    the defining module and attribute for every public name. That works for
    the three constants as well as the functions — a float has no
    ``__module__`` to ask, which is how the first version of this test failed
    on exactly those three and nothing else.
    """
    orblet = importlib.import_module("orblet")
    lazy = orblet._LAZY_EXPORTS
    assert name in lazy, f"{name} is in __all__ but not in _LAZY_EXPORTS"
    module_name, attr = lazy[name]
    assert module_name.startswith("orblet"), f"{name} resolves outside orblet: {module_name}"
    defining = importlib.import_module(module_name)
    assert getattr(defining, attr) is getattr(orblet, name), (
        f"orblet.{name} is not the object {module_name}.{attr} defines"
    )


def test_front_door_map_and_all_agree():
    """Every lazily resolved name is public, and every public name resolves lazily."""
    orblet = importlib.import_module("orblet")
    assert set(orblet._LAZY_EXPORTS) == set(orblet.__all__)
