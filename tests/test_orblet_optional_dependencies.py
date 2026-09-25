"""orblet's optional dependencies really are optional.

orblet's declared dependencies are numpy, scipy, astropy and matplotlib. Four
other packages appear in its code — emcee, corner, pandas and jplephem — and
each must be reachable only from inside a function, so that installing the
four core dependencies gives a working package rather than one that dies on
the first interesting import.

A module-scope import of an optional package is easy to miss: ``import
orblet`` stays clean because the front door resolves its names lazily, so
the cost appears only on the submodule import. One module importing pandas
at the top would make every helper in it unusable without pandas, though
most need only numpy and scipy. This file checks both what the import
statements say and what actually happens when the extras are absent.

WHY A SUBPROCESS. The behavioural checks need the optional packages to be
genuinely absent while orblet modules are imported fresh. Doing that in-process
means dropping orblet modules from ``sys.modules`` and re-importing them, and
that is a known trap here: ``from package import submodule`` reads the PARENT
PACKAGE ATTRIBUTE, so a re-import binds a new object there while
``monkeypatch`` restores only ``sys.modules`` — leaving two live copies of a
module and breaking identity checks in unrelated tests. It has bitten twice. A
fresh interpreter has no such problem and is a truer simulation of the install
being claimed.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ORBLET_DIR = REPO_ROOT / "src" / "orblet"

#: Installed with orblet. A module-scope import of these is fine.
CORE_DEPENDENCIES = frozenset({"numpy", "scipy", "astropy", "matplotlib"})

#: NOT installed with orblet. Reachable only from inside a function.
OPTIONAL_DEPENDENCIES = frozenset({"emcee", "corner", "pandas", "jplephem"})


# --------------------------------------------------------------------------
# The static half: what the import statements say.
# --------------------------------------------------------------------------

def _is_type_checking_guard(stmt: ast.stmt) -> bool:
    """``if TYPE_CHECKING:`` — never runs, so it requires nothing at runtime.

    ``typing.TYPE_CHECKING`` is False when the interpreter runs; the block exists
    so a type checker can resolve an annotation without the import costing
    anything. Both the bare and the dotted spelling appear in the wild.
    """
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _guards_import_error(stmt: ast.stmt) -> bool:
    """``try: import x / except ImportError:`` — runs, but cannot fail the import."""
    if not isinstance(stmt, ast.Try):
        return False
    for handler in stmt.handlers:
        names: list[str] = []
        if isinstance(handler.type, ast.Name):
            names = [handler.type.id]
        elif isinstance(handler.type, ast.Tuple):
            names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
        if {"ImportError", "ModuleNotFoundError"} & set(names):
            return True
    return False


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level packages that importing this module would really require.

    Counted: every import statement executing at module scope unprotected.
    Skipped, because neither can make the import fail — ``if TYPE_CHECKING:``
    blocks and imports inside a ``try``/``except ImportError``. Imports nested in
    a function or class are lazy by construction and never reached here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tops: set[str] = set()

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            tops.add(node.module.split(".")[0])

    for stmt in tree.body:
        if _is_type_checking_guard(stmt) or _guards_import_error(stmt):
            continue
        collect(stmt)
        # A plain module-level if/with/try still executes, so look inside it.
        if isinstance(stmt, (ast.Try, ast.If, ast.With)):
            for inner in ast.walk(stmt):
                collect(inner)
    return tops


def test_every_module_scope_import_is_declared():
    """Nothing outside orblet, the core four and the standard library.

    Read from the allowed side rather than a forbidden list: a NEW dependency
    nobody declared shows up here instead of being discovered at install time.
    """
    allowed = CORE_DEPENDENCIES | {"orblet", "__future__"}
    undeclared: dict[str, set[str]] = {}

    for path in sorted(ORBLET_DIR.rglob("*.py")):
        for top in sorted(_module_scope_imports(path)):
            if top in allowed or top in sys.stdlib_module_names:
                continue
            undeclared.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))

    assert not undeclared, (
        "orblet imports these at module scope and they are neither the core "
        "four, the standard library, nor orblet itself. Either add them to "
        "CORE_DEPENDENCIES here and to orblet's pyproject in the same commit, "
        "or make the import lazy: "
        + "; ".join(f"{k} ({', '.join(sorted(v))})" for k, v in undeclared.items())
    )


def _all_imports(path: Path) -> set[str]:
    """Top-level packages imported anywhere in the module, lazily or not."""
    tops: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            tops.add(node.module.split(".")[0])
    return tops


def test_every_import_anywhere_is_declared():
    """Even a lazy import must name a declared dependency or extra.

    A function-local import is invisible at ``import orblet`` time and fails
    only when that function runs, on an install that lacks the package. So
    every import, at any depth, must be orblet, the standard library, a core
    dependency or a declared optional extra.
    """
    allowed = CORE_DEPENDENCIES | OPTIONAL_DEPENDENCIES | {"orblet", "__future__"}
    undeclared: dict[str, set[str]] = {}
    for path in sorted(ORBLET_DIR.rglob("*.py")):
        for top in sorted(_all_imports(path)):
            if top in allowed or top in sys.stdlib_module_names:
                continue
            undeclared.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))

    assert not undeclared, (
        "orblet imports these and they are neither declared nor the standard "
        "library: " + "; ".join(f"{k} ({', '.join(sorted(v))})" for k, v in undeclared.items())
    )


# --------------------------------------------------------------------------
# The behavioural half: what actually happens when they are missing.
# --------------------------------------------------------------------------

#: Runs in a FRESH interpreter with the optional packages made unavailable, and
#: prints one JSON object. Kept as a payload rather than a file so the checks and
#: the assertions that read them stay side by side.
_PROBE = r'''
import json, sys

# The core four load FIRST, deliberately: astropy probes for pandas while
# importing its own table machinery. That is astropy's business, not a claim
# about orblet, and blocking beforehand would measure the wrong thing.
import numpy, scipy.stats, astropy.timeseries, matplotlib

BLOCKED = {"emcee", "corner", "pandas", "jplephem"}

class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in BLOCKED:
            raise ModuleNotFoundError("No module named %r" % top)
        return None

for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]
sys.meta_path.insert(0, Blocker())
# A None entry is how Python spells "not installed": an import raises
# ModuleNotFoundError, and importlib.util.find_spec returns None instead of
# raising, which is what astropy's optional-dependency probes expect.
for name in BLOCKED:
    sys.modules[name] = None

out = {"blocked_ok": [], "import_failures": [], "checks": {}}

for name in sorted(BLOCKED):
    try:
        __import__(name)
    except ModuleNotFoundError:
        out["blocked_ok"].append(name)

# 1. Every orblet module must import.
import importlib, pathlib
root = pathlib.Path(MODULE_ROOT)
names = []
for p in sorted(root.rglob("*.py")):
    parts = list(p.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    names.append(".".join(parts))
out["module_count"] = len(names)
for name in names:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        out["import_failures"].append("%s: %s" % (name, exc))

# 2. Period search runs on the core install.
try:
    import orblet.periodogram as pg
    rng = numpy.random.default_rng(7)
    t = numpy.sort(rng.uniform(0.0, 200.0, 40))
    rv = 5.0 * numpy.sin(2 * numpy.pi * t / 13.3)
    err = numpy.full_like(rv, 0.2)
    res = pg.compute_lomb_scargle_periodogram(rv, err, t)
    out["checks"]["period_search_ran"] = res is not None
except Exception as exc:
    out["checks"]["periodogram_error"] = "%s: %s" % (type(exc).__name__, exc)

# 2b. The demo bundle builds without jplephem: its parallax factors come from
#     astropy's built-in ephemeris, which needs no JPL kernel.
try:
    from orblet.simulate.bundles import load_simulated_inputs
    b = load_simulated_inputs(seed=0)
    out["checks"]["demo_bundle_built"] = bool(
        numpy.all(numpy.isfinite(b.astro_data["parallax_factor_al"])))
except Exception as exc:
    out["checks"]["demo_bundle_error"] = "%s: %s" % (type(exc).__name__, exc)

# 3. chain_stats: three helpers work; the table one fails helpfully.
try:
    import orblet.chain_stats as cs
    rng = numpy.random.default_rng(11)
    s = rng.normal(size=2000)
    out["checks"]["quantiles"] = sorted(cs.chain_quantiles(s))
    out["checks"]["credible_interval_is_dict"] = isinstance(
        cs.chain_credible_interval(s), dict
    )
    out["checks"]["circular_summary"] = sorted(
        cs.chain_circular_summary(rng.uniform(0, 2 * numpy.pi, 2000))
    )
    try:
        cs.chain_summary_table({"P": s})
    except ImportError as exc:
        out["checks"]["table_error_message"] = str(exc)
    else:
        out["checks"]["table_error_message"] = None
except Exception as exc:
    out["checks"]["chain_stats_error"] = "%s: %s" % (type(exc).__name__, exc)

print(json.dumps(out))
'''


@pytest.fixture(scope="module")
def probe() -> dict:
    """Run the probe once in a fresh interpreter and hand back its findings."""
    code = f"MODULE_ROOT = {str(ORBLET_DIR)!r}\n" + _PROBE
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"the probe interpreter failed:\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_probe_really_blocked_the_optional_packages(probe):
    """Guard the guard: if nothing was blocked, everything below is vacuous."""
    assert sorted(probe["blocked_ok"]) == sorted(OPTIONAL_DEPENDENCIES), (
        "the probe did not manage to hide all four optional packages, so the "
        f"checks below prove nothing: {probe['blocked_ok']}"
    )
    assert probe["module_count"] >= 25, (
        f"the probe found only {probe['module_count']} orblet modules"
    )


def test_every_orblet_module_imports_without_the_optional_dependencies(probe):
    """The invariant, measured rather than inferred from import statements.

    Stronger than the AST walk above, because it cannot be fooled by a spelling
    that walk does not model — a conditional import, a re-export chain, an
    ``importlib`` call inside a module body.
    """
    assert not probe["import_failures"], (
        "these orblet modules cannot be imported on an install carrying only "
        f"{sorted(CORE_DEPENDENCIES)}:\n  "
        + "\n  ".join(probe["import_failures"])
    )


def test_demo_bundle_builds_without_jplephem(probe):
    """Demo data needs no JPL kernel: the built-in ephemeris is the default."""
    checks = probe["checks"]
    assert "demo_bundle_error" not in checks, checks.get("demo_bundle_error")
    assert checks["demo_bundle_built"]


def test_period_search_runs_on_the_core_install(probe):
    """A period search needs nothing beyond the four core dependencies."""
    checks = probe["checks"]
    assert "periodogram_error" not in checks, checks.get("periodogram_error")
    assert checks["period_search_ran"]


def test_chain_stats_helpers_run_without_pandas(probe):
    """Three of the four helpers need only numpy and scipy — prove it."""
    checks = probe["checks"]
    assert "chain_stats_error" not in checks, checks.get("chain_stats_error")
    assert checks["quantiles"] == ["q025", "q500", "q975"]
    assert checks["credible_interval_is_dict"]
    assert checks["circular_summary"] == ["circmean_rad", "circstd_rad"]


def test_chain_summary_table_says_how_to_get_pandas(probe):
    """The one function that cannot degrade must fail helpfully, not cryptically.

    A bare ``ModuleNotFoundError: No module named 'pandas'`` says something is
    missing, but not that it is OPTIONAL, nor that the same numbers are
    available without it.
    """
    message = probe["checks"]["table_error_message"]
    assert message is not None, (
        "chain_summary_table returned a value with pandas blocked, which cannot "
        "be right"
    )
    assert "orblet[tables]" in message, "name the extra that provides pandas"
    assert "chain_quantiles" in message, "name the way to the same numbers"
