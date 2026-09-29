"""Time the PDC periodogram sweep: orblet's ``compute_pdc_periodogram``
against the frozen pre-optimisation reference in ``tests/_pdc_reference.py``.

A script, not a test: timings depend on the machine. Synthetic data only.

    python benchmarks/pdc_sweep.py            # quick: 597 epochs, 500 periods
    python benchmarks/pdc_sweep.py --full     # 597 and 824 epochs, 5000 periods

Implementations are interleaved within each repeat (reference, orblet,
reference, ...) so a slow drift of the machine hits both alike; the median
over repeats is reported. Peak memory is measured in a separate, shorter
run under ``tracemalloc`` (which slows the code, so it is never timed).
While the two implementations are the same code, the ratio column shows
the timing noise of the machine.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
from _pdc_reference import reference_pdc_periodogram  # noqa: E402

import orblet  # noqa: E402
from orblet.periodogram import (  # noqa: E402
    compute_pdc_periodogram,
    scalar_distance_matrix,
    scan_angle_distance_matrix,
)

IMPLS = {"reference": reference_pdc_periodogram, "orblet": compute_pdc_periodogram}


def _inputs(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1800.0, n))
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    x = 3.0 * np.sin(2.0 * np.pi * t / 211.0) + 0.5 * np.cos(psi) + rng.normal(0.0, 1.0, n)
    return t, scalar_distance_matrix(x), scan_angle_distance_matrix(psi)


def _call(fn, t, periods, obs, nui, mode):
    if mode == "semi":
        return fn(t, periods, obs, nuisance_dist=nui, partial_mode="semi")
    return fn(t, periods, obs)


def _peak_mib(fn, t, periods, obs, nui, mode) -> float:
    tracemalloc.start()
    _call(fn, t, periods, obs, nui, mode)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2**20


def _environment() -> str:
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas = f"{blas.get('name')} {blas.get('version', '')}".strip()
    except Exception:
        blas = "unknown"
    return (f"orblet {orblet.__version__} | NumPy {np.__version__} | BLAS {blas} | "
            f"Python {platform.python_version()} | {platform.machine()} "
            f"{platform.processor() or ''} | {platform.platform()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="597 and 824 epochs, 5000 periods")
    ap.add_argument("--epochs", type=int, nargs="+", help="override the epoch counts")
    ap.add_argument("--periods", type=int, help="override the number of trial periods")
    ap.add_argument("--repeats", type=int, help="override the number of repeats")
    ap.add_argument("--modes", nargs="+", default=["none", "semi"], choices=["none", "semi"])
    args = ap.parse_args(argv)

    epochs = args.epochs or ([597, 824] if args.full else [597])
    n_periods = args.periods or (5000 if args.full else 500)
    repeats = args.repeats or (3 if args.full else 5)
    periods = np.geomspace(20.0, 1000.0, n_periods)

    print(_environment())
    print(f"{n_periods} trial periods, {repeats} interleaved repeats, median reported\n")
    print(f"{'epochs':>6} {'mode':>5} {'reference s':>12} {'orblet s':>9} {'speed-up':>9} "
          f"{'ms/period':>10} {'peak MiB ref':>13} {'peak MiB orblet':>16}")

    for n in epochs:
        t, obs, nui = _inputs(n)
        for mode in args.modes:
            times = {name: [] for name in IMPLS}
            _call(compute_pdc_periodogram, t, periods[:20], obs, nui, mode)  # warm-up
            for _ in range(repeats):
                for name, fn in IMPLS.items():
                    t0 = time.perf_counter()
                    _call(fn, t, periods, obs, nui, mode)
                    times[name].append(time.perf_counter() - t0)
            med = {k: statistics.median(v) for k, v in times.items()}
            peak = {k: _peak_mib(fn, t, periods[:20], obs, nui, mode) for k, fn in IMPLS.items()}
            print(f"{n:>6} {mode:>5} {med['reference']:>12.2f} {med['orblet']:>9.2f} "
                  f"{med['reference'] / med['orblet']:>8.2f}x "
                  f"{1e3 * med['orblet'] / n_periods:>10.2f} "
                  f"{peak['reference']:>13.1f} {peak['orblet']:>16.1f}")
            spread = {k: (min(v), max(v)) for k, v in times.items()}
            print(f"{'':>13} range: reference {spread['reference'][0]:.2f}–{spread['reference'][1]:.2f} s, "
                  f"orblet {spread['orblet'][0]:.2f}–{spread['orblet'][1]:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
