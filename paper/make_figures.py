"""Regenerate the paper's figures in ``paper/figs/``.

The two schematics need nothing; the result figures read ``results/runs.csv``
(``python -m experiments.collect``). With ``--checkpoints``, the pole maps,
their per-layer statistics, and the prediction traces are also drawn from the
fixed-configuration runs in ``runs/fixed`` (checkpoints and saved test predictions).

Usage::

    python paper/make_figures.py [--results results/runs.csv] [--checkpoints [--runs-dir runs/fixed]]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"
SCHEMATICS = ["fig_stability_concept", "fig_architecture"]
RESULTS = ["fig_benchmark_discrete", "fig_near_unit_circle_difficulty", "fig_extrapolation", "fig_ablation",
           "fig_benchmark_1d", "fig_training_time"]
CHECKPOINTS = ["fig_zplane_poles", "pole_map_stats", "fig_prediction_traces"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, help="results table (default: results/runs.csv)")
    parser.add_argument("--checkpoints", action="store_true", help="also draw the checkpoint figures")
    parser.add_argument("--runs-dir", type=Path, help="fixed-configuration runs (default: runs/fixed)")
    args = parser.parse_args()
    jobs = [[FIGURES / f"{name}.py"] for name in SCHEMATICS]
    extra = ["--results", str(args.results)] if args.results else []
    jobs += [[FIGURES / f"{name}.py", *extra] for name in RESULTS]
    if args.checkpoints:
        extra = ["--runs-dir", str(args.runs_dir)] if args.runs_dir else []
        jobs += [[FIGURES / "requires_checkpoints" / f"{name}.py", *extra] for name in CHECKPOINTS]
    for job in jobs:
        subprocess.run([sys.executable, *map(str, job)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
