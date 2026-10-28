"""Read ``results/runs.csv`` (written by ``python -m experiments.collect``) and
aggregate runs over seeds. Shared by ``paper/make_tables.py`` and ``paper/figures``.

Aggregates are mean and population standard deviation (ddof=0) over the stated
seeds; a cell that misses a seed stops with the list of seeds found.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import protocols as P  # noqa: E402

DEFAULT_RESULTS = REPO / "results" / "runs.csv"
SEEDS = list(P.SEEDS)
SELECTION_SEEDS = list(P.SELECTION_SEEDS)
PUBLIC_REPORT_SEEDS = list(P.PUBLIC_REPORT_SEEDS)
ONE_D_SEEDS = list(P.ONE_D_SEEDS)
TASKS = ["resonant_arma", "iir_cascade", "nonlinear_narx"]


def load(path: Path | None = None) -> "Runs":
    path = Path(path or DEFAULT_RESULTS)
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m experiments.collect` first")
    return Runs(path)


class Runs:
    def __init__(self, path: Path):
        with open(path, newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def get(self, study, *, task=None, model=None, tag=None, seeds=None, where=None):
        """Rows of one cell; with ``seeds``, require exactly one run per seed."""
        out = [r for r in self.rows if r["study"] == study
               and (task is None or r["task"] == task)
               and (model is None or r["model"] == model)
               and (tag is None or r["tag"] == tag)
               and (where is None or where(r))]
        what = "/".join(str(x) for x in (study, task, model, tag) if x is not None)
        if seeds is not None:
            out = [r for r in out if int(r["seed"]) in seeds]
            found = sorted(int(r["seed"]) for r in out)
            if found != sorted(seeds):
                raise SystemExit(f"{what}: expected one run per seed {sorted(seeds)}, found seeds {found}. "
                                 "Run the missing entries and rerun `python -m experiments.collect`.")
        if not out:
            raise SystemExit(f"{what}: no runs in the results file")
        return out

    def tags(self, study, *, task, model):
        """Configuration tags of a study cell in run order."""
        return list(dict.fromkeys(r["tag"] for r in self.get(study, task=task, model=model)))


class Cell:
    """Aggregate of one configuration over seeds."""

    def __init__(self, rows):
        self.rows = rows
        test = [float(r["test_rel_l2"]) for r in rows] if all(r["test_rel_l2"] != "" for r in rows) else None
        val = [float(r["best_val_rel_l2"]) for r in rows]
        self.val, self.val_sd = statistics.fmean(val), statistics.pstdev(val)
        if test is not None:
            self.test, self.test_sd = statistics.fmean(test), statistics.pstdev(test)
            gap = [t - v for t, v in zip(test, val)]
            self.gap, self.gap_sd = statistics.fmean(gap), statistics.pstdev(gap)
        counts = {int(r["parameter_count"]) for r in rows}
        if len(counts) != 1:
            raise SystemExit(f"parameter count differs across seeds: {sorted(counts)}")
        self.params = counts.pop()
        radii = [float(r["max_pole_radius"]) for r in rows if r["max_pole_radius"] != ""]
        self.max_pole = max(radii) if radii else None

    def config(self, name):
        values = {self.rows[0][name]} | {r[name] for r in self.rows}
        if len(values) != 1:
            raise SystemExit(f"{name} differs across seeds: {values}")
        return values.pop()


def cfg_int(cell, name):
    return int(float(cell.config(name)))


def best_of(cells: dict, attr: str = "test"):
    return min(cells, key=lambda k: getattr(cells[k], attr))


def select_min(runs, study, task, model, seeds=SELECTION_SEEDS):
    """Validation-only selection: lowest mean best-validation error over ``seeds``;
    ties keep the earlier candidate of the grid."""
    scores = {tag: Cell(runs.get(study, task=task, model=model, tag=tag, seeds=seeds)).val
              for tag in runs.tags(study, task=task, model=model)}
    chosen = None
    for tag, score in scores.items():
        if chosen is None or score < scores[chosen]:
            chosen = tag
    return chosen, scores
