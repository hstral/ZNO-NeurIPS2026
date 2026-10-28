"""Checks that the experiment manifests encode the protocols reported in the paper."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import build  # noqa: E402
from experiments import protocols as P  # noqa: E402

# Runs per study (Table 8; the unmodified LNO runs of Appendix H are not counted there).
STUDY_RUNS = {
    "matched": 75, "zno-default": 15, "suffix-control": 60, "iir-optimizer": 20, "capacity": 180,
    "tuning-selection": 132, "tuning-confirm": 45, "lno-reference": 15, "lno-unconstrained": 6,
    "fixed": 45, "difficulty": 75, "extrapolation": 90, "difficulty-extrapolation": 150,
    "public-selection": 339, "public": 120, "public-s5-selection": 75, "public-s5": 25,
    "classical": 20, "emps-interventions": 20, "tanks-interventions": 15, "ablation": 100, "one-d": 54,
}


def _ctx(**kwargs):
    defaults = dict(layout=build.Layout("data", "runs"), zno_backend="triton", device="cpu",
                    selected=None, extra_args=[])
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _value(entry, flag):
    return entry["argv"][entry["argv"].index(flag) + 1]


def test_study_run_counts():
    counts = {study: len(build.build(study, _ctx())) for study in build.STUDIES}
    assert counts == STUDY_RUNS
    assert sum(counts.values()) == 1676


def test_runs_are_unique_across_studies():
    paths = [e["summary_path"] for study in build.STUDIES for e in build.build(study, _ctx())]
    assert len(paths) == len(set(paths))


def test_tuning_grid_has_44_candidates_and_132_runs():
    entries = build.build("tuning-selection", _ctx())
    assert len({(e["case"], e["model"], e["tag"]) for e in entries}) == 44
    assert {e["seed"] for e in entries} == set(P.SELECTION_SEEDS)
    assert not any(e["test_access"] for e in entries)


def test_public_grid_has_113_candidates():
    entries = build.build("public-selection", _ctx())
    assert len({(e["case"], e["model"], e["tag"]) for e in entries}) == 113
    assert {e["seed"] for e in entries} == set(P.SELECTION_SEEDS)
    assert not any(e["test_access"] for e in entries)


def test_matched_comparison_uses_common_task_loss():
    entries = build.build("matched", _ctx())
    assert Counter(e["model"] for e in entries) == Counter(
        {"fno": 15, "lno_stable": 15, "s4d_official": 15, "zno": 15, "s5_official": 15})
    for entry in entries:
        if entry["model"] == "zno":
            assert _value(entry, "--lambda-pole") == "0" and _value(entry, "--lambda-roll") == "0"
        if entry["model"] == "s4d_official":
            assert _value(entry, "--state-dim") == "12"
        if entry["model"] == "s5_official":
            assert _value(entry, "--learning-rate") == "1e-3"


def test_public_seeds_and_s5_recipe():
    entries = build.build("public", _ctx())
    assert {e["seed"] for e in entries} == set(P.PUBLIC_TRAIN_SEEDS)
    s5 = build.build("public-s5", _ctx())
    assert {e["seed"] for e in s5} == set(P.PUBLIC_REPORT_SEEDS)
    assert all(_value(e, "--learning-rate") == "2e-3" and e["test_access"] for e in s5)
    selection = build.build("public-s5-selection", _ctx())
    assert not any(e["test_access"] for e in selection)


def test_zno_kernel_backend_flag():
    for backend in ("triton", "triton_fast"):
        entries = [e for e in build.build("matched", _ctx(zno_backend=backend)) if e["model"] == "zno"]
        assert entries and all(_value(e, "--zno-kernel-backend") == backend for e in entries)


def test_selection_studies_use_test_free_views():
    for study in ("tuning-selection", "public-selection"):
        for entry in build.build(study, _ctx()):
            assert "selection_views" in _value(entry, "--data-root")


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("passed", name)
