"""End-to-end check of the result pipeline without training (CPU, about 20 s).

Writes all manifests, a synthetic ``summary.json`` for every entry (model
configurations and parameter counts as the runners build them, random errors),
then runs ``experiments.collect`` and ``paper/make_tables.py`` and checks that
every table is written and that the parameter counts in Table 1 are those of
the paper.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import build  # noqa: E402
from experiments.manifest import write_manifest  # noqa: E402
from experiments.verify_parameter_counts import s5_counts, torch_counts, torch_model_config  # noqa: E402

TABLES = [
    "tab_matched_symmetric", "tab_matched_recipe", "tab_compute", "tab_suffix_control", "tab_iir_optimizer",
    "tab_budget_configs", "tab_budget", "tab_tuned_validation", "tab_tuned_grid", "tab_lno",
    "tab_extrapolation", "tab_s5_capacity", "tab_public_configs", "tab_classical", "tab_public_gap",
    "tab_public", "tab_emps_interventions", "tab_tanks_interventions", "tab_ablation",
]


def _summary(entry: dict, rng: random.Random, cache: dict) -> dict:
    errors = {"best_val_rel_l2": rng.uniform(0.1, 0.9), "test_rel_l2": rng.uniform(0.1, 0.9),
              "best_epoch": 1, "train_time_seconds": rng.uniform(10, 20), "seed": entry["seed"]}
    if entry["runner"] == "classical":
        return {**errors, "parameter_count": 8, "selected_config": {"tag": "synthetic"}}
    key = (entry["runner"], entry["case"], entry["model"], tuple(entry["model_args"]))
    if entry["runner"] == "s5":
        if key not in cache:
            cache[key] = s5_counts(entry)
        argv = entry["argv"]
        config = {name: int(argv[argv.index(flag) + 1])
                  for name, flag in (("d_model", "--d-model"), ("ssm_size_base", "--ssm-size-base"))}
        lr = float(argv[argv.index("--learning-rate") + 1]) if "--learning-rate" in argv else 1e-3
        return {**errors, "model_config": config, "parameter_count_real": cache[key][0],
                "training_config": {"learning_rate": lr, "epochs": 1}}
    if key not in cache:
        cache[key] = (asdict(torch_model_config(entry)[0]), *torch_counts(entry))
    config, real, elements = cache[key]
    return {**errors, "model_config": config, "parameter_count": real, "parameter_count_elements": elements,
            "max_pole_radius": rng.uniform(0.5, 0.99) if entry["model"] == "zno" else None}


def test_collect_and_tables(tmp_path):
    ctx = argparse.Namespace(layout=build.Layout(str(tmp_path / "data"), str(tmp_path / "runs")),
                             zno_backend="triton", device="cpu", selected=None, extra_args=[])
    rng, cache = random.Random(0), {}
    for study in build.STUDIES:
        entries = build.build(study, ctx)
        write_manifest(tmp_path / "runs" / "manifests" / f"{study}.json", entries)
        for entry in entries:
            path = Path(entry["summary_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_summary(entry, rng, cache)))

    results = tmp_path / "results" / "runs.csv"
    subprocess.run([sys.executable, "-m", "experiments.collect", "--manifests", str(tmp_path / "runs" / "manifests"),
                    "--output", str(results)], cwd=REPO_ROOT, check=True)
    assert len(results.read_text().splitlines()) == 1 + 1676
    tables = tmp_path / "tables"
    subprocess.run([sys.executable, "paper/make_tables.py", "--results", str(results), "--out", str(tables)],
                   cwd=REPO_ROOT, check=True)
    assert sorted(p.stem for p in tables.glob("*.tex")) == sorted(TABLES)
    main = (tables / "tab_matched_symmetric.tex").read_text()
    for count in (r"8\,525", r"8\,643", r"8\,577", r"8\,741", r"8\,657", r"8\,837", r"8\,537"):
        assert count in main
    assert "& GPU & 1650 &" in (tables / "tab_compute.tex").read_text()
