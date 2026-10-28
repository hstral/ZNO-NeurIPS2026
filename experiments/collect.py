"""Collect completed runs into one table: ``results/runs.csv``.

Reads every manifest in ``--manifests`` (default ``runs/manifests``) and the
``summary.json`` of each completed entry, and writes one row per run with its
study, task, model, configuration tag, seed, configuration fields, real
parameter count, best-validation and test errors, training time, and pole
statistics. The table and figure scripts in ``paper/`` read only this file.

Runs of selection studies use test-free data views; their ``test_rel_l2`` is
left empty. Parameter counts are real degrees of freedom (complex scalars count
twice); for summaries written before ``parameter_count_elements`` existed, the
real count is recomputed from the stored model configuration.

Usage::

    python -m experiments.collect                       # runs/manifests -> results/runs.csv
    python -m experiments.collect --allow-missing       # partial results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

from .manifest import load_manifest, resolve
from .protocols import REPO_ROOT

FIELDS = [
    "study", "task", "case", "model", "tag", "seed", "runner", "budget", "length",
    "width", "depth", "modes", "state_dim", "rank", "num_poles", "fir_order", "safe_radius",
    "d_model", "ssm_size_base", "learning_rate", "epochs", "config",
    "parameter_count", "parameter_count_elements", "best_val_rel_l2", "test_rel_l2",
    "best_epoch", "train_time_seconds", "max_pole_radius", "mean_pole_radius", "numerical_failure",
    "summary_path",
]
CONFIG_FIELDS = ("width", "depth", "modes", "state_dim", "rank", "num_poles", "fir_order", "safe_radius",
                 "d_model", "ssm_size_base", "learning_rate", "epochs")


@lru_cache(maxsize=None)
def _registries():
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark_1d.configs import build_case_registry as one_d_registry
    from benchmark_discrete.configs import build_case_registry as discrete_registry

    return discrete_registry(REPO_ROOT), one_d_registry(REPO_ROOT)


def _recount(entry: dict, summary: dict) -> tuple[int, int]:
    """Real and element counts rebuilt from a stored PyTorch model configuration."""
    import torch

    from benchmark_1d.parameters import parameter_counts

    discrete, one_d = _registries()
    config = summary["model_config"]
    if entry["runner"] == "one_d":
        from benchmark_1d.configs import OperatorConfig
        from benchmark_1d.models import build_model

        case = one_d[entry["case"]]
        model = build_model(model_name=entry["model"], config=OperatorConfig(**config),
                            grid=torch.linspace(0, 1, 16), use_grid=case.use_grid)
    else:
        from benchmark_discrete.configs import OperatorConfig
        from benchmark_discrete.models import build_model

        known = OperatorConfig.__dataclass_fields__
        model = build_model(model_name=entry["model"],
                            config=OperatorConfig(**{k: v for k, v in config.items() if k in known}),
                            grid=torch.arange(16, dtype=torch.float32), use_grid=False,
                            input_channels=int(summary["data_shape"]["train"][2]))
    counts = parameter_counts(model)
    return counts["parameter_count"], counts["parameter_count_elements"]


def row_for(entry: dict, summary: dict) -> dict:
    config = summary.get("model_config") or {}
    if entry["runner"] == "s5":
        real = elements = summary.get("parameter_count_real")
    elif entry["runner"] == "classical":
        real, elements = summary.get("parameter_count"), ""  # number of model terms
    elif "parameter_count_elements" in summary:
        real, elements = summary["parameter_count"], summary["parameter_count_elements"]
    else:
        real, elements = _recount(entry, summary)
    row = {
        "study": entry["study"], "task": entry["task"], "case": entry["case"], "model": entry["model"],
        "tag": entry["tag"], "seed": entry["seed"], "runner": entry["runner"],
        "budget": entry.get("budget", ""), "length": entry.get("length", ""),
        "config": (summary.get("selected_config") or {}).get("tag", "") if entry["runner"] == "classical" else "",
        "parameter_count": real, "parameter_count_elements": elements,
        "best_val_rel_l2": summary.get("best_val_rel_l2"),
        "test_rel_l2": summary.get("test_rel_l2") if entry.get("test_access", True) else "",
        "best_epoch": summary.get("best_epoch"),
        "train_time_seconds": summary.get("train_time_seconds"),
        "max_pole_radius": summary.get("max_pole_radius", ""),
        "mean_pole_radius": summary.get("mean_pole_radius", ""),
        "numerical_failure": summary.get("numerical_failure") or "",
        "summary_path": entry["summary_path"],
    }
    for name in CONFIG_FIELDS:
        row[name] = config.get(name, "")
    if entry["runner"] == "s5":
        row["learning_rate"] = (summary.get("training_config") or {}).get("learning_rate", "")
        row["epochs"] = (summary.get("training_config") or {}).get("epochs", "")
    return {k: ("" if v is None else v) for k, v in row.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", type=Path, default=Path("runs/manifests"))
    parser.add_argument("--output", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    manifests = sorted(resolve(str(args.manifests)).glob("*.json"))
    if not manifests:
        raise SystemExit(f"no manifests in {args.manifests}; run `python -m experiments.build all` first")
    rows, missing = [], {}
    for path in manifests:
        for entry in load_manifest(path):
            summary_path = resolve(entry["summary_path"])
            if not summary_path.exists():
                missing[path.stem] = missing.get(path.stem, 0) + 1
                continue
            rows.append(row_for(entry, json.loads(summary_path.read_text(encoding="utf-8"))))
    for study, count in sorted(missing.items()):
        print(f"missing: {study}: {count} runs without summary.json")
    if missing and not args.allow_missing:
        raise SystemExit("some runs are missing; rerun them or pass --allow-missing")
    output = resolve(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} runs from {len(manifests)} manifests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
