"""Run manifests: explicit, reviewable lists of training commands.

A manifest is a JSON list of entries. Each entry records the study, task,
model, configuration tag, seed, the runner it needs (``discrete``, ``one_d``,
``s5``, or ``classical``), whether the run may access the official test set,
the exact command (``argv``, relative to the repository root), and the path of
the ``summary.json`` it produces.

Run entries locally, one at a time or all sequentially::

    python -m experiments.manifest run --manifest runs/manifests/matched.json --index 0
    python -m experiments.manifest run --manifest runs/manifests/matched.json --all --skip-existing

On a cluster, dispatch ``--index i`` for every index of the manifest. S5 entries
need the JAX environment and all other entries the PyTorch environment; use
``--runner s5`` or ``--exclude-runner s5`` to split a mixed manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .protocols import CASE_TO_TASK, REPO_ROOT


def _entry(*, study, case, model, tag, seed, runner, test_access, model_args, argv, summary_path, **extra):
    return {
        "id": Path(summary_path).parent.name,
        "study": study,
        "case": case,
        "task": CASE_TO_TASK.get(case, case),
        "model": model,
        "tag": tag,
        "seed": seed,
        "runner": runner,
        "test_access": test_access,
        "model_args": list(model_args),
        "argv": argv,
        "summary_path": str(summary_path),
        **extra,
    }


def discrete_entry(*, study, case, model, tag, seed, data_root, output_root, model_args,
                   device, test_access=True, **extra) -> dict:
    """Entry for ``run_discrete_operator.py`` (FNO, LNO, S4D, ZNO)."""
    run_name = f"{study}-{case}-{model}-{tag}-seed{seed}"
    argv = ["run_discrete_operator.py", "--case", case, "--model", model,
            "--data-root", str(data_root), "--output-dir", str(output_root),
            "--run-name", run_name, "--seed", str(seed), "--device", device, *model_args]
    return _entry(study=study, case=case, model=model, tag=tag, seed=seed, runner="discrete",
                  test_access=test_access, model_args=model_args, argv=argv,
                  summary_path=Path(output_root) / case / model / run_name / "summary.json", **extra)


def one_d_entry(*, study, case, model, tag, seed, output_root, model_args, device) -> dict:
    """Entry for ``run_1d_operator.py`` (1D LNO ODE benchmark)."""
    run_name = f"{study}-{case}-{model}-{tag}-seed{seed}"
    argv = ["run_1d_operator.py", "--case", case, "--model", model, "--output-dir", str(output_root),
            "--run-name", run_name, "--seed", str(seed), "--device", device, *model_args]
    return _entry(study=study, case=case, model=model, tag=tag, seed=seed, runner="one_d",
                  test_access=True, model_args=model_args, argv=argv,
                  summary_path=Path(output_root) / case / model / run_name / "summary.json")


def s5_entry(*, study, case, tag, seed, view_root, output_root, model_args, evaluate_test,
             run_kind, **extra) -> dict:
    """Entry for the official S5 core (``python -m s5_baseline.train``).

    ``view_root/<case>/train_val.npz`` and ``test.npz`` are written by
    ``python -m s5_baseline.export_views``.
    """
    run_name = f"{study}-{case}-s5-{tag}-seed{seed}"
    view = Path(view_root) / case
    argv = ["-m", "s5_baseline.train", "--train-val-file", str(view / "train_val.npz"),
            "--output-dir", str(output_root), "--run-name", run_name,
            "--task-name", CASE_TO_TASK.get(case, case), "--run-kind", run_kind,
            "--seed", str(seed), *model_args]
    if evaluate_test:
        argv += ["--test-file", str(view / "test.npz"), "--evaluate-test"]
    return _entry(study=study, case=case, model="s5_official", tag=tag, seed=seed, runner="s5",
                  test_access=evaluate_test, model_args=model_args, argv=argv,
                  summary_path=Path(output_root) / run_name / "summary.json", **extra)


def classical_entry(*, study, case, family, data_root, output_root) -> dict:
    """Entry for one SysIdentPy NARMAX family (deterministic, single fit)."""
    run_name = f"{study}-{case}-{family}"
    argv = ["run_external_sysid_baseline.py", "--suite", case, "--model", family,
            "--data-root", str(data_root), "--output-dir", str(output_root),
            "--run-name", run_name, "--seed", "0"]
    return _entry(study=study, case=case, model=family, tag=family, seed=0, runner="classical",
                  test_access=True, model_args=(), argv=argv,
                  summary_path=Path(output_root) / case / family / run_name / "summary.json")


def write_manifest(path: Path, entries: list[dict]) -> None:
    ids = [entry["summary_path"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate runs in manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for entry in entries:
        counts[entry["runner"]] = counts.get(entry["runner"], 0) + 1
    print(f"wrote {path}: {len(entries)} runs ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})")


def load_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def run_entry(entry: dict, *, dry_run: bool = False) -> int:
    command = [sys.executable, *entry["argv"]]
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    print(json.dumps({
        "id": entry["id"],
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 1),
        "summary_exists": resolve(entry["summary_path"]).exists(),
    }), flush=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute manifest entries")
    run.add_argument("--manifest", type=Path, required=True)
    which = run.add_mutually_exclusive_group(required=True)
    which.add_argument("--index", type=int)
    which.add_argument("--all", action="store_true")
    run.add_argument("--runner", action="append", help="only run entries of this runner (repeatable)")
    run.add_argument("--exclude-runner", action="append", help="skip entries of this runner (repeatable)")
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    show = sub.add_parser("status", help="count completed entries")
    show.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    if args.command == "status":
        done = sum(resolve(entry["summary_path"]).exists() for entry in entries)
        print(f"{done}/{len(entries)} runs have a summary.json")
        return 0
    selected = entries if args.all else [entries[args.index]]
    if args.runner:
        selected = [e for e in selected if e["runner"] in args.runner]
    if args.exclude_runner:
        selected = [e for e in selected if e["runner"] not in args.exclude_runner]
    failures = 0
    for entry in selected:
        if args.skip_existing and resolve(entry["summary_path"]).exists():
            continue
        failures += int(run_entry(entry, dry_run=args.dry_run) != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
