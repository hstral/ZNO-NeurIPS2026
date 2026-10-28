"""Validation-only configuration selection from a completed selection manifest.

For every (case, model) the candidate with the lowest mean best-validation
relative L2 over the selection seeds is chosen. Test errors are never read.

Tie-breaking (``--tie-break``):
  * ``order`` (default; controlled tuning audit): strict minimum, so an exact
    tie keeps the candidate that appears first in the manifest.
  * ``params``: ties go to the smaller real parameter count, then the tag
    (public benchmarks).

Usage::

    python -m experiments.select --manifest runs/manifests/tuning-selection.json \
        --output-dir runs/tuning
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .manifest import load_manifest, resolve


def _summary_count(summary: dict) -> int:
    return int(summary.get("parameter_count_real", summary.get("parameter_count", -1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tie-break", choices=("order", "params"), default="order")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    order: dict[tuple[str, str, str], int] = {}
    model_args: dict[tuple[str, str, str], list[str]] = {}
    scores: dict[tuple[str, str, str], list[tuple[int, float, int]]] = defaultdict(list)
    missing = []
    for position, entry in enumerate(entries):
        key = (entry["case"], entry["model"], entry["tag"])
        order.setdefault(key, position)
        model_args[key] = entry["model_args"]
        path = resolve(entry["summary_path"])
        if not path.exists():
            missing.append(str(path))
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("best_val_rel_l2") is None:
            raise SystemExit(f"Run without a finite validation checkpoint: {path}")
        scores[key].append((int(entry["seed"]), float(summary["best_val_rel_l2"]), _summary_count(summary)))
    if missing:
        raise SystemExit(f"{len(missing)} selection runs are missing; first: {missing[0]}")

    rows = []
    for key, values in scores.items():
        seeds = sorted(seed for seed, _, _ in values)
        expected = sorted({int(e["seed"]) for e in entries if (e["case"], e["model"], e["tag"]) == key})
        if seeds != expected:
            raise SystemExit(f"Incomplete seeds for {key}: {seeds}")
        vals = np.asarray([value for _, value, _ in values])
        rows.append({
            "case": key[0], "model": key[1], "tag": key[2],
            "seeds": ",".join(map(str, seeds)),
            "parameter_count": values[0][2],
            "mean_best_val_rel_l2": float(vals.mean()),
            "std_best_val_rel_l2_population": float(vals.std(ddof=0)),
            "model_args": model_args[key],
            "manifest_order": order[key],
        })

    selected = []
    for case, model in sorted({(row["case"], row["model"]) for row in rows}):
        candidates = [row for row in rows if row["case"] == case and row["model"] == model]
        if args.tie_break == "order":
            key_fn = lambda row: (row["mean_best_val_rel_l2"], row["manifest_order"])
        else:
            key_fn = lambda row: (row["mean_best_val_rel_l2"], row["parameter_count"], row["tag"])
        selected.append(min(candidates, key=key_fn))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_rule": "lowest mean best-validation relative L2 over the selection seeds",
        "tie_break": args.tie_break,
        "test_metrics_read": False,
        "selected": selected,
    }
    (args.output_dir / "selected.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "selection_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case", "model", "tag", "seeds", "parameter_count", "mean_best_val_rel_l2",
                  "std_best_val_rel_l2_population", "selected", "model_args"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        chosen = {(row["case"], row["model"], row["tag"]) for row in selected}
        for row in sorted(rows, key=lambda r: (r["case"], r["model"], r["mean_best_val_rel_l2"])):
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             "selected": int((row["case"], row["model"], row["tag"]) in chosen),
                             "model_args": " ".join(row["model_args"])})
    for row in selected:
        print(f"{row['case']:34s} {row['model']:14s} {row['tag']:16s} val={row['mean_best_val_rel_l2']:.6f}")


if __name__ == "__main__":
    main()
