"""Write test-free data views for validation-only selection runs.

Each view copies the training and validation arrays of a benchmark case and
replaces the official test arrays by the first validation sample, because the
runner evaluates a test split after training. The official test arrays are
never copied, so selection runs cannot read them.

Usage::

    python -m experiments.build_selection_views --source-root data/controlled \
        --output-root data/selection_views/controlled
    python -m experiments.build_selection_views --source-root data/public \
        --output-root data/selection_views/public --cases public
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocols import CONTROLLED_CASES, PUBLIC_CASES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_view(source_root: Path, output_root: Path, case: str) -> dict:
    source_dir = source_root / case
    output_dir = output_root / case
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = source_dir / "data.npz"
    output_file = output_dir / "data.npz"
    with np.load(source_file) as data:
        required = {"x_train", "y_train", "x_val", "y_val", "grid"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{source_file} is missing {sorted(missing)}")
        np.savez_compressed(
            output_file,
            x_train=np.asarray(data["x_train"]),
            y_train=np.asarray(data["y_train"]),
            x_val=np.asarray(data["x_val"]),
            y_val=np.asarray(data["y_val"]),
            x_test=np.asarray(data["x_val"][:1]),
            y_test=np.asarray(data["y_val"][:1]),
            grid=np.asarray(data["grid"]),
        )
    meta_path = source_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({
        "selection_view": True,
        "official_test_arrays_included": False,
        "runner_test_proxy": "first_validation_sample",
    })
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return {"case": case, "selection_sha256": sha256(output_file), "official_test_arrays_included": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", choices=("controlled", "public"), default="controlled")
    args = parser.parse_args()
    cases = CONTROLLED_CASES if args.cases == "controlled" else PUBLIC_CASES
    records = [build_view(args.source_root.resolve(), args.output_root.resolve(), case)
               for case in cases.values()]
    path = args.output_root.resolve() / "selection_views_manifest.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
