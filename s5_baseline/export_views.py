"""Export test-isolated S5 data views for a set of benchmark_discrete cases.

For each case ``<data-root>/<case>/data.npz`` this writes
``<output-root>/<case>/train_val.npz`` (no test arrays),
``<output-root>/<case>/test.npz`` and a manifest with array hashes, so the JAX
and PyTorch pipelines are guaranteed to see identical arrays.

Usage::

    python -m s5_baseline.export_views --data-root data/controlled \
        --output-root data/s5_views/controlled --cases \
        resonant_arma_near_unit_circle_v4 high_order_iir_cascade_v5 narx_softplus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_views import export_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    index = {}
    for case in args.cases:
        manifest = export_case(args.data_root.resolve() / case, args.output_root.resolve() / case)
        index[case] = {
            "train_val_sha256": manifest["train_val_sha256"],
            "test_sha256": manifest["test_sha256"],
            "metric_ignore_prefix_steps": manifest["metric_ignore_prefix_steps"],
        }
        print(f"exported {case}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
