"""Quick end-to-end smoke test (tiny NARX data, one epoch per model).

Exercises data generation, ZNO (common task loss and the unbounded radius map),
FNO, official S4D, and the stabilized LNO through ``run_discrete_operator.py``,
and checks that every summary reports real parameter counts. Pass ``--s5`` to
also train the official S5 core (requires the JAX environment and
``scripts/fetch_official_s5.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--s5", action="store_true", help="also run the official S5 core")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    smoke_root = Path(os.environ.get("ZNO_SMOKE_OUTPUT_ROOT", root / "smoke_outputs")).resolve()
    data_root = smoke_root / "data"
    run_root = smoke_root / "runs"

    run(
        [
            sys.executable,
            "benchmark_discrete/generate_suite.py",
            "--suite",
            "narx_softplus",
            "--output-dir",
            str(data_root),
            "--sequence-length",
            "64",
            "--train-size",
            "16",
            "--val-size",
            "4",
            "--test-size",
            "4",
            "--overwrite",
        ],
        cwd=root,
    )

    common = [
        sys.executable,
        "run_discrete_operator.py",
        "--case",
        "narx_softplus",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(run_root),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--max-train-samples",
        "8",
        "--max-val-samples",
        "4",
        "--max-test-samples",
        "4",
        "--batch-size",
        "4",
        "--width",
        "4",
        "--depth",
        "1",
        "--seed",
        "0",
    ]

    # On CPU the layer falls back to the PyTorch reference scan automatically.
    zno = ["--rank", "2", "--num-poles", "4", "--fir-order", "2"]
    runs = {
        "smoke_zno": ["--model", "zno", *zno],
        "smoke_zno_task_loss": ["--model", "zno", *zno, "--lambda-roll", "0", "--lambda-pole", "0"],
        "smoke_zno_unbounded": ["--model", "zno", *zno, "--zno-pole-parameterization", "z_unbounded"],
        "smoke_zno_high_init": ["--model", "zno", *zno, "--zno-pole-init-radius-low", "0.90",
                                "--zno-pole-init-radius-high", "0.995"],
        "smoke_fno": ["--model", "fno", "--modes", "8"],
        "smoke_s4d": ["--model", "s4d_official", "--state-dim", "4"],
        "smoke_lno_stable": ["--model", "lno_stable", "--modes", "4"],
    }
    for name, extra in runs.items():
        run(common + ["--run-name", name, *extra], cwd=root)
        model = extra[1]
        summary = json.loads((run_root / "narx_softplus" / model / name / "summary.json").read_text())
        assert summary["parameter_count"] >= summary["parameter_count_elements"] > 0, summary
        print(f"  {name}: real parameters {summary['parameter_count']}, "
              f"elements {summary['parameter_count_elements']}, test {summary['test_rel_l2']:.4f}")

    if args.s5:
        views = smoke_root / "s5_views"
        run([sys.executable, "-m", "s5_baseline.export_views", "--data-root", str(data_root),
             "--output-root", str(views), "--cases", "narx_softplus"], cwd=root)
        run([sys.executable, "-m", "s5_baseline.train",
             "--train-val-file", str(views / "narx_softplus" / "train_val.npz"),
             "--test-file", str(views / "narx_softplus" / "test.npz"), "--evaluate-test",
             "--output-dir", str(run_root / "s5"), "--run-name", "smoke_s5", "--seed", "0",
             "--epochs", str(args.epochs), "--batch-size", "4", "--d-model", "4",
             "--n-layers", "4", "--ssm-size-base", "4"], cwd=root)
    print(f"Smoke test completed. Outputs: {run_root}")


if __name__ == "__main__":
    main()
