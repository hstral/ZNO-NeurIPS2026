"""Train ZNO for a few epochs with the ``triton`` and ``triton_fast`` backends from the same
seed and compare the per-epoch curves and final errors (8.5k configuration, task loss).

Usage::

    python tools/compare_zno_backends.py --data-root data/controlled --epochs 5 \
        --output-dir runs/backend_check [--max-rel 1e-4]

With ``--max-rel`` the script exits with status 1 if any compared quantity differs by more
than that relative amount. Over the full 600 epochs the float32 rounding differences can
grow, so individual seeds may reach different, statistically equivalent optima.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ["resonant_arma_near_unit_circle_v4", "high_order_iir_cascade_v5", "narx_softplus"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/controlled")
    parser.add_argument("--output-dir", default="runs/backend_check")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rel", type=float, default=None)
    args = parser.parse_args()
    worst = 0.0
    for case in CASES:
        runs = {}
        for backend in ("triton", "triton_fast"):
            name = f"backend-{backend}-s{args.seed}"
            cmd = [sys.executable, "run_discrete_operator.py", "--case", case, "--model", "zno",
                   "--data-root", args.data_root, "--output-dir", args.output_dir,
                   "--run-name", name, "--seed", str(args.seed), "--epochs", str(args.epochs),
                   "--width", "20", "--depth", "4", "--rank", "8", "--num-poles", "40", "--fir-order", "4",
                   "--lambda-roll", "0", "--lambda-pole", "0", "--rollout-context-steps", "512",
                   "--zno-kernel-backend", backend]
            # Pass the parent environment explicitly: numpy's MKL service may set
            # MKL_THREADING_LAYER=INTEL at the C level, which breaks torch+libgomp in the child.
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, env=dict(os.environ))
            run_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else ROOT / args.output_dir
            run_dir = run_dir / case / "zno" / name
            runs[backend] = {
                "train": np.loadtxt(run_dir / "train_error.txt"),
                "val": np.loadtxt(run_dir / "vali_error.txt"),
                "summary": json.loads((run_dir / "summary.json").read_text()),
            }
        old, new = runs["triton"], runs["triton_fast"]
        print(f"== {case} ({args.epochs} epochs, seed {args.seed})")
        for key in ("train", "val"):
            rel = np.abs(new[key] - old[key]) / np.abs(old[key])
            print(f"  {key:5s} per-epoch relative difference: max {rel.max():.2e}")
            worst = max(worst, float(rel.max()))
        for key in ("best_val_rel_l2", "test_rel_l2"):
            a, b = old["summary"][key], new["summary"][key]
            print(f"  {key:16s} triton {a:.8f}  triton_fast {b:.8f}  relative difference {abs(b - a) / a:.2e}")
            worst = max(worst, abs(b - a) / a)
        print(f"  train_time_seconds triton {old['summary']['train_time_seconds']:.1f}  "
              f"triton_fast {new['summary']['train_time_seconds']:.1f}")
    print(f"max relative difference: {worst:.2e}")
    if args.max_rel is not None:
        print("PASS" if worst <= args.max_rel else "FAIL")
        sys.exit(0 if worst <= args.max_rel else 1)


if __name__ == "__main__":
    main()
