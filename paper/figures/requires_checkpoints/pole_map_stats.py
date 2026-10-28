"""Per-layer statistics of the learned ZNO poles shown in fig_zplane_poles.pdf.

Uses the checkpoint selection and pole recovery of ``fig_zplane_poles.py`` (the
ZNO run with the lowest best-validation error among seeds 0-4 of each fixed
configuration) and writes one row per task and layer: the number of stored poles
(one per conjugate pair), the largest radius, the counts above radius 0.8, 0.9
and 0.95, the phase range of the poles with radius above 0.8, and the fraction of
poles within 0.05 of z = rho_max/2 (zero radius logit, zero phase). The appendix
on pole maps quotes these statistics.

Usage: python paper/figures/requires_checkpoints/pole_map_stats.py [--runs-dir runs/fixed]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_zplane_poles as fz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=fz.REPO / "runs" / "fixed")
    ap.add_argument("--output", type=Path, default=fz.REPO / "results" / "pole_map_stats.csv")
    args = ap.parse_args()
    rows = []
    for case, _ in fz.CASES:
        ckpt, run = fz.load_best_ckpt(args.runs_dir, case)
        rho_max = ckpt["model_config"].get("radius_limit", 0.999)
        poles, _ = fz.recover_poles(ckpt["model_state_dict"], rho_max)
        for layer, layer_poles in enumerate(poles):
            p = np.asarray(layer_poles).ravel()
            p = p[np.imag(p) >= 0]  # one representative per conjugate pair
            r, phi = np.abs(p), np.abs(np.angle(p)) / math.pi
            hi = phi[r > 0.8]
            rows.append({
                "task": case, "run": run.name, "layer": layer, "n_poles": len(p),
                "max_radius": round(float(r.max()), 4), "n_gt_0p8": int((r > 0.8).sum()),
                "n_gt_0p9": int((r > 0.9).sum()), "n_gt_0p95": int((r > 0.95).sum()),
                "phase_min_gt_0p8_over_pi": round(float(hi.min()), 4) if len(hi) else "",
                "phase_max_gt_0p8_over_pi": round(float(hi.max()), 4) if len(hi) else "",
                "frac_near_half_rho_max": round(float(np.mean(np.abs(p - rho_max / 2) < 0.05)), 3),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(", ".join(f"{k}={v}" for k, v in row.items() if k != "run"))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
