"""Figure 4: near-unit-circle difficulty sweep.

Relative l2 test error of the fixed FNO / S4D / ZNO configurations (study
``difficulty``, seeds 0-4) in five pole-radius bins, labelled by the radius
interval and the memory horizon 1 / (1 - r) it implies.

Usage: python paper/figures/fig_near_unit_circle_difficulty.py [--results results/runs.csv]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import MODEL_COLORS, MODEL_LABELS, apply_style  # noqa: E402
from runs_table import SEEDS, Cell, P, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figs"
MODELS = [("fno", "fno"), ("s4d", "s4d_official"), ("zno", "zno")]


def radius_bins():
    """(task, r_lo, r_hi) per bin; the task names encode the interval, e.g. r0900_0930."""
    out = []
    for task in P.DIFFICULTY_CASES:
        lo, hi = task[1:].split("_")
        out.append((task, int(lo) / 1000.0, int(hi) / 1000.0))
    return sorted(out, key=lambda b: b[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    runs = load(ap.parse_args().results)

    apply_style()
    bins = radius_bins()
    xs = np.arange(len(bins))
    xlabels = [f"[{lo:.3f},{hi:.3f}]\n({1 / (1 - lo):.0f}-{1 / (1 - hi):.0f})" for _, lo, hi in bins]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for key, model in MODELS:
        cells = [Cell(runs.get("difficulty", task=task, model=model, seeds=SEEDS)) for task, _, _ in bins]
        ax.errorbar(xs, [c.test for c in cells], yerr=[c.test_sd for c in cells], marker="o", markersize=5,
                    linewidth=1.6, capsize=2.5, color=MODEL_COLORS[key], label=MODEL_LABELS[key])
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("pole-radius interval (memory horizon, steps)")
    ax.set_ylabel(r"Relative $\ell_2$ error")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_near_unit_circle_difficulty.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
