"""Figure 6: internal ZNO ablation, one knob at a time away from the anchor (study
``ablation``, seeds 0-4) on the resonant ARMA and sixth-order IIR tasks.

Usage: python paper/figures/fig_ablation.py [--results results/runs.csv]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style  # noqa: E402
from runs_table import SEEDS, Cell, P, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figs"
CASES = list(P.ABLATION_CASES)
TASK_LABELS = {"resonant_arma": "Resonant ARMA", "iir_cascade": "High-order IIR cascade"}


def poles(key):
    return "/".join(str(P.ABLATION_CASES[case][key]) for case in CASES)


# (tag, axis label) per knob; "anchor" is the shared reference configuration.
TAG_GROUPS = {
    "num_poles": [("poles-low", poles("low")), ("anchor", poles("anchor")), ("poles-high", poles("high"))],
    "rank": [("rank4", "4"), ("anchor", "8"), ("rank12", "12")],
    "fir_order": [("fir0", "0"), ("fir2", "2"), ("anchor", "4"), ("fir8", "8")],
    "safe_radius": [("safe0.90", "0.90"), ("anchor", "0.95"), ("safe0.99", "0.99")],
}
GROUP_TITLES = {
    "num_poles": "(a) Number of poles per layer",
    "rank": "(b) Low-rank factor $r$",
    "fir_order": "(c) FIR order $F$",
    "safe_radius": r"(d) Safe radius $\rho_{\mathrm{safe}}$",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    runs = load(ap.parse_args().results)

    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2))
    bar_width = 0.36
    for idx, knob in enumerate(["num_poles", "rank", "fir_order", "safe_radius"]):
        ax = axes[idx]
        tags = TAG_GROUPS[knob]
        x = np.arange(len(tags))
        ymax = 0.0
        for j, case in enumerate(CASES):
            task = P.CASE_TO_TASK[case]
            cells = [Cell(runs.get("ablation", task=task, model="zno", tag=t, seeds=SEEDS)) for t, _ in tags]
            means, stds = [c.test for c in cells], [c.test_sd for c in cells]
            ymax = max(ymax, max(m + s for m, s in zip(means, stds)))
            ax.bar(x + (j - 0.5) * bar_width, means, yerr=stds, width=bar_width, capsize=2.5,
                   color=["#d62728", "#1f77b4"][j], label=TASK_LABELS[task], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in tags])
        ax.set_title(GROUP_TITLES[knob])
        ax.set_ylabel(r"Relative $\ell_2$ error" if idx == 0 else "")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_ylim(0, ymax * (1.30 if idx == 0 else 1.16))
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_ablation.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
