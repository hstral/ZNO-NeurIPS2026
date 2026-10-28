"""Figure 3: synthetic discrete-time benchmark, 8.5k comparison and capacity study.

  (a) about 8.5k real parameters, all models trained on the task loss (study ``matched``);
  (b-d) real budgets 4k, 8.5k, 16k, 32k (study ``capacity``; the 8.5k point is the
      8.5k comparison, with ZNO on its default objective from ``zno-default``), plotted
      against the real parameter count.
Error bars are population standard deviations over seeds 0-4.

Usage: python paper/figures/fig_benchmark_discrete.py [--results results/runs.csv]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style  # noqa: E402
from make_tables import BUDGETS, KEY, budget_cell  # noqa: E402
from runs_table import SEEDS, TASKS, Cell, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figs" / "fig_benchmark_discrete.pdf"
TASK_LABEL = {
    "resonant_arma": "Resonant ARMA",
    "iir_cascade": "Sixth-order IIR cascade",
    "nonlinear_narx": "Nonlinear NARX",
}
BAR_MODELS = ["FNO", "LNO", "S4D", "S5", "ZNO"]
LINE_MODELS = ["FNO", "S4D", "S5", "ZNO"]
# FNO/ZNO keep the paper's established hues; S4D is a darker brown, S5 a purple
# and the stabilized LNO a teal, so that every pair stays separable under
# simulated colour-vision deficiency. Curve markers add a second encoding.
COLOR = {"FNO": "#ff7f0e", "LNO": "#17becf", "S4D": "#5c3a2e", "S5": "#9467bd", "ZNO": "#d62728"}
MARKER = {"FNO": "o", "S4D": "s", "S5": "D", "ZNO": "^"}
LABEL = {"FNO": "FNO", "LNO": "Stabilized LNO", "S4D": "S4D", "S5": "S5", "ZNO": "ZNO (ours)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    args = ap.parse_args()
    runs = load(args.results)

    def main_cell(t, m):
        return Cell(runs.get("matched", task=t, model=KEY[m], seeds=SEEDS))

    apply_style()
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
                         "legend.fontsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=(7.0, 3.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.34)

    # (a) all models at approximately 8.5k real parameters, task loss only.
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(TASKS))
    width = 0.16
    for i, m in enumerate(BAR_MODELS):
        cells = [main_cell(t, m) for t in TASKS]
        ax.bar(x + (i - 2) * width, [c.test for c in cells], width=width * 0.92,
               yerr=[c.test_sd for c in cells], capsize=2, color=COLOR[m], edgecolor="black",
               linewidth=0.4, label=LABEL[m], error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel(r"Relative $\ell_2$ test error")
    ax.set_ylim(0, 1.18)
    ax.set_title("(a) About 8.5k real parameters, task loss", loc="left")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=5, frameon=False, handlelength=1.6, columnspacing=1.0)

    # (b-d) capacity sensitivity against the real parameter count.
    for j, t in enumerate(TASKS):
        ax = fig.add_subplot(gs[1, j])
        for m in LINE_MODELS:
            cells = [budget_cell(runs, b, t, m) for b in BUDGETS]
            ax.errorbar([c.params for c in cells], [c.test for c in cells], yerr=[c.test_sd for c in cells],
                        color=COLOR[m], marker=MARKER[m], markersize=4.2, linewidth=1.3, capsize=1.8,
                        elinewidth=0.8, label=LABEL[m], markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xscale("log")
        ax.set_xticks([4e3, 8e3, 16e3, 32e3])
        ax.set_xticklabels(["4k", "8k", "16k", "32k"])
        ax.minorticks_off()
        ax.set_xlabel("Real parameters (log scale)")
        if j == 0:
            ax.set_ylabel(r"Relative $\ell_2$ test error")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_title(f"({'bcd'[j]}) {TASK_LABEL[t]}", loc="left")
        if t == "nonlinear_narx":
            # FNO sits near 0.35; zoom on the recurrent models and say so.
            fno = [budget_cell(runs, b, t, "FNO").test for b in BUDGETS]
            ax.set_ylim(0.3088, 0.3132)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
            ax.text(0.97, 0.21, f"FNO: {min(fno):.3f}–{max(fno):.3f}, off scale", transform=ax.transAxes,
                    ha="right", va="center", fontsize=7, color="#444444")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None})
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
