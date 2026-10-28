"""Figure 7: 1D LNO ODE benchmark, matched-budget configurations (study ``one-d``;
six cases, LNO/FNO/ZNO, seeds 0-2; error bars are population standard deviations).

Usage: python paper/figures/fig_benchmark_1d.py [--results results/runs.csv]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style  # noqa: E402
from runs_table import ONE_D_SEEDS, Cell, P, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figs" / "fig_benchmark_1d.pdf"
CASE_LABEL = {
    "pendulum_c0": "Pendulum\n" r"$c{=}0$",
    "pendulum_c05": "Pendulum\n" r"$c{=}0.5$",
    "duffing_c0": "Duffing\n" r"$c{=}0$",
    "duffing_c05": "Duffing\n" r"$c{=}0.5$",
    "lorenz_rho5": "Lorenz\n" r"$\rho{=}5$",
    "lorenz_rho10": "Lorenz\n" r"$\rho{=}10$",
}
MODELS = list(P.ONE_D_MODELS)
# Same hues as figs/fig_benchmark_discrete.pdf.
COLOR = {"lno": "#17becf", "fno": "#ff7f0e", "zno": "#d62728"}
LABEL = {"lno": "LNO", "fno": "FNO", "zno": "ZNO (ours)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    runs = load(ap.parse_args().results)

    apply_style()
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    cases = list(P.ONE_D_CASES)
    x = np.arange(len(cases))
    width = 0.26
    for i, m in enumerate(MODELS):
        cells = [Cell(runs.get("one-d", task=c, model=m, seeds=ONE_D_SEEDS)) for c in cases]
        ax.bar(x + (i - 1) * width, [c.test for c in cells], width=width * 0.92, yerr=[c.test_sd for c in cells],
               capsize=2, color=COLOR[m], edgecolor="black", linewidth=0.4, label=LABEL[m],
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in cases])
    ax.set_ylabel(r"Relative $L_2$ test error")
    ax.set_ylim(0, 1.25)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
