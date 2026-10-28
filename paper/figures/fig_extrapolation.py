"""Figure 5: long-horizon evaluation, trained at T=2048 and evaluated at 2048, 4096, 8192.

T=2048 is the study ``fixed``; T=4096 and 8192 are the study ``extrapolation``
(same configurations and seeds 0-4, longer test sequences).

Usage: python paper/figures/fig_extrapolation.py [--results results/runs.csv]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import MODEL_COLORS, MODEL_LABELS, apply_style  # noqa: E402
from runs_table import SEEDS, TASKS, Cell, P, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figs"
TASK_LABELS = {
    "resonant_arma": "Resonant ARMA (near unit circle)",
    "iir_cascade": "Sixth-order IIR cascade",
    "nonlinear_narx": "Nonlinear NARX",
}
MODELS = [("fno", "fno"), ("s4d", "s4d_official"), ("zno", "zno")]
LENGTHS = [2048, *P.EXTRAPOLATION_LENGTHS]


def cell(runs, task, model, length):
    if length == 2048:
        return Cell(runs.get("fixed", task=task, model=model, seeds=SEEDS))
    return Cell(runs.get("extrapolation", task=task, model=model, tag=f"T{length}", seeds=SEEDS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    runs = load(ap.parse_args().results)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4), sharey=False)
    for ax, task in zip(axes, TASKS):
        for key, model in MODELS:
            cells = [cell(runs, task, model, n) for n in LENGTHS]
            ax.errorbar(LENGTHS, [c.test for c in cells], yerr=[c.test_sd for c in cells], marker="o",
                        linewidth=1.6, markersize=5.5, color=MODEL_COLORS[key], label=MODEL_LABELS[key], capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS)
        ax.set_xticklabels([str(n) for n in LENGTHS])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Evaluation sequence length $T$")
        ax.set_ylabel(r"Relative $\ell_2$ error")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axes[0].legend(loc="upper left", ncol=1, frameon=True)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "fig_extrapolation.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
