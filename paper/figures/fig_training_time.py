"""Figure 8: Table 1 errors against the training time of the same runs.

Each point is one task and model of the 8.5k comparison (study ``matched``, seeds
0-4): mean cumulative epoch-loop time per run against the mean relative l2 test
error, with population standard deviations as error bars. Times depend on the GPU.

Usage: python paper/figures/fig_training_time.py [--results results/runs.csv]
"""
import argparse
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import MODEL_COLORS, MODEL_LABELS, apply_style  # noqa: E402
from runs_table import SEEDS, TASKS, Cell, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figs" / "fig_training_time.pdf"
MODELS = ["FNO", "Stabilized LNO", "S4D", "S5", "ZNO"]
KEYS = dict(zip(MODELS, ["fno", "lno", "s4d", "s5", "zno"]))
RUN_MODEL = dict(zip(MODELS, ["fno", "lno_stable", "s4d_official", "s5_official", "zno"]))
TASK_LABELS = dict(zip(TASKS, ["Resonant ARMA", "Sixth-order IIR", "Nonlinear NARX"]))
MARKERS = dict(zip(TASKS, ["o", "s", "^"]))


def points(runs):
    out = []
    for task in TASKS:
        for model in MODELS:
            cell = Cell(runs.get("matched", task=task, model=RUN_MODEL[model], seeds=SEEDS))
            seconds = [float(r["train_time_seconds"]) for r in cell.rows]
            out.append(dict(task=task, model=model, seconds=statistics.fmean(seconds),
                            error=cell.test, error_sd=cell.test_sd))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None)
    runs = load(ap.parse_args().results)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for point in points(runs):
        ax.errorbar(point["seconds"], point["error"], yerr=point["error_sd"], marker=MARKERS[point["task"]],
                    markersize=9, color=MODEL_COLORS[KEYS[point["model"]]], markeredgecolor="black",
                    markeredgewidth=0.7, linestyle="none", capsize=2.5, alpha=0.95)
    handles = [Line2D([], [], marker="s", markersize=9, linestyle="none",
                      markerfacecolor=MODEL_COLORS[KEYS[model]], markeredgecolor="black",
                      label="Stabilized LNO" if model == "Stabilized LNO" else MODEL_LABELS[KEYS[model]])
               for model in MODELS]
    tasks = [Line2D([], [], marker=MARKERS[task], markersize=9, linestyle="none",
                    markerfacecolor="lightgray", markeredgecolor="black", label=TASK_LABELS[task])
             for task in TASKS]
    first = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Model", frameon=True)
    ax.add_artist(first)
    ax.legend(handles=tasks, loc="lower left", bbox_to_anchor=(1.02, 0.0), title="Task", frameon=True)
    ax.set_xlabel("Training time per run (seconds, epoch loop, mean over 5 seeds)")
    ax.set_ylabel(r"Relative $\ell_2$ test error")
    ax.set_title("Accuracy vs. training time")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
