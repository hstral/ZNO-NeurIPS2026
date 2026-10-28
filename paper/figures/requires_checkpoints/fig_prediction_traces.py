"""Figure 10: predicted response traces of FNO / S4D / ZNO on the synthetic tasks.

For each task and model, uses the fixed-configuration run (study ``fixed``,
``runs/fixed/<case>/<model>/<run>/predictions_test.npz``) with the lowest
best-validation error over seeds 0-4 and plots one test sample (drawn with
``numpy.random.seed(0)``).

Usage: python paper/figures/requires_checkpoints/fig_prediction_traces.py [--runs-dir runs/fixed]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import MODEL_COLORS, MODEL_LABELS, apply_style  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[2] / "figs"

CASES = [
    ("resonant_arma_near_unit_circle_v4", "Resonant ARMA (near unit circle)"),
    ("high_order_iir_cascade_v5", "Sixth-order IIR cascade"),
    ("narx_softplus", "Nonlinear NARX (softplus+tanh)"),
]
MODELS = ["fno", "s4d", "zno"]


def pick_best_run(runs_dir: Path, case: str, model: str) -> Path:
    model_dir = runs_dir / case / ("s4d_official" if model == "s4d" else model)
    best = None
    best_val = float("inf")
    for run in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        s = run / "summary.json"
        if not s.exists():
            continue
        with s.open() as f:
            info = json.load(f)
        v = info.get("best_val_rel_l2", float("inf"))
        if v < best_val:
            best_val = v
            best = run
    if best is None:
        raise FileNotFoundError(f"No completed {model} runs in {model_dir}")
    return best


def load_pred(run: Path):
    z = np.load(run / "predictions_test.npz")
    return z["y_test"], z["y_pred"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs" / "fixed")
    runs_dir = ap.parse_args().runs_dir
    apply_style()
    fig, axes = plt.subplots(len(CASES), 1, figsize=(10.5, 6.6))
    if len(CASES) == 1:
        axes = [axes]
    np.random.seed(0)
    idx = None
    for row, (case, title) in enumerate(CASES):
        preds = {}
        truth = None
        for model in MODELS:
            run = pick_best_run(runs_dir, case, model)
            yt, yp = load_pred(run)
            if truth is None:
                truth = yt
                if idx is None:
                    idx = int(np.random.randint(0, truth.shape[0]))
            preds[model] = yp
        t = np.arange(truth.shape[1])
        ax = axes[row]
        for model in MODELS:
            ax.plot(t, preds[model][idx], color=MODEL_COLORS[model],
                    lw=0.95, alpha=0.72, label=MODEL_LABELS[model])
        ax.plot(
            t,
            truth[idx],
            color="black",
            lw=1.05,
            alpha=1.0,
            linestyle=(0, (3.0, 2.0)),
            label="Ground truth",
            zorder=4,
        )
        ax.set_title(title)
        ax.set_xlabel("Discrete time index $n$")
        ax.set_ylabel("Response $y_n$")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        if row == 0:
            ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True, parents=True)
    out = OUT / "fig_prediction_traces.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
