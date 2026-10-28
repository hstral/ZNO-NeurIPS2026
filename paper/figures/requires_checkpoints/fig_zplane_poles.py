"""Figure 9: z-plane poles of trained ZNO checkpoints, all layers.

Loads, per synthetic task, the checkpoint of the ZNO fixed-configuration run
(study ``fixed``, ``runs/fixed/<case>/zno/<run>/model_best.pt``) with the lowest
best-validation error over seeds 0-4, reconstructs the stable poles from the
checkpoint parameterization, and scatters them inside the unit disk, with marker
size proportional to the residue magnitude.

Usage: python paper/figures/requires_checkpoints/fig_zplane_poles.py [--runs-dir runs/fixed]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[2] / "figs"

CASES = [
    ("resonant_arma_near_unit_circle_v4", "Resonant ARMA"),
    ("high_order_iir_cascade_v5", "High-order IIR cascade"),
    ("narx_softplus", "Nonlinear NARX"),
]


def load_best_ckpt(runs_dir: Path, case: str):
    model_dir = runs_dir / case / "zno"
    runs = sorted(p for p in model_dir.iterdir() if p.is_dir())
    best = None
    best_val = float("inf")
    for run in runs:
        summary = run / "summary.json"
        if not summary.exists():
            continue
        with summary.open() as f:
            info = json.load(f)
        val = info.get("best_val_rel_l2", float("inf"))
        if val < best_val:
            best_val = val
            best = run
    if best is None:
        raise FileNotFoundError(f"No ZNO runs found for {case}")
    ckpt = torch.load(best / "model_best.pt", map_location="cpu", weights_only=False)
    return ckpt, best


def recover_poles(state_dict, radius_limit):
    """Return list of (per-layer complex pole tensor) arrays of shape [rank, num_poles]."""
    layers = {}
    for key, value in state_dict.items():
        if not key.startswith("layers."):
            continue
        _, idx, name = key.split(".", 2)
        idx = int(idx)
        layers.setdefault(idx, {})[name] = value
    per_layer_poles = []
    per_layer_residues = []
    for idx in sorted(layers):
        entries = layers[idx]
        poles = []
        residues = []
        if "pair_radius" in entries:
            radius = radius_limit * torch.sigmoid(entries["pair_radius"])
            pair_poles = torch.polar(radius, entries["pair_phase"])
            pair_res = torch.complex(entries["pair_residue_real"], entries["pair_residue_imag"])
            poles.extend([pair_poles, pair_poles.conj()])
            residues.extend([pair_res, pair_res.conj()])
        if "real_radius" in entries:
            rr = radius_limit * torch.sigmoid(entries["real_radius"])
            real_pole = torch.complex(rr, torch.zeros_like(rr))
            real_res = torch.complex(entries["real_residue"], torch.zeros_like(entries["real_residue"]))
            poles.append(real_pole)
            residues.append(real_res)
        per_layer_poles.append(torch.cat(poles, dim=-1).detach().cpu().numpy())
        per_layer_residues.append(torch.cat(residues, dim=-1).detach().cpu().numpy())
    return per_layer_poles, per_layer_residues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs" / "fixed")
    runs_dir = ap.parse_args().runs_dir
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, (case, title) in zip(axes, CASES):
        ckpt, run = load_best_ckpt(runs_dir, case)
        radius_limit = ckpt["model_config"].get("radius_limit", 0.999)
        safe_radius = ckpt["model_config"].get("safe_radius", 0.95)
        poles, residues = recover_poles(ckpt["model_state_dict"], radius_limit)
        # Draw unit circle and safe radius.
        theta = np.linspace(0, 2 * math.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.1)
        ax.plot(safe_radius * np.cos(theta), safe_radius * np.sin(theta),
                color="gray", lw=0.8, linestyle="--")
        ax.axhline(0, color="lightgray", lw=0.5)
        ax.axvline(0, color="lightgray", lw=0.5)
        depth = len(poles)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, depth))
        for idx, p in enumerate(poles):
            real = p.real.ravel()
            imag = p.imag.ravel()
            # Weight marker size by residue magnitude (capped for readability).
            mag = np.abs(residues[idx]).ravel()
            sizes = 8 + 55 * (mag / (mag.max() + 1e-9))
            ax.scatter(real, imag, s=sizes, c=[colors[idx]], alpha=0.70,
                       edgecolors="none", label=f"layer {idx}")
        ax.set_title(title)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\operatorname{Re}(z)$")
        ax.set_ylabel(r"$\operatorname{Im}(z)$")
        ax.grid(False)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(r"Learned ZNO poles on the $z$-plane (unit circle = stability boundary, dashed = safe radius)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT.mkdir(exist_ok=True, parents=True)
    out = OUT / "fig_zplane_poles.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
