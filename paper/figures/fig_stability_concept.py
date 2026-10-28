"""Conceptual figure contrasting FNO (unit circle / frequency), LNO (left-half plane), ZNO (unit disk)."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style

OUT = Path(__file__).resolve().parents[1] / "figs"
OUT.mkdir(exist_ok=True, parents=True)


def draw_fno(ax):
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.2)
    ax.axhline(0, color="lightgray", lw=0.6)
    ax.axvline(0, color="lightgray", lw=0.6)
    modes = np.linspace(0, np.pi, 10)[1:]
    for m in modes:
        ax.plot(np.cos(m), np.sin(m), "o", color="#ff7f0e", markersize=6)
        ax.plot(np.cos(m), -np.sin(m), "o", color="#ff7f0e", markersize=6)
    ax.set_title("(a) FNO: frequency modes $\\{e^{i\\omega_k}\\}$")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}$"); ax.set_ylabel(r"$\operatorname{Im}$")


def draw_lno(ax):
    # Continuous-time s-plane: poles must have Re(s) <= 0 for stability.
    ax.axvspan(-2, 0, color="#e7f5e3", alpha=0.8)
    ax.axvspan(0, 2, color="#fdecea", alpha=0.7)
    ax.axvline(0, color="black", lw=1.0)
    ax.axhline(0, color="lightgray", lw=0.6)
    # Sample stable poles.
    np.random.seed(0)
    xs = -0.1 - 1.0 * np.random.rand(14)
    ys = 2.0 * np.random.rand(14) - 1.0
    ax.plot(xs, ys, "o", color="#1f77b4", markersize=6)
    ax.plot(xs, -ys, "o", color="#1f77b4", markersize=4, alpha=0.5)
    ax.text(-1.85, 0.92, "stable", fontsize=9, color="#277023")
    ax.text(0.55, 0.92, "unstable", fontsize=9, color="#9a2a22")
    ax.set_title("(b) LNO: $s$-plane poles $\\mu_n$")
    ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}(s)$"); ax.set_ylabel(r"$\operatorname{Im}(s)$")


def draw_zno(ax):
    theta = np.linspace(0, 2 * math.pi, 400)
    ax.fill(np.cos(theta), np.sin(theta), color="#e7f5e3", alpha=0.8)
    # Unstable region outside unit circle -> shown as very light red band
    R = 1.5
    outer = np.array([(R * math.cos(a), R * math.sin(a)) for a in theta])
    inner = np.array([(math.cos(a), math.sin(a)) for a in theta[::-1]])
    ring = np.vstack([outer, inner])
    ax.fill(ring[:, 0], ring[:, 1], color="#fdecea", alpha=0.7, edgecolor="none")
    ax.plot(np.cos(theta), np.sin(theta), color="black", lw=1.2)
    # Sample stable poles inside disk.
    np.random.seed(1)
    radii = 0.3 + 0.65 * np.random.rand(14)
    phases = np.random.uniform(0.08, math.pi - 0.08, size=14)
    xs = radii * np.cos(phases)
    ys = radii * np.sin(phases)
    ax.plot(xs, ys, "o", color="#d62728", markersize=6)
    ax.plot(xs, -ys, "o", color="#d62728", markersize=4, alpha=0.5)
    ax.axhline(0, color="lightgray", lw=0.6)
    ax.axvline(0, color="lightgray", lw=0.6)
    ax.text(-0.95, 1.32, "unstable (|z|>1)", fontsize=9, color="#9a2a22")
    ax.text(-0.35, 0.05, "stable disk", fontsize=9, color="#277023")
    ax.set_title("(c) ZNO: $z$-plane poles $p_k=\\rho_k e^{i\\phi_k}$, $\\rho_k<1$")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}(z)$"); ax.set_ylabel(r"$\operatorname{Im}(z)$")


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    draw_fno(axes[0])
    draw_lno(axes[1])
    draw_zno(axes[2])
    fig.tight_layout()
    out = OUT / "fig_stability_concept.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
