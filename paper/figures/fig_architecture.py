"""ZNO architecture schematic rendered with matplotlib (no TikZ dependency)."""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from _style import apply_style

OUT = Path(__file__).resolve().parents[1] / "figs"
OUT.mkdir(exist_ok=True, parents=True)


ARROW = "#2f3437"
TEXT = "#202124"
SUBTEXT = "#3f4447"
FORMULA = "#2f3437"


def box3(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    formula=None,
    fc="#eaf2fb",
    ec="#23527c",
    title_size=10.8,
    subtitle_size=9.4,
    formula_size=8.8,
    formula_yshift=0.0,
):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.055",
        linewidth=1.1, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(rect)
    cx = x + w / 2
    if subtitle is None and formula is None:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=TEXT)
    elif formula is None:
        ax.text(cx, y + h * 0.60, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=TEXT)
        ax.text(cx, y + h * 0.34, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=SUBTEXT)
    elif subtitle is None:
        ax.text(cx, y + h * 0.60, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=TEXT)
        ax.text(cx, y + h * 0.33 + formula_yshift, formula, ha="center", va="center",
                fontsize=formula_size, color=FORMULA)
    else:
        ax.text(cx, y + h * 0.68, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=TEXT)
        ax.text(cx, y + h * 0.47, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=SUBTEXT)
        ax.text(cx, y + h * 0.24 + formula_yshift, formula, ha="center", va="center",
                fontsize=formula_size, color=FORMULA)


def arrow(ax, x1, y1, x2, y2, lw=1.05, color=ARROW, connectionstyle="arc3"):
    ax.annotate("",
                xy=(x2, y2), xycoords="data",
                xytext=(x1, y1), textcoords="data",
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                shrinkA=2, shrinkB=2,
                                connectionstyle=connectionstyle))


def sum_node(ax, x, y, radius=0.18, fontsize=13.5):
    circ = mpatches.Circle((x, y), radius=radius, facecolor="#ffffff",
                           edgecolor="#455a64", linewidth=1.25)
    ax.add_patch(circ)
    ax.text(x, y - 0.005, "+", ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#263238")


def main():
    apply_style()
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11.0,
        "axes.titlesize": 11.6,
    })
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8.0, 4.4),
        gridspec_kw=dict(height_ratios=[0.68, 1.25], hspace=0.04),
    )
    # ---------- Top: full network pipeline ----------
    ax_top.set_xlim(0, 13.35)
    ax_top.set_ylim(0.40, 2.05)
    ax_top.axis("off")
    y0 = 0.80
    h = 0.90
    boxes = [
        (0.20, "Input", r"$x_{0:T-1}$", 1.55, "#fff4df", "#b7791f"),
        (2.05, "Lift", r"$P$", 1.05, "#eaf2fb", "#2f5f8f"),
        (3.40, "ZNO block 1", "rational layer", 1.75, "#e8f5e9", "#2e7d32"),
        (5.45, "ZNO block 2", "rational layer", 1.75, "#e8f5e9", "#2e7d32"),
        (7.75, r"ZNO block $L$", "rational layer", 1.75, "#e8f5e9", "#2e7d32"),
        (9.80, "Projection", r"$Q$", 1.22, "#eaf2fb", "#2f5f8f"),
        (11.32, "Output", r"$\hat{y}_{0:T-1}$", 1.75, "#fff4df", "#b7791f"),
    ]
    last_x = None
    for x, title, subtitle, w, fc, ec in boxes:
        box3(ax_top, x, y0, w, h, title, subtitle, fc=fc, ec=ec,
             title_size=10.8, subtitle_size=9.4)
        if last_x is not None:
            arrow(ax_top, last_x, y0 + h / 2, x, y0 + h / 2)
        last_x = x + w
    ax_top.text(7.48, y0 + h / 2 + 0.05, r"$\cdots$", ha="center", va="center",
                fontsize=12.5, color=SUBTEXT)
    ax_top.set_title("(a) Causal network with per-time-step residual ZNO blocks", pad=0)

    # ---------- Bottom: inside one ZNO layer ----------
    ax_bot.set_xlim(0, 13.0)
    ax_bot.set_ylim(0, 2.72)
    ax_bot.axis("off")
    main_y = 1.42
    skip_y = 0.32
    h = 0.82
    center_y = main_y + h / 2
    skip_cy = skip_y + h / 2

    # A subtle band makes the dynamic branch visually distinct from the skip path.
    branch_x, branch_w = 1.88, 8.43
    branch = mpatches.FancyBboxPatch(
        (branch_x, 1.20), branch_w, 1.25,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=0.8, facecolor="#fff5f5", edgecolor="#f1c4c4",
        alpha=0.58, zorder=0,
    )
    ax_bot.add_patch(branch)
    ax_bot.text(branch_x + branch_w / 2, 2.36, "Rational branch",
                ha="center", va="center", fontsize=10.0, color="#8a3030",
                fontweight="bold")

    hidden = (0.25, main_y, 1.42)
    bproj = (1.96, main_y, 1.34)
    scan = (3.59, main_y, 2.25)
    readout = (6.13, main_y, 2.00)
    aproj = (8.42, main_y, 1.75)
    add_x = 10.74
    output = (11.15, main_y, 1.55)
    skip = (4.25, skip_y, 2.15)

    box3(ax_bot, *hidden, h, "Hidden state", None, r"$h^{(\ell)}_n\in\mathbb{R}^{w}$",
         fc="#eaf2fb", ec="#2f5f8f", title_size=10.8, formula_size=8.7)
    box3(ax_bot, *bproj, h, "Low-rank\nprojection", None, r"$B:\,w\to r$",
         fc="#fff4df", ec="#b7791f", title_size=10.0, formula_size=8.7,
         formula_yshift=-0.065)
    box3(ax_bot, *scan, h, "Stable pole scan", "per latent filter",
         r"$s_{a,k,n}=p_{a,k}s_{a,k,n-1}+b_{a,n}$",
         fc="#ffe9e9", ec="#9b2c2c", title_size=10.2, subtitle_size=9.0,
         formula_size=8.1)
    box3(ax_bot, *readout, h, "Residues + FIR", "latent readout",
         r"$q_{a,n}=\mathrm{Re}\,\Sigma_k c_{a,k}s_{a,k,n}+\mathrm{FIR}_a$",
         fc="#ffe9e9", ec="#9b2c2c", title_size=10.2, subtitle_size=9.0,
         formula_size=7.6, formula_yshift=-0.03)
    box3(ax_bot, *aproj, h, "Output projection", None, r"$A:\,r\to w$",
         fc="#fff4df", ec="#b7791f", title_size=9.8, formula_size=8.7)
    box3(ax_bot, *skip, h, "Pointwise skip", None, r"$W:\,w\to w$",
         fc="#eef2f5", ec="#607d8b", title_size=10.8, formula_size=8.7)
    box3(ax_bot, *output, h, "Layer output", None, r"$h^{(\ell+1)}_n$",
         fc="#eaf2fb", ec="#2f5f8f", title_size=10.8, formula_size=8.7)
    sum_node(ax_bot, add_x, center_y, radius=0.15, fontsize=11.5)

    # Main rational path.
    path = [hidden, bproj, scan, readout, aproj]
    for left, right in zip(path[:-1], path[1:]):
        arrow(ax_bot, left[0] + left[2], center_y, right[0], center_y)
    arrow(ax_bot, aproj[0] + aproj[2], center_y, add_x - 0.16, center_y)
    arrow(ax_bot, add_x + 0.16, center_y, output[0], center_y)

    # Parallel skip path.
    arrow(ax_bot, hidden[0] + hidden[2] * 0.45, main_y, skip[0], skip_cy,
          connectionstyle="angle3,angleA=-90,angleB=180")
    arrow(ax_bot, skip[0] + skip[2], skip_cy, add_x, center_y - 0.20,
          connectionstyle="angle3,angleA=0,angleB=-90")

    ax_bot.set_title("(b) One residual ZNO layer: low-rank rational branch plus skip", pad=0)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.065)
    out = OUT / "fig_architecture.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
