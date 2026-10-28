"""Shared matplotlib style for all ZNO paper figures."""
import matplotlib as mpl


MODEL_COLORS = {
    "lno": "#17becf",
    "fno": "#ff7f0e",
    "s4d": "#8c564b",
    "s5": "#9467bd",
    "zno": "#d62728",
    "truth": "#2ca02c",
    "pred": "#d62728",
}

MODEL_LABELS = {
    "lno": "LNO",
    "fno": "FNO",
    "s4d": "S4D",
    "s5": "S5",
    "zno": "ZNO (ours)",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
