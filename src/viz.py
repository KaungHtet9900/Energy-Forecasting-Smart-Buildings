"""Shared plotting style for publication-quality figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# colorblind-friendly palette
PALETTE = sns.color_palette("colorblind")
MODEL_COLORS = {
    "persistence": "#999999",
    "seasonal_naive": "#777777",
    "linear": "#56B4E9",
    "lightgbm": "#009E73",
    "lstm": "#0072B2",
    "gru": "#56B4E9",
    "tcn": "#E69F00",
    "cnn_lstm": "#D55E00",
    "transformer": "#CC79A7",
    "attn_lstm": "#F0E442",
    "tft": "#D55E00",
}


def setup_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
    })


def savefig(fig, path, also_pdf: bool = True) -> None:
    fig.savefig(path)
    if also_pdf:
        pdf = str(path).rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf)
    plt.close(fig)
