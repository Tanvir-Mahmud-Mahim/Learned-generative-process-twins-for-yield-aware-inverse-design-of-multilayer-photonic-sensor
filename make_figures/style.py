"""Shared IEEE-transaction figure style for FabGAN-ID."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette
BLUE = "#0B5394"      # primary (ours)
CYAN = "#3FA7D6"      # secondary ours / variant
ORANGE = "#E07B39"    # baseline 1
RED = "#B02E2E"       # baseline 2 / true process
GRAY = "#8A8A8A"      # neutral baseline
GREEN = "#2E8B57"     # outcome / good
LIGHT = "#EAF1F8"

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "lines.linewidth": 1.4,
    "figure.dpi": 300,
    "savefig.dpi": 1000,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

W1, W2 = 3.45, 7.10   # IEEE single / double column widths (inches)


def panel_label(ax, s, dx=-0.115, dy=1.14):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bar_values(ax, bars, fmt="%.3f", dy=0.0005, fontsize=6.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt % h,
                ha="center", va="bottom", fontsize=fontsize)
