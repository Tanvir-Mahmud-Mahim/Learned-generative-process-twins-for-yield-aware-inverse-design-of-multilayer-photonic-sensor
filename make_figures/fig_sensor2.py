"""Fig. 1: professional sensor-context figure (redesigned).

(a) fluorescence-biosensor readout chain (vertical optical train, clean
    label lanes, no text overlap);
(b) cross-section of the actual optimized 20-layer variable-index stack with
    incident/reflected/transmitted rays, dimension brace and index colorbar;
(c) annotated spectral response.
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Polygon, Circle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import BLUE, CYAN, ORANGE, RED, GRAY, GREEN, LIGHT, W2, despine

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")

EXC = "#2E8B57"      # excitation green
EMI = "#C0392B"      # emission red


def arrow(ax, x1, y1, x2, y2, color, lw=1.6, mut=10, ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=mut, color=color, lw=lw,
                                 ls=ls, alpha=alpha, zorder=6,
                                 shrinkA=0, shrinkB=0))


def panel_a(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    LX = 0.44          # optical axis x
    # --- laser ---
    ax.add_patch(FancyBboxPatch((0.02, 0.845), 0.20, 0.115,
                                boxstyle="round,pad=0.010", fc="#E8F4E8",
                                ec=EXC, lw=1.1))
    ax.text(0.12, 0.900, "laser\n532 nm", ha="center", va="center",
            fontsize=6.8, color="#1A5B38", fontweight="bold")
    # beam laser -> cuvette
    arrow(ax, 0.238, 0.885, LX - 0.062, 0.885, EXC, lw=2.4)
    # --- cuvette with fluorophores ---
    ax.add_patch(Rectangle((LX - 0.075, 0.80), 0.15, 0.185, fc="#EAF4FB",
                           ec="#456", lw=1.0, zorder=4))
    for fx, fy in [(LX - 0.035, 0.90), (LX + 0.02, 0.935), (LX + 0.035, 0.85),
                   (LX - 0.02, 0.845), (LX + 0.0, 0.885)]:
        ax.add_patch(Circle((fx, fy), 0.010, fc=EMI, ec="none", zorder=5))
    ax.text(LX + 0.11, 0.965, "analyte + fluorophores", fontsize=6.6,
            ha="left", va="center")
    # --- downward: emission + residual excitation ---
    arrow(ax, LX + 0.022, 0.795, LX + 0.022, 0.575, EMI, lw=2.4)
    arrow(ax, LX - 0.022, 0.795, LX - 0.022, 0.575, EXC, lw=1.3)
    ax.text(LX + 0.09, 0.71, "fluorescence\nemission\n($\\lambda>560$ nm)",
            fontsize=6.2, color=EMI, ha="left", va="center")
    ax.text(LX - 0.09, 0.71, "residual\nexcitation", fontsize=6.2, color=EXC,
            ha="right", va="center")
    # --- notch filter slab (this work) ---
    ax.add_patch(Rectangle((LX - 0.17, 0.495), 0.34, 0.075, fc=LIGHT,
                           ec=BLUE, lw=1.6, zorder=5))
    for k in range(1, 6):
        ax.plot([LX - 0.17, LX + 0.17],
                [0.495 + k * 0.0125, 0.495 + k * 0.0125],
                color=BLUE, lw=0.35, alpha=0.55, zorder=6)
    ax.text(LX + 0.20, 0.532, "variable-index SiN$_x$\nnotch filter"
            " (this work)", fontsize=6.6, color=BLUE, ha="left",
            va="center", fontweight="bold")
    # --- after filter ---
    # blocked excitation: stops at slab
    ax.plot([LX - 0.022], [0.452], marker="x", ms=6.5, mew=1.7, color=EXC,
            zorder=7)
    ax.text(LX - 0.085, 0.452, "excitation\nblocked", fontsize=6.2, color=EXC,
            ha="right", va="center")
    # transmitted emission
    arrow(ax, LX + 0.022, 0.492, LX + 0.022, 0.262, EMI, lw=2.4)
    ax.text(LX + 0.09, 0.38, "emission\ntransmitted", fontsize=6.2,
            color=EMI, ha="left", va="center")
    # --- photodetector ---
    ax.add_patch(Polygon([[LX - 0.13, 0.255], [LX + 0.13, 0.255],
                          [LX + 0.085, 0.145], [LX - 0.085, 0.145]],
                         fc="#F3E7D3", ec="#333333", lw=1.0, zorder=5))
    # sensing element stripe inside the detector
    ax.add_patch(Rectangle((LX - 0.115, 0.225), 0.23, 0.022, fc="#C9A96A",
                           ec="none", zorder=6))
    ax.text(LX, 0.108, "photodetector", ha="center", va="center",
            fontsize=6.4, zorder=7)
    ax.text(LX, 0.028, "SNR set by stopband leakage (yield-critical)",
            ha="center", fontsize=5.8, style="italic", color="#555555")


def panel_b(ax, d, n):
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(1.6, 2.4)
    X0, X1 = 0.10, 0.86
    tot = float(np.sum(d)) * 1000
    y = 0.0
    for i in range(len(d)):
        ax.add_patch(Rectangle((X0, y), X1 - X0, d[i] * 1000,
                               fc=cmap(norm(n[i])), ec="white", lw=0.4))
        y += d[i] * 1000
    # substrate + air
    ax.add_patch(Rectangle((X0, -300), X1 - X0, 300, fc="#DDDDDD",
                           ec="#999999", lw=0.5))
    ax.text((X0 + X1) / 2, -150, "fused-SiO$_2$ substrate", ha="center",
            va="center", fontsize=6.4, color="#444444")
    ax.text(X0 + 0.02, tot + 210, "air", ha="left", va="center",
            fontsize=6.2, color="#666666", style="italic")
    # rays
    xin = (X0 + X1) / 2 - 0.11
    arrow(ax, xin, tot + 430, xin, tot + 30, "#222222", lw=1.5, mut=9)
    ax.text(xin - 0.03, tot + 510, "incident", ha="right", fontsize=6.0)
    xr = (X0 + X1) / 2 + 0.11
    arrow(ax, xr, tot + 30, xr, tot + 430, "#999999", lw=1.0, mut=8,
          ls=(0, (4, 2)))
    ax.text(xr + 0.03, tot + 510, "reflected", ha="left", fontsize=6.0,
            color="#888888")
    xt = (X0 + X1) / 2
    arrow(ax, xt, -310, xt, -600, "#222222", lw=1.5, mut=9)
    ax.text(xt, -700, "transmitted $T(\\lambda)$", ha="center",
            fontsize=6.0)
    # dimension brace (right) with horizontal two-line label
    bx = X1 + 0.05
    ax.annotate("", xy=(bx, tot), xytext=(bx, 0),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#333333"))
    ax.text(bx + 0.04, tot * 0.52, "$N=20$\nlayers", ha="left",
            va="center", fontsize=5.9)
    ax.text(bx + 0.04, tot * 0.18, "%.2f $\\mu$m" % (tot / 1000), ha="left",
            va="center", fontsize=5.9)
    # callout for one layer (upper left)
    ci = 15
    ycal = float(np.sum(d[:ci]) + d[ci] / 2) * 1000
    ax.annotate("layer $i$:\n$t_i \\in [20,120]$ nm\n$n_i \\in [1.6,2.4]$",
                xy=(X0, ycal), xytext=(-0.42, tot + 60), fontsize=5.8,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#555555",
                                shrinkA=2, shrinkB=1))
    # colorbar (horizontal, below)
    cax = ax.inset_axes([0.16, -0.075, 0.62, 0.038])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.gcf().colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"layer index $n_i(\lambda_0)$", fontsize=6.0, labelpad=2)
    cb.ax.tick_params(labelsize=5.7)
    ax.set_xlim(-0.44, 1.22)
    ax.set_ylim(-780, tot + 600)
    ax.axis("off")


def panel_c(ax, lam, T):
    ax.plot(lam, T, color=BLUE, lw=1.6, zorder=5)
    ax.axvspan(517, 547, color=EXC, alpha=0.15, zorder=1)
    ax.axvspan(577, 800, color=EMI, alpha=0.06, zorder=1)
    ax.axvspan(400, 487, color=EMI, alpha=0.06, zorder=1)
    ax.annotate("excitation\nstop band", xy=(532, 0.12), xytext=(437, 0.36),
                fontsize=6.4, color=EXC, ha="center",
                arrowprops=dict(arrowstyle="->", color=EXC, lw=0.8))
    ax.text(688, 0.62, "emission\npassband", fontsize=6.4, color=EMI,
            ha="center")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmittance $T$")
    ax.set_xlim(395, 805)
    ax.set_ylim(0, 1.06)
    despine(ax)
    ax.grid(lw=0.3, alpha=0.4)


def main():
    import jax.numpy as jnp
    from src import tmm_jax
    r = json.load(open(os.path.join(RES, "exp4_inverse.json")))
    d = np.array(r["theta_d"]); n = np.array(r["theta_n"])
    T = np.asarray(tmm_jax.transmittance_jit(jnp.asarray(d), jnp.asarray(n)))
    lam = np.asarray(tmm_jax.LAM_UM) * 1000

    fig = plt.figure(figsize=(W2, 2.50))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.16, 1.06],
                          wspace=0.30, left=0.015, right=0.985, top=0.90,
                          bottom=0.14)
    axa = fig.add_subplot(gs[0]); panel_a(axa)
    axb = fig.add_subplot(gs[1]); panel_b(axb, d, n)
    axc = fig.add_subplot(gs[2]); panel_c(axc, lam, T)
    for a, s, dx in [(axa, "(a)", 0.01), (axb, "(b)", -0.30), (axc, "(c)", -0.17)]:
        a.text(dx, 1.07, s, transform=a.transAxes, fontsize=9.5,
               fontweight="bold", va="top")
    fig.savefig(os.path.join(FIG, "fig_sensor.pdf"), bbox_inches=None)
    plt.close(fig)
    print("made sensor (redesign)")


if __name__ == "__main__":
    main()
