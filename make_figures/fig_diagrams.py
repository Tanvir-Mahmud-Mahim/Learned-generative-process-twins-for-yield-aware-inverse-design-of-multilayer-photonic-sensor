"""Professional schematic figures: framework, sensor system, graphical
abstract.  All data-bearing elements are drawn from the released results."""

import json
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Polygon, Circle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (BLUE, CYAN, ORANGE, RED, GRAY, GREEN, LIGHT, W1, W2,
                   panel_label, despine)

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------- helpers --
def stage_box(ax, x, y, w, h, title, lines, title_fc, body_fc="white",
              dashed=False, title_fs=9.2, body_fs=8.2):
    """Rounded box with a colored title bar."""
    ls = (0, (3, 2)) if dashed else "-"
    ax.add_patch(FancyBboxPatch((x + 0.006, y - 0.012), w, h,
                                boxstyle="round,pad=0.008", fc="#00000018",
                                ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                fc=body_fc, ec="#333333", lw=0.9, ls=ls,
                                zorder=2))
    th = 0.055
    ax.add_patch(FancyBboxPatch((x, y + h - th), w, th,
                                boxstyle="round,pad=0.008", fc=title_fc,
                                ec="none", zorder=3))
    ax.add_patch(Rectangle((x - 0.004, y + h - th - 0.008), w + 0.008, th * 0.5,
                           fc=title_fc, ec="none", zorder=3))
    ax.text(x + w / 2, y + h - th / 2 + 0.002, title, ha="center",
            va="center", fontsize=title_fs, fontweight="bold", color="white",
            zorder=4)
    ax.text(x + w / 2, y + (h - th) / 2, "\n".join(lines), ha="center",
            va="center", fontsize=body_fs, zorder=4, linespacing=1.35)


def arr(ax, x1, y1, x2, y2, color="#333333", lw=1.2, ls="-", label=None,
        lab_dx=0, lab_dy=0.028, fs=8.6, style="-|>", mut=11, zorder=5,
        connectionstyle=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=mut, color=color, lw=lw,
                                 ls=ls, zorder=zorder,
                                 connectionstyle=connectionstyle or "arc3"))
    if label:
        ax.text((x1 + x2) / 2 + lab_dx, (y1 + y2) / 2 + lab_dy, label,
                ha="center", va="bottom", fontsize=fs, color=color,
                style="italic")


# ---------------------------------------------------------- framework -----
def fig_framework():
    fig, ax = plt.subplots(figsize=(W2, 2.58))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # top row: the three optimization stages + outcome
    stage_box(ax, 0.005, 0.56, 0.235, 0.38, "STAGE 1",
              [r"Probe-seeded", r"adjoint inverse design",
               r"exact differentiable TMM", r"$360$ solver queries"], BLUE)
    stage_box(ax, 0.30, 0.56, 0.22, 0.38, "STAGE 2",
              [r"Pathwise CVaR$_{5\%}$", r"robustification",
               r"$\nabla_\theta$ through twin", r"$+$ solver jointly"], BLUE)
    stage_box(ax, 0.58, 0.56, 0.235, 0.38, "STAGE 3",
              [r"Spec-conditioned", r"correction policy",
               r"analytic policy gradients", r"(chain-GCN encoder)"], BLUE)
    stage_box(ax, 0.85, 0.30, 0.145, 0.46, "OUTCOME",
              [r"Yield-qualified", r"sensor front-end", "",
               r"$P_5$, CVaR$_{5\%}$", r"$+7.2\%$ vs. nominal"], GREEN,
              body_fs=7.4)

    # bottom row: data -> twin ; hidden process
    stage_box(ax, 0.005, 0.06, 0.235, 0.36, "PROCESS DATA",
              [r"$400$ historical traces", r"(recipe $\rightarrow$ outcome)",
               r"in-situ monitoring"], ORANGE)
    stage_box(ax, 0.30, 0.06, 0.22, 0.36, "FabGAN TWIN",
              [r"conditional WGAN-GP", r"$+$ moment matching",
               r"$+$ tail calibration", r"differentiable sampler"], RED)
    stage_box(ax, 0.58, 0.06, 0.235, 0.36, "TRUE PROCESS",
              [r"held out from", r"all learning",
               r"$2{,}000$ MC draws", r"(evaluation only)"], GRAY,
              dashed=True)

    # flows
    arr(ax, 0.244, 0.75, 0.296, 0.75, label=r"$\theta^\star$", lab_dy=0.045, fs=7.4, lw=1.4)
    arr(ax, 0.524, 0.75, 0.576, 0.75, label=r"$\theta_{rob}$", lab_dy=0.045, fs=7.4, lw=1.4)
    arr(ax, 0.819, 0.75, 0.848, 0.68, lw=1.4)
    arr(ax, 0.244, 0.24, 0.296, 0.24, label="train", lab_dy=0.04, fs=7.4, lw=1.4,
        color=ORANGE)
    arr(ax, 0.44, 0.42, 0.44, 0.56, label=r"$\tilde{\theta}\sim G_\phi(z,\theta)$",
        lab_dx=0.105, lab_dy=-0.055, lw=1.4, color=RED)
    arr(ax, 0.6975, 0.42, 0.6975, 0.56, color=GRAY, ls=(0, (3, 2)),
        label="evaluate", lab_dx=0.068, lab_dy=-0.055, fs=8.0)
    arr(ax, 0.819, 0.20, 0.878, 0.30, color=GRAY, ls=(0, (3, 2)))
    # backward gradient arrow (differentiability)
    arr(ax, 0.355, 0.55, 0.355, 0.43, color=BLUE, ls=(0, (1, 1.2)),
        style="<|-", label=r"$\partial J/\partial\theta$", lab_dx=-0.072,
        lab_dy=-0.058, fs=8.0)

    ax.text(0.5, 1.05, "learn what cannot be simulated  |  "
            "differentiate what can", ha="center", va="top", fontsize=9.6,
            style="italic", color="#333333")
    fig.savefig(os.path.join(FIG, "fig_framework.pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


# ------------------------------------------------------------- sensor -----
def fig_sensor():
    import jax.numpy as jnp
    from src import tmm_jax
    r = json.load(open(os.path.join(RES, "exp4_inverse.json")))
    d = np.array(r["theta_d"])
    n = np.array(r["theta_n"])
    T = np.asarray(tmm_jax.transmittance_jit(jnp.asarray(d), jnp.asarray(n)))
    lam = np.asarray(tmm_jax.LAM_UM) * 1000

    fig = plt.figure(figsize=(W2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.80, 1.25], wspace=0.55)

    # (a) fluorescence biosensor readout chain
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # laser
    ax.add_patch(Rectangle((0.06, 0.60), 0.17, 0.20, fc="#DFF0D8",
                           ec="#333333", lw=0.8))
    ax.text(0.145, 0.70, "532 nm\nlaser", ha="center", va="center", fontsize=6.5)
    # sample cell with fluorophores
    ax.add_patch(Rectangle((0.40, 0.55), 0.15, 0.30, fc="#EAF6FB",
                           ec="#333333", lw=0.8))
    for fx, fy in [(0.44, 0.63), (0.50, 0.74), (0.47, 0.68), (0.52, 0.60)]:
        ax.add_patch(Circle((fx, fy), 0.011, fc=RED, ec="none"))
    ax.text(0.475, 0.90, "analyte +\nfluorophores", ha="center", fontsize=6.5)
    # excitation beam
    arr(ax, 0.23, 0.70, 0.40, 0.70, color=GREEN, lw=2.0, style="-|>", mut=9)
    ax.text(0.315, 0.74, "excitation", fontsize=6, color=GREEN, ha="center")
    # emission + scattered excitation downward
    arr(ax, 0.475, 0.55, 0.475, 0.40, color=RED, lw=2.0, style="-|>", mut=9)
    arr(ax, 0.445, 0.55, 0.445, 0.40, color=GREEN, lw=1.2, style="-|>", mut=7)
    ax.text(0.60, 0.47, "emission (>560 nm)\n+ residual excitation",
            fontsize=6, ha="left", va="center")
    # notch filter = our stack
    ax.add_patch(Rectangle((0.36, 0.30), 0.23, 0.09, fc=LIGHT, ec=BLUE,
                           lw=1.4))
    ax.text(0.475, 0.345, "variable-index SiN$_x$\nnotch front-end (this work)",
            ha="center", va="center", fontsize=6, color=BLUE)
    # after filter: only emission passes
    arr(ax, 0.475, 0.30, 0.475, 0.16, color=RED, lw=2.0, style="-|>", mut=9)
    ax.plot([0.415, 0.415], [0.30, 0.24], color=GREEN, lw=1.2)
    ax.plot([0.400, 0.430], [0.24, 0.24], color=GREEN, lw=1.2)
    ax.text(0.395, 0.20, "excitation\nblocked", fontsize=5.5, color=GREEN,
            ha="right")
    # detector
    ax.add_patch(Polygon([[0.40, 0.16], [0.55, 0.16], [0.52, 0.05],
                          [0.43, 0.05]], fc="#F4E8D8", ec="#333333", lw=0.8))
    ax.text(0.475, 0.105, "photo-\ndetector", ha="center", va="center",
            fontsize=6)
    panel_label(ax, "(a)", dx=0.0, dy=1.02)

    # (b) stack cross-section, colored by layer index (actual optimum)
    ax = fig.add_subplot(gs[1])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(1.6, 2.4)
    y = 0.0
    for i in range(len(d)):
        ax.add_patch(Rectangle((0, y), 1.0, d[i] * 1000, fc=cmap(norm(n[i])),
                               ec="white", lw=0.3))
        y += d[i] * 1000
    total = y
    ax.add_patch(Rectangle((0, -220), 1.0, 220, fc="#D9D9D9", ec="none"))
    ax.text(0.5, -110, "fused-SiO$_2$ substrate", ha="center", va="center",
            fontsize=6.5)
    ax.annotate("", xy=(1.10, total), xytext=(1.10, 0),
                arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(1.16, total / 2, "%.2f $\\mu$m" % (total / 1000), rotation=90,
            va="center", fontsize=6.5)
    arr(ax, 0.5, total + 420, 0.5, total + 60, color="#333333", lw=1.2, mut=8)
    ax.text(0.62, total + 240, "incident light", ha="left", va="center",
            fontsize=6.5)
    cax = ax.inset_axes([-0.42, 0.12, 0.09, 0.58])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label(r"$n(\lambda_0)$", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    ax.set_xlim(-0.05, 1.35)
    ax.set_ylim(-240, total + 520)
    ax.axis("off")
    panel_label(ax, "(b)", dx=-0.16, dy=1.02)

    # (c) spectral response with sensor annotations
    ax = fig.add_subplot(gs[2])
    ax.plot(lam, T, color=BLUE, lw=1.5)
    ax.axvspan(517, 547, color=GREEN, alpha=0.18)
    ax.axvspan(577, 800, color=RED, alpha=0.07)
    ax.annotate("excitation\nblocked", xy=(532, 0.10), xytext=(462, 0.30),
                fontsize=6.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
    ax.text(680, 0.55, "emission\npassband", fontsize=6.5, color=RED,
            ha="center")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmittance $T$")
    ax.set_ylim(0, 1.05)
    despine(ax)
    ax.grid(lw=0.3, alpha=0.4)
    panel_label(ax, "(c)", dx=-0.16, dy=1.02)

    fig.savefig(os.path.join(FIG, "fig_sensor.pdf"))
    plt.close(fig)


# --------------------------------------------------- graphical abstract ----
def fig_abstract():
    import jax
    import jax.numpy as jnp
    from src import process, fabgan

    fig = plt.figure(figsize=(3.9, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.30], hspace=0.62,
                          wspace=0.50, left=0.115, right=0.97, top=0.90,
                          bottom=0.135)

    # top strip: pipeline
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [("process\ntraces", ORANGE), ("FabGAN\ntwin", RED),
             ("differentiable\nCVaR loop", BLUE),
             ("yield-qualified\nsensor", GREEN)]
    xw = 0.205
    for i, (txt, c) in enumerate(steps):
        x = 0.012 + i * 0.258
        ax.add_patch(FancyBboxPatch((x, 0.08), xw, 0.80,
                                    boxstyle="round,pad=0.014", fc=c,
                                    ec="none", alpha=0.13))
        ax.add_patch(FancyBboxPatch((x, 0.08), xw, 0.80,
                                    boxstyle="round,pad=0.014", fc="none",
                                    ec=c, lw=1.3))
        ax.text(x + xw / 2, 0.48, txt, ha="center", va="center",
                fontsize=6.8, color=c, fontweight="bold")
        if i < 3:
            arr(ax, x + xw + 0.013, 0.48, x + 0.252, 0.48, lw=1.4, mut=9)
    ax.text(0.5, 1.14, "learn the process $\\cdot$ differentiate the physics",
            ha="center", va="bottom", fontsize=7.5, style="italic")

    # bottom-left: error marginals (true vs twin), real data
    ax = fig.add_subplot(gs[1, 0])
    bench = np.load(os.path.join(ROOT, "data", "benchmark_designs.npz"))
    d, n = bench["d_um"][7], bench["n0"][7]
    rng = np.random.default_rng(3)
    Dt, _ = process.corrupt_ensemble(d, n, 250, rng)
    with open(os.path.join(RES, "fabgan_vanilla.pkl"), "rb") as f:
        gp = jax.tree.map(jnp.asarray, pickle.load(f))
    Dg, _ = fabgan.sample_fabgan(gp, jax.random.PRNGKey(1), d, n, 250)
    bins = np.linspace(-8, 14, 40)
    ax.hist((Dt / d - 1).ravel() * 100, bins=bins, density=True, alpha=0.60,
            color=RED, label="true process")
    ax.hist((Dg / d - 1).ravel() * 100, bins=bins, density=True, alpha=0.55,
            color=BLUE, label="learned twin")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.6)
    ax.set_xlabel("thickness error (%)", fontsize=6.3, labelpad=1)
    ax.set_ylabel("density (log)", fontsize=6.3, labelpad=1)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right",
              handlelength=1.1, borderaxespad=0.1)
    ax.set_title("GAN captures real error tails", fontsize=6.6, pad=3)
    ax.tick_params(labelsize=5.8)
    despine(ax)

    # bottom-right: yield CDFs, real data
    ax = fig.add_subplot(gs[1, 1])
    Jn = np.load(os.path.join(RES, "exp5_samples_nominal.npy"))
    Jf = np.load(os.path.join(RES, "exp5_samples_fabgan.npy"))
    for J, c, l in [(Jn, GRAY, "nominal"), (Jf, BLUE, "FabGAN-robust")]:
        xs = np.sort(J)
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=c, label=l, lw=1.5)
    ax.axhline(0.05, color="#333333", lw=0.5, ls=":")
    ax.annotate("+7.2%\nCVaR$_{5\\%}$", xy=(0.912, 0.05),
                xytext=(0.775, 0.40), fontsize=6.2, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax.set_xlim(0.75, 0.97)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("fabricated merit $J$", fontsize=6.3, labelpad=1)
    ax.set_ylabel("CDF", fontsize=6.3, labelpad=1)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left",
              handlelength=1.1, borderaxespad=0.1)
    ax.set_title("yield tail lifted (true process)", fontsize=6.6, pad=3)
    ax.tick_params(labelsize=5.8)
    despine(ax)

    fig.savefig(os.path.join(FIG, "fig_abstract.pdf"), bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"framework": fig_framework, "sensor": fig_sensor,
           "abstract": fig_abstract}
    for k, f in fns.items():
        if which in ("all", k):
            f()
            print("made", k)
