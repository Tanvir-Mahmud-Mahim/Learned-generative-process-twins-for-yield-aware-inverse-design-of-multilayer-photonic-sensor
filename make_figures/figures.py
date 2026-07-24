"""Publication data figures for the FabGAN-ID article (IEEE style).

Every figure is generated from the JSON/NPY results produced by the
experiments -- no hand-entered numbers.  Schematics live in fig_diagrams.py.
"""

import json
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (BLUE, CYAN, ORANGE, RED, GRAY, GREEN, W1, W2,
                   panel_label, despine, bar_values)

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

TWIN_COLORS = {"gauss_diag": GRAY, "gauss_full": ORANGE,
               "gan_vanilla": BLUE, "fabgan_tail": CYAN}
TWIN_LABELS = {"gauss_diag": "Gauss.\n(diag.)", "gauss_full": "Gauss.\n(full)",
               "gan_vanilla": "FabGAN\n(ours)", "fabgan_tail": "FabGAN-\nTC"}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ------------------------------------------------------------- inverse ----
def fig_inverse():
    r = load("exp4_inverse.json")
    fig, ax0 = plt.subplots(figsize=(W1, 1.90))
    ts = np.asarray(r["traj_seeded"])
    tr = np.asarray(r["traj_random"])
    ax0.plot(np.arange(1, len(ts) + 1), ts, color=BLUE, lw=1.6,
             label="probe-seeded adjoint (ours)")
    ax0.plot(np.arange(1, len(tr) + 1), tr, color=GRAY, ls="--", lw=1.3,
             label="random search")
    ax0.axvline(200, color="#333333", lw=0.6, ls=":")
    ax0.annotate("refinement\nstarts", xy=(200, 0.905), xytext=(148, 0.885),
                 fontsize=5.9, color="#333333", ha="right", va="center",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="#333333"))
    ax0.axhline(r["J_seeded"], color=BLUE, lw=0.5, ls=":")
    ax0.annotate(r"$J^\star = %.3f$" % r["J_seeded"],
                 xy=(320, r["J_seeded"]), xytext=(40, 0.985),
                 fontsize=7, color=BLUE, va="top",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))
    ax0.annotate("equal-budget gap:\n$%.3f$ vs. $%.3f$"
                 % (r["J_seeded"], r["J_random_search"]),
                 xy=(356, r["J_random_search"] + 0.004), xytext=(42, 0.745),
                 fontsize=6.2, color="#555555", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7,
                                 connectionstyle="arc3,rad=-0.15"))
    ax0.set_xlabel("solver queries", fontsize=7)
    ax0.set_ylabel("best-so-far merit $J$", fontsize=7)
    ax0.set_ylim(0.55, 1.02)
    ax0.tick_params(labelsize=6)
    ax0.legend(frameon=False, loc="lower right", fontsize=5.8)
    despine(ax0)
    ax0.grid(lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_inverse.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- twin ----
def fig_twin():
    r = load("exp3_fabgan.json")
    names = ["gauss_diag", "gauss_full", "gan_vanilla", "fabgan_tail"]
    labels = ["Gauss.\ndiag.", "Gauss.\nfull", "FabGAN\n(ours)", "TC\nvariant"]
    metrics = [("P5_abs_err_mean", r"$P_5$ estimation error"),
               ("CVaR5_abs_err_mean", r"$\mathrm{CVaR}_{5\%}$ estim. error")]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.92))
    fig.subplots_adjust(wspace=0.30)
    for a, (m, t), pl in zip(ax, metrics, ["(a)", "(b)"]):
        vals = [r[n][m] for n in names]
        bars = a.bar(range(4), vals,
                     color=[TWIN_COLORS[n] for n in names], width=0.66,
                     edgecolor="white", lw=0.5)
        bars[2].set_hatch("//")
        bars[2].set_edgecolor("white")
        # value labels above every bar
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v + 0.0004, "%.4f" % v,
                   ha="center", va="bottom", fontsize=5.6)
        # improvement callout above the FabGAN bar
        best_base = min(vals[0], vals[1])
        red = 100 * (1 - vals[2] / best_base)
        a.plot([1.45, 2.55], [best_base, best_base], color="#555555",
               lw=0.7, ls=(0, (3, 2)))
        a.annotate("$-%d\\%%$" % round(red), xy=(2, vals[2] + 0.0016),
                   xytext=(2, best_base + 0.0028), ha="center", fontsize=7.5,
                   color=BLUE, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
        a.set_xticks(range(4))
        a.set_xticklabels(labels, fontsize=5.9)
        a.set_title(t, fontsize=6.9, pad=3)
        a.set_ylim(0, max(vals) * 1.38)
        a.tick_params(labelsize=5.8)
        despine(a)
        a.grid(axis="y", lw=0.3, alpha=0.4)
        panel_label(a, pl, dx=-0.16, dy=1.17)
    ax[0].set_ylabel("mean absolute error", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_twin.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- lowdata ----
def fig_lowdata():
    r = load("exp3b_lowdata.json")
    Ms = [50, 100, 400]
    fig, ax = plt.subplots(figsize=(W1, 2.25))
    for tw, lab, c, mk in [(0, "FabGAN", BLUE, "o"),
                           (20, "FabGAN-TC (tail-calibrated)", CYAN, "s")]:
        v = [r["M%d_tail%d" % (M, tw)]["induced_perf_W1_mean"] for M in Ms]
        ax.plot(Ms, v, marker=mk, color=c, label=lab, ms=4.5, lw=1.5)
        for M, vv in zip(Ms, v):
            ax.annotate("%.4f" % vv, (M, vv), textcoords="offset points",
                        xytext=(0, 5), fontsize=5.8, ha="center", color=c)
    ax.axvspan(42, 66, color=CYAN, alpha=0.10)
    ax.text(44.5, 0.01345, "scarce-trace regime:\ntail calibration\npays off",
            fontsize=5.9, color="#2A7F9E", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(Ms); ax.set_xticklabels(Ms)
    ax.set_xlabel("historical process traces $M$")
    ax.set_ylabel(r"$W_1$(true, twin) of induced $J$")
    ax.legend(frameon=False, loc="upper right")
    despine(ax); ax.grid(lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_lowdata.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- yield ----
def fig_yield():
    names = ["nominal", "gauss_diag", "gauss_full", "fabgan"]
    labels = ["nominal $\\theta^\\star$", "Gauss. (diag.) twin",
              "Gauss. (full) twin", "FabGAN twin (ours)"]
    colors = ["#333333", GRAY, ORANGE, BLUE]
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.05))
    for nm, l, c in zip(names, labels, colors):
        J = np.load(os.path.join(RES, "exp5_samples_%s.npy" % nm))
        xs = np.sort(J)
        ax[0].plot(xs, np.linspace(0, 1, len(xs)), color=c, label=l,
                   lw=1.1 if nm == "nominal" else 1.6)
    ax[0].axhline(0.05, color="#333333", lw=0.5, ls=":")
    ax[0].text(0.772, 0.065, r"$5\%$ tail", fontsize=6.2)
    ax[0].set_xlabel("merit $J$ under the hidden true process", fontsize=7)
    ax[0].set_ylabel("empirical CDF", fontsize=7)
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    despine(ax[0]); ax[0].grid(lw=0.3, alpha=0.4)
    panel_label(ax[0], "(a)")
    # inset: zoom on the lower tail
    axi = ax[0].inset_axes([0.58, 0.08, 0.40, 0.42])
    axi.set_facecolor('white')
    axi.patch.set_alpha(1.0)
    for nm, c in zip(names, colors):
        J = np.load(os.path.join(RES, "exp5_samples_%s.npy" % nm))
        xs = np.sort(J)
        axi.plot(xs, np.linspace(0, 1, len(xs)), color=c, lw=1.2)
    axi.set_xlim(0.84, 0.95); axi.set_ylim(0, 0.08)
    axi.tick_params(labelsize=5)
    axi.set_title("lower tail", fontsize=5.5, pad=1)
    axi.grid(lw=0.2, alpha=0.4)

    r = load("exp5_yield.json")
    show = ["gauss_diag", "gauss_full", "fabgan", "fabgan_meanvar"]
    lbl = ["Gauss.\n(diag.)\nCVaR", "Gauss.\n(full)\nCVaR",
           "FabGAN\nCVaR\n(ours)", "FabGAN\n$\\mu-\\sigma$"]
    cols = [GRAY, ORANGE, BLUE, CYAN]
    x = np.arange(len(show)); wdt = 0.38
    base = r["nominal"]
    p5 = [(r[s]["P5"] - base["P5"]) / base["P5"] * 100 for s in show]
    cv = [(r[s]["CVaR5"] - base["CVaR5"]) / base["CVaR5"] * 100 for s in show]
    b1 = ax[1].bar(x - wdt / 2, p5, wdt, label="$P_5$ floor",
                   color=cols, alpha=0.55, edgecolor="white")
    b2 = ax[1].bar(x + wdt / 2, cv, wdt, label=r"$\mathrm{CVaR}_{5\%}$",
                   color=cols, edgecolor="white")
    bar_values(ax[1], b1, fmt="%.1f", dy=0.05, fontsize=6)
    bar_values(ax[1], b2, fmt="%.1f", dy=0.05, fontsize=6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=5.6)
    ax[1].set_ylabel("gain over nominal (%)", fontsize=7)
    ax[1].legend(frameon=False, fontsize=5.8, loc="upper left")
    despine(ax[1]); ax[1].grid(axis="y", lw=0.3, alpha=0.4)
    panel_label(ax[1], "(b)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_yield.pdf"))
    plt.close(fig)


# -------------------------------------------------------------- policy ----
def fig_policy():
    rb = load("exp6b_apg.json")
    rc = load("exp6c_dense.json")
    with open(os.path.join(RES, "apg_log.pkl"), "rb") as f:
        apg_log = np.asarray(pickle.load(f))
    sac = load("exp6_policy_sac.json")
    sac_log = np.asarray(sac["train_log"])
    specs = [k for k in rb if k != "train_log"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].plot(apg_log[:, 0], apg_log[:, 1], color=BLUE, lw=1.6,
               label="APG through differentiable loop (ours)")
    ax2 = ax[0].twiny()
    ax2.plot(sac_log[:, 0], sac_log[:, 1], "s--", color=RED, ms=3.5, lw=1.2,
             label="model-free SAC")
    ax2.set_xlabel("SAC environment steps", fontsize=7, color=RED)
    ax2.tick_params(colors=RED, labelsize=6.5)
    ax2.spines["top"].set_color(RED)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5,
                 loc="center right")
    ax[0].annotate("converges", xy=(300, 0.878), xytext=(180, 0.80),
                   fontsize=6.5, color=BLUE,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax[0].annotate("diverges", xy=(330, 0.665), xytext=(230, 0.72),
                   fontsize=6.5, color=RED,
                   arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax[0].set_xlabel("APG gradient steps")
    ax[0].set_ylabel("mean CVaR reward (train specs)")
    despine(ax[0]); ax[0].grid(lw=0.3, alpha=0.4)
    panel_label(ax[0], "(a)")

    x = np.arange(len(specs)); wdt = 0.2
    series = [
        ([100 * (rb[s]["recovery_static"] or 0) for s in specs], "static\ncorrection", GRAY),
        ([100 * (rb[s]["recovery_sac"] or 0) for s in specs], "SAC policy", RED),
        ([100 * (rb[s]["recovery_apg"] or 0) for s in specs], "APG (5 specs)", CYAN),
        ([100 * (rc[s]["recovery_apg_dense"] or 0) for s in specs], "APG (11 specs)", BLUE)]
    for i, (v, l, c) in enumerate(series):
        ax[1].bar(x + (i - 1.5) * wdt, v, wdt, label=l, color=c,
                  edgecolor="white", lw=0.4)
    ax[1].axhline(0, color="#333333", lw=0.7)
    ax[1].axhline(100, color=GREEN, lw=0.8, ls=":")
    ax[1].text(2.35, 104, "per-spec re-opt. = 100%", fontsize=5.8,
               color=GREEN, ha="right")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["%d nm" % (float(s) * 1000) for s in specs])
    ax[1].set_ylabel(r"CVaR$_{5\%}$ gain recovery (%)")
    ax[1].set_ylim(-620, 160)
    ax[1].legend(frameon=False, fontsize=6, ncol=4, loc="upper center",
                 bbox_to_anchor=(0.5, -0.14), columnspacing=0.9,
                 handlelength=1.2)
    despine(ax[1]); ax[1].grid(axis="y", lw=0.3, alpha=0.4)
    panel_label(ax[1], "(b)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_policy.pdf"))
    plt.close(fig)


# ----------------------------------------------------------- surrogate ----
def fig_surrogate():
    r = load("exp7_surrogate.json")
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    protos = ["row", "comb"]
    labels = ["row split\n(interpolation)", "held-out design\n(generalization)"]
    x = np.arange(2); wdt = 0.36
    dat = [r["data_%s" % p]["r2"] for p in protos]
    phy = [r["phys_%s" % p]["r2"] for p in protos]
    b1 = ax.bar(x - wdt / 2, dat, wdt, label="data-only surrogate",
                color=GRAY, edgecolor="white")
    b2 = ax.bar(x + wdt / 2, phy, wdt, label="recursion-informed surrogate",
                color=ORANGE, edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2,
                    h + (0.04 if h > 0 else -0.10), "%.2f" % h,
                    ha="center", fontsize=6.5)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.text(0.42, -0.22, "worse than\npredicting\nthe mean", fontsize=6.0,
            ha="center", va="top", style="italic", color="#444444")
    ax.axhline(1.0, color=BLUE, lw=0.9, ls=":")
    ax.text(0.02, 1.04, "exact differentiable solver: $R^2 = 1$ and "
            "%.0f$\\times$ faster" % (1 / r["speedup_surrogate_vs_tmm"]),
            fontsize=6.3, color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(-1.15, 1.30)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    despine(ax); ax.grid(axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_surrogate.pdf"))
    plt.close(fig)


# -------------------------------------------------------------- process ----
def fig_process():
    import jax
    import jax.numpy as jnp
    from src import process, fabgan
    bench = np.load(os.path.join(ROOT, "data", "benchmark_designs.npz"))
    d, n = bench["d_um"][7], bench["n0"][7]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.05))
    xs = np.arange(1, len(d) + 1)
    for i in range(6):
        _, nt = process.corrupt(d, n, rng)
        ax[0].step(xs, nt, where="mid", color=RED, alpha=0.35, lw=0.7,
                   label="true process (6 runs)" if i == 0 else None)
    with open(os.path.join(RES, "fabgan_vanilla.pkl"), "rb") as fh:
        gp = jax.tree.map(jnp.asarray, pickle.load(fh))
    Dg, Ng = fabgan.sample_fabgan(gp, jax.random.PRNGKey(1), d, n, 6)
    for i in range(6):
        ax[0].step(xs, Ng[i], where="mid", color=BLUE, alpha=0.35, lw=0.7,
                   label="FabGAN twin (6 samples)" if i == 0 else None)
    ax[0].step(xs, n, where="mid", color="#222222", lw=1.5, label="recipe")
    ax[0].set_xlabel("layer index $i$", fontsize=7, labelpad=1)
    ax[0].set_ylabel(r"layer index $n_i(\lambda_0)$", fontsize=7)
    ax[0].set_ylim(1.55, 2.62)
    ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[0].legend(frameon=False, fontsize=5.6, loc="upper right", ncol=1,
                 handlelength=1.4, borderaxespad=0.2)
    ax[0].tick_params(labelsize=6)
    despine(ax[0]); ax[0].grid(lw=0.3, alpha=0.4)
    panel_label(ax[0], "(a)", dx=-0.16, dy=1.09)

    K = 300
    Dt, Nt = process.corrupt_ensemble(d, n, K, rng)
    Dg, Ng = fabgan.sample_fabgan(gp, jax.random.PRNGKey(2), d, n, K)
    et = (Dt / d - 1).ravel() * 100
    eg = (Dg / d - 1).ravel() * 100
    bins = np.linspace(-8, 15, 55)
    ax[1].hist(et, bins=bins, density=True, alpha=0.55, color=RED,
               label="true process")
    ax[1].hist(eg, bins=bins, density=True, alpha=0.55, color=BLUE,
               label="FabGAN twin")
    ax[1].annotate("particulate\nheavy tail", xy=(11.0, 2.5e-3),
                   xytext=(11.2, 1.3e-1), fontsize=5.8, ha="center",
                   arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[1].annotate("systematic bias\n+ right skew", xy=(2.4, 1.5e-1),
                   xytext=(-6.6, 3.5e-1), fontsize=5.8, ha="center",
                   arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[1].set_xlabel("thickness error (%)", fontsize=7, labelpad=1)
    ax[1].set_ylabel("density (log)", fontsize=7)
    ax[1].set_yscale("log")
    ax[1].set_ylim(2e-4, 3.2)
    ax[1].legend(frameon=False, fontsize=5.8, loc="center right")
    ax[1].tick_params(labelsize=6)
    despine(ax[1]); ax[1].grid(lw=0.3, alpha=0.4)
    panel_label(ax[1], "(b)", dx=-0.16, dy=1.09)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(os.path.join(FIG, "fig_process.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"inverse": fig_inverse, "twin": fig_twin, "lowdata": fig_lowdata,
           "yield": fig_yield, "policy": fig_policy,
           "surrogate": fig_surrogate, "process": fig_process}
    for k, f in fns.items():
        if which in ("all", k):
            try:
                f()
                print("made", k)
            except Exception as e:
                print("SKIP", k, repr(e))
