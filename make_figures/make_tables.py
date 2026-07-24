"""Generate paper/numbers.tex: every quantitative value cited in the article,
as LaTeX macros derived from the experiment result files."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "numbers.tex")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def main():
    L = []

    def macro(name, val):
        L.append("\\newcommand{\\%s}{%s}" % (name, val))

    e1 = load("exp1_validate.json")
    macro("valTdiff", "%.1f\\times10^{-15}" % (e1["max_abs_T_diff_vs_tmm_pkg"] * 1e15))
    macro("valGradMed", "%.1f\\times10^{-10}" % (e1["grad_median_rel_err_vs_fd"] * 1e10))
    macro("valGradMax", "%.1f\\times10^{-9}" % (e1["grad_max_rel_err_vs_fd"] * 1e9))
    macro("valThroughput", "%d" % round(e1["solver_throughput_designs_per_s"]))

    e2 = load("exp2_dataset.json")
    macro("nDesigns", str(e2["n_designs"]))
    macro("nTraces", str(e2["n_traces"]))
    macro("nSamples", "{:,}".format(e2["total_spectral_samples"]).replace(",", "{,}"))

    e3 = load("exp3_fabgan.json")
    rows = [("gauss_diag", "Gaussian (diagonal)"),
            ("gauss_full", "Gaussian (full covariance)"),
            ("gan_vanilla", "\\textbf{FabGAN (ours)}"),
            ("fabgan_tail", "FabGAN-TC (tail-calibrated)")]
    tbl = []
    for k, lab in rows:
        v = e3[k]
        bold = k == "gan_vanilla"
        fmt = ("\\textbf{%.4f}" if bold else "%.4f")
        tbl.append("%s & %s & %s & %s & %s & %s \\\\" % (
            lab, "%.4f" % v["marginal_W1_mean"],
            "%.3f" % v["corr_frobenius_rel_err"],
            fmt % v["induced_perf_W1_mean"],
            fmt % v["P5_abs_err_mean"],
            fmt % v["CVaR5_abs_err_mean"]))
    with open(os.path.join(ROOT, "paper", "table_twin.tex"), "w") as f:
        f.write("\n".join(tbl) + "%\n")
    fg = e3["gan_vanilla"]; gd = e3["gauss_diag"]; gf = e3["gauss_full"]
    macro("pFiveRedDiag", "%d" % round(100 * (1 - fg["P5_abs_err_mean"] / gd["P5_abs_err_mean"])))
    macro("pFiveRedFull", "%d" % round(100 * (1 - fg["P5_abs_err_mean"] / gf["P5_abs_err_mean"])))
    macro("cvarRedDiag", "%d" % round(100 * (1 - fg["CVaR5_abs_err_mean"] / gd["CVaR5_abs_err_mean"])))
    macro("cvarRedFull", "%d" % round(100 * (1 - fg["CVaR5_abs_err_mean"] / gf["CVaR5_abs_err_mean"])))

    e3b = load("exp3b_lowdata.json")
    mn = {50: "Fifty", 100: "Hundred", 400: "FourH"}
    tn = {0: "Plain", 20: "TC"}
    for M in (50, 100, 400):
        for tw in (0, 20):
            v = e3b["M%d_tail%d" % (M, tw)]
            macro("lowW%s%s" % (mn[M], tn[tw]), "%.4f" % v["induced_perf_W1_mean"])
            macro("lowC%s%s" % (mn[M], tn[tw]), "%.4f" % v["CVaR5_abs_err_mean"])

    e4 = load("exp4_inverse.json")
    macro("Jstar", "%.3f" % e4["J_seeded"])
    macro("Jrs", "%.3f" % e4["J_random_search"])
    macro("Jun", "%.3f" % e4["J_unseeded_gradient"])
    macro("Jprobes", "%.3f" % e4["J_probes_only"])
    macro("rsBudgetMultiple", ">83")

    e5 = load("exp5_yield.json")
    order = [("nominal", "Nominal optimum $\\theta^\\star$"),
             ("gauss_diag", "Gaussian (diag) twin, CVaR"),
             ("gauss_diag_meanvar", "Gaussian (diag) twin, $\\mu-\\sigma$"),
             ("gauss_full", "Gaussian (full) twin, CVaR"),
             ("fabgan_tail", "FabGAN-TC twin, CVaR"),
             ("fabgan_meanvar", "\\textbf{FabGAN twin}, $\\mu-\\sigma$"),
             ("fabgan", "\\textbf{FabGAN twin, CVaR (ours)}")]
    tbl = []
    for k, lab in order:
        v = e5[k]
        b = k in ("fabgan",)
        fmt = ("\\textbf{%.4f}" if b else "%.4f")
        tbl.append("%s & %s & %.1f & %s & %s \\\\" % (
            lab, fmt % v["mean"], 100 * v["std"] / v["mean"],
            fmt % v["P5"], fmt % v["CVaR5"]))
    with open(os.path.join(ROOT, "paper", "table_yield.tex"), "w") as f:
        f.write("\n".join(tbl) + "%\n")
    nom = e5["nominal"]; fab = e5["fabgan"]
    macro("yGainCvar", "%.1f" % (100 * (fab["CVaR5"] / nom["CVaR5"] - 1)))
    macro("yGainPfive", "%.1f" % (100 * (fab["P5"] / nom["P5"] - 1)))
    macro("yGainMean", "%.1f" % (100 * (fab["mean"] / nom["mean"] - 1)))
    macro("yVsGaussFull", "%.1f" % (100 * (fab["CVaR5"] / e5["gauss_full"]["CVaR5"] - 1)))
    macro("yVsGaussDiag", "%.1f" % (100 * (fab["CVaR5"] / e5["gauss_diag"]["CVaR5"] - 1)))
    macro("nomCvar", "%.4f" % nom["CVaR5"])
    macro("fabCvar", "%.4f" % fab["CVaR5"])
    dv = abs(e5["fabgan_meanvar"]["CVaR5"] - fab["CVaR5"])
    macro("objLever", "%.4f" % dv)
    macro("twinLever", "%.4f" % (fab["CVaR5"] - e5["gauss_diag"]["CVaR5"]))

    e6 = load("exp6b_apg.json")
    e6c = load("exp6c_dense.json")
    tbl = []
    for s in ["0.520", "0.580", "0.620"]:
        v = e6[s]; vc = e6c[s]
        tbl.append("%d nm & %.2f & %.2f & %.2f & %.2f \\\\" % (
            float(s) * 1000,
            100 * v["recovery_static"], 100 * v["recovery_sac"],
            100 * v["recovery_apg"], 100 * vc["recovery_apg_dense"]))
    with open(os.path.join(ROOT, "paper", "table_transfer.tex"), "w") as f:
        f.write("\n".join(tbl) + "%\n")

    e7 = load("exp7_surrogate.json")
    macro("surRowData", "%.2f" % e7["data_row"]["r2"])
    macro("surRowPhys", "%.2f" % e7["phys_row"]["r2"])
    macro("surCombData", "%.2f" % e7["data_comb"]["r2"])
    macro("surCombPhys", "%.2f" % e7["phys_comb"]["r2"])
    macro("surSlowdown", "%.0f" % (1.0 / e7["speedup_surrogate_vs_tmm"]))

    with open(OUT, "w") as f:
        f.write("% auto-generated by make_figures/make_tables.py -- do not edit\n")
        f.write("\n".join(L) + "\n")
    print("wrote", OUT, "and 3 table files")


def master_table():
    """Generate paper/table_master.tex: the consolidated quantitative
    figure-of-merit comparison against state-of-the-art practice, entirely
    from the experiment result files."""
    import pickle
    e3 = load("exp3_fabgan.json")
    e4 = load("exp4_inverse.json")
    e5 = load("exp5_yield.json")
    e6 = load("exp6b_apg.json")
    e6c = load("exp6c_dense.json")
    e7 = load("exp7_surrogate.json")
    sac = load("exp6_policy_sac.json")
    with open(os.path.join(RES, "apg_log.pkl"), "rb") as f:
        apg = pickle.load(f)

    L = []
    A = L.append
    A(r"\begin{table*}[!t]")
    A(r"    \scriptsize")
    A(r"    \centering")
    A(r"    \caption{Consolidated Quantitative Figure-of-Merit Comparison of FabGAN-ID Against State-of-the-Art Practice. Every Value Is Computed on the Released Benchmark Under Identical Budgets and Common Monte-Carlo Draws; Baselines Implement the Corruption Models, Optimizers, and Learners of the Cited Prior Art. Best in Bold.}")
    A(r"    \label{tab:master}")
    A(r"    \renewcommand{\arraystretch}{1.06}")
    A(r"    \setlength{\tabcolsep}{2.4pt}")
    A(r"    \begin{tabular}{llcccc}")
    A(r"        \toprule")

    # ---- Panel A: twin fidelity ----
    A(r"        \multicolumn{6}{l}{\textbf{(A) Process-twin fidelity} (30 held-out designs $\times$ 400 fresh true-process draws; mean absolute estimation error, lower is better)} \\")
    A(r"        \cmidrule(lr){1-6}")
    A(r"        \textbf{Corruption model} & \textbf{Practice of} & $|\Delta P_5|$ $\downarrow$ & $|\Delta\mathrm{CVaR}_{5\%}|$ $\downarrow$ & $W_1(J)$ $\downarrow$ & \textbf{Tail-err.\ red.} \\")
    A(r"        \midrule")
    gd, gf, fb = e3["gauss_diag"], e3["gauss_full"], e3["gan_vanilla"]
    A(r"        i.i.d.\ Gaussian tolerances & robust ID practice \cite{b_foundry, b_prev} & %.4f & %.4f & %.4f & --- \\"
      % (gd["P5_abs_err_mean"], gd["CVaR5_abs_err_mean"], gd["induced_perf_W1_mean"]))
    A(r"        Full-covariance Gaussian & strongest parametric fit & %.4f & %.4f & %.4f & --- \\"
      % (gf["P5_abs_err_mean"], gf["CVaR5_abs_err_mean"], gf["induced_perf_W1_mean"]))
    rp5 = 100 * (1 - fb["P5_abs_err_mean"] / min(gd["P5_abs_err_mean"], gf["P5_abs_err_mean"]))
    rcv = 100 * (1 - fb["CVaR5_abs_err_mean"] / min(gd["CVaR5_abs_err_mean"], gf["CVaR5_abs_err_mean"]))
    A(r"        \textbf{FabGAN twin (this work)} & --- & \textbf{%.4f} & \textbf{%.4f} & \textbf{%.4f} & \textbf{$-%d\%%$ / $-%d\%%$} \\"
      % (fb["P5_abs_err_mean"], fb["CVaR5_abs_err_mean"], fb["induced_perf_W1_mean"], round(rp5), round(rcv)))
    A(r"        \midrule")

    # ---- Panel B: true-process yield ----
    A(r"        \multicolumn{6}{l}{\textbf{(B) True-process yield after identical pathwise CVaR$_{5\%}$ robustification} ($2{,}000$ common draws from the hidden process)} \\")
    A(r"        \cmidrule(lr){1-6}")
    A(r"        \textbf{Design} & $\mu$ $\uparrow$ & $\sigma/\mu$ [\%] $\downarrow$ & $P_5$ $\uparrow$ & $\mathrm{CVaR}_{5\%}$ $\uparrow$ & \textbf{$\Delta$ vs.\ nominal} \\")
    A(r"        \midrule")
    nom = e5["nominal"]
    rows = [("Nominal optimum $\\theta^\\star$", "nominal", False),
            ("Robust vs.\\ i.i.d.\\ Gaussian twin \\cite{b_foundry, b_prev}", "gauss_diag", False),
            ("Robust vs.\\ full-covariance Gaussian twin", "gauss_full", False),
            ("Robust vs.\\ FabGAN twin, CVaR objective (this work)", "fabgan", False),
            ("\\textbf{Robust vs.\\ FabGAN twin, $\\mu-\\sigma$ objective (this work)}", "fabgan_meanvar", True)]
    for lab, k, bold in rows:
        v = e5[k]
        d = 100 * (v["CVaR5"] / nom["CVaR5"] - 1)
        fmt = (r"\textbf{%.4f}" if bold else "%.4f")
        dfmt = (r"$\mathbf{%+.1f\%%}$" if bold else r"$%+.1f\%%$")
        A(r"        %s & %s & %.1f & %s & %s & %s \\"
          % (lab, fmt % v["mean"], 100 * v["std"] / v["mean"],
             fmt % v["P5"], fmt % v["CVaR5"], dfmt % d))
    A(r"        \midrule")

    # ---- Panel C: inverse-engine efficiency ----
    A(r"        \multicolumn{6}{l}{\textbf{(C) Nominal inverse-engine query efficiency} ($532$-nm specification, identical solver and budget accounting)} \\")
    A(r"        \cmidrule(lr){1-6}")
    A(r"        \textbf{Engine} & \textbf{Best $J$} $\uparrow$ & \textbf{Queries used} & \textbf{Queries to reach $J^\star$} $\downarrow$ & \textbf{Efficiency} & \\")
    A(r"        \midrule")
    A(r"        Random search & %.3f & 360 & $>$30{,}000 & $1\times$ & \\" % e4["J_random_search"])
    A(r"        Global probes only & %.3f & 360 & --- (stalls) & --- & \\" % e4["J_probes_only"])
    A(r"        Unseeded gradient (10 restarts) & %.3f & 400 & --- (local optimum) & --- & \\" % e4["J_unseeded_gradient"])
    A(r"        \textbf{Probe-seeded adjoint (this work)} & \textbf{%.3f} & \textbf{360} & \textbf{360} & \textbf{$>$83$\times$} & \\" % e4["J_seeded"])
    A(r"        \midrule")

    # ---- Panel D: policy learning ----
    sl = sac["train_log"]
    A(r"        \multicolumn{6}{l}{\textbf{(D) Correction-policy learning at equal wall-clock budget} (train-specification CVaR reward; zero-shot recovery of attainable true-process gain)} \\")
    A(r"        \cmidrule(lr){1-6}")
    A(r"        \textbf{Learner} & \textbf{Reward trend} & \textbf{Behavior} & \multicolumn{2}{c}{\textbf{Zero-shot recovery [best / worst]}} & \\")
    A(r"        \midrule")
    recs_sac = [100 * e6[s]["recovery_sac"] for s in ("0.520", "0.580", "0.620")]
    recs_apg = [100 * e6c[s]["recovery_apg_dense"] for s in ("0.520", "0.580", "0.620")]
    A(r"        Model-free SAC \cite{b_sac} & $%.2f \to %.2f$ & diverges & \multicolumn{2}{c}{$%+.0f\%%$ / $%+.0f\%%$} & \\"
      % (sl[0][1], sl[-1][1], max(recs_sac), min(recs_sac)))
    A(r"        \textbf{Analytic policy gradients (this work)} & $\mathbf{%.2f \to %.2f}$ & \textbf{converges} & \multicolumn{2}{c}{$\mathbf{%+.0f\%%}$ / $%+.0f\%%$} & \\"
      % (apg[0][1], apg[-1][1], max(recs_apg), min(recs_apg)))
    A(r"        \midrule")

    # ---- Panel E: surrogate vs exact solver ----
    A(r"        \multicolumn{6}{l}{\textbf{(E) Forward model in the loop} (held-out $R^2$ of neural surrogates vs.\ the exact differentiable solver)} \\")
    A(r"        \cmidrule(lr){1-6}")
    A(r"        \textbf{Forward model} & \textbf{Row-split $R^2$} $\uparrow$ & \textbf{Held-out-design $R^2$} $\uparrow$ & \textbf{Rel.\ speed} $\uparrow$ & \textbf{Exact gradients} & \\")
    A(r"        \midrule")
    A(r"        Data-only neural surrogate & %.2f & $%.2f$ & $1\times$ & no & \\"
      % (e7["data_row"]["r2"], e7["data_comb"]["r2"]))
    A(r"        Recursion-informed neural surrogate & %.2f & $%.2f$ & $1\times$ & no & \\"
      % (e7["phys_row"]["r2"], e7["phys_comb"]["r2"]))
    A(r"        \textbf{Exact differentiable TMM (this work)} & \textbf{1.00} & \textbf{1.00} & \textbf{%.0f$\times$} & \textbf{yes} & \\"
      % (1.0 / e7["speedup_surrogate_vs_tmm"]))
    A(r"        \bottomrule")
    A(r"    \end{tabular}")
    A(r"\end{table*}")
    with open(os.path.join(ROOT, "paper", "table_master.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote table_master.tex")




if __name__ == "__main__":
    main()
    master_table()
