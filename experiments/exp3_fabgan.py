"""E3: Process-twin fidelity study (chunked / resumable).

Trains: (i) i.i.d. diagonal Gaussian, (ii) full-covariance Gaussian,
(iii) vanilla conditional WGAN-GP, (iv) tail-calibrated FabGAN (ours),
all on the SAME 400 historical traces.

Fidelity metrics on held-out designs (never in the trace set), against 400
fresh true-process draws per design:
  - mean per-dimension Wasserstein-1 distance of error marginals,
  - Frobenius error of the error correlation matrix,
  - induced-performance distribution: W1(J_true, J_twin), and absolute errors
    of the P5 floor and CVaR_5% estimates -- the yield-decisive statistics.

Run repeatedly until it prints DONE (each invocation runs a bounded slice).
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, process, fabgan
from experiments import common

GAN_STEPS = 3000
SLICE_S = 30
STATE = "exp3_state.pkl"


def yield_stats(J):
    Js = np.sort(J)
    q = max(1, int(np.ceil(0.05 * len(J))))
    return float(np.percentile(J, 5)), float(Js[:q].mean())


def load_inputs():
    dat = np.load(os.path.join(common.DATA, "process_traces.npz"))
    rd, rn, fd, fn = dat["recipe_d"], dat["recipe_n"], dat["fab_d"], dat["fab_n"]
    X = fabgan.errors_from_traces(rd, rn, fd, fn)
    C = np.asarray(fabgan.norm_recipe(jnp.asarray(rd), jnp.asarray(rn)))
    stop, pas = tmm_jax.band_masks(common.CENTER)
    rng = np.random.default_rng(common.SEED + 1)
    cal_idx = rng.choice(rd.shape[0], 24, replace=False)
    Jall = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(fd),
                                                jnp.asarray(fn), stop, pas))
    Jnom = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(rd),
                                                jnp.asarray(rn), stop, pas))
    dJ = Jall - Jnom            # per-trace performance degradation
    jr_low = np.sort(dJ)[:24]   # pooled 6% lower tail of degradation
    cal_jnom = Jnom[cal_idx]
    return (rd, rn, fd, fn, X, C, stop, pas, rd[cal_idx], rn[cal_idx],
            cal_jnom, jr_low)


def main():
    t_wall = time.time()
    st = {}
    if os.path.exists(os.path.join(common.RESULTS, STATE)):
        st = common.load_pickle(STATE)
    phase = st.get("phase", "gan_v")
    (rd, rn, fd, fn, X, C, stop, pas, cal_d, cal_n, cal_jnom,
     jr_low) = load_inputs()

    if phase in ("gan_v", "gan_t"):
        tail_w = 0.0 if phase == "gan_v" else 20.0
        seed = common.SEED if phase == "gan_v" else common.SEED + 7
        resume = st.get("trainer")
        gp, dp, out = fabgan.train_fabgan(
            jax.random.PRNGKey(seed), C, X, cal_d, cal_n, cal_jnom, jr_low,
            stop, pas, steps=GAN_STEPS, tail_w=tail_w, resume=resume,
            max_seconds=SLICE_S)
        if not out["done"]:
            st["phase"] = phase
            st["trainer"] = out
            common.save_pickle(STATE, st)
            print("PROGRESS %s step %d/%d" % (phase, out["t"], GAN_STEPS))
            return
        name = "fabgan_vanilla.pkl" if phase == "gan_v" else "fabgan_tail.pkl"
        common.save_pickle(name, jax.tree.map(np.asarray, gp))
        common.save_pickle(name.replace(".pkl", "_hist.pkl"), out["hist"])
        st = {"phase": "gan_t" if phase == "gan_v" else "metrics"}
        common.save_pickle(STATE, st)
        print("PROGRESS finished %s" % phase)
        return

    # ---------------- metrics phase (chunked over held-out designs) --------
    twins = {"gauss_diag": fabgan.GaussianTwin(X, diagonal=True),
             "gauss_full": fabgan.GaussianTwin(X, diagonal=False)}
    common.save_pickle("gauss_twins.pkl", twins)
    gp_v = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_vanilla.pkl"))
    gp_t = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_tail.pkl"))

    bench = np.load(os.path.join(common.DATA, "benchmark_designs.npz"))
    used = set(map(tuple, np.round(rd[::common.RUNS_PER_RECIPE], 9)))
    held = [i for i in range(bench["d_um"].shape[0])
            if tuple(np.round(bench["d_um"][i], 9)) not in used][:30]
    Hd, Hn = bench["d_um"][held], bench["n0"][held]

    names = ["gauss_diag", "gauss_full", "gan_vanilla", "fabgan_tail"]
    acc = st.get("acc")
    if acc is None:
        acc = {n: {"w1s": [], "w1_perf": [], "p5": [], "cv": [],
                   "Xt": [], "Xg": []} for n in names}
    i0 = st.get("design_i", 0)
    rng = np.random.default_rng(common.SEED + 100 + i0)
    key = jax.random.PRNGKey(common.SEED + 200 + i0)
    K = 400

    for i in range(i0, len(held)):
        d, n = Hd[i], Hn[i]
        Dt, Nt = process.corrupt_ensemble(d, n, K, rng)
        Jt = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(Dt),
                                                  jnp.asarray(Nt), stop, pas))
        Xt = np.concatenate([Dt / d - 1, Nt - n], axis=1)
        for name in names:
            key, k = jax.random.split(key)
            if name == "gan_vanilla":
                Dg, Ng = fabgan.sample_fabgan(gp_v, k, d, n, K)
            elif name == "fabgan_tail":
                Dg, Ng = fabgan.sample_fabgan(gp_t, k, d, n, K)
            else:
                Dg, Ng = twins[name].sample(rng, d, n, K)
            Dg = np.clip(Dg, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
            Ng = np.clip(Ng, 1.40, 2.60)
            Jg = np.asarray(tmm_jax.notch_merit_batch(
                jnp.asarray(Dg), jnp.asarray(Ng), stop, pas))
            a = acc[name]
            a["w1_perf"].append(wasserstein_distance(Jt, Jg))
            p5t, cvt = yield_stats(Jt)
            p5g, cvg = yield_stats(Jg)
            a["p5"].append(abs(p5t - p5g))
            a["cv"].append(abs(cvt - cvg))
            Xg = np.concatenate([Dg / d - 1, Ng - n], axis=1)
            for j in range(Xt.shape[1]):
                a["w1s"].append(wasserstein_distance(Xt[:, j], Xg[:, j]))
            a["Xt"].append(Xt)
            a["Xg"].append(Xg)
        if time.time() - t_wall > SLICE_S and i + 1 < len(held):
            common.save_pickle(STATE, {"phase": "metrics", "design_i": i + 1,
                                       "acc": acc})
            print("PROGRESS metrics design %d/%d" % (i + 1, len(held)))
            return

    res = {}
    for name in names:
        a = acc[name]
        Xt_all = np.vstack(a["Xt"])
        Xg_all = np.vstack(a["Xg"])
        ct = np.corrcoef(Xt_all.T)
        cg = np.corrcoef(Xg_all.T)
        res[name] = {
            "marginal_W1_mean": float(np.mean(a["w1s"])),
            "corr_frobenius_rel_err": float(np.linalg.norm(ct - cg)
                                            / np.linalg.norm(ct)),
            "induced_perf_W1_mean": float(np.mean(a["w1_perf"])),
            "P5_abs_err_mean": float(np.mean(a["p5"])),
            "CVaR5_abs_err_mean": float(np.mean(a["cv"])),
        }
        print(name, res[name])
    res["_held_out_design_indices"] = [int(h) for h in held]
    common.save_json("exp3_fabgan.json", res)
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE")


if __name__ == "__main__":
    main()
