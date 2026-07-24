"""E3b: Trace-scarcity study -- does physics-in-the-loop tail calibration
compensate for limited historical process data?

Trains the vanilla conditional WGAN-GP and the tail-calibrated FabGAN on
M in {50, 100, 400} historical traces (same architecture, same steps), and
evaluates tail fidelity (P5 / CVaR5 absolute error of the induced performance
distribution) on 15 held-out designs x 400 true-process draws.
Chunked/resumable: run until DONE.
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

GAN_STEPS = 2000
SLICE_S = 30
STATE = "exp3b_state.pkl"
SIZES = [50, 100, 400]
TAILS = [0.0, 20.0]


def yield_stats(J):
    Js = np.sort(J)
    q = max(1, int(np.ceil(0.05 * len(J))))
    return float(np.percentile(J, 5)), float(Js[:q].mean())


def main():
    t_wall = time.time()
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {"job": 0, "res": {}}

    dat = np.load(os.path.join(common.DATA, "process_traces.npz"))
    rd, rn, fd, fn = dat["recipe_d"], dat["recipe_n"], dat["fab_d"], dat["fab_n"]
    stop, pas = tmm_jax.band_masks(common.CENTER)

    jobs = [(M, tw) for M in SIZES for tw in TAILS]
    while st["job"] < len(jobs):
        M, tw = jobs[st["job"]]
        tag = "M%d_tail%d" % (M, int(tw))
        rng = np.random.default_rng(common.SEED + 31)
        sub = rng.choice(rd.shape[0], M, replace=False)
        rds, rns, fds, fns = rd[sub], rn[sub], fd[sub], fn[sub]
        X = fabgan.errors_from_traces(rds, rns, fds, fns)
        C = np.asarray(fabgan.norm_recipe(jnp.asarray(rds), jnp.asarray(rns)))
        ncal = min(24, M)
        cal_idx = rng.choice(M, ncal, replace=False)
        Jall = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(fds),
                                                    jnp.asarray(fns), stop, pas))
        Jnom = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(rds),
                                                    jnp.asarray(rns), stop, pas))
        dJ = Jall - Jnom
        qn = max(3, int(np.ceil(0.06 * M)))
        jr_low = np.sort(dJ)[:qn]
        resume = st.get("trainer")
        gp, dp, out = fabgan.train_fabgan(
            jax.random.PRNGKey(common.SEED + 900 + st["job"]), C, X,
            rds[cal_idx], rns[cal_idx], Jnom[cal_idx], jr_low, stop, pas,
            steps=GAN_STEPS, tail_w=tw, resume=resume, max_seconds=SLICE_S)
        if not out["done"]:
            st["trainer"] = out
            common.save_pickle(STATE, st)
            print("PROGRESS %s step %d/%d" % (tag, out["t"], GAN_STEPS))
            return
        st.pop("trainer", None)

        # evaluate tail fidelity on held-out designs
        bench = np.load(os.path.join(common.DATA, "benchmark_designs.npz"))
        used = set(map(tuple, np.round(rd[::common.RUNS_PER_RECIPE], 9)))
        held = [i for i in range(bench["d_um"].shape[0])
                if tuple(np.round(bench["d_um"][i], 9)) not in used][:15]
        Hd, Hn = bench["d_um"][held], bench["n0"][held]
        key = jax.random.PRNGKey(common.SEED + 500)
        rng2 = np.random.default_rng(common.SEED + 501)
        K = 400
        p5e, cve, w1p = [], [], []
        for d, n in zip(Hd, Hn):
            Dt, Nt = process.corrupt_ensemble(d, n, K, rng2)
            Jt = np.asarray(tmm_jax.notch_merit_batch(
                jnp.asarray(Dt), jnp.asarray(Nt), stop, pas))
            key, k = jax.random.split(key)
            Dg, Ng = fabgan.sample_fabgan(gp, k, d, n, K)
            Dg = np.clip(Dg, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
            Ng = np.clip(Ng, 1.40, 2.60)
            Jg = np.asarray(tmm_jax.notch_merit_batch(
                jnp.asarray(Dg), jnp.asarray(Ng), stop, pas))
            p5t, cvt = yield_stats(Jt)
            p5g, cvg = yield_stats(Jg)
            p5e.append(abs(p5t - p5g))
            cve.append(abs(cvt - cvg))
            w1p.append(wasserstein_distance(Jt, Jg))
        st["res"][tag] = {"M": M, "tail_w": tw,
                          "P5_abs_err_mean": float(np.mean(p5e)),
                          "CVaR5_abs_err_mean": float(np.mean(cve)),
                          "induced_perf_W1_mean": float(np.mean(w1p))}
        print(tag, st["res"][tag])
        st["job"] += 1
        common.save_pickle(STATE, st)
        if time.time() - t_wall > SLICE_S:
            print("PROGRESS job %d/%d" % (st["job"], len(jobs)))
            return

    common.save_json("exp3b_lowdata.json", st["res"])
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE")


if __name__ == "__main__":
    main()
