"""E5: The money experiment -- twin fidelity converts to true-process yield.

The nominal optimum at 532 nm is robustified by identical pathwise CVaR_5%
ascent, differing ONLY in the process twin that supplies corruption samples:
  (a) i.i.d. diagonal Gaussian, (b) full-covariance Gaussian,
  (c) FabGAN (conditional moment-matched WGAN-GP, ours),
  (d) FabGAN-TC (tail-calibrated variant, ablation),
plus (e) mean-variance objective under the FabGAN twin (objective ablation).

All resulting designs (and the nominal) are evaluated on the HELD-OUT TRUE
process with 2000 common Monte-Carlo draws: mean, sigma/mu, P5, CVaR_5%.
Chunked/resumable: run until DONE.
"""

import os
import sys

import time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, fabgan, adjoint
from experiments import common

STATE = "exp5_state.pkl"
K, STEPS = 256, 250
JOBS = ["nominal", "gauss_diag", "gauss_full", "fabgan", "fabgan_tail",
        "fabgan_meanvar", "gauss_diag_meanvar"]


def main():
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {"job": 0, "out": {}}

    inv = common.load_json("exp4_inverse.json")
    d0 = np.array(inv["theta_d"]); n0 = np.array(inv["theta_n"])
    stop, pas = tmm_jax.band_masks(common.CENTER)
    gp_v = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_vanilla.pkl"))
    gp_t = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_tail.pkl"))
    twins = common.load_pickle("gauss_twins.pkl")

    t_wall = time.time()
    while st["job"] < len(JOBS):
        name = JOBS[st["job"]]
        vg = None
        zdim = fabgan.DIM_Z
        if name == "nominal":
            d, n = d0, n0
        elif name == "gauss_diag":
            vg = adjoint.make_gauss_objective(twins["gauss_diag"].mu,
                                              twins["gauss_diag"].L,
                                              stop, pas, K, 0.05)
            zdim = fabgan.DIM_X
        elif name == "gauss_full":
            vg = adjoint.make_gauss_objective(twins["gauss_full"].mu,
                                              twins["gauss_full"].L,
                                              stop, pas, K, 0.05)
            zdim = fabgan.DIM_X
        elif name == "fabgan":
            vg = adjoint.make_gan_objective(gp_v, stop, pas, K, 0.05)
            zdim = fabgan.DIM_Z
        elif name == "fabgan_tail":
            vg = adjoint.make_gan_objective(gp_t, stop, pas, K, 0.05)
            zdim = fabgan.DIM_Z
        elif name == "fabgan_meanvar":
            vg = adjoint.make_gan_objective(gp_v, stop, pas, K, 0.05,
                                            mean_variance=True, beta=1.0)
            zdim = fabgan.DIM_Z
        else:  # gauss_diag_meanvar
            vg = adjoint.make_gauss_objective(twins["gauss_diag"].mu,
                                              twins["gauss_diag"].L,
                                              stop, pas, K, 0.05,
                                              mean_variance=True, beta=1.0)
            zdim = fabgan.DIM_X

        if vg is not None:
            d, n, out = adjoint.robustify(d0, n0, vg, zdim, K, steps=STEPS,
                                          resume=st.get("trainer"),
                                          max_seconds=max(5.0, 34 - (time.time() - t_wall)))
            if not out["done"]:
                st["trainer"] = out
                common.save_pickle(STATE, st)
                print("PROGRESS %s step %d/%d" % (name, out["t"], STEPS))
                return
            st.pop("trainer", None)

        stt = adjoint.evaluate_true(np.asarray(d), np.asarray(n),
                                    common.CENTER, K=2000,
                                    seed=common.SEED + 5)
        rec = {k: v for k, v in stt.items() if k != "samples"}
        rec["d"] = np.asarray(d).tolist()
        rec["n"] = np.asarray(n).tolist()
        st["out"][name] = rec
        np.save(os.path.join(common.RESULTS, "exp5_samples_%s.npy" % name),
                stt["samples"])
        print(name, {k: round(v, 5) for k, v in rec.items()
                     if isinstance(v, float)})
        st["job"] += 1
        common.save_pickle(STATE, st)

    common.save_json("exp5_yield.json", st["out"])
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE")


if __name__ == "__main__":
    main()
