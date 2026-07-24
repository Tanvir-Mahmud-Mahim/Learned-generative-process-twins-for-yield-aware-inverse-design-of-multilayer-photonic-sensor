"""E7: Transfer-matrix-informed neural surrogate (the PINN pillar).

Protocol study mirroring PARL-ID: the same surrogate is evaluated under
  (i) row splitting (random wavelength points of every design held out) and
  (ii) strict held-out-design splitting (entire designs excluded),
each with physics weight lam_w = 0 (data-only) and lam_w > 0
(recursion-informed).  Also reports amortization speedup vs the exact TMM.
Chunked/resumable: run until DONE.
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, surrogate
from experiments import common

STEPS = 2000
SLICE_S = 30
STATE = "exp7_state.pkl"
JOBS = [("data", 0.0, "comb"), ("phys", 1e-2, "comb"),
        ("data", 0.0, "row"), ("phys", 1e-2, "row")]


def main():
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {"job": 0, "out": {}}

    bench = np.load(os.path.join(common.DATA, "benchmark_designs.npz"))
    D, Nn, T = bench["d_um"], bench["n0"], bench["T"]
    R, L = T.shape
    rng = np.random.default_rng(common.SEED + 4)
    perm = rng.permutation(R)
    n_test = int(0.2 * R)
    te, tr = perm[:n_test], perm[n_test:]
    lam_mask = np.random.default_rng(common.SEED + 5).random((R, L)) < 0.8

    while st["job"] < len(JOBS):
        tag, lam_w, proto = JOBS[st["job"]]
        name = "%s_%s" % (tag, proto)
        if proto == "comb":
            args = (D[tr], Nn[tr], T[tr])
            kw = {}
        else:
            args = (D, Nn, T)
            kw = {"lam_mask": lam_mask}
        params, out = surrogate.train_surrogate(
            jax.random.PRNGKey(common.SEED + st["job"]), *args, steps=STEPS,
            lam_w=lam_w, lr=1e-3, resume=st.get("trainer"), max_seconds=SLICE_S, **kw)
        if not out["done"]:
            st["trainer"] = out
            common.save_pickle(STATE, st)
            print("PROGRESS %s step %d/%d" % (name, out["t"], STEPS))
            return
        st.pop("trainer", None)

        if proto == "comb":
            pred = np.stack([surrogate.predict_spectrum(params, D[i], Nn[i])
                             for i in te])
            r2 = surrogate.r2_score(T[te].ravel(), pred.ravel())
            mae = float(np.mean(np.abs(T[te] - pred)))
        else:
            idx3 = np.arange(0, R, 3)
            pred = np.stack([surrogate.predict_spectrum(params, D[i], Nn[i])
                             for i in idx3])
            m = ~lam_mask[idx3]
            r2 = surrogate.r2_score(T[idx3][m].ravel(), pred[m].ravel())
            mae = float(np.mean(np.abs(T[idx3][m] - pred[m])))
        st["out"][name] = {"r2": float(r2), "mae": mae,
                           "hist_tail": out["hist"][-2:]}
        print(name, st["out"][name])
        if st["job"] == 1:   # keep the physics/comb surrogate for reuse
            common.save_pickle("surrogate_phys.pkl",
                               jax.tree.map(np.asarray, params))
        st["job"] += 1
        common.save_pickle(STATE, st)

    # amortization timing
    params = jax.tree.map(jnp.asarray, common.load_pickle("surrogate_phys.pkl"))
    db = jnp.asarray(D[:256]); nb = jnp.asarray(Nn[:256])
    tmm_jax.transmittance_batch(db, nb).block_until_ready()
    t0 = time.time(); tmm_jax.transmittance_batch(db, nb).block_until_ready()
    t_tmm = time.time() - t0
    lam = jnp.asarray(tmm_jax.LAM_UM)
    lamn = jnp.asarray(surrogate.lam_to_norm(np.asarray(tmm_jax.LAM_UM)))
    vpred = jax.jit(jax.vmap(lambda c: surrogate._v_predT(params, c, lam, lamn)))
    Cb = surrogate.norm_recipe(db, nb)
    vpred(Cb).block_until_ready()
    t0 = time.time(); vpred(Cb).block_until_ready()
    t_sur = time.time() - t0
    st["out"]["speedup_surrogate_vs_tmm"] = float(t_tmm / t_sur)
    st["out"]["t_tmm_256"] = float(t_tmm)
    st["out"]["t_surrogate_256"] = float(t_sur)
    common.save_json("exp7_surrogate.json", st["out"])
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE")


if __name__ == "__main__":
    main()
