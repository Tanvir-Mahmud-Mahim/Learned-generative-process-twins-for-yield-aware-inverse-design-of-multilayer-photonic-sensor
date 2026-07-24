"""E7b: rerun the recursion-informed surrogate under the held-out-design
protocol with random-design collocation (the released code path), updating
exp7_surrogate.json in place.  Chunked: run until DONE."""
import os, sys
import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import surrogate
from experiments import common

STATE = "exp7b_state.pkl"

def main():
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {}
    bench = np.load(os.path.join(common.DATA, "benchmark_designs.npz"))
    D, Nn, T = bench["d_um"], bench["n0"], bench["T"]
    R = T.shape[0]
    rng = np.random.default_rng(common.SEED + 4)
    perm = rng.permutation(R)
    n_test = int(0.2 * R)
    te, tr = perm[:n_test], perm[n_test:]
    params, out = surrogate.train_surrogate(
        jax.random.PRNGKey(common.SEED + 1), D[tr], Nn[tr], T[tr],
        steps=2000, lam_w=1e-2, lr=1e-3, resume=st.get("trainer"),
        max_seconds=30)
    if not out["done"]:
        st["trainer"] = out
        common.save_pickle(STATE, st)
        print("PROGRESS step %d/2000" % out["t"])
        return
    pred = np.stack([surrogate.predict_spectrum(params, D[i], Nn[i])
                     for i in te])
    r2 = surrogate.r2_score(T[te].ravel(), pred.ravel())
    mae = float(np.mean(np.abs(T[te] - pred)))
    res = common.load_json("exp7_surrogate.json")
    res["phys_comb"] = {"r2": float(r2), "mae": mae,
                        "hist_tail": out["hist"][-2:]}
    common.save_json("exp7_surrogate.json", res)
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE phys_comb r2 %.4f mae %.4f" % (r2, mae))

if __name__ == "__main__":
    main()
