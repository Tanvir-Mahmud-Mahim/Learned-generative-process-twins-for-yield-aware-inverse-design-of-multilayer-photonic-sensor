"""E6c: dense specification grid for the APG policy.

Hypothesis: zero-shot policy transfer requires dense specification coverage.
Adds 6 nominal optima (510-610 nm) to the 5 original TRAIN specs, retrains
the APG policy on 11 specs, and re-evaluates zero-shot at the same held-out
TEST specs.  Chunked: run until DONE."""
import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, fabgan, adjoint, rl_loop
from experiments import common

STATE = "exp6c_state.pkl"
EXTRA = [0.51, 0.53, 0.55, 0.57, 0.59, 0.61]
STEPS = 400


def main():
    t_wall = time.time()
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {}
    inv = common.load_json("exp4_inverse.json")
    gp = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_vanilla.pkl"))

    bank = dict(inv["bank"])
    extra_bank = st.get("extra_bank", {})
    for c in EXTRA:
        if f"{c:.3f}" in extra_bank:
            continue
        dc, nc, Jc, _ = adjoint.inverse_design(c, seed=common.SEED + 11)
        extra_bank[f"{c:.3f}"] = {"d": dc.tolist(), "n": nc.tolist(), "J": Jc}
        st["extra_bank"] = extra_bank
        common.save_pickle(STATE, st)
        if time.time() - t_wall > 30:
            print("PROGRESS bank %d/%d" % (len(extra_bank), len(EXTRA)))
            return
    bank.update(extra_bank)

    train_specs = sorted(common.TRAIN_SPECS + EXTRA)
    nominal = {}
    for c in train_specs + common.TEST_SPECS:
        e = bank[f"{c:.3f}"]
        nominal[c] = (np.array(e["d"]), np.array(e["n"]))
    specs = [(c, rl_loop.spec_to_norm(c)) for c in train_specs]

    if not os.path.exists(os.path.join(common.RESULTS, "apg_policy_dense.pkl")):
        pp, out = rl_loop.train_policy_apg(gp, specs, nominal, steps=STEPS,
                                           seed=common.SEED + 2,
                                           resume=st.get("trainer"),
                                           max_seconds=28)
        if not out["done"]:
            st["trainer"] = out
            common.save_pickle(STATE, st)
            print("PROGRESS apg %d/%d" % (out["t"], STEPS))
            return
        st.pop("trainer", None)
        common.save_pickle("apg_policy_dense.pkl", jax.tree.map(np.asarray, pp))
        common.save_pickle("apg_dense_log.pkl", out["hist"])
        common.save_pickle(STATE, st)
    pp = jax.tree.map(jnp.asarray, common.load_pickle("apg_policy_dense.pkl"))

    sac = common.load_json("exp6_policy_sac.json")
    res = st.get("res", {"train_log": common.load_pickle("apg_dense_log.pkl"),
                         "n_train_specs": len(train_specs)})
    for c in common.TEST_SPECS:
        tag = f"{c:.3f}"
        if tag in res:
            continue
        d_star, n_star = nominal[c]
        d_pol, n_pol = rl_loop.apply_policy(pp, d_star, n_star,
                                            rl_loop.spec_to_norm(c))
        st_pol = adjoint.evaluate_true(np.asarray(d_pol), np.asarray(n_pol),
                                       c, seed=common.SEED + 9)
        prev = sac[tag]
        gain = prev["reopt"]["CVaR5"] - prev["nominal"]["CVaR5"]
        rec = ((st_pol["CVaR5"] - prev["nominal"]["CVaR5"]) / gain
               if gain > 1e-9 else None)
        res[tag] = {"apg_dense": {k: v for k, v in st_pol.items()
                                  if k != "samples"},
                    "recovery_apg_dense": rec}
        print("spec %.0f nm: dense-grid recovery %.2f" % (c * 1000, rec))
        st["res"] = res
        common.save_pickle(STATE, st)

    common.save_json("exp6c_dense.json", res)
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE")


if __name__ == "__main__":
    main()
