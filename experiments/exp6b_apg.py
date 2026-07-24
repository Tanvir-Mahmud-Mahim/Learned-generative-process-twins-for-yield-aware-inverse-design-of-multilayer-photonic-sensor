"""E6b: Specification-conditioned correction policy via analytic policy
gradients (APG) through the differentiable fabrication loop, vs the
model-free SAC baseline (exp6_policy_sac.json) and static correction
transfer.

The policy trains on the TRAIN specifications only and is queried zero-shot
at held-out TEST specifications; per-spec pathwise CVaR re-optimization is
the upper bound.  Recovery metric as in PARL-ID.
Chunked/resumable: run until DONE.
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, fabgan, adjoint, rl_loop
from experiments import common

STATE = "exp6b_state.pkl"
STEPS = 400


def main():
    t_wall = time.time()
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {}
    inv = common.load_json("exp4_inverse.json")
    yld = common.load_json("exp5_yield.json")
    sac = common.load_json("exp6_policy_sac.json")
    gp = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_vanilla.pkl"))

    bank = inv["bank"]
    nominal = {}
    for c in common.TRAIN_SPECS + common.TEST_SPECS:
        e = bank[f"{c:.3f}"]
        nominal[c] = (np.array(e["d"]), np.array(e["n"]))

    specs = [(c, rl_loop.spec_to_norm(c)) for c in common.TRAIN_SPECS]
    if not os.path.exists(os.path.join(common.RESULTS, "apg_policy.pkl")):
        pp, out = rl_loop.train_policy_apg(gp, specs, nominal, steps=STEPS,
                                           seed=common.SEED,
                                           resume=st.get("trainer"),
                                           max_seconds=30)
        if not out["done"]:
            st["trainer"] = out
            common.save_pickle(STATE, st)
            print("PROGRESS apg step %d/%d" % (out["t"], STEPS))
            return
        st.pop("trainer", None)
        common.save_pickle("apg_policy.pkl", jax.tree.map(np.asarray, pp))
        common.save_pickle("apg_log.pkl", out["hist"])
        common.save_pickle(STATE, st)
    pp = jax.tree.map(jnp.asarray, common.load_pickle("apg_policy.pkl"))
    log = common.load_pickle("apg_log.pkl")

    d_rob = np.array(yld["fabgan"]["d"]); n_rob = np.array(yld["fabgan"]["n"])
    d0 = np.array(inv["theta_d"]); n0 = np.array(inv["theta_n"])
    delta_d, delta_n = d_rob - d0, n_rob - n0

    res = st.get("res", {"train_log": log})
    done_specs = st.get("done_specs", [])
    for c in common.TEST_SPECS:
        tag = f"{c:.3f}"
        if tag in done_specs:
            continue
        stop, pas = tmm_jax.band_masks(c)
        d_star, n_star = nominal[c]
        # reuse nominal/static/reopt evaluations from the SAC run (identical
        # designs and evaluation seed)
        prev = sac[tag]
        d_pol, n_pol = rl_loop.apply_policy(pp, d_star, n_star,
                                            rl_loop.spec_to_norm(c))
        st_pol = adjoint.evaluate_true(np.asarray(d_pol), np.asarray(n_pol),
                                       c, seed=common.SEED + 9)
        gain_att = prev["reopt"]["CVaR5"] - prev["nominal"]["CVaR5"]
        rec_pol = ((st_pol["CVaR5"] - prev["nominal"]["CVaR5"]) / gain_att
                   if gain_att > 1e-9 else None)
        res[tag] = {"nominal": prev["nominal"], "static": prev["static"],
                    "sac_policy": prev["policy"], "reopt": prev["reopt"],
                    "apg_policy": {k: v for k, v in st_pol.items()
                                   if k != "samples"},
                    "recovery_static": prev["recovery_static"],
                    "recovery_sac": prev["recovery_policy"],
                    "recovery_apg": rec_pol}
        print("spec %.0f nm: recovery apg %.2f (static %.2f, sac %.2f)" % (
            c * 1000, rec_pol, prev["recovery_static"],
            prev["recovery_policy"]))
        done_specs.append(tag)
        st["res"] = res
        st["done_specs"] = done_specs
        common.save_pickle(STATE, st)
        if time.time() - t_wall > 32:
            print("PROGRESS eval %s" % tag)
            return

    common.save_json("exp6b_apg.json", res)
    os.remove(os.path.join(common.RESULTS, STATE))
    print("DONE")


if __name__ == "__main__":
    main()
