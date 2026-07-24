"""E6: Specification-conditioned policy vs static correction transfer.

A chain-GCN SAC agent is trained in the FabGAN virtual fabrication loop over
the TRAIN specifications; it is then queried ZERO-SHOT at held-out TEST
specifications and compared against:
  - static: transplanting the 532 nm correction vector (theta_rob - theta*),
  - per-spec: full pathwise CVaR re-optimization at the test spec (upper bd).
Recovery metric (as in PARL-ID): fraction of the attainable true-process
CVaR_5% gain (per-spec minus nominal) recovered by each method.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, fabgan, adjoint, rl_loop
from experiments import common


STATE = "exp6_state.pkl"


def main():
    t_wall = time.time()
    st = common.load_pickle(STATE) if os.path.exists(
        os.path.join(common.RESULTS, STATE)) else {}
    inv = common.load_json("exp4_inverse.json")
    yld = common.load_json("exp5_yield.json")
    gp_t = jax.tree.map(jnp.asarray, common.load_pickle("fabgan_vanilla.pkl"))

    bank = inv["bank"]
    nominal = {}
    for c in common.TRAIN_SPECS + common.TEST_SPECS:
        e = bank[f"{c:.3f}"]
        nominal[c] = (np.array(e["d"]), np.array(e["n"]))

    specs = [(c, rl_loop.spec_to_norm(c)) for c in common.TRAIN_SPECS]
    if not os.path.exists(os.path.join(common.RESULTS, "sac_actor.pkl")):
        ap, log, out = rl_loop.train_sac(gp_t, specs, nominal, steps=4200,
                                         seed=common.SEED,
                                         resume=st.get("trainer"),
                                         max_seconds=30)
        if not out["done"]:
            st["trainer"] = out
            common.save_pickle(STATE, st)
            print("PROGRESS sac step %d/4200" % out["t"])
            return
        st.pop("trainer", None)
        common.save_pickle("sac_actor.pkl", jax.tree.map(np.asarray, ap))
        common.save_pickle("sac_log.pkl", log)
        common.save_pickle(STATE, st)
    ap = jax.tree.map(jnp.asarray, common.load_pickle("sac_actor.pkl"))
    log = common.load_pickle("sac_log.pkl")

    # static correction from the 532 nm robustification of E5
    d_rob = np.array(yld["fabgan"]["d"])
    n_rob = np.array(yld["fabgan"]["n"])
    d0 = np.array(inv["theta_d"]); n0 = np.array(inv["theta_n"])
    delta_d, delta_n = d_rob - d0, n_rob - n0

    key = jax.random.PRNGKey(common.SEED + 3)
    res = st.get("res", {"train_log": log})
    K, steps = 256, 250
    done_specs = st.get("done_specs", [])
    for c in common.TEST_SPECS:
        if f"{c:.3f}" in done_specs:
            continue
        stop, pas = tmm_jax.band_masks(c)
        d_star, n_star = nominal[c]
        # nominal
        st_nom = adjoint.evaluate_true(d_star, n_star, c, seed=common.SEED + 9)
        # static transfer
        d_stat, n_stat = tmm_jax.clip_design(jnp.asarray(d_star + delta_d),
                                             jnp.asarray(n_star + delta_n))
        st_stat = adjoint.evaluate_true(np.asarray(d_stat), np.asarray(n_stat),
                                        c, seed=common.SEED + 9)
        # policy zero-shot
        key, k = jax.random.split(key)
        d_pol, n_pol = rl_loop.policy_correct(ap, d_star, n_star,
                                              rl_loop.spec_to_norm(c), k)
        st_pol = adjoint.evaluate_true(d_pol, n_pol, c, seed=common.SEED + 9)
        # per-spec re-optimization (upper bound)
        vg = adjoint.make_gan_objective(gp_t, stop, pas, K, 0.05)
        d_re, n_re, out_r = adjoint.robustify(
            d_star, n_star, vg, fabgan.DIM_Z, K, steps=steps,
            resume=st.get("reopt"),
            max_seconds=max(5.0, 34 - (time.time() - t_wall)))
        if not out_r["done"]:
            st["reopt"] = out_r
            st["res"] = res
            st["done_specs"] = done_specs
            common.save_pickle(STATE, st)
            print("PROGRESS reopt %.3f step %d/%d" % (c, out_r["t"], steps))
            return
        st.pop("reopt", None)
        st_re = adjoint.evaluate_true(d_re, n_re, c, seed=common.SEED + 9)

        gain_att = st_re["CVaR5"] - st_nom["CVaR5"]
        rec_stat = (st_stat["CVaR5"] - st_nom["CVaR5"]) / gain_att if gain_att > 1e-9 else None
        rec_pol = (st_pol["CVaR5"] - st_nom["CVaR5"]) / gain_att if gain_att > 1e-9 else None
        res[f"{c:.3f}"] = {
            "nominal": {k_: v for k_, v in st_nom.items() if k_ != "samples"},
            "static": {k_: v for k_, v in st_stat.items() if k_ != "samples"},
            "policy": {k_: v for k_, v in st_pol.items() if k_ != "samples"},
            "reopt": {k_: v for k_, v in st_re.items() if k_ != "samples"},
            "recovery_static": rec_stat, "recovery_policy": rec_pol}
        print(f"spec {c*1000:.0f} nm: attainable CVaR5 gain {gain_att:+.4f}; "
              f"recovery static {rec_stat}, policy {rec_pol}")
        done_specs.append(f"{c:.3f}")
        st["res"] = res
        st["done_specs"] = done_specs
        common.save_pickle(STATE, st)

    common.save_json("exp6_policy.json", res)
    os.remove(os.path.join(common.RESULTS, STATE))


if __name__ == "__main__":
    main()
