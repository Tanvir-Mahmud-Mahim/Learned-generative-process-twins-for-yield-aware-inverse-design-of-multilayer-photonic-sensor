"""E4: Nominal inverse design -- probe-seeded adjoint engine vs baselines.

Equal-budget comparison at the primary specification (532 nm notch):
  - random search (equal budget),
  - unseeded gradient ascent (multi-restart),
  - probe-seeded gradient engine (ours),
plus queries-to-match for random search (query-efficiency multiple).
Also computes nominal optima for all train/test specifications (consumed by
E5/E6) into the warm-start bank.
"""

import numpy as np
import jax
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, adjoint
from experiments import common


def unseeded_gradient(center, restarts=10, n_iter=36, seed=3, lr=4e-3):
    import optax
    stop, pas = tmm_jax.band_masks(center)
    merit = lambda dd, nn: tmm_jax.notch_merit(dd, nn, stop, pas)
    vg = jax.jit(jax.value_and_grad(merit, argnums=(0, 1)))
    rng = np.random.default_rng(seed)
    best = -1.0
    for r in range(restarts):
        d, n = adjoint.random_designs(rng, 1)
        dd, nn = jnp.asarray(d[0]), jnp.asarray(n[0])
        opt = optax.adam(lr); st = opt.init((dd, nn))
        for _ in range(n_iter):
            v, g = vg(dd, nn)
            up, st = opt.update(jax.tree.map(lambda x: -x, g), st)
            dd, nn = optax.apply_updates((dd, nn), up)
            dd, nn = tmm_jax.clip_design(dd, nn)
        best = max(best, float(merit(dd, nn)))
    return best


def main():
    budget = 200 + 4 * 40      # probes + refinement iterations
    d_s, n_s, J_s, traj = adjoint.inverse_design(common.CENTER,
                                                 seed=common.SEED)
    d_r, n_r, J_r, traj_r = adjoint.random_search(common.CENTER, budget,
                                                  seed=common.SEED + 1)
    J_u = unseeded_gradient(common.CENTER, restarts=10, n_iter=36)

    # queries for random search to match J_s
    big = 30000
    _, _, _, traj_big = adjoint.random_search(common.CENTER, big,
                                              seed=common.SEED + 2)
    match = np.argmax(traj_big >= J_s) if (traj_big >= J_s).any() else -1
    probes_only = float(np.max(traj[:200]))

    # nominal optima for all specifications (warm-start bank for E5/E6)
    bank = {}
    for c in common.TRAIN_SPECS + common.TEST_SPECS + [common.CENTER]:
        dc, nc, Jc, _ = adjoint.inverse_design(c, seed=common.SEED + 11)
        bank[f"{c:.3f}"] = {"d": dc.tolist(), "n": nc.tolist(), "J": Jc}
        print(f"spec {c*1000:.0f} nm: J* = {Jc:.4f}")

    out = {"budget": budget,
           "J_seeded": J_s, "theta_d": d_s.tolist(), "theta_n": n_s.tolist(),
           "J_random_search": J_r,
           "J_unseeded_gradient": J_u,
           "J_probes_only": probes_only,
           "rs_queries_to_match": int(match) + 1 if match >= 0 else None,
           "efficiency_multiple": (float(match + 1) / budget) if match >= 0 else None,
           "traj_seeded": traj.tolist(),
           "traj_random": traj_r[:budget].tolist(),
           "bank": bank}
    common.save_json("exp4_inverse.json", out)
    print({k: out[k] for k in ["J_seeded", "J_random_search",
                               "J_unseeded_gradient", "rs_queries_to_match",
                               "efficiency_multiple"]})


if __name__ == "__main__":
    main()
