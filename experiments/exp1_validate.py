"""E1: Solver and adjoint validation.

(a) Differentiable TMM vs the open-source `tmm` package (Byrnes) on random
    stacks -- transmittance agreement.
(b) Reverse-mode (adjoint) design gradients vs central finite differences --
    relative error and cosine similarity.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import tmm as tmm_ref

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, materials
from experiments import common


def main():
    rng = np.random.default_rng(common.SEED)
    R = 20
    d = rng.uniform(tmm_jax.T_LO, tmm_jax.T_HI, (R, tmm_jax.N_LAYERS))
    n0 = rng.uniform(tmm_jax.N_LO, tmm_jax.N_HI, (R, tmm_jax.N_LAYERS))

    # (a) against tmm package
    max_abs = 0.0
    for r in range(R):
        T_ours = np.asarray(tmm_jax.transmittance_jit(jnp.asarray(d[r]),
                                                      jnp.asarray(n0[r])))
        for li, lam in enumerate(tmm_jax.LAM_UM[::8]):
            nlist = [1.0] + list(n0[r] * materials.dispersion_shape(lam)) + \
                    [float(materials.n_sio2(lam))]
            dlist = [np.inf] + list(d[r] * 1000.0) + [np.inf]  # nm
            res = tmm_ref.coh_tmm('s', nlist, dlist, 0.0, lam * 1000.0)
            max_abs = max(max_abs, abs(res['T'] - T_ours[li * 8]))

    # (b) adjoint gradient vs central finite differences
    stop, pas = tmm_jax.band_masks(common.CENTER)
    merit = lambda dd, nn: tmm_jax.notch_merit(dd, nn, stop, pas)
    g_d, g_n = jax.grad(merit, argnums=(0, 1))(jnp.asarray(d[0]),
                                               jnp.asarray(n0[0]))
    g_ad = np.concatenate([np.asarray(g_d), np.asarray(g_n)])
    h = 1e-6
    g_fd = np.zeros_like(g_ad)
    for i in range(tmm_jax.N_LAYERS):
        dp, dm = d[0].copy(), d[0].copy()
        dp[i] += h; dm[i] -= h
        g_fd[i] = (float(merit(jnp.asarray(dp), jnp.asarray(n0[0])))
                   - float(merit(jnp.asarray(dm), jnp.asarray(n0[0])))) / (2 * h)
        npp, nm = n0[0].copy(), n0[0].copy()
        npp[i] += h; nm[i] -= h
        g_fd[tmm_jax.N_LAYERS + i] = (
            float(merit(jnp.asarray(d[0]), jnp.asarray(npp)))
            - float(merit(jnp.asarray(d[0]), jnp.asarray(nm)))) / (2 * h)

    rel = np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-30)
    cos = float(np.dot(g_ad, g_fd) / (np.linalg.norm(g_ad) * np.linalg.norm(g_fd)))

    # (c) solver throughput
    db = jnp.asarray(rng.uniform(tmm_jax.T_LO, tmm_jax.T_HI, (2000, 20)))
    nb = jnp.asarray(rng.uniform(tmm_jax.N_LO, tmm_jax.N_HI, (2000, 20)))
    tmm_jax.transmittance_batch(db, nb).block_until_ready()
    t0 = time.time()
    tmm_jax.transmittance_batch(db, nb).block_until_ready()
    thr = 2000 / (time.time() - t0)

    out = {"max_abs_T_diff_vs_tmm_pkg": float(max_abs),
           "grad_median_rel_err_vs_fd": float(np.median(rel)),
           "grad_max_rel_err_vs_fd": float(np.max(rel)),
           "grad_cosine_similarity": cos,
           "solver_throughput_designs_per_s": float(thr)}
    common.save_json("exp1_validate.json", out)
    print(out)


if __name__ == "__main__":
    main()
