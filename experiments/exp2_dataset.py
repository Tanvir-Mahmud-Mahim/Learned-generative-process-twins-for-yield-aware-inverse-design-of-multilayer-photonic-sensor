"""E2: Generate and release the open benchmark dataset.

(a) Design benchmark: N_RECIPES variable-index stacks + exact TMM spectra.
(b) Historical process traces: TRACE_RECIPES recipes x RUNS_PER_RECIPE
    fabricated realizations from the held-out true process -- the only data
    the process twins (GAN / Gaussian) may learn from.
Everything is saved as portable .npz + .csv for public release.
"""

import numpy as np
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tmm_jax, process
from experiments import common


def main():
    rng = np.random.default_rng(common.SEED)

    # (a) design benchmark
    d = rng.uniform(tmm_jax.T_LO, tmm_jax.T_HI, (common.N_RECIPES, tmm_jax.N_LAYERS))
    n = rng.uniform(tmm_jax.N_LO, tmm_jax.N_HI, (common.N_RECIPES, tmm_jax.N_LAYERS))
    T = np.asarray(tmm_jax.transmittance_batch(jnp.asarray(d), jnp.asarray(n)))
    np.savez_compressed(os.path.join(common.DATA, "benchmark_designs.npz"),
                        d_um=d, n0=n, T=T, lam_um=tmm_jax.LAM_UM)

    # (b) historical process traces
    rd_idx = rng.choice(common.N_RECIPES, common.TRACE_RECIPES, replace=False)
    rd, rn, fd, fn = process.trace_dataset(d[rd_idx], n[rd_idx],
                                           common.RUNS_PER_RECIPE, rng)
    np.savez_compressed(os.path.join(common.DATA, "process_traces.npz"),
                        recipe_d=rd, recipe_n=rn, fab_d=fd, fab_n=fn)

    # csv exports for accessibility
    hdr_d = ",".join(f"d{i}" for i in range(tmm_jax.N_LAYERS))
    hdr_n = ",".join(f"n{i}" for i in range(tmm_jax.N_LAYERS))
    np.savetxt(os.path.join(common.DATA, "benchmark_designs.csv"),
               np.hstack([d, n]), delimiter=",",
               header=hdr_d + "," + hdr_n, comments="")

    out = {"n_designs": int(common.N_RECIPES),
           "n_traces": int(rd.shape[0]),
           "spectral_points": int(len(tmm_jax.LAM_UM)),
           "total_spectral_samples": int(common.N_RECIPES * len(tmm_jax.LAM_UM)),
           "T_range": [float(T.min()), float(T.max())]}
    common.save_json("exp2_dataset.json", out)
    print(out)


if __name__ == "__main__":
    main()
