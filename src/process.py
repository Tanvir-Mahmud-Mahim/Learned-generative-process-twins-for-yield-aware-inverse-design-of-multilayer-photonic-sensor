"""Ground-truth virtual deposition process ("black-box fab").

Held out from all learning: the generative process twin (fabgan.py) sees only
sampled corruption traces, exactly as a design team sees historical in-situ
monitor data.  All magnitudes are consistent with the deposition-error scales
studied in Yesilyurt et al., Nanophotonics 12, 993 (2023) (systematic +
random errors of a few percent on layer thickness and index) and with the
in-situ monitoring literature (Wilbrandt et al., Appl. Opt. 47, C49, 2008).

Error mechanisms (deliberately NOT jointly Gaussian, NOT i.i.d., and
design-conditional):
 1. Systematic thickness rate bias: d -> d*(1+beta), beta = +2 %.
 2. Systematic index drift across the stack (tool aging within a run):
    linear gradient of slope a_n = -0.04 across the stack.
 3. Inter-layer intermixing (transitory regions): each layer index is pulled
    toward the previous layer's index by kappa = 0.12.
 4. AR(1)-correlated random thickness noise (rho = 0.6), std 2 % of nominal,
    scaled by sqrt(d_i / 60 nm)  (longer deposition -> larger accumulated
    variance)  -> design-conditional noise.
 5. Skewed index noise: centered log-normal with sigma = 0.02 (right-skewed).
 6. Rare particulate ("flake") events: with probability 5 % per run, one
    random layer receives an additional thickness error of N(+8 %, 2 %).
"""

import numpy as np

BETA_T = 0.02        # systematic thickness rate bias
A_N = -0.04          # systematic index drift slope across the stack
KAPPA = 0.12         # inter-layer intermixing
RHO = 0.6            # AR(1) correlation of thickness noise
SIG_T = 0.02         # relative thickness noise std
SIG_N = 0.02         # index noise scale (log-normal)
P_FLAKE = 0.05       # probability of a particulate event per run
FLAKE_MU, FLAKE_SIG = 0.08, 0.02


def corrupt(d_um, n0, rng):
    """One fabricated realization (d_tilde, n_tilde) of a recipe (d, n)."""
    d = np.asarray(d_um, float).copy()
    n = np.asarray(n0, float).copy()
    N = d.shape[0]

    # 1. systematic rate bias
    d_t = d * (1.0 + BETA_T)

    # 4. AR(1) correlated, design-conditional thickness noise
    scale = SIG_T * np.sqrt(d / 0.060)
    eps = np.zeros(N)
    innov = rng.normal(0.0, 1.0, N) * scale * np.sqrt(1.0 - RHO ** 2)
    eps[0] = rng.normal(0.0, scale[0])
    for i in range(1, N):
        eps[i] = RHO * eps[i - 1] + innov[i]
    d_t = d_t * (1.0 + eps)

    # 6. rare flake event
    if rng.random() < P_FLAKE:
        i = rng.integers(0, N)
        d_t[i] *= 1.0 + rng.normal(FLAKE_MU, FLAKE_SIG)

    # 2. systematic index drift + 3. intermixing + 5. skewed index noise
    grid = (np.arange(N) / (N - 1)) - 0.5
    n_t = n + A_N * grid
    n_prev = np.concatenate([[n_t[0]], n_t[:-1]])
    n_t = n_t + KAPPA * (n_prev - n_t)
    skew = np.exp(rng.normal(0.0, SIG_N, N))          # log-normal, mean>1
    n_t = n_t * skew / np.exp(SIG_N ** 2 / 2.0)       # centered (mean 1)

    return d_t, n_t


def corrupt_ensemble(d_um, n0, K, rng):
    """K fabricated realizations; returns (K,N) thickness and index arrays."""
    D = np.empty((K, len(d_um)))
    Nn = np.empty((K, len(n0)))
    for k in range(K):
        D[k], Nn[k] = corrupt(d_um, n0, rng)
    return D, Nn


def trace_dataset(recipes_d, recipes_n, runs_per_recipe, rng):
    """Historical process traces: [(recipe_d, recipe_n, fab_d, fab_n), ...].

    Returns arrays of shape (M, N): recipe thicknesses/indices and the
    corresponding observed (fabricated) values -- the ONLY data the process
    twin is allowed to learn from.
    """
    R, N = recipes_d.shape
    M = R * runs_per_recipe
    rd = np.repeat(recipes_d, runs_per_recipe, axis=0)
    rn = np.repeat(recipes_n, runs_per_recipe, axis=0)
    fd = np.empty((M, N))
    fn = np.empty((M, N))
    for m in range(M):
        fd[m], fn[m] = corrupt(rd[m], rn[m], rng)
    return rd, rn, fd, fn
