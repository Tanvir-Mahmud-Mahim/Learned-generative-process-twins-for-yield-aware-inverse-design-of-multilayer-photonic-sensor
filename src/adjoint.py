"""Adjoint inverse engine and robustification optimizers.

- Exact design gradients by reverse-mode autodiff through the differentiable
  TMM (the discrete adjoint of the transfer recursion), validated against
  central finite differences in experiments/exp1_validate.py.
- Probe-seeded projected-Adam inverse design (global probing + gradient
  refinement), with an equal-budget random-search baseline.
- Pathwise CVaR robustification: stochastic gradient ascent of the empirical
  CVaR_alpha of the merit under corruptions sampled from a given process twin
  (FabGAN or Gaussian) -- gradients flow through both the TMM and, for the
  GAN twin, the generator itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import fabgan, tmm_jax

jax.config.update("jax_enable_x64", True)

N = tmm_jax.N_LAYERS


# ---------------- nominal inverse design ----------------
def random_designs(rng, R):
    d = rng.uniform(tmm_jax.T_LO, tmm_jax.T_HI, (R, N))
    n = rng.uniform(tmm_jax.N_LO, tmm_jax.N_HI, (R, N))
    return d, n


def inverse_design(center_um, seed=0, n_probe=200, n_seed=4, n_iter=40,
                   lr=4e-3):
    """Probe-seeded gradient engine.  Query budget = n_probe + 2*n_seed*n_iter
    merit-gradient evaluations (each gradient costs one forward+adjoint).

    Returns (d*, n*, J*, trajectory of best-so-far J per query).
    """
    stop, pas = tmm_jax.band_masks(center_um)
    rng = np.random.default_rng(seed)
    d, n = random_designs(rng, n_probe)
    J = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(d), jnp.asarray(n),
                                             stop, pas))
    order = np.argsort(J)[::-1]
    best_traj = np.maximum.accumulate(J)

    merit = lambda dd, nn: tmm_jax.notch_merit(dd, nn, stop, pas)
    vg = jax.jit(jax.value_and_grad(merit, argnums=(0, 1)))

    best = (None, None, -1.0)
    traj = list(best_traj)
    for s in order[:n_seed]:
        dd, nn = jnp.asarray(d[s]), jnp.asarray(n[s])
        opt = optax.adam(lr)
        st = opt.init((dd, nn))
        for _ in range(n_iter):
            J_c, g = vg(dd, nn)
            up, st = opt.update(jax.tree.map(lambda x: -x, g), st)
            dd, nn = optax.apply_updates((dd, nn), up)
            dd, nn = tmm_jax.clip_design(dd, nn)
            jc = float(J_c)
            # one query per gradient iteration (one forward + one adjoint
            # solve, the standard adjoint budget accounting)
            traj.append(max(traj[-1], jc))
            if jc > best[2]:
                best = (np.asarray(dd), np.asarray(nn), jc)
    # final merit of polished designs
    for _ in range(1):
        jf = float(merit(jnp.asarray(best[0]), jnp.asarray(best[1])))
        if jf > best[2]:
            best = (best[0], best[1], jf)
    return best[0], best[1], best[2], np.asarray(traj)


def random_search(center_um, budget, seed=1, chunk=2000):
    stop, pas = tmm_jax.band_masks(center_um)
    rng = np.random.default_rng(seed)
    Js, Ds, Ns = [], [], []
    done = 0
    while done < budget:
        b = min(chunk, budget - done)
        d, n = random_designs(rng, b)
        J = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(d),
                                                 jnp.asarray(n), stop, pas))
        Js.append(J); Ds.append(d); Ns.append(n)
        done += b
    J = np.concatenate(Js); d = np.vstack(Ds); n = np.vstack(Ns)
    besti = int(np.argmax(J))
    return d[besti], n[besti], float(J[besti]), np.maximum.accumulate(J)


# ---------------- CVaR robustification (pathwise) ----------------
def _cvar(vals, alpha):
    K = vals.shape[0]
    q = max(1, int(np.ceil(alpha * K)))
    return jnp.mean(jnp.sort(vals)[:q])


def make_gan_objective(gp, stop, pas, K, alpha, mean_variance=False, beta=1.0):
    """Empirical CVaR (or mu - beta*sigma) of merit under FabGAN corruptions."""

    def obj(dn, z):
        d, n = dn
        c = fabgan.norm_recipe(d, n)
        x = fabgan.gen_forward(gp, z, jnp.tile(c, (K, 1)))
        dt, nt = fabgan.apply_errors(d, n, x)
        dt = jnp.clip(dt, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
        nt = jnp.clip(nt, 1.40, 2.60)
        J = tmm_jax.notch_merit_batch(dt, nt, stop, pas)
        if mean_variance:
            return jnp.mean(J) - beta * jnp.std(J)
        return _cvar(J, alpha)

    return jax.jit(jax.value_and_grad(obj, argnums=0))


def make_gauss_objective(mu, L, stop, pas, K, alpha, mean_variance=False,
                         beta=1.0):
    """Same objective under a Gaussian twin (reparameterized)."""
    mu = jnp.asarray(mu)
    L = jnp.asarray(L)

    def obj(dn, z):
        d, n = dn
        x = mu + z @ L.T
        dt = d * (1.0 + x[:, :N])
        nt = n + x[:, N:]
        dt = jnp.clip(dt, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
        nt = jnp.clip(nt, 1.40, 2.60)
        J = tmm_jax.notch_merit_batch(dt, nt, stop, pas)
        if mean_variance:
            return jnp.mean(J) - beta * jnp.std(J)
        return _cvar(J, alpha)

    return jax.jit(jax.value_and_grad(obj, argnums=0))


def robustify(d0, n0, vg_obj, z_dim, K, steps=150, lr=2e-3, seed=0,
              resume=None, max_seconds=None):
    """Projected stochastic CVaR ascent from the nominal optimum
    (resumable)."""
    import time as _time
    t_start = _time.time()
    opt = optax.adam(lr)
    if resume is None:
        key = jax.random.PRNGKey(seed)
        dn = (jnp.asarray(d0), jnp.asarray(n0))
        st = opt.init(dn)
        t0, hist = 0, []
    else:
        key, dn, st, t0, hist = (resume["key"], resume["dn"], resume["st"],
                                 resume["t"], resume["hist"])
    for t in range(t0, steps):
        key, k = jax.random.split(key)
        z = jax.random.normal(k, (K, z_dim))
        v, g = vg_obj(dn, z)
        up, st = opt.update(jax.tree.map(lambda x: -x, g), st)
        dn = optax.apply_updates(dn, up)
        dn = tmm_jax.clip_design(*dn)
        if t % 10 == 0:
            hist.append((t, float(v)))
        if max_seconds is not None and _time.time() - t_start > max_seconds:
            return None, None, {"key": key, "dn": dn, "st": st, "t": t + 1,
                                "hist": hist, "done": False}
    return np.asarray(dn[0]), np.asarray(dn[1]), {"hist": hist, "done": True,
                                                  "dn": dn}


# ---------------- true-process evaluation ----------------
def evaluate_true(d, n, center_um, K=2000, seed=123):
    """Yield FOMs of a design under the held-out TRUE process."""
    from . import process
    stop, pas = tmm_jax.band_masks(center_um)
    rng = np.random.default_rng(seed)
    D, Nn = process.corrupt_ensemble(d, n, K, rng)
    J = np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D), jnp.asarray(Nn),
                                             stop, pas))
    q = max(1, int(np.ceil(0.05 * K)))
    Js = np.sort(J)
    return {
        "mean": float(J.mean()),
        "std": float(J.std()),
        "P5": float(np.percentile(J, 5)),
        "CVaR5": float(Js[:q].mean()),
        "samples": J,
    }
