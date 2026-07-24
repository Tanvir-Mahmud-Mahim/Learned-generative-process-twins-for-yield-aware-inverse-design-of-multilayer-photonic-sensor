"""FabGAN: conditional, tail-calibrated generative process twin.

A conditional WGAN-GP learns the joint distribution of fabrication errors
    x = (d_tilde/d - 1, n_tilde - n)  in R^{2N}
conditioned on the recipe c = (d, n) (normalized), from historical process
traces.  Novelty: a *physics-in-the-loop tail-calibration* term evaluated
through the differentiable TMM solver -- generated corruptions are pushed to
reproduce the lower quantiles of the *induced performance distribution* on a
bank of calibration designs, i.e., the twin is trained to be faithful exactly
where manufacturing yield is decided.

Baselines implemented alongside: (i) i.i.d. diagonal Gaussian fit,
(ii) full-covariance Gaussian fit, (iii) vanilla conditional WGAN-GP
(identical architecture, no tail calibration).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import tmm_jax

jax.config.update("jax_enable_x64", True)

N = tmm_jax.N_LAYERS
DIM_X = 2 * N          # error vector (relative thickness err, absolute index err)
DIM_C = 2 * N          # condition (normalized recipe)
DIM_Z = 32

G_HID = (128, 128)
D_HID = (128, 128)


# ---------------- utilities ----------------
def norm_recipe(d, n):
    dn = (d - tmm_jax.T_LO) / (tmm_jax.T_HI - tmm_jax.T_LO) * 2 - 1
    nn = (n - tmm_jax.N_LO) / (tmm_jax.N_HI - tmm_jax.N_LO) * 2 - 1
    return jnp.concatenate([dn, nn], axis=-1)


def errors_from_traces(rd, rn, fd, fn):
    """x = (relative thickness error, absolute index error)."""
    return np.concatenate([fd / rd - 1.0, fn - rn], axis=-1)


def apply_errors(d, n, x):
    """Reconstruct fabricated (d_tilde, n_tilde) from recipe and error vector."""
    et = x[..., :N]
    en = x[..., N:]
    return d * (1.0 + et), n + en


# ---------------- MLP ----------------
def init_mlp(key, sizes):
    params = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1),
                         zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i)
        params.append((w, jnp.zeros(o)))
    return params


def mlp(params, x, act=jax.nn.leaky_relu):
    for w, b in params[:-1]:
        x = act(x @ w + b)
    w, b = params[-1]
    return x @ w + b


# ---------------- generator / critic ----------------
def gen_forward(gp, z, c):
    h = jnp.concatenate([z, c], axis=-1)
    out = mlp(gp, h)
    # scale head keeps outputs in a plausible physical range (up to ~30% / 0.3)
    return 0.30 * jnp.tanh(out)


def crit_forward(dp, x, c):
    h = jnp.concatenate([x, c], axis=-1)
    return mlp(dp, h)[..., 0]


def init_models(key):
    kg, kd = jax.random.split(key)
    gp = init_mlp(kg, (DIM_Z + DIM_C,) + G_HID + (DIM_X,))
    dp = init_mlp(kd, (DIM_X + DIM_C,) + D_HID + (1,))
    return gp, dp


# ---------------- WGAN-GP losses ----------------
def critic_loss(dp, gp, xr, c, z, key, gp_weight=10.0):
    xf = gen_forward(gp, z, c)
    lr = crit_forward(dp, xr, c)
    lf = crit_forward(dp, xf, c)
    eps = jax.random.uniform(key, (xr.shape[0], 1))
    xi = eps * xr + (1 - eps) * xf

    def critic_on(x):
        return crit_forward(dp, x, c).sum()

    g = jax.grad(critic_on)(xi)
    gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=-1) + 1e-12)
    pen = jnp.mean((gnorm - 1.0) ** 2)
    return jnp.mean(lf) - jnp.mean(lr) + gp_weight * pen


def _induced_perf(x, cal_d, cal_n, stop_mask, pass_mask):
    """Merit of calibration designs under error vectors x (paired batch)."""
    dt, nt = apply_errors(cal_d, cal_n, x)
    dt = jnp.clip(dt, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
    nt = jnp.clip(nt, 1.40, 2.60)
    return tmm_jax.notch_merit_batch(dt, nt, stop_mask, pass_mask)


def gen_loss(gp, dp, c, z, xr=None, mm_w=5.0):
    xf = gen_forward(gp, z, c)
    adv = -jnp.mean(crit_forward(dp, xf, c))
    if xr is None:
        return adv
    # pooled first/second-moment matching (helps correlated error structure)
    mg, mr = jnp.mean(xf, 0), jnp.mean(xr, 0)
    cg = (xf - mg).T @ (xf - mg) / xf.shape[0]
    cr = (xr - mr).T @ (xr - mr) / xr.shape[0]
    mm = jnp.sum((mg - mr) ** 2) + jnp.sum((cg - cr) ** 2)
    return adv + mm_w * mm


def gen_loss_tail(gp, dp, c, z, xr, cal_d, cal_n, cal_c, cal_jnom, zc,
                  jr_sorted_low, stop_mask, pass_mask, tail_w):
    """Adversarial loss + physics-in-the-loop tail-calibration penalty.

    jr_sorted_low: sorted lower-quantile merits of the calibration designs
    under REAL process traces (precomputed, fixed target).  The generator's
    induced lower quantiles on the same designs are matched to them.
    """
    adv = gen_loss(gp, dp, c, z, xr)
    # zc: (Kc, DIM_Z) with Kc >> q; recipes tiled cyclically over the bank so
    # the generated tail fraction matches the real trace tail fraction.
    reps = zc.shape[0] // cal_c.shape[0]
    cal_c_t = jnp.tile(cal_c, (reps, 1))
    cal_d_t = jnp.tile(cal_d, (reps, 1))
    cal_n_t = jnp.tile(cal_n, (reps, 1))
    cal_jnom_t = jnp.tile(cal_jnom, (reps,))
    xg = gen_forward(gp, zc, cal_c_t)
    jg = _induced_perf(xg, cal_d_t, cal_n_t, stop_mask, pass_mask)
    # degradation relative to each design's nominal merit: isolates the
    # process tail from intrinsic design quality
    dj = jg - cal_jnom_t
    q = jr_sorted_low.shape[0]
    jg_low = jnp.sort(dj)[:q]
    tail = jnp.mean((jg_low - jr_sorted_low) ** 2)
    return adv + tail_w * tail, (adv, tail)


# ---------------- training ----------------
def train_fabgan(key, cond, xreal, cal_d, cal_n, cal_jnom, jr_low_sorted,
                 stop_mask, pass_mask, steps=4000, batch=128,
                 tail_w=0.0, n_critic=3, lr=1e-4, resume=None,
                 max_seconds=None):
    """Train conditional WGAN-GP; tail_w > 0 activates tail calibration.

    cond, xreal: (M, DIM_C), (M, DIM_X) training traces.
    cal_d, cal_n: (B_cal, N) calibration design bank (recipes).
    jr_low_sorted: (q,) sorted lower-quantile real-process merits of the bank.
    resume: optional state dict from a previous (interrupted) call.
    max_seconds: if set, returns a resumable state once the budget is hit.
    """
    import time as _time
    t_start = _time.time()
    optg = optax.adam(lr, b1=0.5, b2=0.9)
    optd = optax.adam(lr, b1=0.5, b2=0.9)
    if resume is None:
        key, k0 = jax.random.split(key)
        gp, dp = init_models(k0)
        sg, sd = optg.init(gp), optd.init(dp)
        t0 = 0
        hist = []
    else:
        gp, dp = resume["gp"], resume["dp"]
        sg, sd = resume["sg"], resume["sd"]
        key = resume["key"]
        t0 = resume["t"]
        hist = resume["hist"]
    cond = jnp.asarray(cond)
    xreal = jnp.asarray(xreal)
    cal_c = norm_recipe(jnp.asarray(cal_d), jnp.asarray(cal_n))
    M = cond.shape[0]

    @jax.jit
    def dstep(dp, sd, gp, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (batch,), 0, M)
        z = jax.random.normal(k2, (batch, DIM_Z))
        l, g = jax.value_and_grad(critic_loss)(dp, gp, xreal[idx], cond[idx], z, k3)
        up, sd = optd.update(g, sd)
        return optax.apply_updates(dp, up), sd, l

    @jax.jit
    def gstep_plain(gp, sg, dp, key):
        k1, k2 = jax.random.split(key)
        idx = jax.random.randint(k1, (batch,), 0, M)
        z = jax.random.normal(k2, (batch, DIM_Z))
        l, g = jax.value_and_grad(gen_loss)(gp, dp, cond[idx], z, xreal[idx])
        up, sg = optg.update(g, sg)
        return optax.apply_updates(gp, up), sg, l, 0.0

    @jax.jit
    def gstep_tail(gp, sg, dp, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (batch,), 0, M)
        z = jax.random.normal(k2, (batch, DIM_Z))
        zc = jax.random.normal(k3, (16 * cal_c.shape[0], DIM_Z))
        (l, (adv, tail)), g = jax.value_and_grad(
            gen_loss_tail, has_aux=True)(
            gp, dp, cond[idx], z, xreal[idx], jnp.asarray(cal_d),
            jnp.asarray(cal_n), cal_c, jnp.asarray(cal_jnom), zc,
            jnp.asarray(jr_low_sorted), stop_mask, pass_mask, tail_w)
        up, sg = optg.update(g, sg)
        return optax.apply_updates(gp, up), sg, adv, tail

    warmup = steps // 4        # adversarial-only warmup before tail term
    for t in range(t0, steps):
        for _ in range(n_critic):
            key, k = jax.random.split(key)
            dp, sd, dl = dstep(dp, sd, gp, k)
        key, k = jax.random.split(key)
        if tail_w > 0.0 and t >= warmup and t % 4 == 0:
            gp, sg, gl, tl = gstep_tail(gp, sg, dp, k)
        else:
            gp, sg, gl, tl = gstep_plain(gp, sg, dp, k)
        if t % 200 == 0:
            hist.append((t, float(dl), float(gl), float(tl)))
        if max_seconds is not None and _time.time() - t_start > max_seconds:
            return None, None, {"gp": gp, "dp": dp, "sg": sg, "sd": sd,
                                "key": key, "t": t + 1, "hist": hist,
                                "done": False}
    return gp, dp, {"gp": gp, "dp": dp, "hist": hist, "done": True,
                    "t": steps, "sg": sg, "sd": sd, "key": key}


def sample_fabgan(gp, key, d, n, K):
    """K fabricated realizations of recipe (d, n) from the twin."""
    c = norm_recipe(jnp.asarray(d), jnp.asarray(n))
    z = jax.random.normal(key, (K, DIM_Z))
    x = gen_forward(gp, z, jnp.tile(c, (K, 1)))
    dt, nt = apply_errors(jnp.asarray(d), jnp.asarray(n), x)
    return np.asarray(dt), np.asarray(nt)


# ---------------- Gaussian baselines ----------------
class GaussianTwin:
    """Parametric Gaussian corruption model fitted to the same traces."""

    def __init__(self, xreal, diagonal=True):
        self.mu = xreal.mean(axis=0)
        if diagonal:
            self.cov = np.diag(xreal.var(axis=0) + 1e-12)
        else:
            self.cov = np.cov(xreal.T) + 1e-9 * np.eye(xreal.shape[1])
        self.L = np.linalg.cholesky(self.cov)

    def sample(self, rng, d, n, K):
        x = self.mu + rng.normal(size=(K, DIM_X)) @ self.L.T
        dt = d * (1.0 + x[:, :N])
        nt = n + x[:, N:]
        return dt, nt
