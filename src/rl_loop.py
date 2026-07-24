"""Specification-conditioned robustification policy: chain-GCN soft actor-critic
trained in the virtual fabrication loop with the FabGAN process twin.

The multilayer stack is encoded as a chain device-graph (one node per layer,
edges between adjacent layers), the 1-D analog of the device-graph encoding in
PARL-ID: node features are (current layer params, nominal-optimum layer
params, specification), a two-layer GCN mean-pools to an embedding whose size
is independent of the layer count, and SAC learns bounded design corrections
that maximize the CVaR_alpha yield reward under twin-sampled corruptions.

The static-correction-transfer baseline (transplanting the correction vector
learned at one specification onto others) is evaluated against the policy's
zero-shot corrections at held-out specifications.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import fabgan, tmm_jax

jax.config.update("jax_enable_x64", True)

N = tmm_jax.N_LAYERS
FEAT = 5            # (d_cur, n_cur, d_star, n_star, spec)
EMB = 64
ACT_DIM = 2 * N
STEP_SCALE = 0.05   # bounded correction per step, fraction of design box
EP_LEN = 6
GAMMA = 0.9

# normalized path-graph adjacency with self loops: D^{-1/2}(A+I)D^{-1/2}
_A = np.eye(N)
for i in range(N - 1):
    _A[i, i + 1] = _A[i + 1, i] = 1.0
_Dm12 = np.diag(1.0 / np.sqrt(_A.sum(1)))
A_HAT = jnp.asarray(_Dm12 @ _A @ _Dm12)

T_RANGE = tmm_jax.T_HI - tmm_jax.T_LO
N_RANGE = tmm_jax.N_HI - tmm_jax.N_LO


def _norm_d(d):
    return (d - tmm_jax.T_LO) / T_RANGE * 2 - 1


def _norm_n(n):
    return (n - tmm_jax.N_LO) / N_RANGE * 2 - 1


def state_features(d_cur, n_cur, d_star, n_star, spec_norm):
    """(N, FEAT) node-feature matrix."""
    return jnp.stack([_norm_d(d_cur), _norm_n(n_cur),
                      _norm_d(d_star), _norm_n(n_star),
                      jnp.full((N,), spec_norm)], axis=-1)


# ---------------- networks ----------------
def _init_linear(key, i, o, scale=None):
    s = scale if scale is not None else jnp.sqrt(1.0 / i)
    return (jax.random.normal(key, (i, o)) * s, jnp.zeros(o))


def init_encoder(key):
    k1, k2 = jax.random.split(key)
    return {"w1": _init_linear(k1, FEAT, EMB), "w2": _init_linear(k2, EMB, EMB)}


def encode(enc, X):
    """X: (..., N, FEAT) -> (..., EMB) via 2-layer GCN + mean pool."""
    h = jax.nn.relu(A_HAT @ X @ enc["w1"][0] + enc["w1"][1])
    h = jax.nn.relu(A_HAT @ h @ enc["w2"][0] + enc["w2"][1])
    return h.mean(axis=-2)


def init_actor(key):
    k0, k1, k2, k3 = jax.random.split(key, 4)
    return {"enc": init_encoder(k0),
            "l1": _init_linear(k1, EMB, 128),
            "l2": _init_linear(k2, 128, 128),
            "out": _init_linear(k3, 128, 2 * ACT_DIM, scale=1e-2)}


def actor_dist(ap, X):
    e = encode(ap["enc"], X)
    h = jax.nn.relu(e @ ap["l1"][0] + ap["l1"][1])
    h = jax.nn.relu(h @ ap["l2"][0] + ap["l2"][1])
    o = h @ ap["out"][0] + ap["out"][1]
    mu, log_std = o[..., :ACT_DIM], jnp.clip(o[..., ACT_DIM:], -5.0, 1.0)
    return mu, log_std


def sample_action(ap, X, key):
    mu, log_std = actor_dist(ap, X)
    std = jnp.exp(log_std)
    z = mu + std * jax.random.normal(key, mu.shape)
    a = jnp.tanh(z)
    logp = jnp.sum(-0.5 * ((z - mu) / std) ** 2 - log_std
                   - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
    logp -= jnp.sum(jnp.log(1 - a ** 2 + 1e-6), axis=-1)
    return a, logp


def init_critic(key):
    k0, k1, k2, k3 = jax.random.split(key, 4)
    return {"enc": init_encoder(k0),
            "l1": _init_linear(k1, EMB + ACT_DIM, 128),
            "l2": _init_linear(k2, 128, 128),
            "out": _init_linear(k3, 128, 1)}


def q_value(cp, X, a):
    e = encode(cp["enc"], X)
    h = jnp.concatenate([e, a], axis=-1)
    h = jax.nn.relu(h @ cp["l1"][0] + cp["l1"][1])
    h = jax.nn.relu(h @ cp["l2"][0] + cp["l2"][1])
    return (h @ cp["out"][0] + cp["out"][1])[..., 0]


# ---------------- environment ----------------
def apply_action(d, n, a):
    d2 = d + STEP_SCALE * T_RANGE * a[:N]
    n2 = n + STEP_SCALE * N_RANGE * a[N:]
    return tmm_jax.clip_design(d2, n2)


def make_reward_fn(gp, K=48, alpha=0.05):
    """CVaR_alpha reward under FabGAN twin corruptions (jitted per spec masks)."""

    @functools.partial(jax.jit, static_argnums=())
    def reward(d, n, stop, pas, key):
        c = fabgan.norm_recipe(d, n)
        z = jax.random.normal(key, (K, fabgan.DIM_Z))
        x = fabgan.gen_forward(gp, z, jnp.tile(c, (K, 1)))
        dt, nt = fabgan.apply_errors(d, n, x)
        dt = jnp.clip(dt, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
        nt = jnp.clip(nt, 1.40, 2.60)
        J = tmm_jax.notch_merit_batch(dt, nt, stop, pas)
        q = max(1, int(np.ceil(alpha * K)))
        return jnp.mean(jnp.sort(J)[:q])

    return reward


# ---------------- SAC training ----------------
def train_sac(gp, specs, nominal, steps=4000, batch=128, buffer_cap=20000,
              lr=3e-4, seed=0, K=48, alpha=0.05, log_every=250,
              resume=None, max_seconds=None):
    """specs: list of (center_um, spec_norm); nominal: dict center->(d*,n*).

    Returns actor params and training log (resumable).
    """
    import time as _time
    t_start = _time.time()
    target_entropy = -float(ACT_DIM)
    opt_a = optax.adam(lr)
    opt_c = optax.adam(lr)
    opt_al = optax.adam(lr)
    key = jax.random.PRNGKey(seed)
    key, ka, kc1, kc2 = jax.random.split(key, 4)
    ap = init_actor(ka)
    c1, c2 = init_critic(kc1), init_critic(kc2)
    t1 = jax.tree.map(lambda x: x, c1)
    t2 = jax.tree.map(lambda x: x, c2)
    log_alpha = jnp.asarray(0.0)
    sa, sc1, sc2 = opt_a.init(ap), opt_c.init(c1), opt_c.init(c2)
    sal = opt_al.init(log_alpha)

    reward_fn = make_reward_fn(gp, K=K, alpha=alpha)
    masks = {c: tmm_jax.band_masks(c) for c, _ in specs}

    # replay buffer (numpy)
    buf_X = np.zeros((buffer_cap, N, FEAT))
    buf_a = np.zeros((buffer_cap, ACT_DIM))
    buf_r = np.zeros(buffer_cap)
    buf_X2 = np.zeros((buffer_cap, N, FEAT))
    buf_done = np.zeros(buffer_cap)
    ptr, size = 0, 0

    @jax.jit
    def critic_update(c1, c2, sc1, sc2, ap, t1, t2, log_alpha, X, a, r, X2,
                      done, key):
        a2, logp2 = sample_action(ap, X2, key)
        qt = jnp.minimum(q_value(t1, X2, a2), q_value(t2, X2, a2))
        y = r + GAMMA * (1 - done) * (qt - jnp.exp(log_alpha) * logp2)

        def lc(cp):
            return jnp.mean((q_value(cp, X, a) - y) ** 2)

        l1, g1 = jax.value_and_grad(lc)(c1)
        l2, g2 = jax.value_and_grad(lc)(c2)
        u1, sc1 = opt_c.update(g1, sc1)
        u2, sc2 = opt_c.update(g2, sc2)
        return (optax.apply_updates(c1, u1), optax.apply_updates(c2, u2),
                sc1, sc2, l1 + l2)

    @jax.jit
    def actor_update(ap, sa, c1, c2, log_alpha, sal, X, key):
        def la(ap):
            a, logp = sample_action(ap, X, key)
            q = jnp.minimum(q_value(c1, X, a), q_value(c2, X, a))
            return jnp.mean(jnp.exp(log_alpha) * logp - q), logp

        (l, logp), g = jax.value_and_grad(la, has_aux=True)(ap)
        up, sa = opt_a.update(g, sa)
        ap = optax.apply_updates(ap, up)

        def lal(la_):
            return -jnp.mean(jnp.exp(la_) * (jax.lax.stop_gradient(logp)
                                             + target_entropy))

        gl = jax.grad(lal)(log_alpha)
        ual, sal = opt_al.update(gl, sal)
        log_alpha = optax.apply_updates(log_alpha, ual)
        return ap, sa, log_alpha, sal, l

    rng = np.random.default_rng(seed)
    log = []
    ep_rewards = []
    env_steps = 0
    if resume is not None:
        (ap, c1, c2, t1, t2, log_alpha, sa, sc1, sc2, sal, buf_X, buf_a,
         buf_r, buf_X2, buf_done, ptr, size, env_steps, log,
         ep_rewards, key) = resume["blob"]
        rng = resume["rng"]
    while env_steps < steps:
        if max_seconds is not None and _time.time() - t_start > max_seconds:
            return None, None, {"blob": (ap, c1, c2, t1, t2, log_alpha, sa,
                                         sc1, sc2, sal, buf_X, buf_a, buf_r,
                                         buf_X2, buf_done, ptr, size,
                                         env_steps, log, ep_rewards, key),
                                "rng": rng, "done": False, "t": env_steps}
        cidx = rng.integers(0, len(specs))
        center, spec_norm = specs[cidx]
        stop, pas = masks[center]
        d_star, n_star = nominal[center]
        d_cur, n_cur = jnp.asarray(d_star), jnp.asarray(n_star)
        ep_r = 0.0
        for t in range(EP_LEN):
            X = state_features(d_cur, n_cur, jnp.asarray(d_star),
                               jnp.asarray(n_star), spec_norm)
            key, k1, k2 = jax.random.split(key, 3)
            if env_steps < 300:      # uniform exploration warmup
                a = jnp.asarray(rng.uniform(-1, 1, ACT_DIM))
            else:
                a, _ = sample_action(ap, X, k1)
            d2, n2 = apply_action(d_cur, n_cur, a)
            r = float(reward_fn(d2, n2, stop, pas, k2))
            X2 = state_features(d2, n2, jnp.asarray(d_star),
                                jnp.asarray(n_star), spec_norm)
            done = 1.0 if t == EP_LEN - 1 else 0.0
            buf_X[ptr], buf_a[ptr], buf_r[ptr] = np.asarray(X), np.asarray(a), r
            buf_X2[ptr], buf_done[ptr] = np.asarray(X2), done
            ptr = (ptr + 1) % buffer_cap
            size = min(size + 1, buffer_cap)
            d_cur, n_cur = d2, n2
            ep_r += r
            env_steps += 1

            if size >= 512:
                idx = rng.integers(0, size, batch)
                key, k3, k4 = jax.random.split(key, 3)
                c1, c2, sc1, sc2, lc = critic_update(
                    c1, c2, sc1, sc2, ap, t1, t2, log_alpha,
                    jnp.asarray(buf_X[idx]), jnp.asarray(buf_a[idx]),
                    jnp.asarray(buf_r[idx]), jnp.asarray(buf_X2[idx]),
                    jnp.asarray(buf_done[idx]), k3)
                ap, sa, log_alpha, sal, la_ = actor_update(
                    ap, sa, c1, c2, log_alpha, sal, jnp.asarray(buf_X[idx]), k4)
                t1 = jax.tree.map(lambda t, s: 0.995 * t + 0.005 * s, t1, c1)
                t2 = jax.tree.map(lambda t, s: 0.995 * t + 0.005 * s, t2, c2)

        ep_rewards.append(ep_r / EP_LEN)
        if len(ep_rewards) % log_every == 0:
            log.append((env_steps, float(np.mean(ep_rewards[-log_every:]))))
    return ap, log, {"done": True, "ep_rewards": ep_rewards, "t": env_steps}


def policy_correct(ap, d_star, n_star, spec_norm, key, n_steps=EP_LEN):
    """Zero-shot deterministic rollout of the learned policy."""
    d_cur, n_cur = jnp.asarray(d_star), jnp.asarray(n_star)
    for _ in range(n_steps):
        X = state_features(d_cur, n_cur, jnp.asarray(d_star),
                           jnp.asarray(n_star), spec_norm)
        mu, _ = actor_dist(ap, X)
        a = jnp.tanh(mu)
        d_cur, n_cur = apply_action(d_cur, n_cur, a)
    return np.asarray(d_cur), np.asarray(n_cur)


def spec_to_norm(center_um, lo=0.46, hi=0.64):
    return (center_um - lo) / (hi - lo) * 2 - 1


# ---------------- analytic policy gradients (differentiable fab loop) ------
def init_policy(key, hidden=128):
    """Specification-conditioned one-shot correction policy:
    chain-GCN encoder over (d*, n*, spec) node features + MLP head -> bounded
    correction Delta in [-1,1]^{2N} scaled by CORR_SCALE of the design box."""
    k0, k1, k2, k3 = jax.random.split(key, 4)
    enc = {"w1": _init_linear(k0, 3, EMB), "w2": _init_linear(k1, EMB, EMB)}
    return {"enc": enc,
            "l1": _init_linear(k2, EMB, hidden),
            "out": _init_linear(k3, hidden, ACT_DIM, scale=1e-3)}


CORR_SCALE = 0.15


def policy_delta(pp, d_star, n_star, spec_norm):
    X = jnp.stack([_norm_d(d_star), _norm_n(n_star),
                   jnp.full((N,), spec_norm)], axis=-1)
    h = jax.nn.relu(A_HAT @ X @ pp["enc"]["w1"][0] + pp["enc"]["w1"][1])
    h = jax.nn.relu(A_HAT @ h @ pp["enc"]["w2"][0] + pp["enc"]["w2"][1])
    e = h.mean(axis=-2)
    h2 = jax.nn.relu(e @ pp["l1"][0] + pp["l1"][1])
    a = jnp.tanh(h2 @ pp["out"][0] + pp["out"][1])
    dd = CORR_SCALE * T_RANGE * a[:N]
    dn = CORR_SCALE * N_RANGE * a[N:]
    return dd, dn


def apply_policy(pp, d_star, n_star, spec_norm):
    dd, dn = policy_delta(pp, jnp.asarray(d_star), jnp.asarray(n_star),
                          spec_norm)
    return tmm_jax.clip_design(jnp.asarray(d_star) + dd,
                               jnp.asarray(n_star) + dn)


def train_policy_apg(gp, specs, nominal, steps=400, K=128, alpha=0.05,
                     lr=1e-3, seed=0, resume=None, max_seconds=None):
    """Analytic policy gradients: the CVaR_alpha reward under FabGAN-twin
    corruptions is differentiated THROUGH the generator and the TMM solver
    back into the policy parameters (possible because every stage of the
    virtual fabrication loop is differentiable -- unlike FEM-based loops,
    which require model-free RL)."""
    import time as _time
    import optax as _optax
    t_start = _time.time()
    q = max(1, int(np.ceil(alpha * K)))
    masks = [tmm_jax.band_masks(c) for c, _ in specs]
    spec_norms = jnp.asarray([s for _, s in specs])
    Dstars = jnp.asarray(np.stack([nominal[c][0] for c, _ in specs]))
    Nstars = jnp.asarray(np.stack([nominal[c][1] for c, _ in specs]))
    stops = jnp.asarray(np.stack([np.asarray(m[0]) for m in masks]))
    passes = jnp.asarray(np.stack([np.asarray(m[1]) for m in masks]))

    opt = _optax.adam(lr)
    if resume is None:
        pp = init_policy(jax.random.PRNGKey(seed))
        st = opt.init(pp)
        t0, hist = 0, []
        key = jax.random.PRNGKey(seed + 1)
    else:
        pp, st, t0, hist, key = (resume["pp"], resume["st"], resume["t"],
                                 resume["hist"], resume["key"])

    from . import fabgan as _fabgan

    def spec_reward(pp, ds, ns, sn, stop, pas, z):
        d2, n2 = apply_policy(pp, ds, ns, sn)
        c = _fabgan.norm_recipe(d2, n2)
        x = _fabgan.gen_forward(gp, z, jnp.tile(c, (K, 1)))
        dt, nt = _fabgan.apply_errors(d2, n2, x)
        dt = jnp.clip(dt, 0.5 * tmm_jax.T_LO, 2.0 * tmm_jax.T_HI)
        nt = jnp.clip(nt, 1.40, 2.60)
        J = tmm_jax.notch_merit_batch(dt, nt, stop, pas)
        return jnp.mean(jnp.sort(J)[:q])

    def loss(pp, key):
        zs = jax.random.normal(key, (len(specs), K, _fabgan.DIM_Z))
        r = jax.vmap(spec_reward, in_axes=(None, 0, 0, 0, 0, 0, 0))(
            pp, Dstars, Nstars, spec_norms, stops, passes, zs)
        return -jnp.mean(r)

    vg = jax.jit(jax.value_and_grad(loss))
    for t in range(t0, steps):
        key, k = jax.random.split(key)
        l, g = vg(pp, k)
        up, st = opt.update(g, st)
        pp = optax.apply_updates(pp, up)
        if t % 20 == 0:
            hist.append((t, float(-l)))
        if max_seconds is not None and _time.time() - t_start > max_seconds:
            return None, {"pp": pp, "st": st, "t": t + 1, "hist": hist,
                          "key": key, "done": False}
    return pp, {"pp": pp, "hist": hist, "done": True, "t": steps, "key": key}
